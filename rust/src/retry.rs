use std::collections::HashSet;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use rand::Rng;
use tonic::Status;

/// Callback type invoked on every retryable error. Receives the host string.
pub type ThrottleCallback = Arc<dyn Fn(String) + Send + Sync>;

/// Configuration for retry behavior on gRPC calls.
#[derive(Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (0 = no retries, just the initial call).
    pub max_retries: u32,
    /// Initial backoff duration before the first retry.
    pub initial_backoff: Duration,
    /// Maximum backoff duration cap.
    pub max_backoff: Duration,
    /// Backoff multiplier (retained for API compatibility; no longer used in delay
    /// computation since decorrelated jitter was adopted in DX-0153).
    #[allow(dead_code)]
    pub multiplier: u32,
    /// gRPC status codes that trigger a retry. Defaults to UNAVAILABLE, RESOURCE_EXHAUSTED,
    /// ABORTED — Pinecone data-plane operations (upsert, query, fetch, delete-by-id, update)
    /// are idempotent and safe to retry on these transient codes.
    pub retryable_codes: HashSet<i32>,
    /// Optional callback invoked with the host string on every retryable error.
    /// Receives the host string (e.g. "my-index-abc123.svc.pinecone.io") on each
    /// retryable failure so the caller can update per-host rate-limit state.
    /// In transport.rs this wraps a Python ``Py<PyAny>`` callable under `Python::with_gil`.
    pub on_throttle: Option<ThrottleCallback>,
    /// Host string passed to `on_throttle` callback (parsed from endpoint at construction).
    pub host: String,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 5,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_millis(1600),
            multiplier: 2,
            retryable_codes: [
                tonic::Code::Unavailable as i32,
                tonic::Code::ResourceExhausted as i32,
                tonic::Code::Aborted as i32,
            ]
            .into_iter()
            .collect(),
            on_throttle: None,
            host: String::new(),
        }
    }
}

/// Parse a server-supplied retry pushback hint from a tonic Status's trailers.
///
/// Looks for `grpc-retry-pushback-ms` first (gRPC-native, milliseconds), then
/// `retry-after` (HTTP-style, seconds) as a fallback. Returns `None` if neither
/// is present or the value cannot be parsed as a non-negative number.
///
/// HTTP-date values in `retry-after` are not parsed and return `None`.
pub fn parse_pushback(status: &Status) -> Option<Duration> {
    let metadata = status.metadata();

    // gRPC-native: milliseconds
    if let Some(v) = metadata.get("grpc-retry-pushback-ms") {
        if let Ok(s) = v.to_str() {
            if let Ok(ms) = s.trim().parse::<f64>() {
                if ms >= 0.0 && ms.is_finite() {
                    return Some(Duration::from_millis(ms as u64));
                }
            }
        }
    }

    // HTTP fallback: seconds (delta-seconds form only; HTTP-date returns None)
    if let Some(v) = metadata.get("retry-after") {
        if let Ok(s) = v.to_str() {
            if let Ok(secs) = s.trim().parse::<f64>() {
                if secs >= 0.0 && secs.is_finite() {
                    return Some(Duration::from_secs_f64(secs));
                }
            }
        }
    }

    None
}

/// Add a uniform smear on top of the server-supplied pushback so concurrent
/// clients don't all wake at the same instant.
///
/// Returns `min(max_backoff, pushback + uniform(0, pushback * 0.5))`.
fn smear_pushback(pushback: Duration, max_backoff: Duration) -> Duration {
    let pb_ms = pushback.as_millis() as u64;
    let smear_max = pb_ms / 2;
    let smear_ms = if smear_max == 0 {
        0
    } else {
        rand::rng().random_range(0..=smear_max)
    };
    let total = Duration::from_millis(pb_ms.saturating_add(smear_ms));
    std::cmp::min(total, max_backoff)
}

/// Decorrelated jitter (AWS-recommended pattern): uniform(base, prev*3)
/// capped at max_backoff. Less self-correlation across retries than plain
/// full jitter, which spreads fleet retries better.
fn decorrelated_jitter(base: Duration, prev_delay: Duration, max_backoff: Duration) -> Duration {
    let base_ms = base.as_millis() as u64;
    let upper_unbounded = prev_delay
        .as_millis()
        .saturating_mul(3)
        .min(u64::MAX as u128) as u64;
    let upper_capped = std::cmp::min(upper_unbounded, max_backoff.as_millis() as u64);
    let upper = std::cmp::max(base_ms, upper_capped); // guard against base > cap misconfig
    if upper == base_ms {
        return Duration::from_millis(base_ms);
    }
    let ms = rand::rng().random_range(base_ms..=upper);
    Duration::from_millis(ms)
}

/// Execute an async gRPC operation with retry on transient error codes.
///
/// Uses decorrelated jitter on the backoff path and smear on server-supplied
/// pushback hints (`grpc-retry-pushback-ms` / `retry-after`).
///
/// Retries on any code listed in `config.retryable_codes` (default: UNAVAILABLE,
/// RESOURCE_EXHAUSTED, ABORTED). All other error codes are returned immediately without retry.
pub async fn retry_on_transient<F, Fut, T>(
    config: &RetryConfig,
    mut operation: F,
) -> Result<T, Status>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, Status>>,
{
    let mut attempt = 0u32;
    let mut prev_delay = config.initial_backoff;

    loop {
        match operation().await {
            Ok(val) => return Ok(val),
            Err(status) if config.retryable_codes.contains(&(status.code() as i32)) => {
                if let Some(cb) = &config.on_throttle {
                    cb(config.host.clone());
                }
                if attempt >= config.max_retries {
                    return Err(status);
                }
                let delay = if let Some(pushback) = parse_pushback(&status) {
                    smear_pushback(pushback, config.max_backoff)
                } else {
                    decorrelated_jitter(config.initial_backoff, prev_delay, config.max_backoff)
                };
                tracing::debug!("retry attempt {} sleeping for {:?}", attempt + 1, delay);
                prev_delay = delay;
                tokio::time::sleep(delay).await;
                attempt += 1;
            }
            Err(status) => return Err(status),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    fn test_config(max_retries: u32) -> RetryConfig {
        RetryConfig {
            max_retries,
            initial_backoff: Duration::from_millis(1), // fast for tests
            max_backoff: Duration::from_millis(10),
            multiplier: 2,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn retry_occurs_on_unavailable() {
        let call_count = Arc::new(AtomicU32::new(0));
        let count = call_count.clone();

        let config = test_config(5);
        let result = retry_on_transient(&config, || {
            let count = count.clone();
            async move {
                let n = count.fetch_add(1, Ordering::SeqCst);
                if n < 2 {
                    Err(Status::unavailable("service unavailable"))
                } else {
                    Ok::<&str, Status>("success")
                }
            }
        })
        .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
        // Initial call + 2 retries = 3 total calls
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn no_retry_on_deadline_exceeded() {
        let call_count = Arc::new(AtomicU32::new(0));
        let count = call_count.clone();

        let config = test_config(5);
        let result = retry_on_transient(&config, || {
            let count = count.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Err::<(), Status>(Status::deadline_exceeded("timeout"))
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::DeadlineExceeded);
        // Only 1 call, no retries
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn retry_occurs_on_resource_exhausted() {
        let call_count = Arc::new(AtomicU32::new(0));
        let count = call_count.clone();

        let config = test_config(5);
        let result = retry_on_transient(&config, || {
            let count = count.clone();
            async move {
                let n = count.fetch_add(1, Ordering::SeqCst);
                if n < 2 {
                    Err(Status::resource_exhausted("rate limited"))
                } else {
                    Ok::<(), Status>(())
                }
            }
        })
        .await;

        assert!(result.is_ok());
        // Initial call + 2 retries = 3 total calls
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn retry_occurs_on_aborted() {
        let call_count = Arc::new(AtomicU32::new(0));
        let count = call_count.clone();

        let config = test_config(5);
        let result = retry_on_transient(&config, || {
            let count = count.clone();
            async move {
                let n = count.fetch_add(1, Ordering::SeqCst);
                if n < 2 {
                    Err(Status::aborted("conflict"))
                } else {
                    Ok::<(), Status>(())
                }
            }
        })
        .await;

        assert!(result.is_ok());
        // Initial call + 2 retries = 3 total calls
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn no_retry_on_internal() {
        let call_count = Arc::new(AtomicU32::new(0));
        let count = call_count.clone();

        let config = test_config(5);
        let result = retry_on_transient(&config, || {
            let count = count.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Err::<(), Status>(Status::internal("oops"))
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::Internal);
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn custom_retryable_codes_override() {
        let call_count = Arc::new(AtomicU32::new(0));
        let count = call_count.clone();

        let config = RetryConfig {
            max_retries: 3,
            initial_backoff: Duration::from_millis(1),
            max_backoff: Duration::from_millis(10),
            multiplier: 2,
            retryable_codes: HashSet::from([tonic::Code::DeadlineExceeded as i32]),
            ..RetryConfig::default()
        };
        let result = retry_on_transient(&config, || {
            let count = count.clone();
            async move {
                let n = count.fetch_add(1, Ordering::SeqCst);
                if n < 2 {
                    Err(Status::deadline_exceeded("timeout"))
                } else {
                    Ok::<(), Status>(())
                }
            }
        })
        .await;

        assert!(result.is_ok());
        // DEADLINE_EXCEEDED is retried under this custom config
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn respects_max_retry_count() {
        let call_count = Arc::new(AtomicU32::new(0));
        let count = call_count.clone();

        let config = test_config(3);
        let result = retry_on_transient(&config, || {
            let count = count.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Err::<(), Status>(Status::unavailable("always unavailable"))
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::Unavailable);
        // 1 initial + 3 retries = 4 total calls
        assert_eq!(call_count.load(Ordering::SeqCst), 4);
    }

    #[tokio::test]
    async fn zero_retries_means_no_retry() {
        let call_count = Arc::new(AtomicU32::new(0));
        let count = call_count.clone();

        let config = test_config(0);
        let result = retry_on_transient(&config, || {
            let count = count.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Err::<(), Status>(Status::unavailable("unavailable"))
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn backoff_delay_increases() {
        // Verify that successive retries take progressively longer.
        // We measure wall-clock time for configs with different retry counts.
        let config_1 = RetryConfig {
            max_retries: 1,
            initial_backoff: Duration::from_millis(10),
            max_backoff: Duration::from_millis(500),
            multiplier: 2,
            ..Default::default()
        };
        let config_3 = RetryConfig {
            max_retries: 3,
            initial_backoff: Duration::from_millis(10),
            max_backoff: Duration::from_millis(500),
            multiplier: 2,
            ..Default::default()
        };

        let start_1 = std::time::Instant::now();
        let _ = retry_on_transient(&config_1, || async {
            Err::<(), Status>(Status::unavailable("unavailable"))
        })
        .await;
        let elapsed_1 = start_1.elapsed();

        let start_3 = std::time::Instant::now();
        let _ = retry_on_transient(&config_3, || async {
            Err::<(), Status>(Status::unavailable("unavailable"))
        })
        .await;
        let elapsed_3 = start_3.elapsed();

        // 3 retries should take longer than 1 retry due to increasing backoff
        assert!(
            elapsed_3 > elapsed_1,
            "3 retries ({elapsed_3:?}) should take longer than 1 retry ({elapsed_1:?})"
        );
    }

    #[tokio::test]
    async fn success_on_first_attempt_returns_immediately() {
        let config = test_config(5);
        let result =
            retry_on_transient(&config, || async { Ok::<&str, Status>("immediate") }).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "immediate");
    }

    #[test]
    fn parse_pushback_grpc_native_milliseconds() {
        let mut status = Status::resource_exhausted("limited");
        status
            .metadata_mut()
            .insert("grpc-retry-pushback-ms", "1500".parse().unwrap());
        assert_eq!(parse_pushback(&status), Some(Duration::from_millis(1500)));
    }

    #[test]
    fn parse_pushback_retry_after_seconds_fallback() {
        let mut status = Status::resource_exhausted("limited");
        status
            .metadata_mut()
            .insert("retry-after", "30".parse().unwrap());
        assert_eq!(parse_pushback(&status), Some(Duration::from_secs(30)));
    }

    #[test]
    fn parse_pushback_grpc_native_takes_precedence_over_retry_after() {
        let mut status = Status::resource_exhausted("limited");
        status
            .metadata_mut()
            .insert("grpc-retry-pushback-ms", "500".parse().unwrap());
        status
            .metadata_mut()
            .insert("retry-after", "30".parse().unwrap());
        assert_eq!(parse_pushback(&status), Some(Duration::from_millis(500)));
    }

    #[test]
    fn parse_pushback_returns_none_when_absent() {
        let status = Status::resource_exhausted("limited");
        assert_eq!(parse_pushback(&status), None);
    }

    #[test]
    fn parse_pushback_returns_none_for_negative_value() {
        let mut status = Status::resource_exhausted("limited");
        status
            .metadata_mut()
            .insert("grpc-retry-pushback-ms", "-100".parse().unwrap());
        assert_eq!(parse_pushback(&status), None);
    }

    #[test]
    fn parse_pushback_returns_none_for_http_date() {
        let mut status = Status::resource_exhausted("limited");
        status.metadata_mut().insert(
            "retry-after",
            "Fri, 31 Dec 2026 23:59:59 GMT".parse().unwrap(),
        );
        assert_eq!(parse_pushback(&status), None);
    }

    #[test]
    fn parse_pushback_handles_float_milliseconds() {
        let mut status = Status::resource_exhausted("limited");
        status
            .metadata_mut()
            .insert("grpc-retry-pushback-ms", "1500.5".parse().unwrap());
        // Truncates to 1500 ms (Duration::from_millis takes u64)
        assert_eq!(parse_pushback(&status), Some(Duration::from_millis(1500)));
    }

    #[test]
    fn parse_pushback_returns_none_for_nan() {
        let mut status = Status::resource_exhausted("limited");
        status
            .metadata_mut()
            .insert("grpc-retry-pushback-ms", "nan".parse().unwrap());
        assert_eq!(parse_pushback(&status), None);
    }

    #[test]
    fn parse_pushback_returns_none_for_invalid_string() {
        let mut status = Status::resource_exhausted("limited");
        status
            .metadata_mut()
            .insert("grpc-retry-pushback-ms", "not-a-number".parse().unwrap());
        assert_eq!(parse_pushback(&status), None);
    }

    #[tokio::test]
    async fn pushback_honored_on_resource_exhausted() {
        let call_count = Arc::new(AtomicU32::new(0));
        let count = call_count.clone();

        let config = test_config(5);
        let result = retry_on_transient(&config, || {
            let count = count.clone();
            async move {
                let n = count.fetch_add(1, Ordering::SeqCst);
                if n < 2 {
                    let mut s = Status::resource_exhausted("limited");
                    // 1ms pushback so the test runs fast
                    s.metadata_mut()
                        .insert("grpc-retry-pushback-ms", "1".parse().unwrap());
                    Err(s)
                } else {
                    Ok::<&str, Status>("ok")
                }
            }
        })
        .await;
        assert!(result.is_ok());
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn pushback_capped_at_max_backoff() {
        // Build a config with a tight cap and a giant pushback; assert the call returns quickly.
        let config = RetryConfig {
            max_retries: 1,
            initial_backoff: Duration::from_millis(1),
            max_backoff: Duration::from_millis(5),
            multiplier: 2,
            ..Default::default()
        };
        let start = std::time::Instant::now();
        let _ = retry_on_transient(&config, || async {
            let mut s = Status::resource_exhausted("limited");
            // 1 hour in milliseconds — would block forever if not capped
            s.metadata_mut()
                .insert("grpc-retry-pushback-ms", "3600000".parse().unwrap());
            Err::<(), Status>(s)
        })
        .await;
        // With max_backoff=5ms, the single retry sleeps at most 5ms;
        // total elapsed must be well under 100ms.
        assert!(
            start.elapsed() < Duration::from_millis(100),
            "elapsed={:?} — pushback not capped",
            start.elapsed()
        );
    }

    #[tokio::test]
    async fn no_pushback_falls_back_to_jitter() {
        // No pushback header → uses jitter backoff (existing behavior).
        let call_count = Arc::new(AtomicU32::new(0));
        let count = call_count.clone();
        let config = test_config(3);
        let result = retry_on_transient(&config, || {
            let count = count.clone();
            async move {
                let n = count.fetch_add(1, Ordering::SeqCst);
                if n < 1 {
                    Err(Status::resource_exhausted("limited"))
                } else {
                    Ok::<(), Status>(())
                }
            }
        })
        .await;
        assert!(result.is_ok());
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn on_throttle_defaults_to_none() {
        let config = RetryConfig::default();
        assert!(config.on_throttle.is_none());
    }

    #[test]
    fn host_defaults_to_empty_string() {
        let config = RetryConfig::default();
        assert_eq!(config.host, "");
    }

    #[test]
    fn host_can_be_set_via_struct_literal() {
        let config = RetryConfig {
            host: "my-index.svc.pinecone.io".into(),
            on_throttle: None,
            ..RetryConfig::default()
        };
        assert_eq!(config.host, "my-index.svc.pinecone.io");
        assert!(config.on_throttle.is_none());
    }

    #[tokio::test]
    async fn callback_invoked_on_every_retryable_error() {
        // Verify callback fires for every retryable error, including the final one
        // that exceeds max_retries. Uses a pure Rust closure — the Python wrapper
        // in transport.rs is tested via Python integration tests.
        let call_count = Arc::new(AtomicU32::new(0));
        let received_hosts: Arc<std::sync::Mutex<Vec<String>>> =
            Arc::new(std::sync::Mutex::new(vec![]));
        let count_clone = call_count.clone();
        let hosts_clone = received_hosts.clone();
        let config = RetryConfig {
            max_retries: 2,
            initial_backoff: Duration::from_millis(1),
            max_backoff: Duration::from_millis(5),
            on_throttle: Some(Arc::new(move |host: String| {
                count_clone.fetch_add(1, Ordering::SeqCst);
                hosts_clone.lock().unwrap().push(host);
            })),
            host: "my-index.svc.pinecone.io".into(),
            ..RetryConfig::default()
        };
        let _ = retry_on_transient(&config, || async {
            Err::<(), Status>(Status::resource_exhausted("limited"))
        })
        .await;
        // 1 initial + 2 retries = 3 attempts; callback fires on all 3
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
        let hosts = received_hosts.lock().unwrap();
        assert!(hosts.iter().all(|h| h == "my-index.svc.pinecone.io"));
    }

    #[tokio::test]
    async fn callback_not_invoked_on_non_retryable_error() {
        let call_count = Arc::new(AtomicU32::new(0));
        let count_clone = call_count.clone();
        let config = RetryConfig {
            max_retries: 3,
            initial_backoff: Duration::from_millis(1),
            max_backoff: Duration::from_millis(5),
            on_throttle: Some(Arc::new(move |_host: String| {
                count_clone.fetch_add(1, Ordering::SeqCst);
            })),
            host: "test-host".into(),
            ..RetryConfig::default()
        };
        let _ = retry_on_transient(&config, || async {
            Err::<(), Status>(Status::not_found("missing"))
        })
        .await;
        // NOT_FOUND is not retryable → callback never fires
        assert_eq!(call_count.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn smear_pushback_within_expected_range() {
        let max = Duration::from_secs(120);
        let pushback = Duration::from_secs(60);
        for _ in 0..200 {
            let d = smear_pushback(pushback, max);
            assert!(d >= pushback, "delay {d:?} < pushback {pushback:?}");
            assert!(d <= Duration::from_millis(60_000 + 30_000));
        }
    }

    #[test]
    fn smear_pushback_capped_at_max_backoff() {
        let max = Duration::from_millis(70_000);
        let pushback = Duration::from_secs(60);
        for _ in 0..200 {
            let d = smear_pushback(pushback, max);
            assert!(d <= max);
        }
    }

    #[test]
    fn smear_pushback_zero_pushback_is_zero() {
        assert_eq!(
            smear_pushback(Duration::ZERO, Duration::from_secs(60)),
            Duration::ZERO
        );
    }

    #[test]
    fn decorrelated_jitter_within_expected_range() {
        let base = Duration::from_millis(100);
        let prev = Duration::from_millis(500);
        let max = Duration::from_secs(60);
        for _ in 0..200 {
            let d = decorrelated_jitter(base, prev, max);
            assert!(d >= base);
            assert!(d <= Duration::from_millis(1500)); // 500 * 3
        }
    }

    #[test]
    fn decorrelated_jitter_capped_at_max_backoff() {
        let base = Duration::from_millis(100);
        let prev = Duration::from_secs(1000); // would push upper to 3000s
        let max = Duration::from_secs(5);
        for _ in 0..200 {
            let d = decorrelated_jitter(base, prev, max);
            assert!(d <= max);
        }
    }

    #[test]
    fn decorrelated_jitter_non_degenerate() {
        let base = Duration::from_millis(100);
        let prev = Duration::from_millis(500);
        let max = Duration::from_secs(60);
        let samples: Vec<u64> = (0..200)
            .map(|_| decorrelated_jitter(base, prev, max).as_millis() as u64)
            .collect();
        let unique: std::collections::HashSet<_> = samples.iter().collect();
        // Should have many distinct values across 200 samples
        assert!(
            unique.len() > 50,
            "samples too clustered: {} unique",
            unique.len()
        );
    }
}
