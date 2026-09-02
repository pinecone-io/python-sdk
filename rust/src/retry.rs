use std::collections::HashSet;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rand::Rng;
use tonic::Status;

/// Callback type invoked on every retryable error. Receives the host string.
pub type ThrottleCallback = Arc<dyn Fn(String) + Send + Sync>;

/// Token bucket bounding how much *extra* load retries may add, per gRFC A6.
///
/// The adaptive limiter on the Python side bounds how many batches are in
/// flight; nothing bounded how many attempts each one costs. During a partial
/// outage — say 20% of calls failing — a flat 6x multiplier means a client sends
/// 2x its baseline volume at exactly the moment the backend has lost capacity.
///
/// Every failure spends a token; every success returns `token_ratio` of one.
/// While the bucket is below half, retries are suppressed and calls fail fast on
/// their first attempt, so a total outage self-limits to one request per call
/// instead of `max_retries + 1`.
pub struct RetryBudget {
    tokens: Mutex<f64>,
    max_tokens: f64,
    token_ratio: f64,
}

impl RetryBudget {
    pub fn new(max_tokens: f64, token_ratio: f64) -> Self {
        Self {
            tokens: Mutex::new(max_tokens),
            max_tokens,
            token_ratio,
        }
    }

    /// Spend a token for a failed attempt. Returns whether a retry is affordable.
    fn withdraw(&self) -> bool {
        let mut tokens = self.tokens.lock().unwrap_or_else(|e| e.into_inner());
        *tokens = (*tokens - 1.0).max(0.0);
        *tokens > self.max_tokens / 2.0
    }

    /// Return a fraction of a token for a successful call.
    fn deposit(&self) {
        let mut tokens = self.tokens.lock().unwrap_or_else(|e| e.into_inner());
        *tokens = (*tokens + self.token_ratio).min(self.max_tokens);
    }
}

impl Default for RetryBudget {
    fn default() -> Self {
        Self::new(DEFAULT_BUDGET_TOKENS, DEFAULT_BUDGET_RATIO)
    }
}

/// Bucket size. Large enough that a short burst of failures on an otherwise
/// healthy channel never suppresses retries, small enough to react within a few
/// hundred calls.
const DEFAULT_BUDGET_TOKENS: f64 = 100.0;

/// Sustained retry overhead permitted once the bucket has drained: 10% of
/// successful traffic, which is the gRFC A6 default.
const DEFAULT_BUDGET_RATIO: f64 = 0.1;

/// Multiple of `initial_backoff` seeding the decorrelated-jitter window.
///
/// Seeding it at `initial_backoff` makes the first retry `uniform(base, 3*base)`
/// — with a 100ms base that is a 200ms window, identical for every client in a
/// fleet. A backend restart returns UNAVAILABLE to everyone at once and carries
/// no trailers, so that narrow window is exactly where a thundering herd forms.
/// Nothing useful recovers in 100ms anyway, so the wider first draw costs a
/// caller little and buys the fleet real dispersion.
const FIRST_RETRY_SPREAD: u32 = 10;

/// Configuration for retry behavior on gRPC calls.
#[derive(Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (0 = no retries, just the initial call).
    pub max_retries: u32,
    /// Initial backoff duration before the first retry.
    pub initial_backoff: Duration,
    /// Maximum backoff duration cap. Bounds both the jitter path and a
    /// server-supplied pushback hint, so it must be large enough to honor a
    /// realistic `grpc-retry-pushback-ms` value.
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
    /// In transport.rs this wraps a Python ``Py<PyAny>`` callable under `Python::attach`.
    pub on_throttle: Option<ThrottleCallback>,
    /// Host string passed to `on_throttle` callback (parsed from endpoint at construction).
    pub host: String,
    /// Shared per-channel retry budget. `None` disables budgeting entirely.
    pub budget: Option<Arc<RetryBudget>>,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 5,
            initial_backoff: Duration::from_millis(100),
            // Matches REST's `RetryConfig.max_wait`. A 1600ms cap silently swallowed
            // server pushback: a `grpc-retry-pushback-ms: 30000` hint was clamped to
            // 1.6s, so we parsed an explicit instruction from the server and then
            // hammered it anyway.
            max_backoff: Duration::from_secs(60),
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
            budget: Some(Arc::new(RetryBudget::default())),
        }
    }
}

/// What a server-supplied pushback hint tells the client to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pushback {
    /// Wait at least this long before retrying.
    Wait(Duration),
    /// The server is shedding load and does not want this call retried at all.
    Stop,
}

/// Parse a server-supplied retry pushback hint from a tonic Status's trailers.
///
/// Looks for `grpc-retry-pushback-ms` first (gRPC-native, milliseconds), then
/// `retry-after` (HTTP-style, seconds) as a fallback. Returns `None` if neither
/// is present or the value cannot be parsed as a number.
///
/// A negative value is `Pushback::Stop`: per gRFC A6 that is the server saying
/// do not retry. Treating it as "no hint" would mean retrying against an
/// explicit instruction from a backend that is trying to shed load.
///
/// HTTP-date values in `retry-after` are not parsed and return `None`.
pub fn parse_pushback(status: &Status) -> Option<Pushback> {
    let metadata = status.metadata();

    // gRPC-native: milliseconds
    if let Some(v) = metadata.get("grpc-retry-pushback-ms") {
        if let Ok(s) = v.to_str() {
            if let Ok(ms) = s.trim().parse::<f64>() {
                if ms.is_finite() {
                    return Some(if ms < 0.0 {
                        Pushback::Stop
                    } else {
                        Pushback::Wait(Duration::from_millis(ms as u64))
                    });
                }
            }
        }
    }

    // HTTP fallback: seconds (delta-seconds form only; HTTP-date returns None)
    if let Some(v) = metadata.get("retry-after") {
        if let Ok(s) = v.to_str() {
            if let Ok(secs) = s.trim().parse::<f64>() {
                if secs.is_finite() {
                    return Some(if secs < 0.0 {
                        Pushback::Stop
                    } else {
                        Pushback::Wait(Duration::from_secs_f64(secs))
                    });
                }
            }
        }
    }

    None
}

/// Add a uniform smear on top of the server-supplied pushback so concurrent
/// clients don't all wake at the same instant.
///
/// Returns `base + uniform(0, max(base / 2, floor))` where `base` is the
/// pushback clamped to `max_backoff`.
///
/// The clamp happens *before* the smear, matching the REST transport. Smearing
/// first and truncating afterwards collapses part of the distribution onto
/// exactly `max_backoff` — with a 60s cap, a 50s pushback puts 60% of a fleet on
/// the same millisecond and a pushback at or above the cap puts all of it there,
/// on the one code path whose whole purpose is dispersal.
///
/// `floor` keeps a zero pushback ("retry immediately", per gRFC A6) from
/// producing a zero-width, perfectly synchronized wave.
fn smear_pushback(pushback: Duration, max_backoff: Duration, floor: Duration) -> Duration {
    let base_ms = std::cmp::min(pushback, max_backoff).as_millis() as u64;
    let smear_max = std::cmp::max(base_ms / 2, floor.as_millis() as u64);
    let smear_ms = if smear_max == 0 {
        0
    } else {
        rand::rng().random_range(0..=smear_max)
    };
    Duration::from_millis(base_ms.saturating_add(smear_ms))
}

/// Decorrelated jitter (AWS-recommended pattern): uniform(base, prev*3)
/// capped at max_backoff. Less self-correlation across retries than plain
/// full jitter, which spreads fleet retries better.
fn decorrelated_jitter(base: Duration, prev_delay: Duration, max_backoff: Duration) -> Duration {
    let (base_ms, upper) = jitter_window(base, prev_delay, max_backoff);
    if upper == base_ms {
        return Duration::from_millis(base_ms);
    }
    let ms = rand::rng().random_range(base_ms..=upper);
    Duration::from_millis(ms)
}

/// The inclusive `[lower, upper]` millisecond window retry `n` draws from:
/// `base` up to three times the delay retry `n - 1` actually slept, capped at
/// `max_backoff`. Split out from [`decorrelated_jitter`] so the escalation rule
/// is assertable without timing a retry loop against the wall clock.
fn jitter_window(base: Duration, prev_delay: Duration, max_backoff: Duration) -> (u64, u64) {
    let base_ms = base.as_millis() as u64;
    let upper_unbounded = prev_delay
        .as_millis()
        .saturating_mul(3)
        .min(u64::MAX as u128) as u64;
    let upper_capped = std::cmp::min(upper_unbounded, max_backoff.as_millis() as u64);
    let upper = std::cmp::max(base_ms, upper_capped); // guard against base > cap misconfig
    (base_ms, upper)
}

/// Execute an async gRPC operation with retry on transient error codes.
///
/// Uses decorrelated jitter on the backoff path and smear on server-supplied
/// pushback hints (`grpc-retry-pushback-ms` / `retry-after`).
///
/// Retries on any code listed in `config.retryable_codes` (default: UNAVAILABLE,
/// RESOURCE_EXHAUSTED, ABORTED). All other error codes are returned immediately without retry.
// result_large_err: `Status` is the error the generated clients hand us and that
// `status_to_py_err` consumes; boxing here would only add wrap/unwrap at every call site.
#[allow(clippy::result_large_err)]
pub async fn retry_on_transient<F, Fut, T>(
    config: &RetryConfig,
    mut operation: F,
) -> Result<T, Status>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, Status>>,
{
    retry_on_transient_request(config, (), |()| operation()).await
}

/// [`retry_on_transient`] for operations that consume a request.
///
/// tonic's generated clients take the request by value, so every attempt needs
/// its own copy. Handing the closure `request.clone()` on each iteration copies
/// the full proto payload once per attempt — at 500 x 1536 f32 that is ~3 MB of
/// memcpy per batch. This clones only while another attempt is still possible
/// and gives the last one the original by value, so `n` attempts cost `n - 1`
/// clones instead of `n`, and a `max_retries: 0` config costs none at all.
///
/// Removing the remaining clone would mean not owning the payload per attempt —
/// a shared or pre-encoded representation the generated client cannot take.
// result_large_err: see `retry_on_transient` — same `tonic::Status` pass-through.
#[allow(clippy::result_large_err)]
pub async fn retry_on_transient_request<F, Fut, T, R>(
    config: &RetryConfig,
    request: R,
    mut operation: F,
) -> Result<T, Status>
where
    R: Clone,
    F: FnMut(R) -> Fut,
    Fut: Future<Output = Result<T, Status>>,
{
    let mut attempt = 0u32;
    let mut prev_delay = config.initial_backoff * FIRST_RETRY_SPREAD;
    let mut pending = request;

    loop {
        // `spare` is None exactly when the retry budget is spent, which is also
        // the attempt that gets the original rather than a copy.
        let (payload, spare) = if attempt < config.max_retries {
            (pending.clone(), Some(pending))
        } else {
            (pending, None)
        };

        match operation(payload).await {
            Ok(val) => {
                if let Some(budget) = &config.budget {
                    budget.deposit();
                }
                return Ok(val);
            }
            Err(status) if config.retryable_codes.contains(&(status.code() as i32)) => {
                if let Some(cb) = &config.on_throttle {
                    cb(config.host.clone());
                }
                let affordable = config.budget.as_ref().is_none_or(|b| b.withdraw());
                let Some(next) = spare else {
                    return Err(status);
                };
                if !affordable {
                    tracing::debug!("retry budget exhausted; failing fast");
                    return Err(status);
                }
                let delay = match parse_pushback(&status) {
                    Some(Pushback::Stop) => {
                        tracing::debug!("server asked us not to retry; failing fast");
                        return Err(status);
                    }
                    Some(Pushback::Wait(pushback)) => {
                        smear_pushback(pushback, config.max_backoff, config.initial_backoff)
                    }
                    None => {
                        decorrelated_jitter(config.initial_backoff, prev_delay, config.max_backoff)
                    }
                };
                tracing::debug!("retry attempt {} sleeping for {:?}", attempt + 1, delay);
                prev_delay = delay;
                tokio::time::sleep(delay).await;
                pending = next;
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

    /// Replaces a wall-clock comparison of a 3-retry run against a 1-retry run.
    /// That inverted on a noisy CI runner, and decorrelated jitter does not
    /// actually guarantee it: three draws from `[base, 3*prev]` can total less
    /// than one draw from the same seeded window. What the schedule does
    /// guarantee is that each retry's window ceiling is three times the delay
    /// just slept, which is what this asserts.
    #[test]
    fn backoff_window_escalates_from_each_delay() {
        let base = Duration::from_millis(10);
        let max_backoff = Duration::from_secs(60);
        let mut prev = base * FIRST_RETRY_SPREAD;

        for retry in 1..=5u32 {
            let (lower, upper) = jitter_window(base, prev, max_backoff);
            assert_eq!(lower, base.as_millis() as u64, "retry {retry} floor");
            assert_eq!(
                upper,
                prev.as_millis() as u64 * 3,
                "retry {retry} ceiling is not 3x the previous delay"
            );
            assert!(
                upper > prev.as_millis() as u64,
                "retry {retry} ceiling {upper} did not escalate past prev {prev:?}"
            );

            let delay = decorrelated_jitter(base, prev, max_backoff);
            assert!(delay >= base, "retry {retry} delay {delay:?} below floor");
            assert!(
                delay.as_millis() as u64 <= upper,
                "retry {retry} delay {delay:?} above ceiling {upper}"
            );
            prev = delay;
        }
    }

    #[test]
    fn backoff_window_grows_with_the_previous_delay() {
        let base = Duration::from_millis(10);
        let max_backoff = Duration::from_millis(900);
        let ceilings: Vec<u64> = [10u64, 20, 40, 80, 160]
            .into_iter()
            .map(|ms| jitter_window(base, Duration::from_millis(ms), max_backoff).1)
            .collect();

        assert_eq!(ceilings, vec![30, 60, 120, 240, 480]);
        for pair in ceilings.windows(2) {
            assert!(pair[1] > pair[0], "ceiling shrank: {pair:?}");
        }

        // Above the cap the ceiling clamps instead of growing without bound.
        let capped = jitter_window(base, Duration::from_secs(1000), max_backoff).1;
        assert_eq!(capped, max_backoff.as_millis() as u64);
    }

    /// The escalation rule above is arithmetic; this pins that the loop really
    /// sleeps it, once per retry. `initial_backoff == max_backoff` collapses the
    /// jitter window to a single point, so the schedule is exact, and paused
    /// time makes the measurement tokio's virtual clock rather than the runner's.
    #[tokio::test(start_paused = true)]
    async fn more_retries_sleep_strictly_longer() {
        fn flat_config(max_retries: u32) -> RetryConfig {
            RetryConfig {
                max_retries,
                initial_backoff: Duration::from_millis(10),
                max_backoff: Duration::from_millis(10),
                multiplier: 2,
                ..Default::default()
            }
        }

        async fn slept(config: &RetryConfig) -> Duration {
            let start = tokio::time::Instant::now();
            let _ = retry_on_transient(config, || async {
                Err::<(), Status>(Status::unavailable("unavailable"))
            })
            .await;
            start.elapsed()
        }

        let one = slept(&flat_config(1)).await;
        let three = slept(&flat_config(3)).await;

        assert_eq!(one, Duration::from_millis(10));
        assert_eq!(three, Duration::from_millis(30));
        assert!(three > one);
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
        assert_eq!(
            parse_pushback(&status),
            Some(Pushback::Wait(Duration::from_millis(1500)))
        );
    }

    #[test]
    fn parse_pushback_retry_after_seconds_fallback() {
        let mut status = Status::resource_exhausted("limited");
        status
            .metadata_mut()
            .insert("retry-after", "30".parse().unwrap());
        assert_eq!(
            parse_pushback(&status),
            Some(Pushback::Wait(Duration::from_secs(30)))
        );
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
        assert_eq!(
            parse_pushback(&status),
            Some(Pushback::Wait(Duration::from_millis(500)))
        );
    }

    #[test]
    fn parse_pushback_returns_none_when_absent() {
        let status = Status::resource_exhausted("limited");
        assert_eq!(parse_pushback(&status), None);
    }

    #[test]
    fn parse_pushback_negative_means_do_not_retry() {
        // gRFC A6: a negative pushback is the server refusing the retry, not a
        // missing hint. Treating it as absent retries against the instruction.
        let mut status = Status::unavailable("busy");
        status
            .metadata_mut()
            .insert("grpc-retry-pushback-ms", "-1".parse().unwrap());
        assert_eq!(parse_pushback(&status), Some(Pushback::Stop));
    }

    #[test]
    fn parse_pushback_negative_retry_after_means_do_not_retry() {
        let mut status = Status::unavailable("busy");
        status
            .metadata_mut()
            .insert("retry-after", "-5".parse().unwrap());
        assert_eq!(parse_pushback(&status), Some(Pushback::Stop));
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
        assert_eq!(
            parse_pushback(&status),
            Some(Pushback::Wait(Duration::from_millis(1500)))
        );
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

    /// Paused time, so the measurement is tokio's virtual clock: the assertion
    /// is the exact schedule the cap produces rather than a wall-clock budget a
    /// stalled runner can blow through. `smear_pushback(1h, 5ms, 1ms)` clamps
    /// the base to 5ms and smears up to `max(5/2, 1) = 2ms` on top of it.
    #[tokio::test(start_paused = true)]
    async fn pushback_capped_at_max_backoff() {
        let config = RetryConfig {
            max_retries: 1,
            initial_backoff: Duration::from_millis(1),
            max_backoff: Duration::from_millis(5),
            multiplier: 2,
            ..Default::default()
        };
        let start = tokio::time::Instant::now();
        let _ = retry_on_transient(&config, || async {
            let mut s = Status::resource_exhausted("limited");
            // 1 hour in milliseconds — would block forever if not capped
            s.metadata_mut()
                .insert("grpc-retry-pushback-ms", "3600000".parse().unwrap());
            Err::<(), Status>(s)
        })
        .await;
        let elapsed = start.elapsed();
        assert!(
            elapsed >= Duration::from_millis(5),
            "elapsed={elapsed:?} — the clamped pushback was not waited out"
        );
        assert!(
            elapsed <= Duration::from_millis(7),
            "elapsed={elapsed:?} — pushback not capped at max_backoff + smear"
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
            let d = smear_pushback(pushback, max, Duration::ZERO);
            assert!(d >= pushback, "delay {d:?} < pushback {pushback:?}");
            assert!(d <= Duration::from_millis(60_000 + 30_000));
        }
    }

    #[test]
    fn smear_pushback_clamps_the_base_then_smears() {
        // The clamp applies to the base, so the smear still disperses a fleet
        // that was told to wait longer than the cap. Clamping afterwards instead
        // would put every one of these samples on exactly `max`.
        let max = Duration::from_millis(70_000);
        let pushback = Duration::from_secs(120);
        let samples: std::collections::HashSet<u64> = (0..200)
            .map(|_| smear_pushback(pushback, max, Duration::ZERO).as_millis() as u64)
            .collect();
        assert!(
            samples.len() > 50,
            "pushback above the cap collapsed to {} distinct delays",
            samples.len()
        );
        for d in &samples {
            assert!(*d >= 70_000, "delay {d} below the clamped base");
            assert!(*d <= 105_000, "delay {d} above base + half");
        }
    }

    #[test]
    fn smear_pushback_at_the_cap_still_disperses() {
        let max = Duration::from_secs(60);
        let samples: std::collections::HashSet<u64> = (0..200)
            .map(|_| smear_pushback(max, max, Duration::ZERO).as_millis() as u64)
            .collect();
        assert!(
            samples.len() > 50,
            "a pushback exactly at the cap produced {} distinct delays",
            samples.len()
        );
    }

    #[test]
    fn smear_pushback_zero_disperses_using_the_floor() {
        // gRFC A6 reads a zero pushback as "retry immediately". Without a floor
        // every client would do so on the same millisecond.
        let floor = Duration::from_millis(100);
        let samples: std::collections::HashSet<u64> = (0..200)
            .map(|_| {
                smear_pushback(Duration::ZERO, Duration::from_secs(60), floor).as_millis() as u64
            })
            .collect();
        assert!(
            samples.len() > 20,
            "zero pushback collapsed to {} delays",
            samples.len()
        );
        assert!(samples.iter().all(|d| *d <= 100));
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
