use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use pinecone_grpc::retry::{
    retry_on_transient, retry_on_transient_request, RetryBudget, RetryConfig, ThrottleCallback,
};
use tonic::Status;

#[tokio::test]
async fn callback_fires_per_retry_attempt() {
    let call_count = Arc::new(AtomicU32::new(0));
    let count = call_count.clone();
    let cb_call_count = Arc::new(AtomicU32::new(0));
    let cb_count = cb_call_count.clone();

    let on_throttle: ThrottleCallback = Arc::new(move |_h: String| {
        cb_count.fetch_add(1, Ordering::SeqCst);
    });

    let config = RetryConfig {
        max_retries: 3,
        initial_backoff: Duration::from_millis(1),
        max_backoff: Duration::from_millis(5),
        on_throttle: Some(on_throttle),
        ..RetryConfig::default()
    };

    let result = retry_on_transient(&config, || {
        let count = count.clone();
        async move {
            let n = count.fetch_add(1, Ordering::SeqCst);
            if n < 3 {
                Err(Status::resource_exhausted("throttled"))
            } else {
                Ok::<(), Status>(())
            }
        }
    })
    .await;

    assert!(result.is_ok());
    // 3 retryable failures before success → callback fires 3 times (once per failure)
    assert_eq!(cb_call_count.load(Ordering::SeqCst), 3);
}

/// Paused time, so the bound is the exact schedule `smear_pushback` produces
/// rather than a wall-clock budget: the base clamps to the 20ms hint and the
/// smear adds up to `max(20/2, 1) = 10ms` on top of it.
#[tokio::test(start_paused = true)]
async fn pushback_smear_produces_delays_within_range() {
    let pushback_ms: u64 = 20;
    let ceiling_ms = pushback_ms + pushback_ms / 2;

    let config = RetryConfig {
        max_retries: 1,
        initial_backoff: Duration::from_millis(1),
        max_backoff: Duration::from_millis(200),
        ..RetryConfig::default()
    };

    let start = tokio::time::Instant::now();
    let _ = retry_on_transient(&config, || async {
        let mut s = Status::resource_exhausted("throttled");
        s.metadata_mut().insert(
            "grpc-retry-pushback-ms",
            pushback_ms.to_string().parse().unwrap(),
        );
        Err::<(), Status>(s)
    })
    .await;
    let elapsed = start.elapsed();

    assert!(
        elapsed >= Duration::from_millis(pushback_ms),
        "elapsed {:?} is less than the {}ms the server asked for",
        elapsed,
        pushback_ms
    );
    assert!(
        elapsed <= Duration::from_millis(ceiling_ms),
        "elapsed {:?} exceeded the smear ceiling {}ms",
        elapsed,
        ceiling_ms
    );
}

#[tokio::test]
async fn callback_exception_does_not_break_retry() {
    // Verify that a callback which handles its own failure silently (matching the
    // transport.rs pattern for Python exceptions:
    //   `if let Err(_) = py_cb.call1(py, (h,)) { /* log and ignore */ }`)
    // does not prevent retry from succeeding.
    let call_count = Arc::new(AtomicU32::new(0));
    let count = call_count.clone();
    let cb_call_count = Arc::new(AtomicU32::new(0));
    let cb_count = cb_call_count.clone();

    let on_throttle: ThrottleCallback = Arc::new(move |_h: String| {
        cb_count.fetch_add(1, Ordering::SeqCst);
        // Simulate a callback that raises (e.g. Python ValueError) — error is discarded,
        // as transport.rs does with `if let Err(e) = py_cb.call1(py, (h,)) {}`.
        let _discarded: Result<(), &str> = Err("ValueError: simulated throttle error");
    });

    let config = RetryConfig {
        max_retries: 2,
        initial_backoff: Duration::from_millis(1),
        max_backoff: Duration::from_millis(5),
        on_throttle: Some(on_throttle),
        host: "test-index.svc.pinecone.io".into(),
        ..RetryConfig::default()
    };

    let result = retry_on_transient(&config, || {
        let count = count.clone();
        async move {
            let n = count.fetch_add(1, Ordering::SeqCst);
            if n < 2 {
                Err(Status::resource_exhausted("throttled"))
            } else {
                Ok::<(), Status>(())
            }
        }
    })
    .await;

    assert!(
        result.is_ok(),
        "retry should succeed despite callback raising"
    );
    assert_eq!(
        cb_call_count.load(Ordering::SeqCst),
        2,
        "callback should fire on each retryable error"
    );
}

#[tokio::test]
async fn host_string_received_by_callback() {
    let expected_host = "my-index-abc123.svc.pinecone.io";
    let received_hosts: Arc<std::sync::Mutex<Vec<String>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let hosts_clone = received_hosts.clone();

    let on_throttle: ThrottleCallback = Arc::new(move |h: String| {
        hosts_clone.lock().unwrap().push(h);
    });

    let config = RetryConfig {
        max_retries: 2,
        initial_backoff: Duration::from_millis(1),
        max_backoff: Duration::from_millis(5),
        on_throttle: Some(on_throttle),
        host: expected_host.to_string(),
        ..RetryConfig::default()
    };

    let _ = retry_on_transient(&config, || async {
        Err::<(), Status>(Status::resource_exhausted("throttled"))
    })
    .await;

    let hosts = received_hosts.lock().unwrap();
    assert!(
        !hosts.is_empty(),
        "callback should have been invoked at least once"
    );
    assert!(
        hosts.iter().all(|h| h == expected_host),
        "callback received unexpected host strings: {:?}",
        hosts
    );
}

/// A server that answers `grpc-retry-pushback-ms: 30000` is telling us to wait 30
/// seconds. Under the old 1600ms `max_backoff` default that hint was clamped to
/// 1.6s and we hammered the server anyway. Runs on paused time, so the 30s is
/// virtual and the test is instant.
#[tokio::test(start_paused = true)]
async fn thirty_second_pushback_is_honored_not_clamped() {
    let pushback_ms: u64 = 30_000;

    let config = RetryConfig {
        max_retries: 1,
        ..RetryConfig::default()
    };
    assert!(
        config.max_backoff >= Duration::from_millis(pushback_ms),
        "default max_backoff {:?} is too small to honor a {}ms pushback hint",
        config.max_backoff,
        pushback_ms
    );

    let start = tokio::time::Instant::now();
    let _ = retry_on_transient(&config, || async {
        let mut s = Status::resource_exhausted("throttled");
        s.metadata_mut().insert(
            "grpc-retry-pushback-ms",
            pushback_ms.to_string().parse().unwrap(),
        );
        Err::<(), Status>(s)
    })
    .await;
    let elapsed = start.elapsed();

    assert!(
        elapsed >= Duration::from_millis(pushback_ms),
        "waited {:?}, which is less than the {}ms the server asked for",
        elapsed,
        pushback_ms
    );
}

/// The cap is still a cap: an explicitly configured `max_wait` bounds a pushback
/// hint that exceeds it. That is the deliberate behavior — the defect was the
/// default being too small to honor realistic hints, not the clamp existing.
#[tokio::test(start_paused = true)]
async fn configured_max_wait_still_bounds_pushback() {
    let config = RetryConfig {
        max_retries: 1,
        max_backoff: Duration::from_secs(2),
        ..RetryConfig::default()
    };

    let start = tokio::time::Instant::now();
    let _ = retry_on_transient(&config, || async {
        let mut s = Status::resource_exhausted("throttled");
        s.metadata_mut()
            .insert("grpc-retry-pushback-ms", "30000".parse().unwrap());
        Err::<(), Status>(s)
    })
    .await;

    assert!(
        start.elapsed() <= Duration::from_secs(3),
        "max_wait=2s should have bounded a 30s pushback, waited {:?}",
        start.elapsed()
    );
}

/// A request whose clones are counted, so the copy-per-attempt contract is
/// pinned rather than assumed.
#[derive(Debug, Default)]
struct CountingRequest {
    clones: Arc<AtomicU32>,
}

impl Clone for CountingRequest {
    fn clone(&self) -> Self {
        self.clones.fetch_add(1, Ordering::SeqCst);
        Self {
            clones: self.clones.clone(),
        }
    }
}

async fn count_clones(config: &RetryConfig, failures: u32) -> (u32, u32) {
    let clones = Arc::new(AtomicU32::new(0));
    let attempts = Arc::new(AtomicU32::new(0));
    let seen = attempts.clone();

    let request = CountingRequest {
        clones: clones.clone(),
    };
    let _ = retry_on_transient_request(config, request, |_r| {
        let seen = seen.clone();
        async move {
            let n = seen.fetch_add(1, Ordering::SeqCst);
            if n < failures {
                Err(Status::unavailable("transient"))
            } else {
                Ok::<(), Status>(())
            }
        }
    })
    .await;

    (
        attempts.load(Ordering::SeqCst),
        clones.load(Ordering::SeqCst),
    )
}

#[tokio::test(start_paused = true)]
async fn retries_disabled_costs_no_clone() {
    let config = RetryConfig {
        max_retries: 0,
        ..RetryConfig::default()
    };

    let (attempts, clones) = count_clones(&config, 0).await;

    assert_eq!(attempts, 1);
    assert_eq!(
        clones, 0,
        "the only attempt should receive the request by value"
    );
}

#[tokio::test(start_paused = true)]
async fn n_attempts_cost_n_minus_one_clones() {
    let config = RetryConfig {
        max_retries: 2,
        ..RetryConfig::default()
    };

    // Fails twice, succeeds on the third and final permitted attempt.
    let (attempts, clones) = count_clones(&config, 2).await;

    assert_eq!(attempts, 3);
    assert_eq!(clones, 2, "the final attempt should take the original");
}

/// The residual cost, stated so a change to it is deliberate: with retries
/// enabled the first attempt cannot know it will succeed, so it still works from
/// a copy. Removing that would mean not owning the payload per attempt, which
/// tonic's generated client cannot do.
#[tokio::test(start_paused = true)]
async fn first_of_several_permitted_attempts_still_copies() {
    let config = RetryConfig {
        max_retries: 5,
        ..RetryConfig::default()
    };

    let (attempts, clones) = count_clones(&config, 0).await;

    assert_eq!(attempts, 1);
    assert_eq!(clones, 1);
}

/// A backend that refuses further retries must be obeyed, not overruled.
#[tokio::test(start_paused = true)]
async fn negative_pushback_stops_retrying_immediately() {
    let calls = Arc::new(AtomicU32::new(0));
    let seen = calls.clone();

    let config = RetryConfig {
        max_retries: 5,
        ..RetryConfig::default()
    };
    let result = retry_on_transient(&config, || {
        let seen = seen.clone();
        async move {
            seen.fetch_add(1, Ordering::SeqCst);
            let mut s = Status::unavailable("shedding load");
            s.metadata_mut()
                .insert("grpc-retry-pushback-ms", "-1".parse().unwrap());
            Err::<(), Status>(s)
        }
    })
    .await;

    assert!(result.is_err());
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "a negative pushback should end the call, not start a retry chain"
    );
}

/// The budget bounds the retry multiplier, which the concurrency limiter cannot.
#[tokio::test(start_paused = true)]
async fn exhausted_budget_stops_retrying() {
    let budget = Arc::new(RetryBudget::new(10.0, 0.1));
    let config = RetryConfig {
        max_retries: 5,
        budget: Some(budget.clone()),
        ..RetryConfig::default()
    };

    let mut attempts_per_call = Vec::new();
    for _ in 0..8 {
        let calls = Arc::new(AtomicU32::new(0));
        let seen = calls.clone();
        let _ = retry_on_transient(&config, || {
            let seen = seen.clone();
            async move {
                seen.fetch_add(1, Ordering::SeqCst);
                Err::<(), Status>(Status::unavailable("down"))
            }
        })
        .await;
        attempts_per_call.push(calls.load(Ordering::SeqCst));
    }

    // Not the full 6: the bucket holds 10 and retries stop at half, so the first
    // call spends 5 tokens getting 5 attempts. The default 100-token bucket
    // leaves a healthy channel's allowance untouched.
    assert!(
        attempts_per_call[0] >= 5,
        "the first call should retry freely, got {} attempts",
        attempts_per_call[0]
    );
    assert_eq!(
        *attempts_per_call.last().unwrap(),
        1,
        "once the bucket drains, calls should fail on the first attempt"
    );
    let total: u32 = attempts_per_call.iter().sum();
    assert!(
        total < 8 * 6,
        "budget did not reduce amplification: {total} requests for 8 calls"
    );
}

/// Successful traffic refills the bucket, so a blip does not disable retries forever.
#[tokio::test(start_paused = true)]
async fn successes_restore_the_budget() {
    let budget = Arc::new(RetryBudget::new(10.0, 0.5));
    let config = RetryConfig {
        max_retries: 5,
        budget: Some(budget.clone()),
        ..RetryConfig::default()
    };

    for _ in 0..8 {
        let _ = retry_on_transient(&config, || async {
            Err::<(), Status>(Status::unavailable("down"))
        })
        .await;
    }

    let drained = Arc::new(AtomicU32::new(0));
    let seen = drained.clone();
    let _ = retry_on_transient(&config, || {
        let seen = seen.clone();
        async move {
            seen.fetch_add(1, Ordering::SeqCst);
            Err::<(), Status>(Status::unavailable("down"))
        }
    })
    .await;
    assert_eq!(
        drained.load(Ordering::SeqCst),
        1,
        "bucket should be drained"
    );

    for _ in 0..40 {
        let _ = retry_on_transient(&config, || async { Ok::<(), Status>(()) }).await;
    }

    let recovered = Arc::new(AtomicU32::new(0));
    let seen = recovered.clone();
    let _ = retry_on_transient(&config, || {
        let seen = seen.clone();
        async move {
            seen.fetch_add(1, Ordering::SeqCst);
            Err::<(), Status>(Status::unavailable("down"))
        }
    })
    .await;

    assert!(
        recovered.load(Ordering::SeqCst) > 1,
        "successful traffic should have restored the retry allowance"
    );
}

/// The first retry is where a fleet re-synchronizes after a backend blip: every
/// client sees the same error at the same moment with no trailers to smear on.
#[tokio::test(start_paused = true)]
async fn first_retry_delays_are_spread_across_a_wide_window() {
    let config = RetryConfig {
        max_retries: 1,
        budget: None,
        ..RetryConfig::default()
    };

    let mut delays = Vec::new();
    for _ in 0..200 {
        let start = tokio::time::Instant::now();
        let _ = retry_on_transient(&config, || async {
            Err::<(), Status>(Status::unavailable("restarting"))
        })
        .await;
        delays.push(start.elapsed().as_millis() as u64);
    }

    let spread = delays.iter().max().unwrap() - delays.iter().min().unwrap();
    assert!(
        spread >= 500,
        "first-retry delays span only {spread}ms; a fleet would re-synchronize"
    );
}
