use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use pinecone_grpc::retry::{retry_on_transient, RetryConfig, ThrottleCallback};
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

#[tokio::test]
async fn pushback_smear_produces_delays_within_range() {
    let pushback_ms: u64 = 20;

    let config = RetryConfig {
        max_retries: 1,
        initial_backoff: Duration::from_millis(1),
        max_backoff: Duration::from_millis(200),
        ..RetryConfig::default()
    };

    let start = Instant::now();
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

    // smear_pushback(20ms, 200ms) returns uniform(20ms, 30ms), well under cap.
    // Lower bound: must wait at least pushback ms.
    // Upper bound: pushback + pushback/2 + generous CI slack (200ms).
    assert!(
        elapsed >= Duration::from_millis(pushback_ms),
        "elapsed {:?} should be >= pushback {}ms",
        elapsed,
        pushback_ms
    );
    assert!(
        elapsed < Duration::from_millis(pushback_ms + pushback_ms / 2 + 200),
        "elapsed {:?} exceeded expected ceiling {}ms",
        elapsed,
        pushback_ms + pushback_ms / 2 + 200
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
