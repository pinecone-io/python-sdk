"""Property-based tests for the retry backoff in pinecone._internal.http_client.

``_compute_backoff`` samples decorrelated jitter with ``random.uniform``, so it
has no single expected value; the guarantee is a bound. For any config,
attempt, and previous delay (including None, tiny, and very large), the result
must be finite and lie within ``[backoff_factor, max_wait]``. Companion to the
retry-after fuzz tests in test_retry_after_fuzz.py.
"""

from __future__ import annotations

import math

from hypothesis import given
from hypothesis import strategies as st

from pinecone._internal.config import RetryConfig
from pinecone._internal.http_client import _compute_backoff


@st.composite
def retry_configs(draw: st.DrawFn) -> RetryConfig:
    backoff_factor = draw(
        st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
    )
    extra = draw(st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False))
    return RetryConfig(backoff_factor=backoff_factor, max_wait=backoff_factor + extra)


_attempt = st.integers(min_value=0, max_value=100)
_prev_delay = st.one_of(
    st.none(),
    st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
)


@given(config=retry_configs(), attempt=_attempt, prev_delay=_prev_delay)
def test_backoff_is_within_bounds(
    config: RetryConfig, attempt: int, prev_delay: float | None
) -> None:
    delay = _compute_backoff(config, attempt, prev_delay)
    assert math.isfinite(delay)
    assert config.backoff_factor - 1e-9 <= delay <= config.max_wait + 1e-9


@given(config=retry_configs(), attempt=_attempt, prev_delay=_prev_delay)
def test_backoff_bounds_hold_across_repeated_draws(
    config: RetryConfig, attempt: int, prev_delay: float | None
) -> None:
    for _ in range(25):
        delay = _compute_backoff(config, attempt, prev_delay)
        assert config.backoff_factor - 1e-9 <= delay <= config.max_wait + 1e-9


@given(config=retry_configs(), attempt=_attempt)
def test_backoff_caps_huge_prev_delay_at_max_wait(config: RetryConfig, attempt: int) -> None:
    delay = _compute_backoff(config, attempt, prev_delay=1e12)
    assert delay <= config.max_wait + 1e-9
