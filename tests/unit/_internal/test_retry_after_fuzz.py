from __future__ import annotations

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pinecone._internal.config import RetryConfig
from pinecone._internal.http_client import _compute_retry_after_delay, _raise_for_status
from pinecone.errors.exceptions import RateLimitError

# Edge cases that Hypothesis won't reliably generate but cover known problematic inputs.
# Note: U+0660 (arabic-indic digit 0) is intentionally excluded — HTTP headers are
# ASCII-only and httpx raises UnicodeEncodeError before our parsing code is reached.
EDGE_CASES = [
    "",  # empty
    "  ",  # whitespace only
    " 60 ",  # leading/trailing whitespace
    "60.0",  # float
    "60.5",  # float with fraction
    "-1",  # negative
    "NaN",  # NaN
    "Infinity",  # Infinity
    "Fri, 31 Dec 2026 23:59:59 GMT",  # HTTP-date
    "60,80",  # multi-value (should reject)
    "1e3",  # scientific notation
    "0",  # zero
    "9" * 100,  # huge number
]


@given(retry_after_text=st.text(alphabet=st.characters(max_codepoint=127)))
def test_compute_retry_after_delay_never_returns_negative(retry_after_text: str) -> None:
    response = httpx.Response(429, headers={"retry-after": retry_after_text})
    cfg = RetryConfig(max_retries=1, backoff_factor=0.1, max_wait=60.0)
    delay = _compute_retry_after_delay(cfg, response, attempt=0, prev_delay=None)
    assert delay >= 0.0


@given(
    retry_after_text=st.one_of(
        st.from_regex(r"[0-9]{1,4}(\.[0-9]{1,3})?", fullmatch=True),  # well-formed floats
        st.text(
            alphabet=st.characters(max_codepoint=127)
        ),  # garbage (ASCII only — httpx requires ASCII headers)
    )
)
def test_compute_retry_after_delay_respects_bounds(retry_after_text: str) -> None:
    response = httpx.Response(429, headers={"retry-after": retry_after_text})
    cfg = RetryConfig(max_retries=1, backoff_factor=0.1, max_wait=60.0)
    delay = _compute_retry_after_delay(cfg, response, attempt=0, prev_delay=None)
    try:
        ra = float(retry_after_text)
        if ra >= 0:
            # Large retry-after values are capped at max_wait to prevent unbounded delays.
            effective_ra = min(ra, cfg.max_wait)
            assert effective_ra <= delay <= effective_ra * 1.5 + 1e-9
            return
    except (ValueError, TypeError):
        pass
    # Fallback to backoff path.
    assert cfg.backoff_factor <= delay <= cfg.max_wait + 1e-9


@pytest.mark.parametrize("retry_after_text", EDGE_CASES)
def test_edge_case_retry_after_values(retry_after_text: str) -> None:
    response = httpx.Response(429, headers={"retry-after": retry_after_text})
    cfg = RetryConfig(max_retries=1, backoff_factor=0.1, max_wait=60.0)
    # Must not raise.
    delay = _compute_retry_after_delay(cfg, response, attempt=0, prev_delay=None)
    # Must be a finite non-negative number bounded above.
    assert 0.0 <= delay <= cfg.max_wait * 1.5 + 1e-9
    assert delay == delay  # not NaN


@pytest.mark.parametrize("retry_after_text", EDGE_CASES)
def test_rate_limit_error_retry_after_normalized(retry_after_text: str) -> None:
    response = httpx.Response(429, headers={"retry-after": retry_after_text}, content=b"{}")
    with pytest.raises(RateLimitError) as exc_info:
        _raise_for_status(response)
    ra = exc_info.value.retry_after
    assert ra is None or (isinstance(ra, float) and ra >= 0 and ra == ra)
