"""gRFC A6 retry-budget tests (issue #76, the REST half of #55).

Parity targets mirror the arithmetic that motivated the budget: an exhausted
budget drops attempts to 1, successes restore retry capacity, and total
attempts under a partial outage stay near 1x instead of the unbudgeted
``0.8 + 0.2 * (max_retries + 1)`` ≈ 2x amplification.

The audit tests at the bottom enforce the one-retry-layer invariant: the SDK
loop in http_client.py is the ONLY in-channel REST retry layer — any
transport-level retries (httpx ``retries=``, urllib3 ``Retry``) would
compound multiplicatively as (r1+1)(r2+1).
"""

from __future__ import annotations

import pathlib
import re

import httpx
import pytest

from pinecone._internal.config import RetryConfig
from pinecone._internal.http_client import (
    _BUDGET_MAX_TOKENS,
    _AsyncRetryTransport,
    _BudgetRegistry,
    _RetryBudget,
    _RetryTransport,
    get_budget_registry,
)

HOST = "budget-test.example.com"
URL = f"https://{HOST}/upsert"


class _ScriptedTransport(httpx.BaseTransport):
    """Returns canned status codes in order, repeating the final one."""

    def __init__(self, statuses: list[int]) -> None:
        self._statuses = list(statuses)
        self.calls = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        idx = min(self.calls, len(self._statuses) - 1)
        self.calls += 1
        return httpx.Response(self._statuses[idx], request=request)


class _AsyncScriptedTransport(httpx.AsyncBaseTransport):
    def __init__(self, statuses: list[int]) -> None:
        self._statuses = list(statuses)
        self.calls = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        idx = min(self.calls, len(self._statuses) - 1)
        self.calls += 1
        return httpx.Response(self._statuses[idx], request=request)


def _drain(budget: _RetryBudget, to_tokens: float = 0.0) -> None:
    while budget.tokens() > to_tokens:
        budget.record_failure()


def _sync_transport(inner: httpx.BaseTransport, max_retries: int = 5) -> _RetryTransport:
    cfg = RetryConfig(max_retries=max_retries, backoff_factor=0.0, max_wait=0.01)
    return _RetryTransport(transport=inner, retry_config=cfg)  # type: ignore[arg-type]


def _async_transport(inner: httpx.AsyncBaseTransport, max_retries: int = 5) -> _AsyncRetryTransport:
    cfg = RetryConfig(max_retries=max_retries, backoff_factor=0.0, max_wait=0.01)
    return _AsyncRetryTransport(transport=inner, retry_config=cfg)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _RetryBudget unit behavior
# ---------------------------------------------------------------------------


def test_budget_starts_full_and_allows_retries() -> None:
    budget = _RetryBudget()
    assert budget.tokens() == _BUDGET_MAX_TOKENS
    assert budget.allows_retry()


def test_failure_spends_one_success_earns_a_tenth() -> None:
    budget = _RetryBudget()
    budget.record_failure()
    assert budget.tokens() == pytest.approx(_BUDGET_MAX_TOKENS - 1.0)
    budget.record_success()
    assert budget.tokens() == pytest.approx(_BUDGET_MAX_TOKENS - 0.9)


def test_retries_suppressed_at_or_below_half() -> None:
    budget = _RetryBudget()
    _drain(budget, to_tokens=_BUDGET_MAX_TOKENS / 2.0)
    assert budget.tokens() == pytest.approx(_BUDGET_MAX_TOKENS / 2.0)
    assert not budget.allows_retry()


def test_tokens_clamp_at_zero_and_max() -> None:
    budget = _RetryBudget()
    _drain(budget)
    budget.record_failure()
    assert budget.tokens() == 0.0
    for _ in range(2000):
        budget.record_success()
    assert budget.tokens() == pytest.approx(_BUDGET_MAX_TOKENS)


# ---------------------------------------------------------------------------
# Registry keying
# ---------------------------------------------------------------------------


def test_registry_normalizes_host_variants_to_one_budget() -> None:
    registry = _BudgetRegistry()
    a = registry.get("API.Example.COM")
    b = registry.get("api.example.com:443")
    c = registry.get("https://api.example.com/path")
    assert a is b is c


def test_registry_isolates_distinct_hosts() -> None:
    registry = _BudgetRegistry()
    assert registry.get("a.example.com") is not registry.get("b.example.com")


# ---------------------------------------------------------------------------
# Sync transport wiring
# ---------------------------------------------------------------------------


def test_exhausted_budget_drops_attempts_to_one_sync() -> None:
    _drain(get_budget_registry().get(HOST))
    inner = _ScriptedTransport([503])
    transport = _sync_transport(inner)

    response = transport.handle_request(httpx.Request("POST", URL))

    assert response.status_code == 503
    assert inner.calls == 1


def test_full_budget_retries_to_success_sync() -> None:
    inner = _ScriptedTransport([503, 503, 200])
    transport = _sync_transport(inner)

    response = transport.handle_request(httpx.Request("POST", URL))

    assert response.status_code == 200
    assert inner.calls == 3


def test_exhausted_budget_fails_fast_on_transport_error_sync() -> None:
    _drain(get_budget_registry().get(HOST))

    class _RaisingTransport(httpx.BaseTransport):
        calls = 0

        def handle_request(self, request: httpx.Request) -> httpx.Response:
            type(self).calls += 1
            raise httpx.ConnectError("boom", request=request)

    inner = _RaisingTransport()
    transport = _sync_transport(inner)

    with pytest.raises(httpx.ConnectError):
        transport.handle_request(httpx.Request("POST", URL))
    assert _RaisingTransport.calls == 1


def test_first_attempt_never_blocked_sync() -> None:
    _drain(get_budget_registry().get(HOST))
    inner = _ScriptedTransport([200])
    transport = _sync_transport(inner)

    response = transport.handle_request(httpx.Request("POST", URL))

    assert response.status_code == 200
    assert inner.calls == 1


def test_success_restores_retry_capacity_sync() -> None:
    budget = get_budget_registry().get(HOST)
    _drain(budget, to_tokens=_BUDGET_MAX_TOKENS / 2.0)
    assert not budget.allows_retry()

    inner = _ScriptedTransport([200])
    transport = _sync_transport(inner)
    for _ in range(11):
        transport.handle_request(httpx.Request("POST", URL))
    assert budget.allows_retry()

    inner2 = _ScriptedTransport([503, 200])
    transport2 = _sync_transport(inner2)
    response = transport2.handle_request(httpx.Request("POST", URL))
    assert response.status_code == 200
    assert inner2.calls == 2


def test_per_host_isolation_sync() -> None:
    _drain(get_budget_registry().get("drained.example.com"))
    inner = _ScriptedTransport([503, 200])
    transport = _sync_transport(inner)

    response = transport.handle_request(httpx.Request("POST", "https://healthy.example.com/upsert"))

    assert response.status_code == 200
    assert inner.calls == 2


def test_budget_shared_across_transport_instances() -> None:
    inner_a = _ScriptedTransport([503])
    inner_b = _ScriptedTransport([503])
    transport_a = _sync_transport(inner_a, max_retries=0)
    transport_b = _sync_transport(inner_b, max_retries=5)
    _drain(get_budget_registry().get(HOST), to_tokens=_BUDGET_MAX_TOKENS / 2.0 + 1.0)

    transport_a.handle_request(httpx.Request("POST", URL))
    response = transport_b.handle_request(httpx.Request("POST", URL))

    assert response.status_code == 503
    assert inner_b.calls == 1


def test_non_retryable_status_is_budget_neutral() -> None:
    budget = get_budget_registry().get(HOST)
    start = budget.tokens()
    inner = _ScriptedTransport([404])
    transport = _sync_transport(inner)

    transport.handle_request(httpx.Request("POST", URL))

    assert budget.tokens() == pytest.approx(start)


# ---------------------------------------------------------------------------
# Async transport wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exhausted_budget_drops_attempts_to_one_async() -> None:
    _drain(get_budget_registry().get(HOST))
    inner = _AsyncScriptedTransport([503])
    transport = _async_transport(inner)

    response = await transport.handle_async_request(httpx.Request("POST", URL))

    assert response.status_code == 503
    assert inner.calls == 1


@pytest.mark.asyncio
async def test_full_budget_retries_to_success_async() -> None:
    inner = _AsyncScriptedTransport([503, 503, 200])
    transport = _async_transport(inner)

    response = await transport.handle_async_request(httpx.Request("POST", URL))

    assert response.status_code == 200
    assert inner.calls == 3


@pytest.mark.asyncio
async def test_first_attempt_never_blocked_async() -> None:
    _drain(get_budget_registry().get(HOST))
    inner = _AsyncScriptedTransport([200])
    transport = _async_transport(inner)

    response = await transport.handle_async_request(httpx.Request("POST", URL))

    assert response.status_code == 200
    assert inner.calls == 1


@pytest.mark.asyncio
async def test_sync_and_async_share_one_ledger() -> None:
    sync_inner = _ScriptedTransport([503])
    sync_t = _sync_transport(sync_inner, max_retries=0)
    budget = get_budget_registry().get(HOST)
    _drain(budget, to_tokens=_BUDGET_MAX_TOKENS / 2.0 + 1.0)

    sync_t.handle_request(httpx.Request("POST", URL))
    assert not budget.allows_retry()

    async_inner = _AsyncScriptedTransport([503, 200])
    async_t = _async_transport(async_inner)
    response = await async_t.handle_async_request(httpx.Request("POST", URL))

    assert response.status_code == 503
    assert async_inner.calls == 1


# ---------------------------------------------------------------------------
# Partial-outage amplification parity (#55's arithmetic)
# ---------------------------------------------------------------------------


def test_partial_outage_total_attempts_bounded_sync() -> None:
    """20% of requests hard-fail every attempt. Unbudgeted, max_retries=6
    yields 0.8 + 0.2*7 = 2.2x attempts; the budget must hold it near 1x."""

    class _PartialOutageTransport(httpx.BaseTransport):
        def __init__(self) -> None:
            self.request_seq = 0
            self.attempts = 0

        def handle_request(self, request: httpx.Request) -> httpx.Response:
            self.attempts += 1
            return httpx.Response(503 if request.extensions["seq"] % 5 == 0 else 200)

    inner = _PartialOutageTransport()
    cfg = RetryConfig(max_retries=6, backoff_factor=0.0, max_wait=0.01)
    transport = _RetryTransport(transport=inner, retry_config=cfg)  # type: ignore[arg-type]

    n_requests = 500
    for seq in range(n_requests):
        request = httpx.Request("POST", URL)
        request.extensions["seq"] = seq
        transport.handle_request(request)

    amplification = inner.attempts / n_requests
    assert amplification <= 1.3, (
        f"amplification {amplification:.3f} — budget failed to bound the multiplier"
    )
    assert amplification >= 1.0


# ---------------------------------------------------------------------------
# Audit: exactly one in-channel REST retry layer
# ---------------------------------------------------------------------------

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[3] / "pinecone"


def _non_legacy_sources() -> list[pathlib.Path]:
    return [p for p in _PACKAGE_ROOT.rglob("*.py") if "_legacy" not in p.parts]


def test_audit_no_urllib3_retry_layer() -> None:
    """urllib3's Retry is a second in-channel retry layer; multipliers
    compound as (r1+1)(r2+1). _legacy is excluded: its request paths are
    not wrapped by the budgeted SDK loop and are frozen for removal."""
    offenders = [
        str(path)
        for path in _non_legacy_sources()
        if re.search(r"^\s*(import urllib3|from urllib3)", path.read_text(), re.MULTILINE)
    ]
    assert not offenders, f"urllib3 imported outside _legacy: {offenders}"


def test_audit_httpx_transports_have_zero_transport_retries() -> None:
    """httpx.(Async)HTTPTransport(retries=N) retries connect errors *inside*
    the channel, underneath the SDK loop and invisible to the budget."""
    pattern = re.compile(r"(?:Async)?HTTPTransport\(([^)]*)\)", re.DOTALL)
    offenders: list[str] = []
    for path in _non_legacy_sources():
        for match in pattern.finditer(path.read_text()):
            if "retries" in match.group(1):
                offenders.append(f"{path}: {match.group(0)[:80]}")
    assert not offenders, f"transport-level retries configured: {offenders}"
