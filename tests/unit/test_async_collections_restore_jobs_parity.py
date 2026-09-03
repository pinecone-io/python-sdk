"""Sync/async parity for collections and restore jobs (#118).

``Collections``/``AsyncCollections`` and ``RestoreJobs``/``AsyncRestoreJobs``
validate through the same callables, build their query strings through the same
``pinecone/_internal/backups_helpers.py``, and decode through the same
adapters, so the pairs should differ only in ``await``. #118's second
acceptance criterion is that this parity is *preserved*, so it is asserted
here rather than eyeballed, on the axes a transport port can quietly break:
identical request snapshots on the wire (method, path, query, body and the
version header), identical signatures and docstring contracts, and identical
exception types and messages for client-side rejections.

Follows ``tests/unit/test_async_backups_parity.py`` (#113 ∥ #114). Both
namespaces share one module because each is small — four methods and two —
and the whole harness is parametrised over the two class pairs, so splitting
them would duplicate the harness rather than the cases.

Two divergences are asserted rather than papered over, exactly as in the
backups parity module:

* Docstring *examples* must differ — the sync lane shows ``pc = Pinecone(...)``
  doctests, the async lane ``async with``. Docstring parity is therefore
  enforced over the caller-facing contract sections (``Args``, ``Returns``,
  ``Raises``) rather than the whole string.
* The ``Raises`` sections name one class under two spellings: sync says
  ``ValidationError``, async says ``PineconeValueError``, and the former is a
  deprecated alias of the latter. Normalised here rather than renamed in the
  shipped sync copy, which is out of scope for this ticket.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from typing import Any

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import API_VERSION_HEADER, CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import AsyncHTTPClient, HTTPClient
from pinecone.async_client.collections import AsyncCollections
from pinecone.async_client.restore_jobs import AsyncRestoreJobs
from pinecone.client.collections import Collections
from pinecone.client.restore_jobs import RestoreJobs

BASE_URL = "https://api.test.pinecone.io"

COLLECTION_NAME = "parity-collection"
JOB_ID = "rj-parity-123"

COLLECTION_PAYLOAD: dict[str, Any] = {
    "name": COLLECTION_NAME,
    "status": "Ready",
    "environment": "us-east1-gcp",
    "size": 10000000,
    "dimension": 1536,
    "vector_count": 120000,
}

RESTORE_JOB_PAYLOAD: dict[str, Any] = {
    "restore_job_id": JOB_ID,
    "backup_id": "bkp-parity-456",
    "target_index_name": "parity-index",
    "target_index_id": "idx-parity-789",
    "status": "Completed",
    "created_at": "2026-07-15T10:30:00Z",
    "completed_at": "2026-07-15T10:35:00Z",
    "percent_complete": 100.0,
}

_COLLECTIONS_METHODS = ["create", "list", "describe", "delete"]
_RESTORE_JOBS_METHODS = ["list", "describe"]

_PAIRS: list[tuple[str, type, type, list[str]]] = [
    ("collections", Collections, AsyncCollections, _COLLECTIONS_METHODS),
    ("restore_jobs", RestoreJobs, AsyncRestoreJobs, _RESTORE_JOBS_METHODS),
]

_METHOD_IDS = [(namespace, method) for namespace, _, _, methods in _PAIRS for method in methods]

_CALLS: dict[tuple[str, str], dict[str, Any]] = {
    ("collections", "create"): {"name": COLLECTION_NAME, "source": "parity-index"},
    ("collections", "list"): {},
    ("collections", "describe"): {"name": COLLECTION_NAME},
    ("collections", "delete"): {"name": COLLECTION_NAME},
    ("restore_jobs", "list"): {"limit": 5},
    ("restore_jobs", "describe"): {"job_id": JOB_ID},
}

_LIST_VARIANTS: list[dict[str, Any]] = [
    {},
    {"limit": 1},
    {"pagination_token": "tok-1"},
    {"limit": 50, "pagination_token": "tok-2"},
]

_ERROR_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("collections", "create", {"name": "", "source": "parity-index"}),
    ("collections", "create", {"name": "Bad_Name", "source": "parity-index"}),
    ("collections", "create", {"name": "-leading", "source": "parity-index"}),
    ("collections", "create", {"name": COLLECTION_NAME, "source": ""}),
    ("collections", "create", {"name": COLLECTION_NAME, "source": "   "}),
    ("collections", "describe", {"name": ""}),
    ("collections", "describe", {"name": "   "}),
    ("collections", "delete", {"name": ""}),
    ("collections", "delete", {"name": "   "}),
    ("restore_jobs", "describe", {"job_id": ""}),
    ("restore_jobs", "describe", {"job_id": "   "}),
]


@pytest.fixture
def sync_namespaces() -> Generator[dict[str, Any]]:
    config = PineconeConfig(api_key="parity-key", host=BASE_URL)
    http = HTTPClient(config, CONTROL_PLANE_API_VERSION)
    yield {"collections": Collections(http=http), "restore_jobs": RestoreJobs(http=http)}
    http.close()


@pytest.fixture
async def async_namespaces() -> AsyncGenerator[dict[str, Any]]:
    config = PineconeConfig(api_key="parity-key", host=BASE_URL)
    http = AsyncHTTPClient(config, CONTROL_PLANE_API_VERSION)
    yield {
        "collections": AsyncCollections(http=http),
        "restore_jobs": AsyncRestoreJobs(http=http),
    }
    await http.close()


def _register_routes() -> None:
    respx.post(f"{BASE_URL}/collections").mock(
        return_value=httpx.Response(201, json=COLLECTION_PAYLOAD)
    )
    respx.get(f"{BASE_URL}/collections").mock(
        return_value=httpx.Response(200, json={"collections": [COLLECTION_PAYLOAD]})
    )
    respx.get(f"{BASE_URL}/collections/{COLLECTION_NAME}").mock(
        return_value=httpx.Response(200, json=COLLECTION_PAYLOAD)
    )
    respx.delete(f"{BASE_URL}/collections/{COLLECTION_NAME}").mock(return_value=httpx.Response(202))
    respx.get(f"{BASE_URL}/restore-jobs").mock(
        return_value=httpx.Response(200, json={"data": [RESTORE_JOB_PAYLOAD]})
    )
    respx.get(f"{BASE_URL}/restore-jobs/{JOB_ID}").mock(
        return_value=httpx.Response(200, json=RESTORE_JOB_PAYLOAD)
    )


def _snapshot(request: httpx.Request) -> dict[str, Any]:
    return {
        "method": request.method,
        "raw_path": request.url.raw_path.decode(),
        "query": dict(request.url.params),
        "body": request.content.decode() if request.content else None,
        "api_version": request.headers[API_VERSION_HEADER],
    }


def _raised(call: Callable[[], object]) -> tuple[type[BaseException], str]:
    try:
        call()
    except Exception as exc:
        return type(exc), str(exc)
    raise AssertionError("expected the call to raise, it returned instead")


async def _raised_async(call: Callable[[], Awaitable[object]]) -> tuple[type[BaseException], str]:
    try:
        await call()
    except Exception as exc:
        return type(exc), str(exc)
    raise AssertionError("expected the call to raise, it returned instead")


def _section(docstring: str | None, heading: str) -> str:
    """Return the ``heading:`` block of a Google-style docstring, dedented."""
    assert docstring is not None, f"missing docstring, cannot compare {heading}"
    lines = inspect.cleandoc(docstring).splitlines()
    try:
        start = lines.index(f"{heading}:")
    except ValueError:
        return ""
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line and not line.startswith(" "):
            break
        body.append(line.rstrip())
    return "\n".join(body).strip()


def _classes(namespace: str) -> tuple[type, type]:
    for name, sync_cls, async_cls, _ in _PAIRS:
        if name == namespace:
            return sync_cls, async_cls
    raise AssertionError(namespace)


@pytest.mark.parametrize(("namespace", "method_name"), _METHOD_IDS)
@respx.mock
async def test_request_snapshot_parity(
    namespace: str,
    method_name: str,
    sync_namespaces: dict[str, Any],
    async_namespaces: dict[str, Any],
) -> None:
    _register_routes()
    kwargs = _CALLS[(namespace, method_name)]

    getattr(sync_namespaces[namespace], method_name)(**kwargs)
    sync_snapshot = _snapshot(respx.calls.last.request)

    await getattr(async_namespaces[namespace], method_name)(**kwargs)
    async_snapshot = _snapshot(respx.calls.last.request)

    assert len(respx.calls) == 2, "each transport must have issued exactly one request"
    assert async_snapshot == sync_snapshot
    assert async_snapshot["api_version"] == CONTROL_PLANE_API_VERSION


@pytest.mark.parametrize("kwargs", _LIST_VARIANTS, ids=range(len(_LIST_VARIANTS)))
@respx.mock
async def test_restore_jobs_list_query_string_parity(
    kwargs: dict[str, Any],
    sync_namespaces: dict[str, Any],
    async_namespaces: dict[str, Any],
) -> None:
    """Both lanes build the listing query through the same shared helper."""
    _register_routes()

    sync_namespaces["restore_jobs"].list(**kwargs)
    sync_snapshot = _snapshot(respx.calls.last.request)

    await async_namespaces["restore_jobs"].list(**kwargs)
    async_snapshot = _snapshot(respx.calls.last.request)

    assert async_snapshot == sync_snapshot


@pytest.mark.parametrize(("namespace", "method_name"), _METHOD_IDS)
def test_parameter_parity(namespace: str, method_name: str) -> None:
    sync_cls, async_cls = _classes(namespace)
    sync_params = dict(inspect.signature(getattr(sync_cls, method_name)).parameters)
    async_params = dict(inspect.signature(getattr(async_cls, method_name)).parameters)

    assert set(sync_params) == set(async_params), (
        f"{namespace}.{method_name}: parameter names differ — "
        f"sync-only={set(sync_params) - set(async_params)}, "
        f"async-only={set(async_params) - set(sync_params)}"
    )

    for name, sync_param in sync_params.items():
        async_param = async_params[name]
        assert sync_param.kind == async_param.kind, (
            f"{namespace}.{method_name}.{name}: kind differs "
            f"(sync={sync_param.kind}, async={async_param.kind})"
        )
        assert sync_param.default == async_param.default, (
            f"{namespace}.{method_name}.{name}: default differs "
            f"(sync={sync_param.default!r}, async={async_param.default!r})"
        )
        assert str(sync_param.annotation) == str(async_param.annotation), (
            f"{namespace}.{method_name}.{name}: annotation differs "
            f"(sync={sync_param.annotation}, async={async_param.annotation})"
        )


@pytest.mark.parametrize(("namespace", "method_name"), _METHOD_IDS)
def test_return_annotation_parity(namespace: str, method_name: str) -> None:
    sync_cls, async_cls = _classes(namespace)
    sync_return = str(inspect.signature(getattr(sync_cls, method_name)).return_annotation)
    async_return = str(inspect.signature(getattr(async_cls, method_name)).return_annotation)

    assert async_return == sync_return, (
        f"{namespace}.{method_name}: return annotation differs "
        f"(sync={sync_return}, async={async_return})"
    )


@pytest.mark.parametrize(("namespace", "method_name"), _METHOD_IDS)
def test_sync_is_blocking_and_async_is_a_coroutine(namespace: str, method_name: str) -> None:
    sync_cls, async_cls = _classes(namespace)
    assert inspect.iscoroutinefunction(getattr(async_cls, method_name))
    assert not inspect.iscoroutinefunction(getattr(sync_cls, method_name))


@pytest.mark.parametrize("namespace", [name for name, _, _, _ in _PAIRS])
def test_no_public_method_drift(namespace: str) -> None:
    sync_cls, async_cls = _classes(namespace)

    def public(cls: type) -> set[str]:
        return {
            name
            for name, _ in inspect.getmembers(cls, callable)
            if not name.startswith("_") or name == "__repr__"
        }

    assert public(sync_cls) == public(async_cls)


@pytest.mark.parametrize("heading", ["Args", "Returns"])
@pytest.mark.parametrize(("namespace", "method_name"), _METHOD_IDS)
def test_docstring_contract_section_parity(namespace: str, method_name: str, heading: str) -> None:
    """The per-parameter and return contracts must read identically in both lanes."""
    sync_cls, async_cls = _classes(namespace)
    sync_section = _section(getattr(sync_cls, method_name).__doc__, heading)
    async_section = _section(getattr(async_cls, method_name).__doc__, heading)

    assert async_section == sync_section, f"{namespace}.{method_name}: {heading} section differs"


@pytest.mark.parametrize(("namespace", "method_name"), _METHOD_IDS)
def test_raises_section_parity_modulo_the_deprecated_alias(
    namespace: str, method_name: str
) -> None:
    sync_cls, async_cls = _classes(namespace)
    sync_raises = _section(getattr(sync_cls, method_name).__doc__, "Raises").replace(
        "ValidationError", "PineconeValueError"
    )
    async_raises = _section(getattr(async_cls, method_name).__doc__, "Raises").replace(
        "ValidationError", "PineconeValueError"
    )

    assert async_raises == sync_raises, f"{namespace}.{method_name}: Raises section differs"


@pytest.mark.parametrize(
    ("namespace", "method_name", "kwargs"),
    _ERROR_CASES,
    ids=[f"{ns}-{name}-{i}" for i, (ns, name, _) in enumerate(_ERROR_CASES)],
)
async def test_validation_error_parity(
    namespace: str,
    method_name: str,
    kwargs: dict[str, Any],
    sync_namespaces: dict[str, Any],
    async_namespaces: dict[str, Any],
) -> None:
    sync_type, sync_message = _raised(
        lambda: getattr(sync_namespaces[namespace], method_name)(**kwargs)
    )
    async_type, async_message = await _raised_async(
        lambda: getattr(async_namespaces[namespace], method_name)(**kwargs)
    )

    assert async_type is sync_type
    assert async_message == sync_message
