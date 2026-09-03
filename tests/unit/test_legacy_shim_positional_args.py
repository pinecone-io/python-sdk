"""The flat 9.x shims keep their 9.1.0 positional signatures.

``pc.create_index("movies", ServerlessSpec(...), 1536)`` and
``pc.configure_index("movies", 4)`` are the call shapes 9.1.0 documented, so
they have to keep binding to the same parameters. Commit 63954a1c restored the
shims' 9.1.0 parameter *lists* but left a bare ``*`` in front of them, which
made ``create_index`` keyword-only and reordered its middle parameters. This
module pins the positional contract by signature and by wire body, and pins
that the 2026-07 resource namespaces stay keyword-only.

``create_index`` also had a trailing 9.1.0 ``schema=``; 10.x deliberately
routes that to the guided ``PineconeTypeError`` pointing at
``pc.indexes.create``, so it is not restored below and the first eight
positions stay identical to 9.1.0's.
"""

from __future__ import annotations

import inspect
from typing import Any

import httpx
import pytest
import respx

from pinecone._client import Pinecone
from pinecone.async_client.pinecone import AsyncPinecone
from pinecone.errors.exceptions import PineconeValueError
from pinecone.models.indexes.specs import ServerlessSpec
from tests.factories import make_index_response

BASE_URL = "https://api.pinecone.io"

_V910_POSITIONAL: dict[str, tuple[str, ...]] = {
    "create_index": (
        "name",
        "spec",
        "dimension",
        "metric",
        "timeout",
        "deletion_protection",
        "vector_type",
        "tags",
    ),
    "create_index_for_model": (
        "name",
        "cloud",
        "region",
        "embed",
        "tags",
        "deletion_protection",
        "read_capacity",
        "schema",
        "timeout",
    ),
    "configure_index": (
        "name",
        "replicas",
        "pod_type",
        "deletion_protection",
        "tags",
        "embed",
        "read_capacity",
        "serverless_read_capacity",
    ),
    "describe_index": ("name",),
    "delete_index": ("name", "timeout"),
    "has_index": ("name",),
    "create_collection": ("name", "source"),
    "describe_collection": ("name",),
    "delete_collection": ("name",),
}

_V910_KEYWORD_ONLY = (
    "create_index_from_backup",
    "create_backup",
    "list_backups",
    "describe_backup",
    "delete_backup",
    "list_restore_jobs",
    "describe_restore_job",
)


@pytest.mark.parametrize("client", [Pinecone, AsyncPinecone], ids=["sync", "async"])
@pytest.mark.parametrize("method", sorted(_V910_POSITIONAL))
def test_v910_parameters_are_positional_in_v910_order(client: type, method: str) -> None:
    params = list(inspect.signature(getattr(client, method)).parameters.values())
    assert params[0].name == "self"
    expected = _V910_POSITIONAL[method]
    actual = tuple(p.name for p in params[1 : 1 + len(expected)])
    assert actual == expected, f"{client.__name__}.{method} positional order drifted"
    for param in params[1 : 1 + len(expected)]:
        assert param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD, (
            f"{client.__name__}.{method}() parameter {param.name!r} is {param.kind.name}, "
            "but 9.1.0 accepted it positionally"
        )


@pytest.mark.parametrize("client", [Pinecone, AsyncPinecone], ids=["sync", "async"])
@pytest.mark.parametrize("method", _V910_KEYWORD_ONLY)
def test_shims_keyword_only_in_v910_stay_keyword_only(client: type, method: str) -> None:
    params = list(inspect.signature(getattr(client, method)).parameters.values())
    kinds = {p.kind for p in params if p.name != "self"}
    assert inspect.Parameter.POSITIONAL_OR_KEYWORD not in kinds


def test_resource_namespaces_stay_keyword_only() -> None:
    pc = Pinecone(api_key="test-key")
    for method in (pc.indexes.create, pc.backups.describe):
        kinds = {p.kind for p in inspect.signature(method).parameters.values()}
        assert inspect.Parameter.POSITIONAL_OR_KEYWORD not in kinds
    with pytest.raises(TypeError):
        pc.indexes.create("movies")  # type: ignore[misc]
    with pytest.raises(TypeError):
        pc.backups.describe("bk-123")  # type: ignore[misc]


def test_flat_shim_positional_call_no_longer_raises_the_keyword_only_guard() -> None:
    pc = Pinecone(api_key="test-key")
    guard = getattr(Pinecone.create_index, "__wrapped__", None)
    assert guard is None, "create_index is still wrapped by the keyword-only guard"
    with respx.mock:
        respx.post(f"{BASE_URL}/indexes").mock(
            return_value=httpx.Response(201, json=make_index_response(name="movies")),
        )
        respx.get(f"{BASE_URL}/indexes/movies").mock(
            return_value=httpx.Response(200, json=make_index_response(name="movies")),
        )
        try:
            pc.create_index("movies", ServerlessSpec(cloud="aws", region="us-east-1"), 1536)
        except PineconeValueError as exc:  # pragma: no cover - regression guard
            pytest.fail(f"positional create_index raised {exc}")


def _create_index_body(call: Any) -> bytes:
    with respx.mock:
        route = respx.post(f"{BASE_URL}/indexes").mock(
            return_value=httpx.Response(201, json=make_index_response(name="movies")),
        )
        respx.get(f"{BASE_URL}/indexes/movies").mock(
            return_value=httpx.Response(200, json=make_index_response(name="movies")),
        )
        call()
        return bytes(route.calls.last.request.content)


async def _create_index_body_async(call: Any) -> bytes:
    with respx.mock:
        route = respx.post(f"{BASE_URL}/indexes").mock(
            return_value=httpx.Response(201, json=make_index_response(name="movies")),
        )
        respx.get(f"{BASE_URL}/indexes/movies").mock(
            return_value=httpx.Response(200, json=make_index_response(name="movies")),
        )
        await call()
        return bytes(route.calls.last.request.content)


def _configure_index_body(call: Any) -> bytes:
    with respx.mock:
        route = respx.patch(f"{BASE_URL}/indexes/movies").mock(
            return_value=httpx.Response(200, json=make_index_response(name="movies")),
        )
        call()
        return bytes(route.calls.last.request.content)


async def _configure_index_body_async(call: Any) -> bytes:
    with respx.mock:
        route = respx.patch(f"{BASE_URL}/indexes/movies").mock(
            return_value=httpx.Response(200, json=make_index_response(name="movies")),
        )
        await call()
        return bytes(route.calls.last.request.content)


def test_create_index_positional_matches_keyword_body() -> None:
    pc = Pinecone(api_key="test-key")
    positional = _create_index_body(
        lambda: pc.create_index("movies", ServerlessSpec(cloud="aws", region="us-east-1"), 1536)
    )
    keyword = _create_index_body(
        lambda: pc.create_index(
            name="movies",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            dimension=1536,
        )
    )
    assert positional == keyword


def test_create_index_all_v910_positionals_bind_to_the_same_parameters() -> None:
    pc = Pinecone(api_key="test-key")
    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    positional = _create_index_body(
        lambda: pc.create_index(
            "movies", spec, 1536, "cosine", -1, "enabled", "dense", {"env": "prod"}
        )
    )
    keyword = _create_index_body(
        lambda: pc.create_index(
            name="movies",
            spec=spec,
            dimension=1536,
            metric="cosine",
            timeout=-1,
            deletion_protection="enabled",
            vector_type="dense",
            tags={"env": "prod"},
        )
    )
    assert positional == keyword


def test_configure_index_positional_matches_keyword_body() -> None:
    pc = Pinecone(api_key="test-key")
    positional = _configure_index_body(lambda: pc.configure_index("movies", 4))
    keyword = _configure_index_body(lambda: pc.configure_index("movies", replicas=4))
    assert positional == keyword


def test_configure_index_positional_pod_type_binds_to_pod_type() -> None:
    pc = Pinecone(api_key="test-key")
    positional = _configure_index_body(lambda: pc.configure_index("movies", 4, "p1.x2"))
    keyword = _configure_index_body(
        lambda: pc.configure_index("movies", replicas=4, pod_type="p1.x2")
    )
    assert positional == keyword


async def test_async_create_index_positional_matches_keyword_body() -> None:
    pc = AsyncPinecone(api_key="test-key")
    try:
        positional = await _create_index_body_async(
            lambda: pc.create_index("movies", ServerlessSpec(cloud="aws", region="us-east-1"), 1536)
        )
        keyword = await _create_index_body_async(
            lambda: pc.create_index(
                name="movies",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                dimension=1536,
            )
        )
    finally:
        await pc.close()
    assert positional == keyword


async def test_async_configure_index_positional_matches_keyword_body() -> None:
    pc = AsyncPinecone(api_key="test-key")
    try:
        positional = await _configure_index_body_async(lambda: pc.configure_index("movies", 4))
        keyword = await _configure_index_body_async(
            lambda: pc.configure_index("movies", replicas=4)
        )
    finally:
        await pc.close()
    assert positional == keyword
