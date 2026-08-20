"""2026-07 conformance for the six db_control index operations.

These claims were deliberately deferred by #131 (sync, PR #209) and #133
(async, PR #258) until #112 flipped ``CONTROL_PLANE_API_VERSION`` to
``2026-07`` — ``claim.assert_api_version`` hardcodes the expected value, so
any claim landed before the flip would have been a red test. Every operation
is claimed twice — once through :class:`Pinecone`, once through
:class:`AsyncPinecone` — because the header has to appear on the wire for
both transports, and both read it from the same ``CONTROL_PLANE_API_VERSION``.

``create_index_for_model`` needs no divergence entry despite #206: the
staleness in the built OAS is confined to the *request* schema
(``CreateIndexForModelRequest``), which the conformance harness never
validates. The method (POST), path (``/indexes/create-for-model``), and
response schema (``IndexModel``) agree across the built spec, the source
spec, and the backend — and the sealed ``IndexModel`` schema admits the
``semantic_text`` field the backend surfaces in the response, so the
backend-shaped fixture below validates against the spec schema cleanly.

Fixture hosts carry the ``https://`` scheme because
``IndexModel.__post_init__`` normalizes bare hosts by prepending it; the
spec types ``host`` as a plain string, so the schemed spelling is a payload
a spec-conformant server could send, and it keeps the round-trip exact.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
import respx

from pinecone import Pinecone
from pinecone._internal.adapters.indexes_adapter import _IndexListEnvelope
from pinecone._internal.constants import DEFAULT_BASE_URL
from pinecone.async_client.pinecone import AsyncPinecone
from pinecone.models.indexes.index import IndexModel
from tests.unit.conformance import api_op

BASE_URL = DEFAULT_BASE_URL
INDEX_NAME = "conformance-index"

INDEX: dict[str, Any] = {
    "name": INDEX_NAME,
    "host": f"https://{INDEX_NAME}-abc123.svc.aws-us-east-1.pinecone.io",
    "status": {"ready": True, "state": "Ready"},
    "deployment": {"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
    "schema": {
        "fields": {"embedding": {"type": "dense_vector", "dimension": 1024, "metric": "cosine"}}
    },
    "deletion_protection": "disabled",
    "tags": {"env": "conformance"},
}

INDEX_FOR_MODEL: dict[str, Any] = {
    **INDEX,
    "name": "conformance-integrated-index",
    "host": "https://conformance-integrated-index-abc123.svc.aws-us-east-1.pinecone.io",
    "schema": {
        "fields": {
            "chunk_text": {
                "type": "semantic_text",
                "model": "multilingual-e5-large",
                "metric": "cosine",
            }
        }
    },
}

INDEX_LIST: dict[str, Any] = {"indexes": [INDEX]}

INDEX_OPTIONALS = ["tags"]

CREATE_SCHEMA: dict[str, Any] = {
    "fields": {"embedding": {"type": "dense_vector", "dimension": 1024, "metric": "cosine"}}
}

EMBED: dict[str, Any] = {
    "model": "multilingual-e5-large",
    "field_map": {"text": "chunk_text"},
}


@pytest.fixture
def pc() -> Iterator[Pinecone]:
    client = Pinecone(api_key="conformance-key")
    yield client
    client.close()


@pytest.fixture
async def async_pc() -> AsyncIterator[AsyncPinecone]:
    client = AsyncPinecone(api_key="conformance-key")
    yield client
    await client.close()


def _conforms(
    claim: Any,
    route: respx.Route,
    model: type,
    payload: dict[str, Any],
    optional_absent: list[str],
) -> None:
    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(model, payload, optional_absent=optional_absent)


def _conforms_bodyless(claim: Any, route: respx.Route, returned: Any) -> None:
    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_no_response_body(returned)


@api_op("db_control:create_index")
def test_create_index(claim: Any, pc: Pinecone, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=INDEX)
    )
    result = pc.indexes.create(name=INDEX_NAME, schema=CREATE_SCHEMA, timeout=-1)
    assert result.name == INDEX_NAME
    _conforms(claim, route, IndexModel, INDEX, INDEX_OPTIONALS)


@api_op("db_control:create_index")
async def test_async_create_index(
    claim: Any, async_pc: AsyncPinecone, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=INDEX)
    )
    result = await async_pc.indexes.create(name=INDEX_NAME, schema=CREATE_SCHEMA, timeout=-1)
    assert result.name == INDEX_NAME
    _conforms(claim, route, IndexModel, INDEX, INDEX_OPTIONALS)


@api_op("db_control:create_index_for_model")
def test_create_index_for_model(claim: Any, pc: Pinecone, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/indexes/create-for-model").mock(
        return_value=httpx.Response(201, json=INDEX_FOR_MODEL)
    )
    result = pc.indexes.create_for_model(
        name="conformance-integrated-index",
        cloud="aws",
        region="us-east-1",
        embed=EMBED,
        timeout=-1,
    )
    assert result.name == "conformance-integrated-index"
    _conforms(claim, route, IndexModel, INDEX_FOR_MODEL, INDEX_OPTIONALS)


@api_op("db_control:create_index_for_model")
async def test_async_create_index_for_model(
    claim: Any, async_pc: AsyncPinecone, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{BASE_URL}/indexes/create-for-model").mock(
        return_value=httpx.Response(201, json=INDEX_FOR_MODEL)
    )
    result = await async_pc.indexes.create_for_model(
        name="conformance-integrated-index",
        cloud="aws",
        region="us-east-1",
        embed=EMBED,
        timeout=-1,
    )
    assert result.name == "conformance-integrated-index"
    _conforms(claim, route, IndexModel, INDEX_FOR_MODEL, INDEX_OPTIONALS)


@api_op("db_control:describe_index")
def test_describe_index(claim: Any, pc: Pinecone, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{BASE_URL}/indexes/{INDEX_NAME}").mock(
        return_value=httpx.Response(200, json=INDEX)
    )
    assert pc.indexes.describe(INDEX_NAME).name == INDEX_NAME
    _conforms(claim, route, IndexModel, INDEX, INDEX_OPTIONALS)


@api_op("db_control:describe_index")
async def test_async_describe_index(
    claim: Any, async_pc: AsyncPinecone, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{BASE_URL}/indexes/{INDEX_NAME}").mock(
        return_value=httpx.Response(200, json=INDEX)
    )
    result = await async_pc.indexes.describe(INDEX_NAME)
    assert result.name == INDEX_NAME
    _conforms(claim, route, IndexModel, INDEX, INDEX_OPTIONALS)


@api_op("db_control:list_indexes")
def test_list_indexes(claim: Any, pc: Pinecone, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(200, json=INDEX_LIST)
    )
    assert [idx.name for idx in pc.indexes.list()] == [INDEX_NAME]
    _conforms(claim, route, _IndexListEnvelope, INDEX_LIST, ["indexes"])


@api_op("db_control:list_indexes")
async def test_async_list_indexes(
    claim: Any, async_pc: AsyncPinecone, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(200, json=INDEX_LIST)
    )
    assert [idx.name async for idx in async_pc.indexes.list()] == [INDEX_NAME]
    _conforms(claim, route, _IndexListEnvelope, INDEX_LIST, ["indexes"])


@api_op("db_control:configure_index")
def test_configure_index(claim: Any, pc: Pinecone, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.patch(f"{BASE_URL}/indexes/{INDEX_NAME}").mock(
        return_value=httpx.Response(200, json=INDEX)
    )
    result = pc.indexes.configure(INDEX_NAME, tags={"env": "conformance"})
    assert result.name == INDEX_NAME
    _conforms(claim, route, IndexModel, INDEX, INDEX_OPTIONALS)


@api_op("db_control:configure_index")
async def test_async_configure_index(
    claim: Any, async_pc: AsyncPinecone, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.patch(f"{BASE_URL}/indexes/{INDEX_NAME}").mock(
        return_value=httpx.Response(200, json=INDEX)
    )
    result = await async_pc.indexes.configure(INDEX_NAME, tags={"env": "conformance"})
    assert result.name == INDEX_NAME
    _conforms(claim, route, IndexModel, INDEX, INDEX_OPTIONALS)


@api_op("db_control:delete_index")
def test_delete_index(claim: Any, pc: Pinecone, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.delete(f"{BASE_URL}/indexes/{INDEX_NAME}").mock(
        return_value=httpx.Response(202)
    )
    _conforms_bodyless(claim, route, pc.indexes.delete(INDEX_NAME, timeout=-1))


@api_op("db_control:delete_index")
async def test_async_delete_index(
    claim: Any, async_pc: AsyncPinecone, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.delete(f"{BASE_URL}/indexes/{INDEX_NAME}").mock(
        return_value=httpx.Response(202)
    )
    returned = await async_pc.indexes.delete(INDEX_NAME, timeout=-1)
    _conforms_bodyless(claim, route, returned)
