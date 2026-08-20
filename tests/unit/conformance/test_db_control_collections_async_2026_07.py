"""2026-07 conformance for the asyncio transport of the collections operations (#118).

``create_collection``, ``list_collections``, ``describe_collection`` and
``delete_collection`` are ``version-bump-only`` for 2026-07: diffing
``db_control_2026-04.oas.yaml`` against ``db_control_2026-07.oas.yaml`` over
these four path items and the ``CollectionModel`` / ``CollectionList``
component schemas leaves exactly one difference — the
``X-Pinecone-Api-Version`` parameter default going ``2026-04`` -> ``2026-07``.
So the thing worth gating is the header and the endpoint shape, which is what
these tests assert.

The sync variants live in ``test_db_control_collections_2026_07.py`` (#117);
both lanes may claim the same operation (see README, "Additional rules"), so
this file adds no operation ids of its own to the coverage numerator. What it
adds is that nothing on the async side asserted the 2026-07 version header or
the endpoint shape before it — ``AsyncCollections`` could have regressed to
another version with a green suite, because the header is built per transport
by ``AsyncHTTPClient``.

Fixtures are restated here rather than imported from the sync module: #117 and
#118 land as independent PRs, so a cross-file import would couple two branches
that do not yet share a merge base. Sync/async fixture drift is instead ruled
out by ``tests/unit/test_async_collections_restore_jobs_parity.py``, which
holds both lanes to byte-identical requests.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import respx

from pinecone._internal.adapters.collections_adapter import _CollectionListEnvelope
from pinecone._internal.constants import DEFAULT_BASE_URL
from pinecone.async_client.pinecone import AsyncPinecone
from pinecone.models.collections.model import CollectionModel
from tests.unit.conformance import api_op

BASE_URL = DEFAULT_BASE_URL
COLLECTION_NAME = "conformance-collection"
SOURCE_INDEX = "conformance-index"

COLLECTION: dict[str, Any] = {
    "name": COLLECTION_NAME,
    "status": "Ready",
    "environment": "us-east1-gcp",
    "size": 10000000,
    "dimension": 1536,
    "vector_count": 120000,
}

COLLECTION_OPTIONALS = ["size", "dimension", "vector_count"]

COLLECTION_LIST: dict[str, Any] = {"collections": [COLLECTION]}


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


@api_op("db_control:create_collection")
async def test_async_create_collection(
    claim: Any, async_pc: AsyncPinecone, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{BASE_URL}/collections").mock(
        return_value=httpx.Response(201, json=COLLECTION)
    )
    result = await async_pc.collections.create(name=COLLECTION_NAME, source=SOURCE_INDEX)
    assert result.name == COLLECTION_NAME
    _conforms(claim, route, CollectionModel, COLLECTION, COLLECTION_OPTIONALS)


@api_op("db_control:list_collections")
async def test_async_list_collections(
    claim: Any, async_pc: AsyncPinecone, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{BASE_URL}/collections").mock(
        return_value=httpx.Response(200, json=COLLECTION_LIST)
    )
    result = await async_pc.collections.list()
    assert result.names() == [COLLECTION_NAME]
    _conforms(claim, route, _CollectionListEnvelope, COLLECTION_LIST, ["collections"])


@api_op("db_control:describe_collection")
async def test_async_describe_collection(
    claim: Any, async_pc: AsyncPinecone, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{BASE_URL}/collections/{COLLECTION_NAME}").mock(
        return_value=httpx.Response(200, json=COLLECTION)
    )
    result = await async_pc.collections.describe(COLLECTION_NAME)
    assert result.status == "Ready"
    _conforms(claim, route, CollectionModel, COLLECTION, COLLECTION_OPTIONALS)


@api_op("db_control:delete_collection")
async def test_async_delete_collection(
    claim: Any, async_pc: AsyncPinecone, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.delete(f"{BASE_URL}/collections/{COLLECTION_NAME}").mock(
        return_value=httpx.Response(202)
    )
    returned = await async_pc.collections.delete(COLLECTION_NAME)
    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_no_response_body(returned)
