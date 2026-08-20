"""2026-07 conformance for the four db_control collection operations.

These claims were deliberately deferred by #117 until #112 flipped
``CONTROL_PLANE_API_VERSION`` to ``2026-07``: the header is the whole point
of a ``version-bump-only`` claim, so claiming these against ``2025-10``
would have certified the wrong constant.

``CollectionModel``, ``CollectionList`` and ``CreateCollectionRequest`` are
byte-identical between ``2025-10`` and ``2026-07`` once descriptions and
examples are stripped, and the four ``/collections`` operations differ only
in the ``X-Pinecone-Api-Version`` default — so no adapter, model, or client
change was needed and the fixtures below are the ``2025-10`` shapes
unchanged. The backend agrees: ``2026-07`` nests ``/collections`` on
``v202604::collections``, which re-exports the ``v202510`` types and
delegates to the ``v202404`` handlers, whose ``CollectionSpec``
(``v202404/collections.rs:78-89`` @ pinecone-db ``cbee5a67fe``) matches the
spec field-for-field with ``size`` / ``vector_count`` / ``dimension``
skipped when absent — which is what the ``optional_absent`` leg pins.

The module-level fixtures are named for reuse: #118 claims the same four
operations through :class:`AsyncPinecone` and can import them rather than
re-inventing a second set of payloads that could drift.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx

from pinecone import Pinecone
from pinecone._internal.adapters.collections_adapter import _CollectionListEnvelope
from pinecone._internal.constants import DEFAULT_BASE_URL
from pinecone.models.collections.model import CollectionModel
from tests.unit.conformance import api_op

BASE_URL = DEFAULT_BASE_URL
COLLECTION_NAME = "conformance-collection"
SOURCE_INDEX = "conformance-index"

COLLECTION: dict[str, Any] = {
    "name": COLLECTION_NAME,
    "status": "Ready",
    "environment": "us-east1-gcp",
    "size": 3126700,
    "dimension": 1024,
    "vector_count": 99,
}

COLLECTION_OPTIONALS = ["size", "dimension", "vector_count"]

COLLECTION_LIST: dict[str, Any] = {"collections": [COLLECTION]}

COLLECTION_LIST_OPTIONALS = ["collections"]


@pytest.fixture
def pc() -> Iterator[Pinecone]:
    client = Pinecone(api_key="conformance-key")
    yield client
    client.close()


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
def test_create_collection(claim: Any, pc: Pinecone, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/collections").mock(
        return_value=httpx.Response(201, json=COLLECTION)
    )
    result = pc.collections.create(name=COLLECTION_NAME, source=SOURCE_INDEX)
    assert result.name == COLLECTION_NAME
    assert json.loads(route.calls.last.request.content) == {
        "name": COLLECTION_NAME,
        "source": SOURCE_INDEX,
    }
    _conforms(claim, route, CollectionModel, COLLECTION, COLLECTION_OPTIONALS)


@api_op("db_control:list_collections")
def test_list_collections(claim: Any, pc: Pinecone, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{BASE_URL}/collections").mock(
        return_value=httpx.Response(200, json=COLLECTION_LIST)
    )
    result = pc.collections.list()
    assert result.names() == [COLLECTION_NAME]
    _conforms(claim, route, _CollectionListEnvelope, COLLECTION_LIST, COLLECTION_LIST_OPTIONALS)


@api_op("db_control:describe_collection")
def test_describe_collection(claim: Any, pc: Pinecone, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{BASE_URL}/collections/{COLLECTION_NAME}").mock(
        return_value=httpx.Response(200, json=COLLECTION)
    )
    result = pc.collections.describe(COLLECTION_NAME)
    assert result.dimension == 1024
    _conforms(claim, route, CollectionModel, COLLECTION, COLLECTION_OPTIONALS)


@api_op("db_control:delete_collection")
def test_delete_collection(claim: Any, pc: Pinecone, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.delete(f"{BASE_URL}/collections/{COLLECTION_NAME}").mock(
        return_value=httpx.Response(202)
    )
    returned = pc.collections.delete(COLLECTION_NAME)
    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_no_response_body(returned)
