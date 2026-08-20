"""The records routes carry the namespace in the URL path, so it must be encoded.

``upsert_records`` and ``search`` address ``/records/namespaces/{namespace}/...``.
The 2026-07 namespace charset is ``^[\\x01-\\x7F]+$``
(``db_data_2026-07.oas.yaml:975-984`` @ apis ``5f808858``), which admits ``/``,
``?``, ``#``, ``%`` and the C0 controls — none of which survive raw
interpolation into a path:

- ``/`` silently addresses a different route (``a/b`` requests
  ``/records/namespaces/a/b/upsert``)
- ``?`` and ``#`` truncate the path
- a control character makes httpx reject the URL outright

So the segment is percent-encoded, exactly as the namespace routes (#119/#120)
and the documents routes already do. These tests pin the encoded request line
for both records operations in both transports; reverting the encoding fails
every case.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import respx

from pinecone import AsyncIndex, Index

INDEX_HOST = "records-path-encoding-abc123.svc.pinecone.io"
BASE_URL = f"https://{INDEX_HOST}"
RECORDS_URL = f"{BASE_URL}/records/namespaces"

SEARCH_RESPONSE: dict[str, Any] = {
    "result": {"hits": [{"_id": "r1", "_score": 0.5, "fields": {}}]},
    "usage": {"read_units": 1},
}

ENCODING_CASES = [
    pytest.param("a/b", "a%2Fb", id="slash_cannot_change_the_route"),
    pytest.param("a b", "a%20b", id="space"),
    pytest.param("a?b#c", "a%3Fb%23c", id="query_and_fragment_delimiters"),
    pytest.param("100%", "100%25", id="percent"),
    pytest.param("\x01", "%01", id="control_character"),
    pytest.param("\x7f", "%7F", id="delete_character"),
]


@pytest.fixture
def index() -> Index:
    return Index(host=INDEX_HOST, api_key="test-key")


@pytest.fixture
async def async_index() -> AsyncIterator[AsyncIndex]:
    client = AsyncIndex(host=INDEX_HOST, api_key="test-key")
    yield client
    await client.close()


@pytest.mark.parametrize(("namespace", "encoded"), ENCODING_CASES)
@respx.mock
def test_upsert_records_encodes_namespace_as_one_path_segment(
    index: Index, namespace: str, encoded: str
) -> None:
    route = respx.post(f"{RECORDS_URL}/{encoded}/upsert").mock(return_value=httpx.Response(201))

    index.upsert_records(namespace=namespace, records=[{"_id": "r1", "text": "hello"}])

    assert route.called
    assert route.calls.last.request.url.raw_path.decode() == (
        f"/records/namespaces/{encoded}/upsert"
    )


@pytest.mark.parametrize(("namespace", "encoded"), ENCODING_CASES)
@respx.mock
def test_search_encodes_namespace_as_one_path_segment(
    index: Index, namespace: str, encoded: str
) -> None:
    route = respx.post(f"{RECORDS_URL}/{encoded}/search").mock(
        return_value=httpx.Response(200, json=SEARCH_RESPONSE)
    )

    index.search(namespace=namespace, top_k=1, inputs={"text": "hello"})

    assert route.called
    assert route.calls.last.request.url.raw_path.decode() == (
        f"/records/namespaces/{encoded}/search"
    )


@pytest.mark.parametrize(("namespace", "encoded"), ENCODING_CASES)
async def test_async_upsert_records_encodes_namespace_as_one_path_segment(
    async_index: AsyncIndex, namespace: str, encoded: str, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{RECORDS_URL}/{encoded}/upsert").mock(
        return_value=httpx.Response(201, content=b"")
    )

    await async_index.upsert_records(namespace=namespace, records=[{"_id": "r1", "text": "hello"}])

    assert route.called
    assert route.calls.last.request.url.raw_path.decode() == (
        f"/records/namespaces/{encoded}/upsert"
    )


@pytest.mark.parametrize(("namespace", "encoded"), ENCODING_CASES)
async def test_async_search_encodes_namespace_as_one_path_segment(
    async_index: AsyncIndex, namespace: str, encoded: str, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(f"{RECORDS_URL}/{encoded}/search").mock(
        return_value=httpx.Response(200, json=SEARCH_RESPONSE)
    )

    await async_index.search(namespace=namespace, top_k=1, inputs={"text": "hello"})

    assert route.called
    assert route.calls.last.request.url.raw_path.decode() == (
        f"/records/namespaces/{encoded}/search"
    )
