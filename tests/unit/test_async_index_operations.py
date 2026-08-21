"""Unit tests for AsyncIndex data-plane methods."""

from __future__ import annotations

from typing import Any

import httpx
import orjson
import pytest
import respx

from pinecone import AsyncIndex
from pinecone.errors.exceptions import PineconeValueError, ValidationError
from pinecone.models.vectors.responses import (
    DescribeIndexStatsResponse,
    FetchResponse,
    ListResponse,
    QueryResponse,
    UpdateResponse,
    UpsertResponse,
)
from pinecone.models.vectors.vector import Vector

INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
UPSERT_URL = f"{INDEX_HOST_HTTPS}/vectors/upsert"
QUERY_URL = f"{INDEX_HOST_HTTPS}/query"
FETCH_URL = f"{INDEX_HOST_HTTPS}/vectors/fetch"
DELETE_URL = f"{INDEX_HOST_HTTPS}/vectors/delete"
UPDATE_URL = f"{INDEX_HOST_HTTPS}/vectors/update"
LIST_URL = f"{INDEX_HOST_HTTPS}/vectors/list"
STATS_URL = f"{INDEX_HOST_HTTPS}/describe_index_stats"


def _make_async_index() -> AsyncIndex:
    return AsyncIndex(host=INDEX_HOST, api_key="test-key")


def _make_query_response(
    *,
    matches: list[dict[str, object]] | None = None,
    namespace: str = "",
    usage: dict[str, int] | None = None,
) -> dict[str, object]:
    """Build a realistic query API response payload."""
    return {
        "matches": matches or [],
        "namespace": namespace,
        "usage": usage or {"readUnits": 5},
    }


def _make_fetch_response(
    *,
    vectors: dict[str, dict[str, Any]] | None = None,
    namespace: str = "",
    usage: dict[str, int] | None = None,
) -> dict[str, object]:
    """Build a realistic fetch API response payload."""
    return {
        "vectors": vectors or {},
        "namespace": namespace,
        "usage": usage or {"readUnits": 5},
    }


# ---------------------------------------------------------------------------
# AsyncIndex.upsert()
# ---------------------------------------------------------------------------


class TestAsyncIndexUpsert:
    """Async upsert operations."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_upsert_basic(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(
                200,
                json={"upsertedCount": 2},
            ),
        )
        idx = _make_async_index()
        result = await idx.upsert(
            vectors=[
                Vector(id="vec1", values=[0.1, 0.2, 0.3]),
                Vector(id="vec2", values=[0.4, 0.5, 0.6]),
            ],
        )

        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 2

        body = orjson.loads(route.calls.last.request.content)
        assert len(body["vectors"]) == 2
        assert body["vectors"][0]["id"] == "vec1"
        assert body["vectors"][1]["id"] == "vec2"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_upsert_with_namespace(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(
                200,
                json={"upsertedCount": 1},
            ),
        )
        idx = _make_async_index()
        await idx.upsert(
            vectors=[Vector(id="vec1", values=[0.1, 0.2])],
            namespace="my-ns",
        )

        body = orjson.loads(route.calls.last.request.content)
        assert body["namespace"] == "my-ns"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_upsert_tuple_format(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(
                200,
                json={"upsertedCount": 2},
            ),
        )
        idx = _make_async_index()
        result = await idx.upsert(
            vectors=[
                ("vec1", [0.1, 0.2, 0.3]),
                ("vec2", [0.4, 0.5, 0.6]),
            ],
        )

        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 2

        body = orjson.loads(route.calls.last.request.content)
        assert body["vectors"][0]["id"] == "vec1"
        assert body["vectors"][0]["values"] == [0.1, 0.2, 0.3]
        assert body["vectors"][1]["id"] == "vec2"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_upsert_dict_format(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(
                200,
                json={"upsertedCount": 2},
            ),
        )
        idx = _make_async_index()
        result = await idx.upsert(
            vectors=[
                {"id": "vec1", "values": [0.1, 0.2]},
                {"id": "vec2", "values": [0.3, 0.4]},
            ],
        )

        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 2

        body = orjson.loads(route.calls.last.request.content)
        assert body["vectors"][0]["id"] == "vec1"
        assert body["vectors"][0]["values"] == [0.1, 0.2]


# ---------------------------------------------------------------------------
# AsyncIndex.query()
# ---------------------------------------------------------------------------


class TestAsyncQuery:
    """Async query operations."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_query_with_vector(self) -> None:
        respx.post(QUERY_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_query_response(
                    matches=[
                        {"id": "vec1", "score": 0.95},
                        {"id": "vec2", "score": 0.80},
                    ],
                ),
            ),
        )
        idx = _make_async_index()
        result = await idx.query(top_k=2, vector=[0.1, 0.2, 0.3])

        assert isinstance(result, QueryResponse)
        assert len(result.matches) == 2
        assert result.matches[0].id == "vec1"
        assert result.matches[0].score == pytest.approx(0.95)
        assert result.matches[1].id == "vec2"

    @respx.mock
    @pytest.mark.asyncio
    async def test_query_with_id(self) -> None:
        route = respx.post(QUERY_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_query_response(
                    matches=[{"id": "vec1", "score": 1.0}],
                ),
            ),
        )
        idx = _make_async_index()
        result = await idx.query(top_k=1, id="vec1")

        assert isinstance(result, QueryResponse)
        assert len(result.matches) == 1

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["id"] == "vec1"
        assert "vector" not in body

    @pytest.mark.asyncio
    async def test_query_top_k_validation(self) -> None:
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="top_k must be between 1 and 10000, got 0"):
            await idx.query(top_k=0, vector=[0.1])

    @pytest.mark.asyncio
    async def test_query_top_k_over_max_validation(self) -> None:
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="top_k must be between 1 and 10000, got 10001"):
            await idx.query(top_k=10_001, vector=[0.1])

    @pytest.mark.asyncio
    async def test_query_both_vector_and_id_raises(self) -> None:
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="not both"):
            await idx.query(top_k=1, vector=[0.1], id="vec1")

    @pytest.mark.asyncio
    async def test_query_neither_vector_nor_id_raises(self) -> None:
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="At least one of vector, id, or sparse_vector"):
            await idx.query(top_k=1)

    @pytest.mark.asyncio
    async def test_query_keyword_only(self) -> None:
        idx = _make_async_index()
        with pytest.raises(PineconeValueError, match="keyword-only"):
            await idx.query(10, [0.1])  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AsyncIndex.fetch()
# ---------------------------------------------------------------------------


class TestAsyncFetch:
    """Async fetch operations."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_multiple_ids(self) -> None:
        respx.get(FETCH_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_fetch_response(
                    vectors={
                        "vec1": {"id": "vec1", "values": [0.1, 0.2, 0.3]},
                        "vec2": {"id": "vec2", "values": [0.4, 0.5, 0.6]},
                    },
                ),
            ),
        )
        idx = _make_async_index()
        result = await idx.fetch(ids=["vec1", "vec2"])

        assert isinstance(result, FetchResponse)
        assert len(result.vectors) == 2
        assert result.vectors["vec1"].id == "vec1"
        assert result.vectors["vec1"].values == pytest.approx([0.1, 0.2, 0.3])
        assert result.vectors["vec2"].id == "vec2"

    @pytest.mark.asyncio
    async def test_fetch_empty_ids_raises(self) -> None:
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="ids must be a non-empty list"):
            await idx.fetch(ids=[])

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_nonexistent_ids_returns_empty(self) -> None:
        respx.get(FETCH_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_fetch_response(vectors={}),
            ),
        )
        idx = _make_async_index()
        result = await idx.fetch(ids=["does-not-exist"])

        assert isinstance(result, FetchResponse)
        assert result.vectors == {}


# ---------------------------------------------------------------------------
# AsyncIndex.delete()
# ---------------------------------------------------------------------------


class TestAsyncDelete:
    """Async delete operations."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_delete_by_ids(self) -> None:
        route = respx.post(DELETE_URL).mock(
            return_value=httpx.Response(200, json={}),
        )
        idx = _make_async_index()
        result = await idx.delete(ids=["vec1", "vec2"])

        assert result is None
        body = orjson.loads(route.calls.last.request.content)
        assert body["ids"] == ["vec1", "vec2"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_delete_all(self) -> None:
        route = respx.post(DELETE_URL).mock(
            return_value=httpx.Response(200, json={}),
        )
        idx = _make_async_index()
        await idx.delete(delete_all=True, namespace="ns")

        body = orjson.loads(route.calls.last.request.content)
        assert body["deleteAll"] is True
        assert body["namespace"] == "ns"

    @respx.mock
    @pytest.mark.asyncio
    async def test_delete_by_filter(self) -> None:
        route = respx.post(DELETE_URL).mock(
            return_value=httpx.Response(200, json={}),
        )
        idx = _make_async_index()
        await idx.delete(filter={"genre": {"$eq": "drama"}})

        body = orjson.loads(route.calls.last.request.content)
        assert body["filter"] == {"genre": {"$eq": "drama"}}

    @pytest.mark.asyncio
    async def test_delete_no_mode_raises(self) -> None:
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="Must specify one of"):
            await idx.delete()

    @pytest.mark.asyncio
    async def test_delete_multiple_modes_raises(self) -> None:
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="Cannot combine"):
            await idx.delete(ids=["x"], delete_all=True)

    @pytest.mark.asyncio
    async def test_delete_keyword_only(self) -> None:
        idx = _make_async_index()
        with pytest.raises(PineconeValueError, match="keyword-only"):
            await idx.delete(["vec1"])  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AsyncIndex.update()
# ---------------------------------------------------------------------------


def _make_update_response(
    *,
    matched_records: int | None = None,
) -> dict[str, object]:
    """Build a realistic update API response payload."""
    resp: dict[str, object] = {}
    if matched_records is not None:
        resp["matchedRecords"] = matched_records
    return resp


class TestAsyncUpdate:
    """Async update operations."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_update_by_id(self) -> None:
        route = respx.post(UPDATE_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_update_response(matched_records=1),
            ),
        )
        idx = _make_async_index()
        result = await idx.update(id="vec1", values=[0.1, 0.2, 0.3])

        assert isinstance(result, UpdateResponse)
        assert result.matched_records == 1
        body = orjson.loads(route.calls.last.request.content)
        assert body["id"] == "vec1"
        assert body["values"] == [0.1, 0.2, 0.3]

    @respx.mock
    @pytest.mark.asyncio
    async def test_update_by_filter(self) -> None:
        route = respx.post(UPDATE_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_update_response(matched_records=5),
            ),
        )
        idx = _make_async_index()
        result = await idx.update(
            filter={"genre": {"$eq": "drama"}},
            set_metadata={"year": 2020},
        )

        assert isinstance(result, UpdateResponse)
        assert result.matched_records == 5
        body = orjson.loads(route.calls.last.request.content)
        assert body["filter"] == {"genre": {"$eq": "drama"}}
        assert body["setMetadata"] == {"year": 2020}

    @pytest.mark.asyncio
    async def test_update_both_id_and_filter_raises(self) -> None:
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="not both"):
            await idx.update(id="vec1", filter={"x": 1})

    @respx.mock
    @pytest.mark.asyncio
    async def test_update_dry_run(self) -> None:
        route = respx.post(UPDATE_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_update_response(matched_records=3),
            ),
        )
        idx = _make_async_index()
        result = await idx.update(
            filter={"genre": {"$eq": "drama"}},
            set_metadata={"year": 2020},
            dry_run=True,
        )

        assert isinstance(result, UpdateResponse)
        assert result.matched_records == 3
        body = orjson.loads(route.calls.last.request.content)
        assert body["dryRun"] is True

    @pytest.mark.asyncio
    async def test_update_neither_id_nor_filter_raises(self) -> None:
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="got neither"):
            await idx.update()

    @pytest.mark.asyncio
    async def test_update_keyword_only(self) -> None:
        idx = _make_async_index()
        with pytest.raises(PineconeValueError, match="keyword-only"):
            await idx.update("vec1")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AsyncIndex.list_paginated()
# ---------------------------------------------------------------------------


class TestAsyncListPaginated:
    """Async list_paginated operations."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_paginated_basic(self) -> None:
        respx.get(LIST_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "vectors": [{"id": "v1"}],
                    "pagination": {"next": None},
                    "namespace": "",
                    "usage": {"readUnits": 1},
                },
            ),
        )
        idx = _make_async_index()
        result = await idx.list_paginated()

        assert isinstance(result, ListResponse)
        assert len(result.vectors) == 1
        assert result.vectors[0].id == "v1"

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_paginated_with_prefix(self) -> None:
        route = respx.get(LIST_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "vectors": [{"id": "doc#1"}],
                    "pagination": {"next": None},
                    "namespace": "",
                    "usage": {"readUnits": 1},
                },
            ),
        )
        idx = _make_async_index()
        await idx.list_paginated(prefix="doc#")

        request_url = str(route.calls.last.request.url)
        assert "prefix=doc" in request_url

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_paginated_with_pagination_token(self) -> None:
        route = respx.get(LIST_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "vectors": [{"id": "v2"}],
                    "pagination": {"next": None},
                    "namespace": "",
                    "usage": {"readUnits": 1},
                },
            ),
        )
        idx = _make_async_index()
        await idx.list_paginated(pagination_token="abc")

        request_url = str(route.calls.last.request.url)
        assert "paginationToken=abc" in request_url

    @pytest.mark.asyncio
    async def test_list_paginated_keyword_only(self) -> None:
        idx = _make_async_index()
        with pytest.raises(PineconeValueError, match="keyword-only"):
            await idx.list_paginated("prefix")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AsyncIndex.list()
# ---------------------------------------------------------------------------


class TestAsyncList:
    """Async list (auto-paginating) operations."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_auto_paginates(self) -> None:
        call_count = 0

        def _side_effect(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(
                    200,
                    json={
                        "vectors": [{"id": "v1"}],
                        "pagination": {"next": "tok2"},
                        "namespace": "",
                        "usage": {"readUnits": 1},
                    },
                )
            return httpx.Response(
                200,
                json={
                    "vectors": [{"id": "v2"}],
                    "pagination": None,
                    "namespace": "",
                    "usage": {"readUnits": 1},
                },
            )

        respx.get(LIST_URL).mock(side_effect=_side_effect)
        idx = _make_async_index()
        pages: list[ListResponse] = [page async for page in idx.list()]

        assert len(pages) == 2
        assert pages[0].vectors[0].id == "v1"
        assert pages[1].vectors[0].id == "v2"

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_single_page(self) -> None:
        respx.get(LIST_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "vectors": [{"id": "v1"}],
                    "pagination": None,
                    "namespace": "",
                    "usage": {"readUnits": 1},
                },
            ),
        )
        idx = _make_async_index()
        pages: list[ListResponse] = [page async for page in idx.list()]

        assert len(pages) == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_skips_empty_pages(self) -> None:
        """Generator skips pages with empty vectors (unified-pag-0003)."""
        call_count = 0

        def _side_effect(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(
                    200,
                    json={
                        "vectors": [{"id": "v1"}],
                        "pagination": {"next": "tok2"},
                        "namespace": "",
                        "usage": {"readUnits": 1},
                    },
                )
            if call_count == 2:
                return httpx.Response(
                    200,
                    json={
                        "vectors": [],
                        "pagination": {"next": "tok3"},
                        "namespace": "",
                        "usage": {"readUnits": 1},
                    },
                )
            return httpx.Response(
                200,
                json={
                    "vectors": [{"id": "v2"}],
                    "namespace": "",
                    "usage": {"readUnits": 1},
                },
            )

        respx.get(LIST_URL).mock(side_effect=_side_effect)
        idx = _make_async_index()
        pages: list[ListResponse] = [page async for page in idx.list()]

        assert len(pages) == 2
        assert pages[0].vectors[0].id == "v1"
        assert pages[1].vectors[0].id == "v2"

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_empty_only_pages_yields_nothing(self) -> None:
        """Generator yields nothing when all pages are empty."""
        respx.get(LIST_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "vectors": [],
                    "namespace": "",
                    "usage": {"readUnits": 1},
                },
            ),
        )
        idx = _make_async_index()
        pages: list[ListResponse] = [page async for page in idx.list()]

        assert len(pages) == 0


# ---------------------------------------------------------------------------
# AsyncIndex.describe_index_stats()
# ---------------------------------------------------------------------------


class TestAsyncDescribeIndexStats:
    """Async describe_index_stats operations."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_stats_without_filter(self) -> None:
        respx.post(STATS_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "namespaces": {"": {"vectorCount": 100}},
                    "dimension": 128,
                    "indexFullness": 0.5,
                    "totalVectorCount": 100,
                },
            ),
        )
        idx = _make_async_index()
        result = await idx.describe_index_stats()

        assert isinstance(result, DescribeIndexStatsResponse)
        assert result.total_vector_count == 100
        assert result.dimension == 128
        assert result.index_fullness == pytest.approx(0.5)
        assert "" in result.namespaces

    @respx.mock
    @pytest.mark.asyncio
    async def test_stats_with_filter(self) -> None:
        route = respx.post(STATS_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "namespaces": {"": {"vectorCount": 10}},
                    "dimension": 128,
                    "indexFullness": 0.5,
                    "totalVectorCount": 10,
                },
            ),
        )
        idx = _make_async_index()
        await idx.describe_index_stats(filter={"genre": {"$eq": "drama"}})

        body = orjson.loads(route.calls.last.request.content)
        assert body["filter"] == {"genre": {"$eq": "drama"}}

    @pytest.mark.asyncio
    async def test_stats_keyword_only(self) -> None:
        idx = _make_async_index()
        with pytest.raises(PineconeValueError, match="keyword-only"):
            await idx.describe_index_stats({"genre": {"$eq": "drama"}})  # type: ignore[misc]
