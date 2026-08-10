"""Unit tests for Index.list_namespaces and list_namespaces_paginated."""

from __future__ import annotations

import httpx
import pytest
import respx

from pinecone import Index
from pinecone.errors.exceptions import PineconeValueError
from pinecone.models.namespaces.models import ListNamespacesResponse

INDEX_HOST = "my-index-abc123.svc.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
NS_LIST_URL = f"{INDEX_HOST_HTTPS}/namespaces"


def _make_index() -> Index:
    return Index(host=INDEX_HOST, api_key="test-key")


# ---------------------------------------------------------------------------
# list_namespaces_paginated
# ---------------------------------------------------------------------------


class TestListNamespacesPaginated:
    """Single-page namespace listing (unified-ns-0005, unified-ns-0006)."""

    @respx.mock
    def test_list_namespaces_paginated_basic(self) -> None:
        respx.get(NS_LIST_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "namespaces": [
                        {"name": "ns1", "record_count": 100},
                        {"name": "ns2", "record_count": 200},
                    ],
                    "total_count": 2,
                },
            ),
        )
        idx = _make_index()
        result = idx.list_namespaces_paginated()

        assert isinstance(result, ListNamespacesResponse)
        assert len(result.namespaces) == 2
        assert result.total_count == 2
        assert result.pagination is None

    @respx.mock
    def test_list_namespaces_paginated_with_pagination(self) -> None:
        respx.get(NS_LIST_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "namespaces": [{"name": "ns1", "record_count": 10}],
                    "pagination": {"next": "tok123"},
                    "total_count": 50,
                },
            ),
        )
        idx = _make_index()
        result = idx.list_namespaces_paginated()

        assert result.pagination is not None
        assert result.pagination.next == "tok123"

    @respx.mock
    def test_list_namespaces_paginated_with_prefix(self) -> None:
        route = respx.get(NS_LIST_URL).mock(
            return_value=httpx.Response(
                200,
                json={"namespaces": [], "total_count": 0},
            ),
        )
        idx = _make_index()
        idx.list_namespaces_paginated(prefix="prod-")

        assert route.called
        request = route.calls[0].request
        assert "prefix=prod-" in str(request.url)

    @respx.mock
    def test_list_namespaces_paginated_with_limit(self) -> None:
        route = respx.get(NS_LIST_URL).mock(
            return_value=httpx.Response(
                200,
                json={"namespaces": [], "total_count": 0},
            ),
        )
        idx = _make_index()
        idx.list_namespaces_paginated(limit=10)

        assert route.called
        request = route.calls[0].request
        assert "limit=10" in str(request.url)

    @respx.mock
    def test_list_namespaces_paginated_with_pagination_token(self) -> None:
        route = respx.get(NS_LIST_URL).mock(
            return_value=httpx.Response(
                200,
                json={"namespaces": [], "total_count": 0},
            ),
        )
        idx = _make_index()
        idx.list_namespaces_paginated(pagination_token="tok123")

        assert route.called
        request = route.calls[0].request
        assert "paginationToken=tok123" in str(request.url)

    @respx.mock
    def test_list_namespaces_paginated_empty(self) -> None:
        respx.get(NS_LIST_URL).mock(
            return_value=httpx.Response(
                200,
                json={"namespaces": [], "total_count": 0},
            ),
        )
        idx = _make_index()
        result = idx.list_namespaces_paginated()

        assert result.namespaces == []
        assert result.total_count == 0


# ---------------------------------------------------------------------------
# list_namespaces (generator)
# ---------------------------------------------------------------------------


class TestListNamespacesGenerator:
    """Auto-paginating namespace listing (unified-ns-0007, unified-ns-0008)."""

    @respx.mock
    def test_list_namespaces_single_page(self) -> None:
        respx.get(NS_LIST_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "namespaces": [
                        {"name": "ns1", "record_count": 10},
                        {"name": "ns2", "record_count": 20},
                    ],
                    "total_count": 2,
                },
            ),
        )
        idx = _make_index()
        pages = list(idx.list_namespaces())

        assert len(pages) == 1
        assert len(pages[0].namespaces) == 2
        assert pages[0].namespaces[0].name == "ns1"
        assert pages[0].namespaces[1].name == "ns2"

    @respx.mock
    def test_list_namespaces_multi_page(self) -> None:
        route = respx.get(NS_LIST_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "namespaces": [{"name": "ns1", "record_count": 10}],
                        "pagination": {"next": "tok"},
                        "total_count": 2,
                    },
                ),
                httpx.Response(
                    200,
                    json={
                        "namespaces": [{"name": "ns2", "record_count": 20}],
                        "total_count": 2,
                    },
                ),
            ],
        )
        idx = _make_index()
        pages = list(idx.list_namespaces())

        assert len(pages) == 2
        assert pages[0].namespaces[0].name == "ns1"
        assert pages[1].namespaces[0].name == "ns2"
        assert route.call_count == 2

    @respx.mock
    def test_list_namespaces_with_prefix(self) -> None:
        route = respx.get(NS_LIST_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "namespaces": [{"name": "prod-ns1", "record_count": 5}],
                    "total_count": 1,
                },
            ),
        )
        idx = _make_index()
        pages = list(idx.list_namespaces(prefix="prod-"))

        assert len(pages) == 1
        request = route.calls[0].request
        assert "prefix=prod-" in str(request.url)


# ---------------------------------------------------------------------------
# Keyword-only enforcement
# ---------------------------------------------------------------------------


class TestKeywordOnly:
    """Verify both methods reject positional arguments."""

    def test_list_namespaces_paginated_keyword_only(self) -> None:
        idx = _make_index()
        with pytest.raises(PineconeValueError, match="keyword-only"):
            idx.list_namespaces_paginated("prod-")  # type: ignore[misc]

    def test_list_namespaces_keyword_only(self) -> None:
        idx = _make_index()
        with pytest.raises(PineconeValueError, match="keyword-only"):
            idx.list_namespaces("prod-")  # type: ignore[misc]
