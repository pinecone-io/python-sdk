"""Unit tests for the GRPCIndex capitalisation alias (BCG-141) and GrpcIndex.describe_namespace."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import pinecone.grpc
from pinecone.grpc import GRPCIndex, GrpcIndex
from pinecone.models.namespaces.models import ListNamespacesResponse, NamespaceDescription

_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"


def _make_grpc_index_via_alias(mock_channel: MagicMock) -> GRPCIndex:
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        return GRPCIndex(
            host="https://x-abc.svc.pinecone.io",
            api_key="k",
        )


def _make_grpc_index(mock_channel: MagicMock) -> GrpcIndex:
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        return GrpcIndex(
            host="https://x-abc.svc.pinecone.io",
            api_key="k",
        )


def test_legacy_GRPCIndex_alias_imports() -> None:  # noqa: N802
    from pinecone.grpc import GRPCIndex as _GRPCIndex  # noqa: F401


def test_legacy_GRPCIndex_alias_is_canonical() -> None:  # noqa: N802
    assert pinecone.grpc.GRPCIndex is pinecone.grpc.GrpcIndex


def test_legacy_GRPCIndex_alias_constructs() -> None:  # noqa: N802
    mock_channel = MagicMock()
    mock_channel.query.return_value = {"matches": [], "namespace": ""}
    idx = _make_grpc_index_via_alias(mock_channel)
    assert isinstance(idx, GrpcIndex)


class TestGrpcDescribeNamespace:
    def test_grpc_describe_namespace_routes_to_channel(self) -> None:
        mock_channel = MagicMock()
        mock_channel.describe_namespace.return_value = {
            "name": "movies",
            "record_count": 42,
        }
        idx = _make_grpc_index(mock_channel)

        result = idx.describe_namespace(name="movies")

        mock_channel.describe_namespace.assert_called_once_with("movies", timeout_s=None)
        assert isinstance(result, NamespaceDescription)
        assert result.name == "movies"
        assert result.record_count == 42
        assert result.schema is None
        assert result.indexed_fields is None

    def test_grpc_describe_namespace_accepts_legacy_namespace_kwarg(self) -> None:
        mock_channel = MagicMock()
        mock_channel.describe_namespace.return_value = {
            "name": "movies",
            "record_count": 7,
        }
        idx = _make_grpc_index(mock_channel)

        result = idx.describe_namespace(namespace="movies")  # type: ignore[call-arg]

        mock_channel.describe_namespace.assert_called_once_with("movies", timeout_s=None)
        assert result.name == "movies"
        assert result.record_count == 7

    def test_grpc_describe_namespace_both_kwargs_raise(self) -> None:
        from pinecone.errors.exceptions import ValidationError

        mock_channel = MagicMock()
        idx = _make_grpc_index(mock_channel)

        with pytest.raises(ValidationError, match=r"name=.*namespace="):
            idx.describe_namespace(name="movies", namespace="movies")  # type: ignore[call-arg]

    def test_grpc_describe_namespace_empty_name_raises(self) -> None:
        from pinecone.errors.exceptions import ValidationError

        mock_channel = MagicMock()
        idx = _make_grpc_index(mock_channel)

        with pytest.raises(ValidationError, match="non-empty"):
            idx.describe_namespace(name="")

    def test_grpc_describe_namespace_whitespace_name_reaches_channel(self) -> None:
        """2026-07 accepts any ASCII name, so a whitespace-only one is describable."""
        mock_channel = MagicMock()
        mock_channel.describe_namespace.return_value = {"name": "   ", "record_count": 0}
        idx = _make_grpc_index(mock_channel)

        result = idx.describe_namespace(name="   ")

        mock_channel.describe_namespace.assert_called_once_with("   ", timeout_s=None)
        assert result.name == "   "

    def test_grpc_describe_namespace_with_schema_and_indexed_fields(self) -> None:
        mock_channel = MagicMock()
        mock_channel.describe_namespace.return_value = {
            "name": "docs",
            "record_count": 100,
            "schema": {"fields": {"genre": {"filterable": True}}},
            "indexed_fields": ["genre", "year"],
        }
        idx = _make_grpc_index(mock_channel)

        result = idx.describe_namespace(name="docs")

        assert result.name == "docs"
        assert result.record_count == 100
        assert result.schema is not None
        assert result.schema.fields["genre"].filterable is True
        assert result.indexed_fields is not None
        assert result.indexed_fields.fields == ["genre", "year"]

    def test_grpc_describe_namespace_passes_timeout(self) -> None:
        mock_channel = MagicMock()
        mock_channel.describe_namespace.return_value = {
            "name": "ns1",
            "record_count": 0,
        }
        idx = _make_grpc_index(mock_channel)

        idx.describe_namespace(name="ns1", timeout=5.0)

        mock_channel.describe_namespace.assert_called_once_with("ns1", timeout_s=5.0)


class TestGrpcCreateNamespace:
    def test_grpc_create_namespace_routes_to_channel(self) -> None:
        mock_channel = MagicMock()
        mock_channel.create_namespace.return_value = {
            "name": "movies",
            "record_count": 0,
        }
        idx = _make_grpc_index(mock_channel)

        result = idx.create_namespace(name="movies")

        mock_channel.create_namespace.assert_called_once_with("movies", None, timeout_s=None)
        assert isinstance(result, NamespaceDescription)
        assert result.name == "movies"
        assert result.record_count == 0

    def test_grpc_create_namespace_forwards_schema(self) -> None:
        mock_channel = MagicMock()
        mock_channel.create_namespace.return_value = {
            "name": "movies",
            "record_count": 0,
            "schema": {"fields": {"genre": {"filterable": True}}},
        }
        idx = _make_grpc_index(mock_channel)

        schema = {"fields": {"genre": {"filterable": True}}}
        result = idx.create_namespace(name="movies", schema=schema)

        mock_channel.create_namespace.assert_called_once_with("movies", schema, timeout_s=None)
        assert isinstance(result, NamespaceDescription)
        assert result.schema is not None
        assert result.schema.fields["genre"].filterable is True

    def test_grpc_create_namespace_positional_name_raises(self) -> None:
        from pinecone.errors.exceptions import PineconeValueError

        mock_channel = MagicMock()
        idx = _make_grpc_index(mock_channel)

        with pytest.raises(PineconeValueError, match="keyword-only"):
            idx.create_namespace("my-ns")  # type: ignore[misc]

    def test_grpc_create_namespace_empty_name_raises(self) -> None:
        from pinecone.errors.exceptions import ValidationError

        mock_channel = MagicMock()
        idx = _make_grpc_index(mock_channel)

        with pytest.raises(ValidationError, match="non-empty"):
            idx.create_namespace(name="")

    def test_grpc_create_namespace_whitespace_name_reaches_channel(self) -> None:
        """2026-07 accepts any ASCII name, so a whitespace-only one is creatable."""
        mock_channel = MagicMock()
        mock_channel.create_namespace.return_value = {"name": "   ", "record_count": 0}
        idx = _make_grpc_index(mock_channel)

        result = idx.create_namespace(name="   ")

        mock_channel.create_namespace.assert_called_once_with("   ", None, timeout_s=None)
        assert result.name == "   "

    def test_grpc_create_namespace_non_string_name_raises(self) -> None:
        from pinecone.errors.exceptions import ValidationError

        mock_channel = MagicMock()
        idx = _make_grpc_index(mock_channel)

        with pytest.raises(ValidationError, match="must be a string"):
            idx.create_namespace(name=123)  # type: ignore[arg-type]

    def test_grpc_create_namespace_passes_timeout(self) -> None:
        mock_channel = MagicMock()
        mock_channel.create_namespace.return_value = {"name": "ns1", "record_count": 0}
        idx = _make_grpc_index(mock_channel)

        idx.create_namespace(name="ns1", timeout=4.5)

        mock_channel.create_namespace.assert_called_once_with("ns1", None, timeout_s=4.5)


class TestGrpcDeleteNamespace:
    def test_grpc_delete_namespace_returns_none(self) -> None:
        mock_channel = MagicMock()
        mock_channel.delete_namespace.return_value = None
        idx = _make_grpc_index(mock_channel)

        result = idx.delete_namespace(name="movies")

        mock_channel.delete_namespace.assert_called_once_with("movies", timeout_s=None)
        assert result is None

    def test_grpc_delete_namespace_accepts_legacy_namespace_kwarg(self) -> None:
        mock_channel = MagicMock()
        mock_channel.delete_namespace.return_value = None
        idx = _make_grpc_index(mock_channel)

        result = idx.delete_namespace(namespace="movies")  # type: ignore[call-arg]

        mock_channel.delete_namespace.assert_called_once_with("movies", timeout_s=None)
        assert result is None

    def test_grpc_delete_namespace_both_kwargs_raise(self) -> None:
        from pinecone.errors.exceptions import ValidationError

        mock_channel = MagicMock()
        idx = _make_grpc_index(mock_channel)

        with pytest.raises(ValidationError, match=r"name=.*namespace="):
            idx.delete_namespace(name="movies", namespace="movies")  # type: ignore[call-arg]

    def test_grpc_delete_namespace_positional_name_raises(self) -> None:
        from pinecone.errors.exceptions import PineconeValueError

        mock_channel = MagicMock()
        idx = _make_grpc_index(mock_channel)

        with pytest.raises(PineconeValueError, match="keyword-only"):
            idx.delete_namespace("my-ns")  # type: ignore[misc]

    def test_grpc_delete_namespace_empty_name_raises(self) -> None:
        from pinecone.errors.exceptions import ValidationError

        mock_channel = MagicMock()
        idx = _make_grpc_index(mock_channel)

        with pytest.raises(ValidationError, match="non-empty"):
            idx.delete_namespace(name="")

    def test_grpc_delete_namespace_whitespace_name_reaches_channel(self) -> None:
        """2026-07 accepts any ASCII name, so a whitespace-only one is deletable."""
        mock_channel = MagicMock()
        mock_channel.delete_namespace.return_value = None
        idx = _make_grpc_index(mock_channel)

        assert idx.delete_namespace(name="   ") is None
        mock_channel.delete_namespace.assert_called_once_with("   ", timeout_s=None)

    def test_grpc_delete_namespace_passes_timeout(self) -> None:
        mock_channel = MagicMock()
        mock_channel.delete_namespace.return_value = None
        idx = _make_grpc_index(mock_channel)

        idx.delete_namespace(name="ns1", timeout=3.0)

        mock_channel.delete_namespace.assert_called_once_with("ns1", timeout_s=3.0)


class TestGrpcListNamespacesPaginated:
    def test_grpc_list_namespaces_paginated_routes_to_channel(self) -> None:
        mock_channel = MagicMock()
        mock_channel.list_namespaces.return_value = {
            "namespaces": [{"name": "ns1", "record_count": 5}],
            "total_count": 1,
        }
        idx = _make_grpc_index(mock_channel)

        result = idx.list_namespaces_paginated()

        assert isinstance(result, ListNamespacesResponse)
        assert len(result.namespaces) == 1
        assert isinstance(result.namespaces[0], NamespaceDescription)
        assert result.namespaces[0].name == "ns1"
        assert result.namespaces[0].record_count == 5
        assert result.total_count == 1
        assert result.pagination is None

    def test_grpc_list_namespaces_paginated_prefix_forwarded(self) -> None:
        mock_channel = MagicMock()
        mock_channel.list_namespaces.return_value = {"namespaces": [], "total_count": 0}
        idx = _make_grpc_index(mock_channel)

        idx.list_namespaces_paginated(prefix="prod-")

        mock_channel.list_namespaces.assert_called_once_with(
            prefix="prod-",
            limit=None,
            pagination_token=None,
            timeout_s=None,
        )

    def test_grpc_list_namespaces_paginated_pagination_token_forwarded(self) -> None:
        mock_channel = MagicMock()
        mock_channel.list_namespaces.return_value = {"namespaces": [], "total_count": 0}
        idx = _make_grpc_index(mock_channel)

        idx.list_namespaces_paginated(pagination_token="tok123")

        mock_channel.list_namespaces.assert_called_once_with(
            prefix=None,
            limit=None,
            pagination_token="tok123",
            timeout_s=None,
        )

    def test_grpc_list_namespaces_paginated_limit_forwarded(self) -> None:
        mock_channel = MagicMock()
        mock_channel.list_namespaces.return_value = {"namespaces": [], "total_count": 0}
        idx = _make_grpc_index(mock_channel)

        idx.list_namespaces_paginated(limit=10)

        mock_channel.list_namespaces.assert_called_once_with(
            prefix=None,
            limit=10,
            pagination_token=None,
            timeout_s=None,
        )

    def test_grpc_list_namespaces_paginated_pagination_next_parsed(self) -> None:
        mock_channel = MagicMock()
        mock_channel.list_namespaces.return_value = {
            "namespaces": [{"name": "a", "record_count": 1}],
            "pagination": {"next": "nexttoken"},
            "total_count": 5,
        }
        idx = _make_grpc_index(mock_channel)

        result = idx.list_namespaces_paginated()

        assert result.pagination is not None
        assert result.pagination.next == "nexttoken"
        assert result.total_count == 5

    def test_grpc_list_namespaces_paginated_timeout_forwarded(self) -> None:
        mock_channel = MagicMock()
        mock_channel.list_namespaces.return_value = {"namespaces": [], "total_count": 0}
        idx = _make_grpc_index(mock_channel)

        idx.list_namespaces_paginated(timeout=2.5)

        mock_channel.list_namespaces.assert_called_once_with(
            prefix=None,
            limit=None,
            pagination_token=None,
            timeout_s=2.5,
        )


class TestGrpcListNamespaces:
    def test_grpc_list_namespaces_yields_pages_single(self) -> None:
        mock_channel = MagicMock()
        idx = _make_grpc_index(mock_channel)

        page = ListNamespacesResponse(
            namespaces=[NamespaceDescription(name="a", record_count=1)],
            pagination=None,
            total_count=1,
        )
        idx.list_namespaces_paginated = MagicMock(return_value=page)

        pages = list(idx.list_namespaces())

        assert pages == [page]
        idx.list_namespaces_paginated.assert_called_once_with(
            prefix=None,
            limit=None,
            pagination_token=None,
            timeout=None,
        )

    def test_grpc_list_namespaces_yields_pages_multi(self) -> None:
        mock_channel = MagicMock()
        idx = _make_grpc_index(mock_channel)

        from pinecone.models.namespaces.models import Pagination

        page1 = ListNamespacesResponse(
            namespaces=[NamespaceDescription(name="a", record_count=1)],
            pagination=Pagination(next="tok1"),
            total_count=2,
        )
        page2 = ListNamespacesResponse(
            namespaces=[NamespaceDescription(name="b", record_count=2)],
            pagination=None,
            total_count=2,
        )
        idx.list_namespaces_paginated = MagicMock(side_effect=[page1, page2])

        pages = list(idx.list_namespaces())

        assert pages == [page1, page2]
        assert idx.list_namespaces_paginated.call_count == 2
        second_call_kwargs = idx.list_namespaces_paginated.call_args_list[1].kwargs
        assert second_call_kwargs["pagination_token"] == "tok1"

    def test_grpc_list_namespaces_skips_empty_pages(self) -> None:
        mock_channel = MagicMock()
        idx = _make_grpc_index(mock_channel)

        from pinecone.models.namespaces.models import Pagination

        empty_page = ListNamespacesResponse(
            namespaces=[],
            pagination=Pagination(next="tok1"),
            total_count=0,
        )
        final_page = ListNamespacesResponse(
            namespaces=[NamespaceDescription(name="a", record_count=1)],
            pagination=None,
            total_count=1,
        )
        idx.list_namespaces_paginated = MagicMock(side_effect=[empty_page, final_page])

        pages = list(idx.list_namespaces())

        assert pages == [final_page]

    def test_grpc_list_namespaces_limit_forwarded(self) -> None:
        mock_channel = MagicMock()
        idx = _make_grpc_index(mock_channel)

        page = ListNamespacesResponse(
            namespaces=[NamespaceDescription(name="a", record_count=1)],
            pagination=None,
            total_count=1,
        )
        idx.list_namespaces_paginated = MagicMock(return_value=page)

        list(idx.list_namespaces(limit=5))

        idx.list_namespaces_paginated.assert_called_once_with(
            prefix=None,
            limit=5,
            pagination_token=None,
            timeout=None,
        )
