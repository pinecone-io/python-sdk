"""Unit tests for backcompat shims: SearchQuery, SearchQueryVector, SearchRerank."""

from __future__ import annotations

import pytest


class TestSearchQueryShim:
    def test_legacy_search_query_importable_from_old_path(self) -> None:
        from pinecone.db_data.dataclasses.search_query import SearchQuery

        obj = SearchQuery(inputs={"text": "x"}, top_k=5)
        assert isinstance(obj, SearchQuery)

    def test_legacy_search_query_is_canonical_type(self) -> None:
        from pinecone.db_data.dataclasses.search_query import SearchQuery as Legacy
        from pinecone.models.vectors.search import SearchQuery as Canonical

        assert Legacy is Canonical

    def test_legacy_search_query_to_dict_excludes_none(self) -> None:
        from pinecone.db_data.dataclasses.search_query import SearchQuery

        obj = SearchQuery(inputs={"text": "x"}, top_k=5)
        result = obj.to_dict()
        assert result == {"inputs": {"text": "x"}, "top_k": 5}
        assert "filter" not in result
        assert "vector" not in result
        assert "id" not in result
        assert "match_terms" not in result

    def test_search_query_to_dict_includes_non_none_optional_fields(self) -> None:
        from pinecone.db_data.dataclasses.search_query import SearchQuery

        obj = SearchQuery(
            inputs={"text": "hello"},
            top_k=10,
            filter={"genre": "action"},
            id="rec-1",
        )
        result = obj.to_dict()
        assert result["inputs"] == {"text": "hello"}
        assert result["top_k"] == 10
        assert result["filter"] == {"genre": "action"}
        assert result["id"] == "rec-1"
        assert "vector" not in result
        assert "match_terms" not in result

    def test_search_query_in_all(self) -> None:
        import pinecone.db_data.dataclasses.search_query as mod

        assert "SearchQuery" in mod.__all__

    def test_search_query_getitem(self) -> None:
        from pinecone.db_data.dataclasses.search_query import SearchQuery

        obj = SearchQuery(inputs={"text": "q"}, top_k=3)
        assert obj["inputs"] == {"text": "q"}
        assert obj["top_k"] == 3

    def test_search_query_contains(self) -> None:
        from pinecone.db_data.dataclasses.search_query import SearchQuery

        obj = SearchQuery(inputs={"text": "q"}, top_k=3)
        assert "inputs" in obj
        assert "top_k" in obj
        assert "filter" in obj
        assert "nonexistent" not in obj

    def test_search_query_iter(self) -> None:
        from pinecone.db_data.dataclasses.search_query import SearchQuery

        obj = SearchQuery(inputs={"text": "q"}, top_k=3)
        fields = list(obj)
        assert "inputs" in fields
        assert "top_k" in fields


class TestSearchQueryVectorShim:
    def test_legacy_search_query_vector_importable_from_old_path(self) -> None:
        from pinecone.db_data.dataclasses.search_query_vector import SearchQueryVector

        obj = SearchQueryVector(values=[0.1, 0.2])
        assert isinstance(obj, SearchQueryVector)

    def test_legacy_search_query_vector_is_canonical_type(self) -> None:
        from pinecone.db_data.dataclasses.search_query_vector import (
            SearchQueryVector as Legacy,
        )
        from pinecone.models.vectors.search import SearchQueryVector as Canonical

        assert Legacy is Canonical

    def test_search_query_vector_to_dict_excludes_none(self) -> None:
        from pinecone.db_data.dataclasses.search_query_vector import SearchQueryVector

        obj = SearchQueryVector(values=[0.1, 0.2, 0.3])
        result = obj.to_dict()
        assert result == {"values": [0.1, 0.2, 0.3]}
        assert "sparse_values" not in result
        assert "sparse_indices" not in result

    def test_search_query_vector_empty_to_dict(self) -> None:
        from pinecone.db_data.dataclasses.search_query_vector import SearchQueryVector

        obj = SearchQueryVector()
        assert obj.to_dict() == {}

    def test_search_query_vector_in_all(self) -> None:
        import pinecone.db_data.dataclasses.search_query_vector as mod

        assert "SearchQueryVector" in mod.__all__

    def test_search_query_vector_with_sparse(self) -> None:
        from pinecone.db_data.dataclasses.search_query_vector import SearchQueryVector

        obj = SearchQueryVector(
            values=[0.1, 0.2],
            sparse_values=[0.5],
            sparse_indices=[3],
        )
        result = obj.to_dict()
        assert result["values"] == [0.1, 0.2]
        assert result["sparse_values"] == [0.5]
        assert result["sparse_indices"] == [3]


class TestSearchRerankShim:
    def test_legacy_search_rerank_importable_from_old_path(self) -> None:
        from pinecone.db_data.dataclasses.search_rerank import SearchRerank

        obj = SearchRerank(model="bge-reranker-v2-m3")
        assert isinstance(obj, SearchRerank)

    def test_legacy_search_rerank_is_canonical_type(self) -> None:
        from pinecone.db_data.dataclasses.search_rerank import SearchRerank as Legacy
        from pinecone.models.vectors.search import SearchRerank as Canonical

        assert Legacy is Canonical

    def test_search_rerank_to_dict_excludes_none(self) -> None:
        from pinecone.db_data.dataclasses.search_rerank import SearchRerank

        obj = SearchRerank(model="bge-reranker-v2-m3")
        result = obj.to_dict()
        assert result == {"model": "bge-reranker-v2-m3"}
        assert "top_n" not in result
        assert "rank_fields" not in result
        assert "parameters" not in result
        assert "query" not in result

    def test_search_rerank_to_dict_includes_non_none_fields(self) -> None:
        from pinecone.db_data.dataclasses.search_rerank import SearchRerank

        obj = SearchRerank(
            model="bge-reranker-v2-m3",
            top_n=5,
            rank_fields=["text"],
            query="hello world",
        )
        result = obj.to_dict()
        assert result["model"] == "bge-reranker-v2-m3"
        assert result["top_n"] == 5
        assert result["rank_fields"] == ["text"]
        assert result["query"] == "hello world"
        assert "parameters" not in result

    def test_search_rerank_in_all(self) -> None:
        import pinecone.db_data.dataclasses.search_rerank as mod

        assert "SearchRerank" in mod.__all__

    def test_search_rerank_getitem(self) -> None:
        from pinecone.db_data.dataclasses.search_rerank import SearchRerank

        obj = SearchRerank(model="test-model", top_n=3)
        assert obj["model"] == "test-model"
        assert obj["top_n"] == 3

    def test_search_rerank_contains(self) -> None:
        from pinecone.db_data.dataclasses.search_rerank import SearchRerank

        obj = SearchRerank(model="test-model")
        assert "model" in obj
        assert "top_n" in obj
        assert "nonexistent" not in obj


class TestDataclassesPackageExports:
    def test_package_exports_search_query(self) -> None:
        import pinecone.db_data.dataclasses as pkg
        from pinecone.models.vectors.search import SearchQuery

        assert pkg.SearchQuery is SearchQuery

    def test_package_exports_search_query_vector(self) -> None:
        import pinecone.db_data.dataclasses as pkg
        from pinecone.models.vectors.search import SearchQueryVector

        assert pkg.SearchQueryVector is SearchQueryVector

    def test_package_exports_search_rerank(self) -> None:
        import pinecone.db_data.dataclasses as pkg
        from pinecone.models.vectors.search import SearchRerank

        assert pkg.SearchRerank is SearchRerank

    def test_package_does_not_export_dictlike(self) -> None:
        import pinecone.db_data.dataclasses as pkg

        assert "DictLike" not in pkg.__all__
        assert not hasattr(pkg, "DictLike")

    def test_package_all_symbols_accessible(self) -> None:
        import pinecone.db_data.dataclasses as pkg

        for name in pkg.__all__:
            assert hasattr(pkg, name), f"__all__ lists {name!r} but it is not an attribute"


class TestAsyncSearchLegacyQueryKwarg:
    """v8 backcompat: AsyncIndex.search/search_records accept query=SearchQuery(...)."""

    @pytest.mark.asyncio
    async def test_search_records_with_legacy_query_kwarg(self) -> None:
        import warnings
        from unittest.mock import AsyncMock, patch

        from pinecone.async_client.async_index import AsyncIndex
        from pinecone.models.vectors.search import SearchQuery, SearchRecordsResponse

        idx = object.__new__(AsyncIndex)
        mock_response = AsyncMock(spec=SearchRecordsResponse)
        with patch.object(
            AsyncIndex, "search", new_callable=AsyncMock, return_value=mock_response
        ) as m:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                await idx.search_records(
                    namespace="ns",
                    query=SearchQuery(inputs={"text": "hi"}, top_k=3),
                )
            assert any(issubclass(w.category, DeprecationWarning) for w in caught)
        m.assert_awaited_once()
        kw = m.await_args.kwargs
        assert kw["namespace"] == "ns"
        assert kw["top_k"] == 3
        assert kw["inputs"] == {"text": "hi"}
        assert kw["query"] is None  # already unpacked by search_records before forwarding

    @pytest.mark.asyncio
    async def test_search_query_and_top_k_both_provided_raises(self) -> None:
        from pinecone.async_client.async_index import AsyncIndex
        from pinecone.models.vectors.search import SearchQuery

        idx = object.__new__(AsyncIndex)
        with pytest.raises(TypeError, match="received both 'query='"):
            await idx.search(
                namespace="ns",
                top_k=5,
                query=SearchQuery(inputs={"text": "x"}, top_k=3),
            )

    @pytest.mark.asyncio
    async def test_search_query_invalid_type_raises(self) -> None:
        from pinecone.async_client.async_index import AsyncIndex

        idx = object.__new__(AsyncIndex)
        with pytest.raises(TypeError, match="must be a SearchQuery or Mapping"):
            await idx.search(namespace="ns", query=12345)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_search_without_top_k_or_query_raises(self) -> None:
        from pinecone.async_client.async_index import AsyncIndex
        from pinecone.errors.exceptions import ValidationError

        idx = object.__new__(AsyncIndex)
        with pytest.raises(ValidationError, match="top_k is required"):
            await idx.search(namespace="ns", inputs={"text": "x"})
