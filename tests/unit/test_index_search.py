"""Unit tests for Index.search() method."""

from __future__ import annotations

import httpx
import pytest
import respx

from pinecone import Index
from pinecone.errors.exceptions import PineconeValueError, ValidationError
from pinecone.models.vectors.search import SearchRecordsResponse

INDEX_HOST = "my-index-abc123.svc.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
SEARCH_URL_NS = f"{INDEX_HOST_HTTPS}/records/namespaces/test-ns/search"

SEARCH_RESPONSE: dict[str, object] = {
    "result": {
        "hits": [
            {"_id": "r1", "_score": 0.95, "fields": {"chunk_text": "hello world"}},
            {"_id": "r2", "_score": 0.82, "fields": {"chunk_text": "foo bar"}},
        ]
    },
    "usage": {"read_units": 5, "embed_total_tokens": 10},
}


def _make_index() -> Index:
    return Index(host=INDEX_HOST, api_key="test-key")


class TestSearch:
    """Index.search() — happy paths and request body verification."""

    @respx.mock
    def test_search_with_text_inputs(self) -> None:
        respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_index()
        response = idx.search(namespace="test-ns", top_k=10, inputs={"text": "hello"})

        assert isinstance(response, SearchRecordsResponse)
        assert len(response.result.hits) == 2
        assert response.result.hits[0].id == "r1"
        assert response.result.hits[0].score == pytest.approx(0.95)
        assert response.usage.read_units == 5

    @respx.mock
    def test_search_with_vector(self) -> None:
        route = respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_index()
        idx.search(namespace="test-ns", top_k=10, vector=[0.1, 0.2, 0.3])

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["query"]["vector"] == {"values": [0.1, 0.2, 0.3]}

    @respx.mock
    def test_search_with_id(self) -> None:
        route = respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_index()
        idx.search(namespace="test-ns", top_k=10, id="vec1")

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["query"]["id"] == "vec1"

    @respx.mock
    def test_search_with_filter(self) -> None:
        route = respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_index()
        idx.search(
            namespace="test-ns",
            top_k=10,
            inputs={"text": "hello"},
            filter={"genre": {"$eq": "sci-fi"}},
        )

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["query"]["filter"] == {"genre": {"$eq": "sci-fi"}}

    @respx.mock
    def test_search_with_fields(self) -> None:
        route = respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_index()
        idx.search(
            namespace="test-ns",
            top_k=10,
            inputs={"text": "hello"},
            fields=["chunk_text", "title"],
        )

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["fields"] == ["chunk_text", "title"]

    @respx.mock
    def test_search_accepts_legacy_query_body(self) -> None:
        route = respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_index()
        query = {
            "inputs": {"text": "hello"},
            "top_k": 10,
            "filter": {"genre": {"$eq": "sci-fi"}},
        }
        idx.search(namespace="test-ns", query=query, fields=["chunk_text", "title"])

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["query"] == query
        assert body["fields"] == ["chunk_text", "title"]

    def test_search_query_body_cannot_mix_direct_query_params(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="query cannot be combined"):
            idx.search(
                namespace="test-ns",
                query={"inputs": {"text": "hello"}, "top_k": 10},
                top_k=10,
            )

    @respx.mock
    def test_search_with_rerank(self) -> None:
        route = respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_index()
        idx.search(
            namespace="test-ns",
            top_k=10,
            inputs={"text": "hello"},
            rerank={
                "model": "bge-reranker-v2-m3",
                "rank_fields": ["chunk_text"],
                "top_n": 5,
            },
        )

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["rerank"]["model"] == "bge-reranker-v2-m3"
        assert body["rerank"]["rank_fields"] == ["chunk_text"]
        assert body["rerank"]["top_n"] == 5

    def test_search_namespace_not_string(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="namespace must be a string"):
            idx.search(namespace=123, top_k=10, inputs={"text": "hello"})  # type: ignore[arg-type]

    def test_search_namespace_empty_string(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="namespace must be a non-empty string"):
            idx.search(namespace="", top_k=10, inputs={"text": "hello"})

    def test_search_namespace_whitespace_only(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="namespace must be a non-empty string"):
            idx.search(namespace="   ", top_k=10, inputs={"text": "hello"})

    def test_search_top_k_validation(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="top_k must be a positive integer"):
            idx.search(namespace="test-ns", top_k=0, inputs={"text": "hello"})

    def test_search_rerank_missing_model(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="model"):
            idx.search(
                namespace="test-ns",
                top_k=10,
                inputs={"text": "hello"},
                rerank={"rank_fields": ["text"]},
            )

    def test_search_rerank_missing_rank_fields(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="rank_fields"):
            idx.search(
                namespace="test-ns",
                top_k=10,
                inputs={"text": "hello"},
                rerank={"model": "bge-reranker-v2-m3"},
            )

    def test_search_no_query_source_raises(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="At least one of inputs, vector, or id"):
            idx.search(namespace="test-ns", top_k=5)

    def test_search_keyword_only(self) -> None:
        idx = _make_index()
        with pytest.raises(TypeError):
            idx.search("ns", 10)  # type: ignore[misc]

    @respx.mock
    def test_search_match_terms_included_in_query_body(self) -> None:
        route = respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_index()
        idx.search(
            namespace="test-ns",
            top_k=10,
            inputs={"text": "hello"},
            match_terms={"strategy": "all", "terms": ["animal", "duck"]},
        )

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["query"]["match_terms"] == {
            "strategy": "all",
            "terms": ["animal", "duck"],
        }

    @respx.mock
    def test_search_match_terms_omitted_when_none(self) -> None:
        route = respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_index()
        idx.search(namespace="test-ns", top_k=10, inputs={"text": "hello"})

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert "match_terms" not in body["query"]

    @respx.mock
    def test_search_default_fields_none(self) -> None:
        route = respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_index()
        idx.search(namespace="test-ns", top_k=10, inputs={"text": "hello"})

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert "fields" not in body

    @respx.mock
    def test_search_with_sparse_vector(self) -> None:
        route = respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_index()
        idx.search(
            namespace="test-ns",
            top_k=3,
            vector={"sparse_indices": [10, 20], "sparse_values": [0.5, 0.3]},
        )

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["query"]["vector"] == {"sparse_indices": [10, 20], "sparse_values": [0.5, 0.3]}

    @respx.mock
    def test_search_with_hybrid_vector(self) -> None:
        route = respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_index()
        idx.search(
            namespace="test-ns",
            top_k=3,
            vector={
                "values": [0.1, 0.2, 0.3],
                "sparse_indices": [10, 20],
                "sparse_values": [0.5, 0.3],
            },
        )

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["query"]["vector"] == {
            "values": [0.1, 0.2, 0.3],
            "sparse_indices": [10, 20],
            "sparse_values": [0.5, 0.3],
        }

    @respx.mock
    def test_search_dense_list_still_wrapped_as_values(self) -> None:
        route = respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_index()
        idx.search(namespace="test-ns", top_k=3, vector=[0.1, 0.2, 0.3])

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["query"]["vector"] == {"values": [0.1, 0.2, 0.3]}

    def test_search_vector_dict_unknown_key_raises(self) -> None:
        idx = _make_index()
        with pytest.raises(PineconeValueError, match="vlaues") as exc_info:
            idx.search(namespace="ns", top_k=3, vector={"vlaues": [0.1]})
        assert "values" in str(exc_info.value)
        assert "sparse_indices" in str(exc_info.value)
        assert "sparse_values" in str(exc_info.value)

    @respx.mock
    def test_search_vector_dict_normalizes_sequences(self) -> None:
        route = respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_index()
        idx.search(namespace="test-ns", top_k=3, vector={"values": (0.1, 0.2)})

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["query"]["vector"] == {"values": [0.1, 0.2]}
        assert isinstance(body["query"]["vector"]["values"], list)

    @respx.mock
    def test_search_vector_dict_passes_full_hybrid(self) -> None:
        route = respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_index()
        idx.search(
            namespace="test-ns",
            top_k=3,
            vector={"values": [0.1, 0.2], "sparse_indices": [10], "sparse_values": [0.9]},
        )

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["query"]["vector"] == {
            "values": [0.1, 0.2],
            "sparse_indices": [10],
            "sparse_values": [0.9],
        }
