"""``embed`` / ``sparse_embed`` scoring on documents search, and ``embed_total_tokens``."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import msgspec
import pytest
import respx

from pinecone.async_client.async_index import AsyncIndex
from pinecone.errors.exceptions import ApiError, PineconeValueError
from pinecone.index import Index
from pinecone.models.documents.requests import SearchDocumentsRequest
from pinecone.models.documents.responses import DocumentSearchUsage, SearchDocumentsResponse
from pinecone.models.documents.score_by import (
    DenseVectorQuery,
    DocumentScoringMethod,
    EmbedQuery,
    QueryStringQuery,
    SparseEmbedQuery,
    TextQuery,
)

INDEX_HOST = "embed-index-abc123.svc.us-east-1-aws.pinecone.io"
BASE_URL = f"https://{INDEX_HOST}"
NS = "ns1"
SEARCH_URL = f"{BASE_URL}/namespaces/{NS}/documents/search"

EMBED_SEARCH_RESPONSE: dict[str, Any] = {
    "matches": [
        {"_id": "d1", "_score": 0.6059723, "genre": "animal"},
        {"_id": "d2", "_score": 0.005859375, "genre": "tech"},
    ],
    "namespace": NS,
    "usage": {"read_units": 1},
}


def _body(route: respx.Route) -> dict[str, Any]:
    return dict(json.loads(route.calls.last.request.content))


@pytest.fixture
def index() -> Iterator[Index]:
    client = Index(host=INDEX_HOST, api_key="test-key")
    yield client
    client.close()


@pytest.fixture
async def async_index() -> AsyncIterator[AsyncIndex]:
    client = AsyncIndex(host=INDEX_HOST, api_key="test-key")
    yield client
    await client.close()


class TestClauses:
    @pytest.mark.parametrize(
        "clause,expected",
        [
            (
                EmbedQuery(field="body", query="quick brown fox"),
                {"type": "embed", "field": "body", "query": "quick brown fox"},
            ),
            (
                SparseEmbedQuery(field="body", query="vector database"),
                {"type": "sparse_embed", "field": "body", "query": "vector database"},
            ),
        ],
        ids=["embed", "sparse_embed"],
    )
    def test_encodes_to_the_spec_shape(
        self, clause: DocumentScoringMethod, expected: dict[str, Any]
    ) -> None:
        assert msgspec.to_builtins(clause) == expected

    @pytest.mark.parametrize(
        "raw,cls",
        [
            ({"type": "embed", "field": "body", "query": "fox"}, EmbedQuery),
            ({"type": "sparse_embed", "field": "body", "query": "fox"}, SparseEmbedQuery),
        ],
    )
    def test_dict_form_converts(self, raw: dict[str, Any], cls: type) -> None:
        assert isinstance(msgspec.convert(raw, type=DocumentScoringMethod), cls)

    @pytest.mark.parametrize("kind", ["embed", "sparse_embed"])
    @pytest.mark.parametrize(
        "extra",
        [
            {"values": [0.1]},
            {"sparse_values": {"indices": [1], "values": [0.5]}},
            {"top_k": 3},
        ],
        ids=["values", "sparse_values", "typo"],
    )
    def test_dict_clause_with_an_unknown_key_rejected(
        self, kind: str, extra: dict[str, Any]
    ) -> None:
        """An unknown key, a vector included, fails rather than being dropped."""
        (key,) = extra
        with pytest.raises(msgspec.ValidationError, match=f"unknown field `{key}`"):
            SearchDocumentsRequest(
                score_by=[{"type": kind, "field": "body", "query": "fox", **extra}], top_k=4
            )

    @pytest.mark.parametrize("cls", [EmbedQuery, SparseEmbedQuery])
    @pytest.mark.parametrize("kwarg", ["values", "sparse_values"])
    def test_typed_clause_takes_only_field_and_query(self, cls: type, kwarg: str) -> None:
        with pytest.raises(TypeError, match=kwarg):
            cls(field="body", query="fox", **{kwarg: ["body"]})

    @pytest.mark.parametrize("cls", [EmbedQuery, SparseEmbedQuery])
    def test_empty_field_rejected(self, cls: type) -> None:
        with pytest.raises(ValueError, match="exactly one field"):
            cls(field="", query="fox")

    @pytest.mark.parametrize("cls", [EmbedQuery, SparseEmbedQuery])
    def test_empty_query_rejected(self, cls: type) -> None:
        with pytest.raises(ValueError, match="non-empty 'query'"):
            cls(field="body", query="")

    @pytest.mark.parametrize(
        "clause",
        [
            EmbedQuery(field="body", query="fox"),
            SparseEmbedQuery(field="body", query="fox"),
            {"type": "embed", "field": "body", "query": "fox"},
        ],
        ids=["embed", "sparse_embed", "embed dict"],
    )
    @pytest.mark.parametrize(
        "other",
        [
            TextQuery(query="fox", fields=["body"]),
            QueryStringQuery(query="body:fox"),
            DenseVectorQuery(field="v", values=[0.1]),
        ],
        ids=["text", "query_string", "dense_vector"],
    )
    def test_must_stand_alone(self, clause: Any, other: DocumentScoringMethod) -> None:
        with pytest.raises(ValueError, match="'embed', and 'sparse_embed' clauses must appear"):
            SearchDocumentsRequest(score_by=[clause, other], top_k=5)
        with pytest.raises(ValueError, match="must appear alone"):
            SearchDocumentsRequest(score_by=[other, clause], top_k=5)


class TestSearchParameters:
    """``parameters`` on an ``embed`` clause overrides the field's read parameters."""

    def test_set_is_sent_alongside_the_other_fields(self) -> None:
        clause = EmbedQuery(
            fields=["body"],
            query="Is the Eiffel Tower taller than 300 m?",
            parameters={"task_type": "fact_verification"},
        )

        assert msgspec.to_builtins(clause) == {
            "type": "embed",
            "field": "body",
            "query": "Is the Eiffel Tower taller than 300 m?",
            "parameters": {"task_type": "fact_verification"},
        }

    def test_unset_is_omitted(self) -> None:
        clause = EmbedQuery(field="body", query="fox")

        assert clause.parameters is None
        assert "parameters" not in msgspec.to_builtins(clause)

    def test_dict_form_converts(self) -> None:
        raw = {
            "type": "embed",
            "field": "body",
            "query": "fox",
            "parameters": {"task_type": "question_answering"},
        }

        clause = msgspec.convert(raw, type=DocumentScoringMethod)

        assert isinstance(clause, EmbedQuery)
        assert clause.parameters == {"task_type": "question_answering"}
        assert msgspec.to_builtins(clause) == raw

    def test_values_are_left_to_the_server(self) -> None:
        """Neither the task nor the key is checked here; the server owns that."""
        parameters = {"task_type": "summarization", "truncate": "NONE"}

        clause = EmbedQuery(field="body", query="fox", parameters=parameters)

        assert msgspec.to_builtins(clause)["parameters"] == parameters

    def test_sparse_embed_takes_none(self) -> None:
        with pytest.raises(TypeError, match="parameters"):
            SparseEmbedQuery(  # type: ignore[call-arg]
                field="body", query="fox", parameters={"task_type": "retrieval_query"}
            )
        with pytest.raises(msgspec.ValidationError, match="unknown field `parameters`"):
            SearchDocumentsRequest(
                score_by=[
                    {
                        "type": "sparse_embed",
                        "field": "body",
                        "query": "fox",
                        "parameters": {"task_type": "retrieval_query"},
                    }
                ],
                top_k=4,
            )


class TestUsage:
    def test_embed_total_tokens_read_when_present(self) -> None:
        payload = {**EMBED_SEARCH_RESPONSE, "usage": {"read_units": 1, "embed_total_tokens": 4}}

        response = SearchDocumentsResponse.from_dict(payload)

        assert response.usage == DocumentSearchUsage(read_units=1, embed_total_tokens=4)
        assert response.to_dict()["usage"] == {"read_units": 1, "embed_total_tokens": 4}

    def test_embed_total_tokens_absent_is_none(self) -> None:
        response = SearchDocumentsResponse.from_dict(EMBED_SEARCH_RESPONSE)

        assert response.usage is not None
        assert response.usage.embed_total_tokens is None
        assert response.to_dict()["usage"] == {"read_units": 1}


class TestSyncSearch:
    @respx.mock
    def test_embed_search_wire_and_response(self, index: Index) -> None:
        route = respx.post(SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json={**EMBED_SEARCH_RESPONSE, "usage": {"read_units": 1, "embed_total_tokens": 4}},
            )
        )

        result = index.documents.search(
            namespace=NS,
            top_k=4,
            score_by=[EmbedQuery(field="body", query="quick brown fox jumps")],
            include_fields=["genre"],
            filter={"genre": {"$eq": "animal"}},
        )

        assert _body(route) == {
            "score_by": [{"type": "embed", "field": "body", "query": "quick brown fox jumps"}],
            "top_k": 4,
            "include_fields": ["genre"],
            "filter": {"genre": {"$eq": "animal"}},
        }
        assert [doc.id for doc in result.matches] == ["d1", "d2"]
        assert result.matches[0].genre == "animal"
        assert result.usage is not None
        assert result.usage.embed_total_tokens == 4

    @respx.mock
    def test_embed_search_with_parameters(self, index: Index) -> None:
        route = respx.post(SEARCH_URL).mock(
            return_value=httpx.Response(200, json=EMBED_SEARCH_RESPONSE)
        )

        index.documents.search(
            namespace=NS,
            top_k=5,
            score_by=[
                EmbedQuery(
                    field="body",
                    query="Is the Eiffel Tower taller than 300 m?",
                    parameters={"task_type": "fact_verification"},
                )
            ],
            include_fields=["body"],
        )

        assert _body(route) == {
            "score_by": [
                {
                    "type": "embed",
                    "field": "body",
                    "query": "Is the Eiffel Tower taller than 300 m?",
                    "parameters": {"task_type": "fact_verification"},
                }
            ],
            "top_k": 5,
            "include_fields": ["body"],
        }

    @respx.mock
    def test_sparse_embed_search_from_a_dict(self, index: Index) -> None:
        route = respx.post(SEARCH_URL).mock(
            return_value=httpx.Response(200, json=EMBED_SEARCH_RESPONSE)
        )

        result = index.documents.search(
            namespace=NS,
            top_k=2,
            score_by=[{"type": "sparse_embed", "field": "body", "query": "Vector database!"}],
        )

        assert _body(route)["score_by"] == [
            {"type": "sparse_embed", "field": "body", "query": "Vector database!"}
        ]
        assert result.usage is not None and result.usage.embed_total_tokens is None

    @respx.mock
    def test_vector_on_a_dict_clause_rejected_before_the_request(self, index: Index) -> None:
        route = respx.post(SEARCH_URL).mock(
            return_value=httpx.Response(200, json=EMBED_SEARCH_RESPONSE)
        )

        with pytest.raises(PineconeValueError, match="unknown field `values`"):
            index.documents.search(
                namespace=NS,
                top_k=4,
                score_by=[{"type": "embed", "field": "body", "query": "fox", "values": [0.1]}],
            )

        assert not route.called

    def test_combined_clause_rejected_before_the_request(self, index: Index) -> None:
        with pytest.raises(PineconeValueError, match="must appear alone"):
            index.documents.search(
                namespace=NS,
                top_k=2,
                score_by=[
                    EmbedQuery(field="body", query="fox"),
                    TextQuery(query="fox", fields=["body"]),
                ],
            )

    @respx.mock
    def test_unstored_include_fields_400_surfaced_intact(self, index: Index) -> None:
        server_message = (
            "Cannot return field with 'store_text: false': 'body'. Its text is embedded "
            "but not stored, so there is no text to return. Remove this name from "
            "'include_fields', or use '*' to return every field the index stores."
        )
        respx.post(SEARCH_URL).mock(
            return_value=httpx.Response(400, json={"code": 3, "message": server_message})
        )

        with pytest.raises(ApiError) as excinfo:
            index.documents.search(
                namespace=NS,
                top_k=2,
                score_by=[EmbedQuery(field="body", query="fox")],
                include_fields=["body"],
            )

        assert server_message in str(excinfo.value)


class TestAsyncSearch:
    @respx.mock
    async def test_embed_search_wire_and_response(self, async_index: AsyncIndex) -> None:
        route = respx.post(SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json={**EMBED_SEARCH_RESPONSE, "usage": {"read_units": 1, "embed_total_tokens": 4}},
            )
        )

        result = await async_index.documents.search(
            namespace=NS,
            top_k=4,
            score_by=[SparseEmbedQuery(field="body", query="quick brown fox")],
        )

        assert _body(route) == {
            "score_by": [{"type": "sparse_embed", "field": "body", "query": "quick brown fox"}],
            "top_k": 4,
        }
        assert result.matches[0].id == "d1"
        assert result.usage is not None and result.usage.embed_total_tokens == 4

    @respx.mock
    async def test_embed_search_with_parameters_from_a_dict(self, async_index: AsyncIndex) -> None:
        route = respx.post(SEARCH_URL).mock(
            return_value=httpx.Response(200, json=EMBED_SEARCH_RESPONSE)
        )
        clause = {
            "type": "embed",
            "field": "body",
            "query": "how does the router pick a shard",
            "parameters": {"task_type": "code_retrieval_query"},
        }

        await async_index.documents.search(namespace=NS, top_k=3, score_by=[clause])

        assert _body(route) == {"score_by": [clause], "top_k": 3}

    async def test_combined_clause_rejected_before_the_request(
        self, async_index: AsyncIndex
    ) -> None:
        with pytest.raises(PineconeValueError, match="must appear alone"):
            await async_index.documents.search(
                namespace=NS,
                top_k=2,
                score_by=[
                    QueryStringQuery(query="body:fox"),
                    {"type": "embed", "field": "body", "query": "fox"},
                ],
            )
