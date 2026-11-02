"""The ``embed`` / ``sparse_embed`` aspects and ``store_text`` on a string field.

The describe payloads are the ones the server returned for the same create
requests in a local run of pinecone-db PRs 20074 / 20229 / 20327: resolved
model, dimension, metric and parameters are echoed, a stored field reports
the full-text-search config it gained by default, and ``store_text`` is always
reported on a search field.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import msgspec.json
import pytest
import respx

from pinecone._internal.adapters.indexes_adapter import IndexesAdapter
from pinecone._internal.config import PineconeConfig
from pinecone._internal.http_client import HTTPClient
from pinecone.client.indexes import Indexes
from pinecone.models.indexes.requests import CreateIndexRequest
from pinecone.models.indexes.schema import (
    FieldEmbedConfig,
    FieldSparseEmbedConfig,
    FullTextSearchConfig,
    IndexSchema,
    StringField,
)
from pinecone.schema_builder import SchemaBuilder
from tests.factories import make_index_response

BASE_URL = "https://api.test.pinecone.io"

DENSE_WRITE = {"dimension": 1024, "input_type": "passage", "truncate": "END"}
DENSE_READ = {"dimension": 1024, "input_type": "query", "truncate": "END"}
SPARSE_WRITE = {"input_type": "passage", "max_tokens_per_sequence": 512, "truncate": "END"}
SPARSE_READ = {"input_type": "query", "max_tokens_per_sequence": 512, "truncate": "END"}
DEFAULT_FTS = {"language": "en", "stemming": False, "stop_words": False}

DESCRIBED_EMBED_STORED: dict[str, Any] = {
    "type": "string",
    "description": None,
    "full_text_search": DEFAULT_FTS,
    "embed": {
        "model": "llama-text-embed-v2",
        "dimension": 1024,
        "metric": "cosine",
        "write_parameters": DENSE_WRITE,
        "read_parameters": DENSE_READ,
    },
    "store_text": True,
}
DESCRIBED_EMBED_UNSTORED: dict[str, Any] = {
    "type": "string",
    "description": None,
    "embed": {
        "model": "llama-text-embed-v2",
        "dimension": 1024,
        "metric": "cosine",
        "write_parameters": DENSE_WRITE,
        "read_parameters": DENSE_READ,
    },
    "store_text": False,
}
DESCRIBED_SPARSE_STORED: dict[str, Any] = {
    "type": "string",
    "description": None,
    "full_text_search": DEFAULT_FTS,
    "sparse_embed": {
        "model": "pinecone-sparse-english-v0",
        "write_parameters": SPARSE_WRITE,
        "read_parameters": SPARSE_READ,
    },
    "store_text": True,
}


def _decode_field(raw: dict[str, Any]) -> StringField:
    schema = msgspec.json.decode(json.dumps({"fields": {"body": raw}}), type=IndexSchema)
    field = schema.fields["body"]
    assert isinstance(field, StringField)
    return field


def _create_fields(schema: dict[str, Any] | IndexSchema) -> dict[str, Any]:
    request = CreateIndexRequest(schema=schema, name="embed-index")
    body: dict[str, Any] = json.loads(IndexesAdapter.to_create_request(request))
    fields: dict[str, Any] = body["schema"]["fields"]
    return fields


class TestDescribeDecode:
    def test_stored_embed_field_keeps_its_resolved_config(self) -> None:
        field = _decode_field(DESCRIBED_EMBED_STORED)

        assert field.embed == FieldEmbedConfig(
            model="llama-text-embed-v2",
            dimension=1024,
            metric="cosine",
            write_parameters=DENSE_WRITE,
            read_parameters=DENSE_READ,
        )
        assert field.sparse_embed is None
        assert field.store_text is True
        assert field.full_text_search == FullTextSearchConfig(
            language="en", stemming=False, stop_words=False
        )
        assert field.filterable is False

    def test_unstored_embed_field_reports_store_text_false(self) -> None:
        field = _decode_field(DESCRIBED_EMBED_UNSTORED)

        assert field.store_text is False
        assert field.full_text_search is None
        assert field.embed is not None
        assert field.embed.dimension == 1024

    def test_sparse_embed_field_keeps_its_resolved_config(self) -> None:
        field = _decode_field(DESCRIBED_SPARSE_STORED)

        assert field.sparse_embed == FieldSparseEmbedConfig(
            model="pinecone-sparse-english-v0",
            write_parameters=SPARSE_WRITE,
            read_parameters=SPARSE_READ,
        )
        assert field.embed is None

    def test_embed_processing_location_is_decoded(self) -> None:
        raw = {
            **DESCRIBED_EMBED_STORED,
            "embed": {**DESCRIBED_EMBED_STORED["embed"], "processing_location": "us"},
        }

        field = _decode_field(raw)

        assert field.embed is not None and field.embed.processing_location == "us"

    def test_one_field_carries_all_three_aspects(self) -> None:
        raw = {
            **DESCRIBED_EMBED_STORED,
            "embed": {**DESCRIBED_EMBED_STORED["embed"], "dimension": 2048},
            "sparse_embed": DESCRIBED_SPARSE_STORED["sparse_embed"],
        }

        field = _decode_field(raw)

        assert field.full_text_search is not None
        assert field.embed is not None and field.embed.dimension == 2048
        assert field.sparse_embed is not None
        assert field.sparse_embed.model == "pinecone-sparse-english-v0"

    def test_plain_fts_field_still_decodes_without_aspects(self) -> None:
        field = _decode_field({"type": "string", "full_text_search": DEFAULT_FTS})

        assert field.embed is None
        assert field.sparse_embed is None
        assert field.store_text is None

    @respx.mock
    def test_describe_surfaces_the_aspects(self) -> None:
        payload = make_index_response(
            name="embed-index",
            schema={"fields": {"title": DESCRIBED_EMBED_STORED, "body": DESCRIBED_SPARSE_STORED}},
        )
        respx.get(f"{BASE_URL}/indexes/embed-index").mock(
            return_value=httpx.Response(200, json=payload)
        )
        http = HTTPClient(PineconeConfig(api_key="k", host=BASE_URL), "2026-07")
        try:
            model = Indexes(http=http).describe("embed-index")
        finally:
            http.close()

        title = model.schema.fields["title"]
        body = model.schema.fields["body"]
        assert isinstance(title, StringField) and isinstance(body, StringField)
        assert title.embed is not None and title.embed.model == "llama-text-embed-v2"
        assert body.sparse_embed is not None
        assert model.schema.to_dict()["fields"]["title"]["store_text"] is True


class TestCreateEncoding:
    @pytest.mark.parametrize(
        "field,expected",
        [
            (StringField(embed=FieldEmbedConfig()), {"type": "string", "embed": {}}),
            (
                StringField(embed=FieldEmbedConfig(model="llama-text-embed-v2", dimension=512)),
                {"type": "string", "embed": {"model": "llama-text-embed-v2", "dimension": 512}},
            ),
            (
                StringField(embed=FieldEmbedConfig(processing_location="us")),
                {"type": "string", "embed": {"processing_location": "us"}},
            ),
            (
                StringField(sparse_embed=FieldSparseEmbedConfig()),
                {"type": "string", "sparse_embed": {}},
            ),
            (
                StringField(store_text=False, embed=FieldEmbedConfig()),
                {"type": "string", "embed": {}, "store_text": False},
            ),
            (
                StringField(store_text=True, full_text_search=FullTextSearchConfig()),
                {"type": "string", "full_text_search": {}, "store_text": True},
            ),
            (StringField(store_text=False), {"type": "string", "store_text": False}),
        ],
        ids=[
            "embed",
            "embed model dimension",
            "embed processing location",
            "sparse",
            "unstored",
            "stored fts",
            "flag only",
        ],
    )
    def test_search_keys_go_on_the_wire_without_filterable(
        self, field: StringField, expected: dict[str, Any]
    ) -> None:
        """The server rejects ``filterable`` alongside an aspect or ``store_text``."""
        assert _create_fields(IndexSchema(fields={"body": field}))["body"] == expected

    def test_explicit_filterable_true_is_still_sent(self) -> None:
        field = StringField(filterable=True, embed=FieldEmbedConfig())

        assert _create_fields(IndexSchema(fields={"body": field}))["body"] == {
            "type": "string",
            "filterable": True,
            "embed": {},
        }

    @pytest.mark.parametrize(
        "raw", [DESCRIBED_EMBED_STORED, DESCRIBED_EMBED_UNSTORED, DESCRIBED_SPARSE_STORED]
    )
    def test_describe_then_create_sends_the_described_field(self, raw: dict[str, Any]) -> None:
        described = msgspec.json.decode(json.dumps({"fields": {"body": raw}}), type=IndexSchema)

        sent = _create_fields(described)["body"]

        assert sent == {k: v for k, v in raw.items() if v is not None}


class TestSchemaBuilder:
    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            ({"embed": True}, {"type": "string", "embed": {}}),
            ({"embed": {}}, {"type": "string", "embed": {}}),
            (
                {"embed": FieldEmbedConfig(dimension=512, read_parameters={"truncate": "START"})},
                {
                    "type": "string",
                    "embed": {"dimension": 512, "read_parameters": {"truncate": "START"}},
                },
            ),
            (
                {"embed": FieldEmbedConfig(processing_location="us")},
                {"type": "string", "embed": {"processing_location": "us"}},
            ),
            ({"sparse_embed": True}, {"type": "string", "sparse_embed": {}}),
            (
                {"sparse_embed": FieldSparseEmbedConfig(model="pinecone-sparse-english-v0")},
                {"type": "string", "sparse_embed": {"model": "pinecone-sparse-english-v0"}},
            ),
            (
                {"embed": True, "store_text": False},
                {"type": "string", "embed": {}, "store_text": False},
            ),
            (
                {"full_text_search": {"stemming": True}, "embed": True, "sparse_embed": True},
                {
                    "type": "string",
                    "full_text_search": {"stemming": True},
                    "embed": {},
                    "sparse_embed": {},
                },
            ),
            ({"store_text": True}, {"type": "string", "store_text": True}),
            ({"embed": False}, {"type": "string", "filterable": False}),
        ],
        ids=[
            "embed true",
            "embed dict",
            "embed struct",
            "embed processing location",
            "sparse true",
            "sparse struct",
            "unstored",
            "all three",
            "flag only",
            "embed false",
        ],
    )
    def test_add_string_field(self, kwargs: dict[str, Any], expected: dict[str, Any]) -> None:
        schema = SchemaBuilder().add_string_field("body", **kwargs).build()

        assert schema["fields"]["body"] == expected

    def test_store_text_is_absent_unless_given(self) -> None:
        schema = SchemaBuilder().add_string_field("body", embed=True).build()

        assert "store_text" not in schema["fields"]["body"]

    def test_explicit_filterable_true_is_kept(self) -> None:
        schema = SchemaBuilder().add_string_field("body", embed=True, filterable=True).build()

        assert schema["fields"]["body"] == {"type": "string", "embed": {}, "filterable": True}

    def test_embed_dict_is_copied(self) -> None:
        config: dict[str, Any] = {"model": "llama-text-embed-v2"}
        builder = SchemaBuilder().add_string_field("body", embed=config)
        config["dimension"] = 512

        assert builder.build()["fields"]["body"]["embed"] == {"model": "llama-text-embed-v2"}
