"""How a typed ``IndexSchema`` encodes a full-text-search string field (#414).

The server reads a string field as filter-only metadata as soon as
``filterable`` is present on it, and discards the ``full_text_search``
config while doing so without reporting an error.
``StringField.filterable`` defaults to ``False`` and used to be emitted
unconditionally, so every full-text-search field built from the typed
models degraded to a filter-only metadata field.

Each expected body below was measured against a faithful copy of the
create-path deserialization chain built at pinecone-db's pinned
``serde 1.0.228`` / ``serde_json 1.0.149`` (pinecone-db ``71dacafc``):
the bodies with no ``filterable`` land in the searchable variant, and the
bodies that carry one land in the metadata variant.
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
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import AsyncHTTPClient, HTTPClient
from pinecone.async_client.indexes import AsyncIndexes
from pinecone.client.indexes import Indexes
from pinecone.models.indexes.requests import ConfigureIndexRequest, CreateIndexRequest
from pinecone.models.indexes.schema import (
    BooleanField,
    DenseVectorField,
    FloatField,
    FullTextSearchConfig,
    IndexSchema,
    IndexSchemaField,
    LegacyMetadataField,
    NgramConfig,
    SparseVectorField,
    StringField,
    StringListField,
)
from tests.factories import make_index_response

BASE_URL = "https://api.test.pinecone.io"


def _create_body(schema: dict[str, Any] | IndexSchema) -> dict[str, Any]:
    request = CreateIndexRequest(schema=schema, name="fts-index")
    body: dict[str, Any] = json.loads(IndexesAdapter.to_create_request(request))
    return body


def _create_field(field: IndexSchemaField, name: str = "body") -> dict[str, Any]:
    fields = _create_body(IndexSchema(fields={name: field}))["schema"]["fields"]
    encoded: dict[str, Any] = fields[name]
    return encoded


_FTS_SHAPES: list[tuple[str, FullTextSearchConfig, dict[str, Any]]] = [
    ("empty config", FullTextSearchConfig(), {}),
    ("language only", FullTextSearchConfig(language="en"), {"language": "en"}),
    (
        "language and stemming",
        FullTextSearchConfig(language="en", stemming=True),
        {"language": "en", "stemming": True},
    ),
    (
        "stemming and stop words",
        FullTextSearchConfig(language="en", stemming=True, stop_words=True),
        {"language": "en", "stemming": True, "stop_words": True},
    ),
    (
        "ngram",
        FullTextSearchConfig(ngram=NgramConfig(min_gram=2, max_gram=3)),
        {"ngram": {"min_gram": 2, "max_gram": 3, "prefix_only": False}},
    ),
    (
        "prefix-only ngram",
        FullTextSearchConfig(ngram=NgramConfig(min_gram=2, max_gram=3, prefix_only=True)),
        {"ngram": {"min_gram": 2, "max_gram": 3, "prefix_only": True}},
    ),
]


@pytest.mark.parametrize(
    "config,expected_config",
    [(config, expected) for _, config, expected in _FTS_SHAPES],
    ids=[label for label, _, _ in _FTS_SHAPES],
)
def test_fts_string_field_omits_filterable(
    config: FullTextSearchConfig, expected_config: dict[str, Any]
) -> None:
    """The caller's search config reaches the server instead of being reinterpreted."""
    assert _create_field(StringField(full_text_search=config)) == {
        "type": "string",
        "full_text_search": expected_config,
    }


def test_fts_string_field_keeps_its_description() -> None:
    encoded = _create_field(
        StringField(
            description="article body", full_text_search=FullTextSearchConfig(language="en")
        )
    )

    assert encoded == {
        "type": "string",
        "description": "article body",
        "full_text_search": {"language": "en"},
    }


def test_explicit_filterable_false_with_fts_omits_filterable() -> None:
    """``filterable=False`` plus a search config describes exactly what a
    search-only field already is, so the search config is what survives."""
    encoded = _create_field(
        StringField(filterable=False, full_text_search=FullTextSearchConfig(language="en"))
    )

    assert encoded == {"type": "string", "full_text_search": {"language": "en"}}


def test_filterable_true_with_fts_sends_both() -> None:
    """No client-side guard on the combination: both keys go on the wire.

    The server then keeps the filter and drops the search config. That is a
    consequence of sending what the caller asked for, not something the SDK
    decides on their behalf.
    """
    encoded = _create_field(
        StringField(filterable=True, full_text_search=FullTextSearchConfig(language="en"))
    )

    assert encoded == {
        "type": "string",
        "filterable": True,
        "full_text_search": {"language": "en"},
    }


@pytest.mark.parametrize(
    "field,expected",
    [
        (StringField(), {"type": "string", "filterable": False}),
        (StringField(filterable=False), {"type": "string", "filterable": False}),
        (StringField(filterable=True), {"type": "string", "filterable": True}),
        (
            StringField(description="d"),
            {"type": "string", "description": "d", "filterable": False},
        ),
    ],
    ids=["bare", "explicit false", "explicit true", "description only"],
)
def test_string_field_without_fts_still_emits_filterable(
    field: StringField, expected: dict[str, Any]
) -> None:
    """A string field with no search config is a metadata field, and the
    server needs ``filterable`` to read it as one."""
    assert _create_field(field) == expected


@pytest.mark.parametrize(
    "field,expected",
    [
        (BooleanField(), {"type": "boolean", "filterable": False}),
        (BooleanField(filterable=True), {"type": "boolean", "filterable": True}),
        (FloatField(), {"type": "float", "filterable": False}),
        (StringListField(), {"type": "string_list", "filterable": False}),
        (StringListField(filterable=True), {"type": "string_list", "filterable": True}),
    ],
    ids=["boolean", "boolean filterable", "float", "string list", "string list filterable"],
)
def test_other_metadata_fields_still_emit_filterable(
    field: IndexSchemaField, expected: dict[str, Any]
) -> None:
    """#374: the server rejects these types outright when the key is absent."""
    assert _create_field(field) == expected


def test_vector_fields_are_untouched() -> None:
    body = _create_body(
        IndexSchema(
            fields={
                "embedding": DenseVectorField(dimension=8, metric="cosine"),
                "sparse_terms": SparseVectorField(),
            }
        )
    )

    assert body["schema"]["fields"] == {
        "embedding": {"type": "dense_vector", "dimension": 8, "metric": "cosine"},
        "sparse_terms": {"type": "sparse_vector"},
    }


HYBRID_SCHEMA = IndexSchema(
    fields={
        "embedding": DenseVectorField(dimension=1536, metric="dotproduct"),
        "sparse_terms": SparseVectorField(),
        "body": StringField(full_text_search=FullTextSearchConfig(language="en")),
    }
)
EXPECTED_HYBRID_FIELDS: dict[str, Any] = {
    "embedding": {"type": "dense_vector", "dimension": 1536, "metric": "dotproduct"},
    "sparse_terms": {"type": "sparse_vector"},
    "body": {"type": "string", "full_text_search": {"language": "en"}},
}


def test_canonical_hybrid_schema_stays_searchable() -> None:
    """Dense plus sparse plus a full-text-search string field: the documented
    hybrid shape, which could not previously be expressed through these models."""
    assert _create_body(HYBRID_SCHEMA)["schema"] == {"fields": EXPECTED_HYBRID_FIELDS}


@pytest.mark.parametrize(
    "schema",
    [
        {"fields": {"body": {"type": "string", "filterable": False, "full_text_search": {}}}},
        {"fields": {"body": {"type": "string", "full_text_search": {"language": "en"}}}},
        {"fields": {"body": {"type": "string"}}},
    ],
    ids=["dict with filterable", "dict without filterable", "bare dict"],
)
def test_dict_schemas_go_verbatim(schema: dict[str, Any]) -> None:
    """A dict schema, including one built by
    :class:`~pinecone.schema_builder.SchemaBuilder`, spells the wire body
    itself and is never rewritten."""
    assert _create_body(schema)["schema"] == schema


def test_configure_request_applies_the_same_rule() -> None:
    request = ConfigureIndexRequest(
        schema=IndexSchema(
            fields={"body": StringField(full_text_search=FullTextSearchConfig(language="en"))}
        )
    )

    body = json.loads(IndexesAdapter.to_configure_request(request))

    assert body == {
        "schema": {"fields": {"body": {"type": "string", "full_text_search": {"language": "en"}}}}
    }


def test_configure_request_without_a_schema_stays_sparse() -> None:
    body = json.loads(IndexesAdapter.to_configure_request(ConfigureIndexRequest(tags={"a": "b"})))

    assert body == {"tags": {"a": "b"}}


def test_to_dict_still_reports_filterable() -> None:
    """``to_dict`` mirrors a describe response, where ``filterable`` is always
    present and meaningful. Only the request encoding changed."""
    schema = IndexSchema(
        fields={
            "body": StringField(full_text_search=FullTextSearchConfig(language="en")),
            "old": LegacyMetadataField(filterable=True),
        }
    )

    assert schema.to_dict() == {
        "fields": {
            "body": {
                "type": "string",
                "description": None,
                "filterable": False,
                "full_text_search": {
                    "language": "en",
                    "stemming": None,
                    "stop_words": None,
                    "ngram": None,
                },
            },
            "old": {"filterable": True},
        }
    }


def test_decoding_a_response_is_unchanged() -> None:
    raw = (
        b'{"fields":{"body":{"type":"string","description":null,"filterable":false,'
        b'"full_text_search":{"language":"en","stemming":false,"stop_words":false,"ngram":null}}}}'
    )

    decoded = msgspec.json.decode(raw, type=IndexSchema)

    body = decoded.fields["body"]
    assert isinstance(body, StringField)
    assert body.filterable is False
    assert body.full_text_search is not None
    assert body.full_text_search.language == "en"


@respx.mock
def test_sync_create_puts_the_searchable_body_on_the_wire() -> None:
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response(name="hybrid"))
    )
    http = HTTPClient(PineconeConfig(api_key="test-key", host=BASE_URL), CONTROL_PLANE_API_VERSION)

    Indexes(http=http).create(name="hybrid", schema=HYBRID_SCHEMA, timeout=-1)

    sent = json.loads(route.calls.last.request.content)
    assert sent["schema"]["fields"] == EXPECTED_HYBRID_FIELDS


@respx.mock
async def test_async_create_puts_the_searchable_body_on_the_wire() -> None:
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response(name="hybrid"))
    )
    http = AsyncHTTPClient(
        PineconeConfig(api_key="test-key", host=BASE_URL), CONTROL_PLANE_API_VERSION
    )
    try:
        await AsyncIndexes(http=http).create(name="hybrid", schema=HYBRID_SCHEMA, timeout=-1)
    finally:
        await http.close()

    sent = json.loads(route.calls.last.request.content)
    assert sent["schema"]["fields"] == EXPECTED_HYBRID_FIELDS
