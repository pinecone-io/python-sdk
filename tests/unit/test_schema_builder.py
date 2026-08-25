"""Unit tests for SchemaBuilder (2026-07 create-schema rules).

Migrated from tests/unit/preview/test_schema_builder.py when the builder
graduated out of the preview package (#106).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from pinecone.schema_builder import SchemaBuilder

# ---------------------------------------------------------------------------
# add_dense_vector_field
# ---------------------------------------------------------------------------


def test_dense_vector_field_basic() -> None:
    schema = SchemaBuilder().add_dense_vector_field("vec", dimension=768, metric="cosine").build()
    assert schema == {
        "fields": {"vec": {"type": "dense_vector", "dimension": 768, "metric": "cosine"}}
    }


def test_dense_vector_field_with_description() -> None:
    schema = (
        SchemaBuilder()
        .add_dense_vector_field("vec", dimension=1536, metric="dotproduct", description="ada-002")
        .build()
    )
    assert schema["fields"]["vec"]["description"] == "ada-002"


def test_dense_vector_field_additional_options() -> None:
    schema = (
        SchemaBuilder()
        .add_dense_vector_field("vec", dimension=64, metric="cosine", extra="val")
        .build()
    )
    assert schema["fields"]["vec"]["extra"] == "val"


def test_dense_vector_field_no_description_omitted() -> None:
    schema = SchemaBuilder().add_dense_vector_field("vec", dimension=64, metric="cosine").build()
    assert "description" not in schema["fields"]["vec"]


# ---------------------------------------------------------------------------
# add_dense_vector_field: dimension range 1..20000 (2026-07)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dimension", [1, 2, 19999, 20000])
def test_dense_vector_dimension_bounds_accepted(dimension: int) -> None:
    schema = (
        SchemaBuilder().add_dense_vector_field("vec", dimension=dimension, metric="cosine").build()
    )
    assert schema["fields"]["vec"]["dimension"] == dimension


@pytest.mark.parametrize("dimension", [0, -1, 20001, 1_000_000])
def test_dense_vector_dimension_out_of_range_rejected(dimension: int) -> None:
    from pinecone.errors.exceptions import PineconeValueError

    with pytest.raises(PineconeValueError) as excinfo:
        SchemaBuilder().add_dense_vector_field("vec", dimension=dimension, metric="cosine")
    message = str(excinfo.value)
    assert "'vec'" in message
    assert "between 1 and 20000 inclusive" in message
    assert str(dimension) in message


def test_dense_vector_dimension_rejection_leaves_builder_unchanged() -> None:
    from pinecone.errors.exceptions import PineconeValueError

    builder = SchemaBuilder()
    with pytest.raises(PineconeValueError):
        builder.add_dense_vector_field("vec", dimension=0, metric="cosine")
    assert builder.build() == {"fields": {}}


# ---------------------------------------------------------------------------
# add_sparse_vector_field
# ---------------------------------------------------------------------------


def test_sparse_vector_field_defaults() -> None:
    schema = SchemaBuilder().add_sparse_vector_field("sparse").build()
    assert schema == {"fields": {"sparse": {"type": "sparse_vector"}}}


def test_sparse_vector_field_emits_no_key_beyond_type_and_description() -> None:
    """#350: the field carried a ``metric`` the create schema has no place for.

    Asserted as an exact key set, not as ``"metric" not in field``, so any
    other unmodelled key reintroduced here fails too.
    """
    bare = SchemaBuilder().add_sparse_vector_field("sparse").build()
    described = SchemaBuilder().add_sparse_vector_field("sparse", description="BM25").build()

    assert set(bare["fields"]["sparse"]) == {"type"}
    assert set(described["fields"]["sparse"]) == {"type", "description"}


@pytest.mark.parametrize("option", ["metric", "dimension"])
def test_sparse_vector_field_rejects_the_keys_it_does_not_have(option: str) -> None:
    """The 9.x spelling fails loudly rather than being dropped on the floor."""
    from pinecone.errors.exceptions import PineconeValueError

    with pytest.raises(PineconeValueError) as excinfo:
        SchemaBuilder().add_sparse_vector_field("sparse", **{option: "dotproduct"})

    message = str(excinfo.value)
    assert f"cannot declare '{option}'" in message
    assert "sparse" in message


def test_sparse_vector_field_with_description() -> None:
    schema = SchemaBuilder().add_sparse_vector_field("sparse", description="BM25").build()
    assert schema["fields"]["sparse"]["description"] == "BM25"


def test_sparse_vector_field_additional_options() -> None:
    schema = SchemaBuilder().add_sparse_vector_field("sparse", extra=True).build()
    assert schema["fields"]["sparse"]["extra"] is True


# ---------------------------------------------------------------------------
# add_string_field
# ---------------------------------------------------------------------------


def test_string_field_bare_call_emits_filterable_metadata_shape() -> None:
    """A bare call has no way to select the search variant, so it must select
    the metadata variant instead (#391) — the previous shape, `{"type":
    "string"}`, matched neither of the server's two variants and was
    rejected outright."""
    schema = SchemaBuilder().add_string_field("title").build()
    field = schema["fields"]["title"]
    assert field == {"type": "string", "filterable": False}


def test_string_field_full_text_search_empty_dict() -> None:
    # Empty dict is valid — signals FTS-enabled with server defaults for all options.
    schema = SchemaBuilder().add_string_field("title", full_text_search={}).build()
    assert schema["fields"]["title"]["full_text_search"] == {}


def test_string_field_full_text_search_with_language() -> None:
    schema = SchemaBuilder().add_string_field("title", full_text_search={"language": "en"}).build()
    assert schema["fields"]["title"]["full_text_search"] == {"language": "en"}


def test_string_field_filterable_true() -> None:
    schema = SchemaBuilder().add_string_field("cat", filterable=True).build()
    assert schema["fields"]["cat"]["filterable"] is True


# ---------------------------------------------------------------------------
# add_string_field: every wire shape the server can actually deserialize
# (#391) — measured against pinecone-db 71dacafc. Before the fix, the four
# cases below all emitted `{"type": "string"}`, which matches neither of the
# server's two `CreateStringSchemaField` variants and is rejected outright.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"description": "d"},
        {"filterable": False},
        {"full_text_search": False},
    ],
    ids=[
        "bare",
        "with description",
        "explicit filterable=False",
        "explicit full_text_search=False",
    ],
)
def test_string_field_previously_rejected_spellings_now_emit_metadata_shape(
    kwargs: dict[str, Any],
) -> None:
    field = SchemaBuilder().add_string_field("s", **kwargs).build()["fields"]["s"]
    assert field["type"] == "string"
    assert field["filterable"] is False
    assert "full_text_search" not in field


def test_string_field_fts_true_omits_filterable() -> None:
    field = SchemaBuilder().add_string_field("t", full_text_search=True).build()["fields"]["t"]
    assert field == {"type": "string", "full_text_search": {}}


def test_string_field_fts_enabled_with_explicit_filterable_false_omits_filterable() -> None:
    field = (
        SchemaBuilder()
        .add_string_field("t", full_text_search=True, filterable=False)
        .build()["fields"]["t"]
    )
    assert field == {"type": "string", "full_text_search": {}}


def test_string_field_omits_full_text_search_when_not_provided() -> None:
    schema = SchemaBuilder().add_string_field("t").build()
    field = schema["fields"]["t"]
    assert "full_text_search" not in field
    # No flat FTS option keys leak to the top level.
    for key in ("language", "stemming", "lowercase", "max_term_len", "stop_words"):
        assert key not in field


def test_string_field_full_text_search_all_options() -> None:
    # lowercase and max_term_len are server-managed but still round-trip through dict.
    cfg = {
        "language": "en",
        "stemming": True,
        "lowercase": False,
        "max_term_len": 40,
        "stop_words": False,
    }
    schema = (
        SchemaBuilder()
        .add_string_field("body", full_text_search=cfg, description="article body")
        .build()
    )
    field = schema["fields"]["body"]
    assert field["full_text_search"] == cfg
    assert field["description"] == "article body"


def test_string_field_full_text_search_true_emits_empty_dict() -> None:
    schema = SchemaBuilder().add_string_field("t", full_text_search=True).build()
    assert schema["fields"]["t"]["full_text_search"] == {}


def test_string_field_kwargs_imply_fts_enabled() -> None:
    schema = SchemaBuilder().add_string_field("t", language="en").build()
    assert schema["fields"]["t"]["full_text_search"] == {"language": "en"}


def test_string_field_kwarg_normalizes_long_alias() -> None:
    schema = SchemaBuilder().add_string_field("t", language="english").build()
    assert schema["fields"]["t"]["full_text_search"] == {"language": "en"}


def test_string_field_kwarg_normalizes_long_alias_in_dict() -> None:
    schema = SchemaBuilder().add_string_field("t", full_text_search={"language": "english"}).build()
    assert schema["fields"]["t"]["full_text_search"] == {"language": "en"}


def test_string_field_kwargs_merge_into_dict() -> None:
    schema = (
        SchemaBuilder()
        .add_string_field("t", full_text_search={"language": "fr"}, stemming=True)
        .build()
    )
    fts = schema["fields"]["t"]["full_text_search"]
    assert fts["language"] == "fr"
    assert fts["stemming"] is True


def test_string_field_kwarg_overrides_dict_for_same_key() -> None:
    schema = (
        SchemaBuilder()
        .add_string_field("t", full_text_search={"language": "fr"}, language="en")
        .build()
    )
    assert schema["fields"]["t"]["full_text_search"]["language"] == "en"


def test_string_field_unknown_language_passes_through_kwarg() -> None:
    schema = SchemaBuilder().add_string_field("t", language="klingon").build()
    assert schema["fields"]["t"]["full_text_search"] == {"language": "klingon"}


def test_string_field_unknown_language_passes_through_dict() -> None:
    schema = SchemaBuilder().add_string_field("t", full_text_search={"language": "klingon"}).build()
    assert schema["fields"]["t"]["full_text_search"] == {"language": "klingon"}


def test_string_field_stop_words_without_stemming_raises() -> None:
    from pinecone.errors.exceptions import PineconeValueError

    with pytest.raises(PineconeValueError) as excinfo:
        SchemaBuilder().add_string_field("t", language="en", stop_words=True)
    message = str(excinfo.value)
    assert "'t'" in message
    assert "stop_words requires stemming to be enabled" in message


def test_string_field_stop_words_with_stemming_passes() -> None:
    schema = (
        SchemaBuilder().add_string_field("t", language="en", stemming=True, stop_words=True).build()
    )
    fts = schema["fields"]["t"]["full_text_search"]
    assert fts["language"] == "en"
    assert fts["stemming"] is True
    assert fts["stop_words"] is True


def test_string_field_stop_words_via_dict_validates() -> None:
    from pinecone.errors.exceptions import PineconeValueError

    with pytest.raises(PineconeValueError, match="stop_words requires stemming to be enabled"):
        SchemaBuilder().add_string_field("t", full_text_search={"stop_words": True})


def test_string_field_does_not_validate_language_stop_words_compat() -> None:
    # Arabic does not support stop words, but that rule is server-side only.
    schema = (
        SchemaBuilder().add_string_field("t", language="ar", stemming=True, stop_words=True).build()
    )
    fts = schema["fields"]["t"]["full_text_search"]
    assert fts["language"] == "ar"
    assert fts["stop_words"] is True


def test_string_field_lowercase_and_max_term_len_pass_through() -> None:
    cfg = {"lowercase": False, "max_term_len": 40}
    schema = SchemaBuilder().add_string_field("t", full_text_search=cfg).build()
    fts = schema["fields"]["t"]["full_text_search"]
    assert fts["lowercase"] is False
    assert fts["max_term_len"] == 40


def test_string_field_full_text_search_dict_is_copied_not_aliased() -> None:
    cfg = {"language": "en"}
    builder = SchemaBuilder().add_string_field("title", full_text_search=cfg)
    cfg["language"] = "fr"  # mutate after the call
    assert builder.build()["fields"]["title"]["full_text_search"] == {"language": "en"}


def test_string_field_additional_options_merged() -> None:
    schema = SchemaBuilder().add_string_field("t", future_param="x").build()
    assert schema["fields"]["t"]["future_param"] == "x"


def test_string_field_additional_options_merged_last() -> None:
    # additional_options override explicit kwargs because they .update() last.
    schema = SchemaBuilder().add_string_field("t", extra_future_key="x").build()
    assert schema["fields"]["t"]["extra_future_key"] == "x"
    assert schema["fields"]["t"]["type"] == "string"


def test_string_field_full_text_and_filterable_together() -> None:
    """No client-side guard on this combination (#391): both keys reach the
    wire, and the server keeps the filter while silently discarding the
    search configuration."""
    schema = (
        SchemaBuilder()
        .add_string_field(
            "title",
            full_text_search={"language": "en"},
            filterable=True,
        )
        .build()
    )
    field = schema["fields"]["title"]
    assert field["full_text_search"] == {"language": "en"}
    assert field["filterable"] is True


# ---------------------------------------------------------------------------
# add_string_field: ngram rules (2026-07)
# ---------------------------------------------------------------------------


def test_string_field_ngram_passes_through() -> None:
    cfg = {"ngram": {"min_gram": 2, "max_gram": 3, "prefix_only": True}}
    schema = SchemaBuilder().add_string_field("t", full_text_search=cfg).build()
    fts = schema["fields"]["t"]["full_text_search"]
    assert fts["ngram"] == {"min_gram": 2, "max_gram": 3, "prefix_only": True}


def test_string_field_ngram_with_explicit_false_stemming_passes() -> None:
    cfg = {"ngram": {"min_gram": 2, "max_gram": 3}, "stemming": False, "stop_words": False}
    schema = SchemaBuilder().add_string_field("t", full_text_search=cfg).build()
    assert schema["fields"]["t"]["full_text_search"] == cfg


@pytest.mark.parametrize("conflicting", [{"stemming": True}, {"stop_words": True}])
def test_string_field_ngram_with_stemming_or_stop_words_raises(
    conflicting: dict[str, Any],
) -> None:
    from pinecone.errors.exceptions import PineconeValueError

    cfg = {"ngram": {"min_gram": 2, "max_gram": 3}, **conflicting}
    with pytest.raises(PineconeValueError) as excinfo:
        SchemaBuilder().add_string_field("t", full_text_search=cfg)
    message = str(excinfo.value)
    assert "'t'" in message
    assert "ngram cannot be combined with stemming or stop_words" in message


def test_string_field_ngram_conflict_detected_across_dict_and_kwargs() -> None:
    from pinecone.errors.exceptions import PineconeValueError

    with pytest.raises(PineconeValueError, match="ngram cannot be combined"):
        SchemaBuilder().add_string_field(
            "t", full_text_search={"ngram": {"min_gram": 2, "max_gram": 3}}, stemming=True
        )


def test_string_field_ngram_check_precedes_stop_words_check() -> None:
    from pinecone.errors.exceptions import PineconeValueError

    cfg = {"ngram": {"min_gram": 2, "max_gram": 3}, "stop_words": True}
    with pytest.raises(PineconeValueError, match="ngram cannot be combined"):
        SchemaBuilder().add_string_field("t", full_text_search=cfg)


# ---------------------------------------------------------------------------
# add_float_field
# ---------------------------------------------------------------------------


def test_float_field_defaults() -> None:
    schema = SchemaBuilder().add_float_field("year").build()
    assert schema["fields"]["year"] == {"type": "float", "filterable": False}


def test_float_field_filterable_false_is_still_emitted() -> None:
    schema = SchemaBuilder().add_float_field("year", filterable=False).build()
    assert schema["fields"]["year"] == {"type": "float", "filterable": False}


def test_float_field_description() -> None:
    schema = SchemaBuilder().add_float_field("year", description="pub year").build()
    assert schema["fields"]["year"]["description"] == "pub year"


def test_float_field_additional_options() -> None:
    schema = SchemaBuilder().add_float_field("year", extra=1).build()
    assert schema["fields"]["year"]["extra"] == 1


# ---------------------------------------------------------------------------
# add_custom_field
# ---------------------------------------------------------------------------


def test_custom_field_stored_verbatim() -> None:
    raw: dict[str, Any] = {"type": "new_type", "foo": 42}
    schema = SchemaBuilder().add_custom_field("experimental", raw).build()
    assert schema["fields"]["experimental"] == {"type": "new_type", "foo": 42}


def test_custom_field_complex_definition() -> None:
    raw: dict[str, Any] = {"type": "new_type", "nested": {"a": 1}, "list_val": [1, 2]}
    schema = SchemaBuilder().add_custom_field("f", raw).build()
    assert schema["fields"]["f"]["type"] == "new_type"
    assert schema["fields"]["f"]["nested"] == {"a": 1}
    assert schema["fields"]["f"]["list_val"] == [1, 2]


# ---------------------------------------------------------------------------
# Re-adding a field name replaces it
# ---------------------------------------------------------------------------


def test_duplicate_field_name_replaces() -> None:
    schema = (
        SchemaBuilder()
        .add_string_field("title")
        .add_dense_vector_field("title", dimension=64, metric="cosine")
        .build()
    )
    assert schema["fields"]["title"]["type"] == "dense_vector"
    assert len(schema["fields"]) == 1


def test_duplicate_field_preserves_last_definition() -> None:
    schema = (
        SchemaBuilder()
        .add_float_field("score")
        .add_float_field("score", filterable=False, description="override")
        .build()
    )
    assert schema["fields"]["score"]["description"] == "override"
    assert schema["fields"]["score"]["filterable"] is False


# ---------------------------------------------------------------------------
# add_* methods return self for chaining
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method_name", "args", "kwargs"),
    [
        ("add_dense_vector_field", ("vec",), {"dimension": 128, "metric": "cosine"}),
        ("add_sparse_vector_field", ("sparse",), {}),
        ("add_string_field", ("title",), {}),
        ("add_string_list_field", ("tags",), {}),
        ("add_float_field", ("year",), {}),
        ("add_boolean_field", ("is_published",), {}),
        ("add_custom_field", ("custom", {"type": "custom"}), {}),
    ],
)
def test_add_methods_return_self_for_chaining(
    method_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    builder = SchemaBuilder()
    method = getattr(builder, method_name)
    assert method(*args, **kwargs) is builder


# ---------------------------------------------------------------------------
# build() idempotency
# ---------------------------------------------------------------------------


def test_build_idempotent_same_content() -> None:
    builder = SchemaBuilder().add_string_field("title")
    result1 = builder.build()
    result2 = builder.build()
    assert result1 == result2


def test_build_returns_copy_not_same_object() -> None:
    builder = SchemaBuilder().add_string_field("title")
    result1 = builder.build()
    result2 = builder.build()
    assert result1 is not result2
    assert result1["fields"] is not result2["fields"]


def test_build_subsequent_mutations_do_not_affect_prior_result() -> None:
    builder = SchemaBuilder().add_string_field("title")
    first = builder.build()
    builder.add_string_field("body")
    second = builder.build()
    assert "body" not in first["fields"]
    assert "body" in second["fields"]


def test_build_empty_schema_returns_empty_fields() -> None:
    result = SchemaBuilder().build()
    assert result == {"fields": {}}
    assert isinstance(result["fields"], dict)


# ---------------------------------------------------------------------------
# JSON round-trip serialization
# ---------------------------------------------------------------------------


def test_build_output_json_round_trips_with_optional_fields_absent() -> None:
    schema = (
        SchemaBuilder()
        .add_dense_vector_field("vec", dimension=8, metric="cosine")
        .add_sparse_vector_field("sparse")
        .add_string_field("title", full_text_search=True)
        .build()
    )
    assert json.loads(json.dumps(schema)) == schema
    for field in schema["fields"].values():
        assert "description" not in field
        assert "filterable" not in field


def test_build_output_json_round_trips_with_all_options_set() -> None:
    schema = (
        SchemaBuilder()
        .add_dense_vector_field("vec", dimension=1536, metric="dotproduct", description="emb")
        .add_string_field(
            "body",
            language="en",
            stemming=True,
            stop_words=True,
            filterable=True,
            description="text",
        )
        .build()
    )
    assert json.loads(json.dumps(schema)) == schema


# ---------------------------------------------------------------------------
# add_string_list_field
# ---------------------------------------------------------------------------


def test_string_list_field_defaults() -> None:
    schema = SchemaBuilder().add_string_list_field("tags").build()
    field = schema["fields"]["tags"]
    assert field == {"type": "string_list", "filterable": False}


def test_string_list_field_filterable_true() -> None:
    schema = SchemaBuilder().add_string_list_field("tags", filterable=True).build()
    assert schema["fields"]["tags"]["filterable"] is True


def test_string_list_field_description() -> None:
    schema = (
        SchemaBuilder().add_string_list_field("tags", description="genres and keywords").build()
    )
    assert schema["fields"]["tags"]["description"] == "genres and keywords"


def test_string_list_field_additional_options_merged() -> None:
    schema = SchemaBuilder().add_string_list_field("tags", future_key="v").build()
    assert schema["fields"]["tags"]["future_key"] == "v"


def test_string_list_field_emits_snake_case_tag_not_brackets() -> None:
    # Sanity-check: the old "string[]" tag must never appear.
    schema = SchemaBuilder().add_string_list_field("tags").build()
    assert schema["fields"]["tags"]["type"] == "string_list"
    assert schema["fields"]["tags"]["type"] != "string[]"


# ---------------------------------------------------------------------------
# add_boolean_field
# ---------------------------------------------------------------------------


def test_boolean_field_defaults() -> None:
    schema = SchemaBuilder().add_boolean_field("is_published").build()
    field = schema["fields"]["is_published"]
    assert field == {"type": "boolean", "filterable": False}


def test_boolean_field_filterable_true() -> None:
    schema = SchemaBuilder().add_boolean_field("is_published", filterable=True).build()
    assert schema["fields"]["is_published"]["filterable"] is True


def test_boolean_field_filterable_false_is_still_emitted() -> None:
    schema = SchemaBuilder().add_boolean_field("is_published", filterable=False).build()
    assert schema["fields"]["is_published"] == {"type": "boolean", "filterable": False}


def test_boolean_field_description() -> None:
    schema = (
        SchemaBuilder().add_boolean_field("is_published", description="visibility flag").build()
    )
    assert schema["fields"]["is_published"]["description"] == "visibility flag"


def test_boolean_field_additional_options_merged() -> None:
    schema = SchemaBuilder().add_boolean_field("is_published", future_key="v").build()
    assert schema["fields"]["is_published"]["future_key"] == "v"


# ---------------------------------------------------------------------------
# Field-name and description validation
# ---------------------------------------------------------------------------


def test_field_name_empty_raises() -> None:
    from pinecone.errors.exceptions import PineconeValueError

    with pytest.raises(PineconeValueError, match="non-empty"):
        SchemaBuilder().add_string_field("")


def test_dense_vector_field_underscore_prefixed_name_accepted() -> None:
    schema = SchemaBuilder().add_dense_vector_field("_values", dimension=8, metric="cosine").build()
    assert schema["fields"]["_values"] == {
        "type": "dense_vector",
        "dimension": 8,
        "metric": "cosine",
    }


def test_field_name_starts_with_dollar_accepted() -> None:
    schema = SchemaBuilder().add_string_field("$illegal").build()
    assert "$illegal" in schema["fields"]


def test_field_name_over_64_bytes_raises() -> None:
    from pinecone.errors.exceptions import PineconeValueError

    long_name = "a" * 65
    with pytest.raises(PineconeValueError, match="too long") as excinfo:
        SchemaBuilder().add_string_field(long_name)
    assert long_name in str(excinfo.value)


def test_field_name_64_bytes_ok() -> None:
    schema = SchemaBuilder().add_string_field("a" * 64).build()
    assert "a" * 64 in schema["fields"]


def test_field_name_with_surrogate_raises_pinecone_error() -> None:
    from pinecone.errors.exceptions import PineconeValueError

    with pytest.raises(PineconeValueError, match="valid Unicode"):
        SchemaBuilder().add_string_field("bad\ud800name")


def test_description_with_surrogate_raises_pinecone_error() -> None:
    from pinecone.errors.exceptions import PineconeValueError

    with pytest.raises(PineconeValueError, match="valid Unicode") as excinfo:
        SchemaBuilder().add_string_field("title", description="bad\ud800desc")
    assert "'title'" in str(excinfo.value)


def test_field_name_multibyte_counts_bytes_not_chars() -> None:
    from pinecone.errors.exceptions import PineconeValueError

    multibyte_name = "é" * 33  # "é" is 2 bytes in UTF-8 -> 66 bytes total
    with pytest.raises(PineconeValueError, match="too long"):
        SchemaBuilder().add_string_field(multibyte_name)


def test_description_over_256_bytes_raises_and_names_field() -> None:
    from pinecone.errors.exceptions import PineconeValueError

    with pytest.raises(PineconeValueError, match="too long") as excinfo:
        SchemaBuilder().add_string_field("title", description="x" * 257)
    assert "'title'" in str(excinfo.value)


def test_description_256_bytes_ok() -> None:
    schema = SchemaBuilder().add_string_field("title", description="x" * 256).build()
    assert schema["fields"]["title"]["description"] == "x" * 256


def test_description_none_ok() -> None:
    schema = SchemaBuilder().add_string_field("title").build()
    assert "description" not in schema["fields"]["title"]


def test_validation_applies_to_every_add_method() -> None:
    from pinecone.errors.exceptions import PineconeValueError

    builders = [
        lambda: SchemaBuilder().add_dense_vector_field("", dimension=4, metric="cosine"),
        lambda: SchemaBuilder().add_sparse_vector_field(""),
        lambda: SchemaBuilder().add_string_field(""),
        lambda: SchemaBuilder().add_string_list_field(""),
        lambda: SchemaBuilder().add_boolean_field(""),
        lambda: SchemaBuilder().add_float_field(""),
        lambda: SchemaBuilder().add_custom_field("", {"type": "x"}),
    ]
    for build in builders:
        with pytest.raises(PineconeValueError):
            build()


# ---------------------------------------------------------------------------
# Graduation: import paths and naming
# ---------------------------------------------------------------------------


def test_schema_builder_importable_from_top_level() -> None:
    from pinecone import SchemaBuilder as TopLevelAlias

    assert TopLevelAlias is SchemaBuilder


def test_preview_schema_builder_name_is_gone() -> None:
    import pinecone.schema_builder

    assert not hasattr(pinecone.schema_builder, "PreviewSchemaBuilder")


# ---------------------------------------------------------------------------
# `filterable` is required on the wire for the three metadata field types (#374)
# ---------------------------------------------------------------------------
#
# All three deserialize into `MetadataSchemaField`, whose `filterable` is a bare
# `bool` with no `#[serde(default)]` (pinecone-db
# `pc-types/src/index_schema_def.rs:399-402` @ 71dacafc), so omitting the key
# makes the whole create body undeserializable.
#
# Exact key sets rather than `"filterable" in field`, so that a *different*
# unmodelled key reappearing here fails too — the #350 defect, which a
# containment assertion would have missed.

_METADATA_FIELD_TYPES = [
    ("add_boolean_field", "boolean"),
    ("add_float_field", "float"),
    ("add_string_list_field", "string_list"),
]


@pytest.mark.parametrize(("method", "wire_type"), _METADATA_FIELD_TYPES)
def test_metadata_field_emits_filterable_at_the_documented_default(
    method: str, wire_type: str
) -> None:
    schema = getattr(SchemaBuilder(), method)("f").build()
    assert schema["fields"]["f"] == {"type": wire_type, "filterable": False}


@pytest.mark.parametrize(("method", "wire_type"), _METADATA_FIELD_TYPES)
@pytest.mark.parametrize("filterable", [True, False])
def test_metadata_field_emits_filterable_for_both_values(
    method: str, wire_type: str, filterable: bool
) -> None:
    schema = getattr(SchemaBuilder(), method)("f", filterable=filterable).build()
    assert schema["fields"]["f"] == {"type": wire_type, "filterable": filterable}


@pytest.mark.parametrize(("method", "wire_type"), _METADATA_FIELD_TYPES)
def test_metadata_field_emits_filterable_alongside_a_description(
    method: str, wire_type: str
) -> None:
    schema = getattr(SchemaBuilder(), method)("f", description="d").build()
    assert schema["fields"]["f"] == {
        "type": wire_type,
        "filterable": False,
        "description": "d",
    }


@pytest.mark.parametrize(("method", "wire_type"), _METADATA_FIELD_TYPES)
def test_metadata_field_filterable_survives_additional_options(method: str, wire_type: str) -> None:
    # `additional_options` is merged last, so confirm it does not displace the
    # required key on its way to the wire.
    schema = getattr(SchemaBuilder(), method)("f", future_key="v").build()
    assert schema["fields"]["f"] == {
        "type": wire_type,
        "filterable": False,
        "future_key": "v",
    }


def test_metadata_fields_agree_with_the_response_side_models() -> None:
    """The builder and the msgspec models must not disagree on this key.

    The models already encode `filterable: false` (they are plain msgspec
    Structs without ``omit_defaults``); only the builder omitted it. Pinning
    the agreement keeps the two request-shaping paths from drifting apart
    again.
    """
    from pinecone.models.indexes.schema import (
        BooleanField,
        FloatField,
        IndexSchema,
        StringListField,
    )

    built = (
        SchemaBuilder()
        .add_boolean_field("b")
        .add_float_field("f")
        .add_string_list_field("t")
        .build()
    )
    modelled = IndexSchema(
        fields={"b": BooleanField(), "f": FloatField(), "t": StringListField()}
    ).to_dict()

    for name in ("b", "f", "t"):
        assert built["fields"][name]["filterable"] is False
        assert modelled["fields"][name]["filterable"] is False
