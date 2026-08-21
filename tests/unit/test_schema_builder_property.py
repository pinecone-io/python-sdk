"""Property-based tests for SchemaBuilder (2026-07 create-schema rules).

Five properties are pinned here (#106, #374):

* Arbitrary unicode field names are accepted iff they are 1-64 UTF-8 bytes
  and do not start with ``$`` or ``_``.
* Language normalization is idempotent, case-insensitive for known codes and
  long-form aliases, and a passthrough for unknown values.
* ``build()`` output is always JSON-serializable, and ``add_*`` then
  ``build()`` round-trips the exact set of field names added.
* Builder reuse: ``build()`` copies — later ``add_*`` calls never mutate
  earlier results.
* ``boolean``/``float``/``string_list`` fields always carry ``filterable``,
  which the backend requires (#374).
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pinecone.errors.exceptions import PineconeValueError
from pinecone.schema_builder import (
    _FTS_LANGUAGES_LONG_TO_SHORT,
    _FTS_LANGUAGES_SHORT,
    SchemaBuilder,
    _normalize_fts_language,
)

# ---------------------------------------------------------------------------
# Property 1: arbitrary unicode field names
# ---------------------------------------------------------------------------


def _name_is_valid(name: str) -> bool:
    try:
        byte_len = len(name.encode("utf-8"))
    except UnicodeEncodeError:
        return False
    return name != "" and not name.startswith("$") and not name.startswith("_") and byte_len <= 64


@given(name=st.text(max_size=80))
def test_field_name_accepted_iff_valid(name: str) -> None:
    builder = SchemaBuilder()
    if _name_is_valid(name):
        schema = builder.add_string_field(name, full_text_search=True).build()
        assert name in schema["fields"]
    else:
        with pytest.raises(PineconeValueError):
            builder.add_string_field(name, full_text_search=True)
        assert builder.build() == {"fields": {}}


@given(name=st.text(alphabet=st.characters(min_codepoint=0x80), min_size=1, max_size=80))
def test_multibyte_field_name_accepted_iff_valid(name: str) -> None:
    builder = SchemaBuilder()
    if _name_is_valid(name):
        schema = builder.add_boolean_field(name).build()
        assert name in schema["fields"]
    else:
        with pytest.raises(PineconeValueError):
            builder.add_boolean_field(name)


# ---------------------------------------------------------------------------
# Property 2: language normalization
# ---------------------------------------------------------------------------

_KNOWN_INPUTS = sorted(_FTS_LANGUAGES_SHORT | set(_FTS_LANGUAGES_LONG_TO_SHORT))


@st.composite
def _random_casing(draw: st.DrawFn) -> str:
    base = draw(st.sampled_from(_KNOWN_INPUTS))
    flips = draw(st.lists(st.booleans(), min_size=len(base), max_size=len(base)))
    return "".join(c.upper() if flip else c for c, flip in zip(base, flips))


@given(language=_random_casing())
def test_known_language_normalizes_case_insensitively(language: str) -> None:
    normalized = _normalize_fts_language(language)
    lowered = language.lower()
    expected = _FTS_LANGUAGES_LONG_TO_SHORT.get(lowered, lowered)
    assert normalized == expected
    assert normalized in _FTS_LANGUAGES_SHORT


@given(language=st.text(max_size=30))
def test_language_normalization_is_idempotent(language: str) -> None:
    once = _normalize_fts_language(language)
    assert _normalize_fts_language(once) == once


@given(language=st.text(max_size=30))
def test_unknown_language_passes_through_unchanged(language: str) -> None:
    lowered = language.lower()
    if lowered in _FTS_LANGUAGES_SHORT or lowered in _FTS_LANGUAGES_LONG_TO_SHORT:
        return
    assert _normalize_fts_language(language) == language


# ---------------------------------------------------------------------------
# Properties 3 & 4: build() serializability, key round-trip, and copy semantics
# ---------------------------------------------------------------------------


def _utf8_encodable(value: str) -> bool:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


_valid_name = st.text(min_size=1, max_size=20).filter(_name_is_valid)
_description = st.one_of(st.none(), st.text(max_size=40).filter(_utf8_encodable))

_add_dense = st.tuples(
    st.just("add_dense_vector_field"),
    _valid_name,
    st.fixed_dictionaries(
        {
            "dimension": st.integers(min_value=1, max_value=20000),
            "metric": st.sampled_from(["cosine", "dotproduct", "euclidean"]),
            "description": _description,
        }
    ),
)
_add_sparse = st.tuples(
    st.just("add_sparse_vector_field"),
    _valid_name,
    st.fixed_dictionaries({"description": _description}),
)
_add_string = st.tuples(
    st.just("add_string_field"),
    _valid_name,
    st.fixed_dictionaries(
        {
            "language": st.one_of(st.none(), st.sampled_from(_KNOWN_INPUTS)),
            "stemming": st.one_of(st.none(), st.booleans()),
            "filterable": st.booleans(),
            "description": _description,
        }
    ),
)
_METADATA_METHODS = ("add_boolean_field", "add_float_field", "add_string_list_field")

_add_metadata = st.tuples(
    st.sampled_from(_METADATA_METHODS),
    _valid_name,
    st.fixed_dictionaries({"filterable": st.booleans(), "description": _description}),
)

_add_call = st.one_of(_add_dense, _add_sparse, _add_string, _add_metadata)


def _apply(builder: SchemaBuilder, call: tuple[str, str, dict[str, Any]]) -> None:
    method_name, name, kwargs = call
    getattr(builder, method_name)(name, **kwargs)


@given(calls=st.lists(_add_call, max_size=8))
def test_build_json_serializable_and_round_trips_field_names(
    calls: list[tuple[str, str, dict[str, Any]]],
) -> None:
    builder = SchemaBuilder()
    for call in calls:
        _apply(builder, call)
    schema = builder.build()

    assert json.loads(json.dumps(schema)) == schema
    assert set(schema["fields"]) == {name for _, name, _ in calls}

    last_call_per_name = {name: (method, kwargs) for method, name, kwargs in calls}
    for name, (method, kwargs) in last_call_per_name.items():
        field = schema["fields"][name]
        if kwargs.get("description") is not None:
            assert field["description"] == kwargs["description"]
        else:
            assert "description" not in field
        if method == "add_dense_vector_field":
            assert field["dimension"] == kwargs["dimension"]
            assert field["metric"] == kwargs["metric"]
        if method == "add_sparse_vector_field":
            assert set(field) <= {"type", "description"}
        if method in _METADATA_METHODS:
            assert field["filterable"] is kwargs["filterable"]


@given(
    initial=st.lists(_add_call, max_size=6),
    later=st.lists(_add_call, min_size=1, max_size=6),
)
def test_later_add_calls_never_mutate_earlier_build_results(
    initial: list[tuple[str, str, dict[str, Any]]],
    later: list[tuple[str, str, dict[str, Any]]],
) -> None:
    builder = SchemaBuilder()
    for call in initial:
        _apply(builder, call)
    first = builder.build()
    snapshot = copy.deepcopy(first)

    for call in later:
        _apply(builder, call)
    builder.build()

    assert first == snapshot


@given(calls=st.lists(_add_metadata, min_size=1, max_size=8))
def test_metadata_fields_always_carry_filterable(
    calls: list[tuple[str, str, dict[str, Any]]],
) -> None:
    """No builder-emitted `boolean`/`float`/`string_list` field ever omits `filterable`.

    The backend deserializes all three into `MetadataSchemaField`, whose
    `filterable` is a bare `bool` with no serde default, so a field that omits
    the key makes the entire create body undeserializable (#374). Stated as an
    invariant over arbitrary builder call sequences because the defect was
    specifically in the `False` branch — the documented default, and the value
    an example-based test is least likely to reach for.
    """
    builder = SchemaBuilder()
    for call in calls:
        _apply(builder, call)
    schema = builder.build()

    last_call_per_name = {name: kwargs for _, name, kwargs in calls}
    for name, kwargs in last_call_per_name.items():
        field = schema["fields"][name]
        assert "filterable" in field
        assert field["filterable"] is kwargs["filterable"]
        assert set(field) <= {"type", "filterable", "description"}
