"""Guard tests: response-only schema field types on the create path (#251).

``integer`` exists in the describe/list response union but not in the
create-schema union, so a describe-then-create round-trip of an
``IntegerField`` is answered by the server with a plain-text ``422``.
:class:`~pinecone.schema_builder.SchemaBuilder` must refuse it locally with
an error that names the field and says the type is response-only.

The undocumented ``number`` alias for ``float`` stays unexposed: the builder
grows no method for it and emits only ``"float"``.
"""

from __future__ import annotations

from typing import Any

import pytest

from pinecone.errors.exceptions import PineconeValueError
from pinecone.models.indexes.schema import IndexSchema, IntegerField
from pinecone.schema_builder import SchemaBuilder

# ---------------------------------------------------------------------------
# The builder never offers an integer field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method_name", ["add_integer_field", "add_number_field"])
def test_builder_offers_no_integer_or_number_method(method_name: str) -> None:
    assert not hasattr(SchemaBuilder(), method_name)


def test_add_float_field_emits_float_not_number_or_integer() -> None:
    schema = SchemaBuilder().add_float_field("year", filterable=True).build()
    assert schema["fields"]["year"]["type"] == "float"


# ---------------------------------------------------------------------------
# add_custom_field refuses integer
# ---------------------------------------------------------------------------


def test_custom_field_rejects_integer_type() -> None:
    with pytest.raises(PineconeValueError) as exc:
        SchemaBuilder().add_custom_field("count", {"type": "integer", "filterable": True})
    message = str(exc.value)
    assert "count" in message
    assert "integer" in message
    assert "response-only" in message
    assert "add_float_field" in message


def test_custom_field_rejection_leaves_builder_unchanged() -> None:
    builder = SchemaBuilder().add_dense_vector_field("vec", dimension=8, metric="cosine")
    with pytest.raises(PineconeValueError):
        builder.add_custom_field("count", {"type": "integer"})
    assert builder.build() == {
        "fields": {"vec": {"type": "dense_vector", "dimension": 8, "metric": "cosine"}}
    }


def test_custom_field_still_accepts_unknown_forward_compatible_types() -> None:
    schema = SchemaBuilder().add_custom_field("f", {"type": "some_future_type"}).build()
    assert schema["fields"]["f"] == {"type": "some_future_type"}


@pytest.mark.parametrize("field_definition", [{}, {"filterable": True}, {"type": 7}])
def test_custom_field_without_string_type_is_not_rejected(
    field_definition: dict[str, Any],
) -> None:
    schema = SchemaBuilder().add_custom_field("f", dict(field_definition)).build()
    assert schema["fields"]["f"] == field_definition


# ---------------------------------------------------------------------------
# additional_options cannot smuggle an integer type past the guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method_name", "kwargs"),
    [
        ("add_dense_vector_field", {"dimension": 8, "metric": "cosine"}),
        ("add_sparse_vector_field", {}),
        ("add_string_field", {"full_text_search": True}),
        ("add_string_list_field", {}),
        ("add_boolean_field", {}),
        ("add_float_field", {}),
    ],
)
def test_additional_options_cannot_override_type_to_integer(
    method_name: str, kwargs: dict[str, Any]
) -> None:
    builder = SchemaBuilder()
    method = getattr(builder, method_name)
    with pytest.raises(PineconeValueError, match="response-only"):
        method("f", **kwargs, type="integer")


# ---------------------------------------------------------------------------
# Round-trip guard: describe-shaped schema -> create builder
# ---------------------------------------------------------------------------


def _replay_described_fields(described: dict[str, Any]) -> dict[str, Any]:
    builder = SchemaBuilder().add_dense_vector_field("vec", dimension=8, metric="cosine")
    for name, definition in described["fields"].items():
        builder.add_custom_field(name, definition)
    return builder.build()


def test_describe_to_create_roundtrip_of_integer_field_fails_client_side() -> None:
    described = IndexSchema(
        fields={"count": IntegerField(filterable=True, description="legacy int")}
    ).to_dict()
    assert described["fields"]["count"]["type"] == "integer"

    with pytest.raises(PineconeValueError) as exc:
        _replay_described_fields(described)

    message = str(exc.value)
    assert "count" in message
    assert "response-only" in message
    assert "422" not in message


def test_roundtrip_succeeds_once_integer_field_is_redeclared_as_float() -> None:
    described = IndexSchema(fields={"count": IntegerField(filterable=True)}).to_dict()
    definition = described["fields"]["count"]

    schema = (
        SchemaBuilder()
        .add_dense_vector_field("vec", dimension=8, metric="cosine")
        .add_float_field("count", filterable=definition["filterable"])
        .build()
    )
    assert schema["fields"]["count"] == {"type": "float", "filterable": True}
