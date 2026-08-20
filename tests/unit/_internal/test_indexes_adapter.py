"""Unit tests for IndexesAdapter decode paths (2026-07)."""

from __future__ import annotations

import orjson
import pytest

from pinecone._internal.adapters.indexes_adapter import IndexesAdapter
from pinecone.errors.exceptions import ResponseParsingError
from pinecone.models.indexes.schema import (
    BooleanField,
    FloatField,
    IntegerField,
    LegacyMetadataField,
    StringListField,
)
from tests.factories import make_index_response


def test_describe_decodes_legacy_untyped_field() -> None:
    data = make_index_response()
    data["schema"] = {"fields": {"old_meta": {"filterable": True}}}

    model = IndexesAdapter.to_index_model(orjson.dumps(data))

    field = model.schema.fields["old_meta"]
    assert isinstance(field, LegacyMetadataField)
    assert field.filterable is True


def test_describe_decodes_metadata_field_types() -> None:
    data = make_index_response()
    data["schema"] = {
        "fields": {
            "count": {"type": "integer", "filterable": True},
            "year": {"type": "float", "filterable": True},
            "active": {"type": "boolean", "filterable": False},
            "genres": {"type": "string_list", "filterable": True},
            "old": {"filterable": False},
        }
    }

    model = IndexesAdapter.to_index_model(orjson.dumps(data))

    assert isinstance(model.schema.fields["count"], IntegerField)
    assert isinstance(model.schema.fields["year"], FloatField)
    assert isinstance(model.schema.fields["active"], BooleanField)
    assert isinstance(model.schema.fields["genres"], StringListField)
    assert isinstance(model.schema.fields["old"], LegacyMetadataField)


def test_describe_unknown_deployment_type_error_lists_allowed_values() -> None:
    data = make_index_response(
        deployment={"deployment_type": "quantum", "cloud": "aws", "region": "us-east-1"}
    )

    with pytest.raises(ResponseParsingError) as exc_info:
        IndexesAdapter.to_index_model(orjson.dumps(data))

    message = str(exc_info.value)
    assert "quantum" in message
    assert "deployment_type" in message
    assert "'managed', 'pod', 'byoc'" in message


def test_describe_unknown_read_capacity_mode_error_lists_allowed_values() -> None:
    data = make_index_response(read_capacity={"mode": "Turbo", "status": {"state": "Ready"}})

    with pytest.raises(ResponseParsingError) as exc_info:
        IndexesAdapter.to_index_model(orjson.dumps(data))

    message = str(exc_info.value)
    assert "Turbo" in message
    assert "'OnDemand', 'Dedicated'" in message


def test_describe_invalid_json_raises_parsing_error() -> None:
    with pytest.raises(ResponseParsingError):
        IndexesAdapter.to_index_model(b"this is not json")


def test_list_mixes_legacy_and_typed_indexes() -> None:
    typed = make_index_response(name="typed-index")
    legacy = make_index_response(name="legacy-index")
    legacy["schema"] = {"fields": {"old": {"filterable": True}}}

    result = IndexesAdapter.to_index_list(orjson.dumps({"indexes": [typed, legacy]}))

    assert result.names() == ["typed-index", "legacy-index"]


def test_list_skips_only_the_malformed_index() -> None:
    good = make_index_response(name="good-index")
    bad = make_index_response(name="bad-index")
    bad["deployment"] = {"deployment_type": "quantum"}

    result = IndexesAdapter.to_index_list(orjson.dumps({"indexes": [good, bad]}))

    assert result.names() == ["good-index"]
