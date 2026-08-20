"""Unit tests for AssistantFileModel wire-format and backwards-compatibility."""

from __future__ import annotations

import msgspec
import pytest

from pinecone.models.assistant.file_model import AssistantFileModel

REMOVED_FIELDS = ["percent_done", "error_message"]


def test_crc32c_hash_alias() -> None:
    """crc32c_hash property returns the same value as content_hash."""
    model = AssistantFileModel(name="f.txt", id="id-1", content_hash="abc123")
    assert model.content_hash == "abc123"
    assert model.crc32c_hash == "abc123"


def test_crc32c_hash_alias_none() -> None:
    """crc32c_hash property returns None when content_hash is None."""
    model = AssistantFileModel(name="f.txt", id="id-1")
    assert model.content_hash is None
    assert model.crc32c_hash is None


def test_wire_key_is_crc32c_hash() -> None:
    """msgspec deserializes the wire key 'crc32c_hash' into the content_hash attribute."""
    raw = b'{"name":"f.txt","id":"id-1","crc32c_hash":"deadbeef"}'
    model = msgspec.json.decode(raw, type=AssistantFileModel)
    assert model.content_hash == "deadbeef"
    assert model.crc32c_hash == "deadbeef"


def test_wire_key_content_hash_ignored() -> None:
    """msgspec silently ignores an unknown 'content_hash' wire key (not the struct key)."""
    raw = b'{"name":"f.txt","id":"id-1","content_hash":"should-be-ignored"}'
    model = msgspec.json.decode(raw, type=AssistantFileModel)
    assert model.content_hash is None
    assert model.crc32c_hash is None


def test_optional_fields_default_none() -> None:
    """All optional fields default to None when absent from the wire response."""
    raw = b'{"name":"f.txt","id":"id-1"}'
    model = msgspec.json.decode(raw, type=AssistantFileModel)
    assert model.metadata is None
    assert model.created_on is None
    assert model.updated_on is None
    assert model.status is None
    assert model.size is None
    assert model.multimodal is None
    assert model.signed_url is None
    assert model.content_hash is None
    assert model.crc32c_hash is None


def test_2026_07_payload_round_trips() -> None:
    """A full 2026-07 payload — size present, non-uuid id, no removed keys — round-trips."""
    payload = {
        "name": "report.pdf",
        "id": "my-own-file-handle",
        "metadata": {"team": "Operations"},
        "created_on": "2026-07-01T12:30:00Z",
        "updated_on": "2026-07-01T12:45:00Z",
        "status": "Available",
        "size": 1048576,
        "signed_url": "https://storage.googleapis.com/bucket/file.pdf",
        "multimodal": True,
    }
    model = msgspec.convert(payload, type=AssistantFileModel)
    assert model.id == "my-own-file-handle"
    assert model.size == 1048576
    assert msgspec.to_builtins(model) == {**payload, "crc32c_hash": None}


def test_size_decodes_int64() -> None:
    """size carries the spec's int64 range, not a 32-bit int."""
    raw = b'{"name":"big.bin","id":"id-1","size":9007199254740993}'
    model = msgspec.json.decode(raw, type=AssistantFileModel)
    assert model.size == 9007199254740993


@pytest.mark.parametrize("field", REMOVED_FIELDS)
def test_removed_field_is_not_a_struct_field(field: str) -> None:
    """The removed fields are genuinely gone from the struct definition."""
    assert field not in AssistantFileModel.__struct_fields__
    assert field not in {f.name for f in msgspec.structs.fields(AssistantFileModel)}


@pytest.mark.parametrize("field", REMOVED_FIELDS)
def test_removed_field_attribute_access_names_replacement(field: str) -> None:
    """Attribute access raises AttributeError pointing at describe_operation."""
    model = AssistantFileModel(name="f.txt", id="id-1")
    with pytest.raises(AttributeError, match="describe_operation"):
        getattr(model, field)


@pytest.mark.parametrize("field", REMOVED_FIELDS)
def test_removed_field_constructor_rejected(field: str) -> None:
    """Constructing with a removed keyword is a TypeError, not a silently kept attribute."""
    with pytest.raises(TypeError):
        AssistantFileModel(name="f.txt", id="id-1", **{field: 1.0})


@pytest.mark.parametrize("field", REMOVED_FIELDS)
def test_removed_field_dict_access_raises_keyerror(field: str) -> None:
    """Dict-style access to a removed field raises KeyError and 'in' is False."""
    model = AssistantFileModel(name="f.txt", id="id-1")
    assert field not in model
    with pytest.raises(KeyError):
        model[field]


@pytest.mark.parametrize("field", REMOVED_FIELDS)
def test_removed_field_absent_from_dict_views(field: str) -> None:
    """keys()/to_dict() no longer advertise the removed fields."""
    model = AssistantFileModel(name="f.txt", id="id-1")
    advertised = model.keys()
    assert field not in advertised
    assert field not in model.to_dict()


def test_unknown_attribute_still_raises_plain_attribute_error() -> None:
    """__getattr__ only special-cases the removed fields."""
    model = AssistantFileModel(name="f.txt", id="id-1")
    with pytest.raises(AttributeError, match="has no attribute 'nope'"):
        model.nope  # type: ignore[attr-defined]


def test_legacy_payload_with_removed_keys_still_decodes() -> None:
    """A 2025-10-shaped payload still decodes; the removed keys are ignored."""
    raw = (
        b'{"name":"f.txt","id":"id-1","status":"ProcessingFailed",'
        b'"percent_done":42.5,"error_message":"boom"}'
    )
    model = msgspec.json.decode(raw, type=AssistantFileModel)
    assert model.status == "ProcessingFailed"
    with pytest.raises(AttributeError, match="describe_operation"):
        model.error_message
