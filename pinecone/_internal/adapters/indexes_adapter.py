"""Adapter for Indexes API responses (2026-07)."""

from __future__ import annotations

import logging
from typing import Any

import msgspec
import orjson

from pinecone._internal.adapters._decode import convert_response, decode_response
from pinecone.errors.exceptions import ResponseParsingError
from pinecone.models.indexes.index import IndexModel
from pinecone.models.indexes.list import IndexList
from pinecone.models.indexes.requests import ConfigureIndexRequest, CreateIndexRequest
from pinecone.models.indexes.schema import UNTYPED_FIELD_TAG

_logger = logging.getLogger(__name__)

_DISCRIMINATOR_HINTS: dict[str, str] = {
    "deployment_type": "Expected one of: 'managed', 'pod', 'byoc'.",
    "mode": "Expected one of: 'OnDemand', 'Dedicated'.",
}


class _IndexListEnvelope(msgspec.Struct, kw_only=True):
    """Internal envelope for the list-indexes response."""

    indexes: list[IndexModel] = []


class _RawIndexListEnvelope(msgspec.Struct, kw_only=True):
    """Envelope capturing each index as raw JSON so the outer parse always
    succeeds even when individual items fail to decode."""

    indexes: list[msgspec.Raw] = []


def _tag_untyped_schema_fields(obj: Any) -> Any:
    """Inject the internal discriminator into schema fields lacking a ``type`` key.

    Legacy metadata fields arrive with no ``type`` discriminator; msgspec
    tagged unions cannot decode them without one.
    """
    if not isinstance(obj, dict):
        return obj
    schema = obj.get("schema")
    if isinstance(schema, dict):
        fields = schema.get("fields")
        if isinstance(fields, dict):
            for field in fields.values():
                if isinstance(field, dict) and "type" not in field:
                    field["type"] = UNTYPED_FIELD_TAG
    return obj


def _enrich_parse_error(exc: ResponseParsingError) -> ResponseParsingError:
    message = exc.message if isinstance(exc.message, str) else str(exc)
    for discriminator, hint in _DISCRIMINATOR_HINTS.items():
        if discriminator in message and hint not in message:
            return ResponseParsingError(f"{message} {hint}", cause=exc.cause)
    return exc


def _drop_none(obj: Any) -> Any:
    """Recursively drop None values from dicts so unset optionals stay off the wire."""
    if isinstance(obj, dict):
        return {k: _drop_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_drop_none(item) for item in obj]
    return obj


class IndexesAdapter:
    """Transforms raw API JSON into IndexModel / IndexList instances."""

    @staticmethod
    def to_create_request(request: CreateIndexRequest) -> bytes:
        """Encode a CreateIndexRequest as JSON bytes with no null-valued keys."""
        return orjson.dumps(_drop_none(msgspec.to_builtins(request)))

    @staticmethod
    def to_configure_request(request: ConfigureIndexRequest) -> bytes:
        """Encode a ConfigureIndexRequest as sparse JSON bytes with no null-valued keys."""
        return orjson.dumps(_drop_none(msgspec.to_builtins(request)))

    @staticmethod
    def to_index_model(data: bytes) -> IndexModel:
        """Decode raw JSON bytes into an IndexModel."""
        try:
            return decode_response(data, IndexModel)
        except ResponseParsingError as first_error:
            try:
                obj = orjson.loads(data)
            except orjson.JSONDecodeError:
                raise _enrich_parse_error(first_error) from first_error
            try:
                return convert_response(_tag_untyped_schema_fields(obj), IndexModel)
            except ResponseParsingError as exc:
                raise _enrich_parse_error(exc) from exc

    @staticmethod
    def to_index_list(data: bytes) -> IndexList:
        """Decode raw JSON bytes from a list-indexes response into an IndexList.

        Falls back to parsing each index independently so a single
        malformed index (e.g. a schema field with an unrecognised type)
        does not fail the entire list call; such indexes are skipped with
        a warning naming the index.
        """
        try:
            envelope = decode_response(data, _IndexListEnvelope)
            return IndexList(envelope.indexes)
        except ResponseParsingError:
            pass

        raw_envelope = decode_response(data, _RawIndexListEnvelope)
        result: list[IndexModel] = []
        for raw in raw_envelope.indexes:
            raw_bytes = bytes(raw)
            try:
                result.append(IndexesAdapter.to_index_model(raw_bytes))
            except ResponseParsingError as exc:
                try:
                    name: str = msgspec.json.decode(raw_bytes, type=dict).get("name", "<unknown>")
                except Exception:
                    name = "<unknown>"
                _logger.warning(
                    "Skipping index %r: cannot parse response (%s). "
                    "This usually means the index was created with an older or "
                    "experimental API that uses a field type not recognised by "
                    "this SDK version.",
                    name,
                    exc,
                )
        return IndexList(result)
