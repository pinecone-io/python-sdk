"""Adapter for Backups API responses."""

from __future__ import annotations

from typing import Any, TypeVar

import msgspec
import orjson
from msgspec import Struct

from pinecone._internal.adapters._decode import convert_response, decode_response
from pinecone.errors.exceptions import ResponseParsingError
from pinecone.models.backups.list import BackupList
from pinecone.models.backups.model import (
    BackupModel,
    CreateIndexFromBackupRequest,
    CreateIndexFromBackupResponse,
)
from pinecone.models.indexes.schema import _tag_untyped_schema_fields
from pinecone.models.vectors.responses import Pagination

T = TypeVar("T")


class _BackupListEnvelope(Struct, kw_only=True):
    """Internal envelope for the list-backups response."""

    data: list[BackupModel] = []
    pagination: Pagination | None = None


def _tag_untyped_backups(obj: Any) -> Any:
    if isinstance(obj, dict) and isinstance(obj.get("data"), list):
        for item in obj["data"]:
            _tag_untyped_schema_fields(item)
        return obj
    return _tag_untyped_schema_fields(obj)


def decode_backups_envelope(data: bytes, envelope_type: type[T]) -> T:
    """Decode a single backup or a list envelope into *envelope_type*.

    Retries once with legacy metadata-schema fields tagged, so a backup
    whose captured schema pre-dates typed schema fields still decodes
    instead of raising. Every backup-bearing response — stable or preview —
    must go through here, because the schema shape is a property of the
    stored backup rather than of the API version that reads it.
    """
    try:
        return decode_response(data, envelope_type)
    except ResponseParsingError:
        try:
            obj = orjson.loads(data)
        except orjson.JSONDecodeError:
            reparsed = False
        else:
            reparsed = True
        if not reparsed:
            raise
        return convert_response(_tag_untyped_backups(obj), envelope_type)


class BackupsAdapter:
    """Transforms raw API JSON into BackupModel / BackupList instances."""

    @staticmethod
    def to_backup(data: bytes) -> BackupModel:
        """Decode raw JSON bytes into a BackupModel."""
        return decode_backups_envelope(data, BackupModel)

    @staticmethod
    def to_backup_list(data: bytes) -> BackupList:
        """Decode raw JSON bytes from a list-backups response into a BackupList."""
        envelope = decode_backups_envelope(data, _BackupListEnvelope)
        return BackupList(envelope.data, pagination=envelope.pagination)

    @staticmethod
    def to_create_index_from_backup_response(data: bytes) -> CreateIndexFromBackupResponse:
        """Decode raw JSON bytes into a CreateIndexFromBackupResponse."""
        return decode_response(data, CreateIndexFromBackupResponse)

    @staticmethod
    def to_create_index_from_backup_request(request: CreateIndexFromBackupRequest) -> bytes:
        """Encode a CreateIndexFromBackupRequest into request-body bytes.

        The struct is ``omit_defaults=True``, so unset optionals never reach
        the wire and the server applies its own defaults.
        """
        return msgspec.json.encode(request)
