"""Adapter for Backup Schedules API responses."""

from __future__ import annotations

from msgspec import Struct

from pinecone._internal.adapters._decode import decode_response
from pinecone._internal.adapters.backups_adapter import decode_backups_envelope
from pinecone.models.backups.list import BackupScheduleHistoryList, BackupScheduleList
from pinecone.models.backups.schedules import BackupScheduleHistoryItem, BackupScheduleModel
from pinecone.models.vectors.responses import Pagination


class _BackupScheduleListEnvelope(Struct, kw_only=True):
    """Internal envelope for the list-backup-schedules response."""

    data: list[BackupScheduleModel] = []
    pagination: Pagination | None = None


class _BackupScheduleHistoryEnvelope(Struct, kw_only=True):
    """Internal envelope for the list-backup-schedule-history response."""

    data: list[BackupScheduleHistoryItem] = []
    pagination: Pagination | None = None


class BackupSchedulesAdapter:
    """Transforms raw API JSON into backup-schedule models and list wrappers."""

    @staticmethod
    def to_schedule(data: bytes) -> BackupScheduleModel:
        """Decode raw JSON bytes into a BackupScheduleModel."""
        return decode_response(data, BackupScheduleModel)

    @staticmethod
    def to_schedule_list(data: bytes) -> BackupScheduleList:
        """Decode a list-schedules response into a BackupScheduleList.

        Schedules carry no captured index schema, so this needs none of the
        legacy-schema tolerance :meth:`to_history_list` does.
        """
        envelope = decode_response(data, _BackupScheduleListEnvelope)
        return BackupScheduleList(envelope.data, pagination=envelope.pagination)

    @staticmethod
    def to_history_list(data: bytes) -> BackupScheduleHistoryList:
        """Decode a schedule-history response into a BackupScheduleHistoryList.

        History rows are backup snapshots, so they carry a captured index
        schema and must go through ``decode_backups_envelope``: the backend
        serves this endpoint from its shared backup handler, whose rows still
        use the pre-typed metadata schema. A plain ``decode_response`` here
        would raise :exc:`ResponseParsingError` against today's backend.
        """
        envelope = decode_backups_envelope(data, _BackupScheduleHistoryEnvelope)
        return BackupScheduleHistoryList(envelope.data, pagination=envelope.pagination)
