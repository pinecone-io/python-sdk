"""Unit tests for the Pinecone.create_index backcompat shim."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pinecone import Pinecone
from pinecone.models.indexes.specs import ServerlessSpec


def _make_pc_with_mock_indexes() -> tuple[Pinecone, MagicMock]:
    pc = Pinecone(api_key="test-key")
    mock_indexes = MagicMock()
    mock_indexes.create = MagicMock(return_value=MagicMock())
    pc._indexes = mock_indexes
    return pc, mock_indexes


def test_create_index_shim_forwards_legacy_kwargs() -> None:
    pc, mock_indexes = _make_pc_with_mock_indexes()
    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    pc.create_index(
        name="test",
        spec=spec,
        dimension=4,
        metric="cosine",
        vector_type="dense",
        deletion_protection="enabled",
        tags={"env": "prod"},
        timeout=-1,
    )

    mock_indexes.create.assert_called_once_with(
        name="test",
        spec=spec,
        dimension=4,
        metric="cosine",
        vector_type="dense",
        deletion_protection="enabled",
        tags={"env": "prod"},
        timeout=-1,
    )


def test_create_index_shim_defaults() -> None:
    pc, mock_indexes = _make_pc_with_mock_indexes()
    pc.create_index(name="test")

    mock_indexes.create.assert_called_once_with(
        name="test",
        spec=None,
        dimension=None,
        metric=None,
        vector_type=None,
        deletion_protection=None,
        tags=None,
        timeout=None,
    )


def test_create_index_shim_rejects_2026_07_only_kwargs_before_delegating() -> None:
    """deployment=/read_capacity=/cmek_id=/schema= must not reach Indexes.create()."""
    pc, mock_indexes = _make_pc_with_mock_indexes()
    for kwargs in (
        {"deployment": {"deployment_type": "managed"}},
        {"cmek_id": "key-1"},
        {"read_capacity": {"mode": "OnDemand"}},
        {"schema": {"fields": {}}},
    ):
        with pytest.raises(TypeError):
            pc.create_index(name="test", **kwargs)
    mock_indexes.create.assert_not_called()


def test_create_index_shim_forwards_legacy_hard_break_kwargs_for_interception() -> None:
    """pods=/source_backup_id= reach the real Indexes.create() guided error."""
    from pinecone.errors.exceptions import PineconeTypeError

    pc = Pinecone(api_key="test-key")
    with pytest.raises(PineconeTypeError):
        pc.create_index(name="test", source_backup_id="bkp-1")
    with pytest.raises(PineconeTypeError):
        pc.create_index(name="test", pods=4)
