"""Unit tests for Pinecone.create_index backcompat shim (2026-07 signature)."""

from __future__ import annotations

from unittest.mock import MagicMock

from pinecone import Pinecone

SCHEMA = {"fields": {"embedding": {"type": "dense_vector", "dimension": 4, "metric": "cosine"}}}


def _make_pc_with_mock_indexes() -> tuple[Pinecone, MagicMock]:
    pc = Pinecone(api_key="test-key")
    mock_indexes = MagicMock()
    mock_indexes.create = MagicMock(return_value=MagicMock())
    pc._indexes = mock_indexes
    return pc, mock_indexes


def test_create_index_shim_forwards_schema() -> None:
    """Shim must forward the 2026-07 schema kwarg to Indexes.create."""
    pc, mock_indexes = _make_pc_with_mock_indexes()
    pc.create_index(name="test", schema=SCHEMA)

    mock_indexes.create.assert_called_once()
    _, kwargs = mock_indexes.create.call_args
    assert kwargs["schema"] == SCHEMA
    assert kwargs["name"] == "test"


def test_create_index_shim_forwards_all_new_kwargs() -> None:
    pc, mock_indexes = _make_pc_with_mock_indexes()
    pc.create_index(
        name="test",
        schema=SCHEMA,
        deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
        read_capacity={"mode": "OnDemand"},
        deletion_protection="enabled",
        tags={"env": "prod"},
        cmek_id="key-1",
        timeout=-1,
    )

    _, kwargs = mock_indexes.create.call_args
    assert kwargs["read_capacity"] == {"mode": "OnDemand"}
    assert kwargs["cmek_id"] == "key-1"
    assert kwargs["timeout"] == -1


def test_create_index_shim_forwards_legacy_kwargs_for_interception() -> None:
    """Legacy kwargs pass through so Indexes.create raises the guided error."""
    pc, mock_indexes = _make_pc_with_mock_indexes()
    pc.create_index(name="test", dimension=1536, spec={"serverless": {}})

    _, kwargs = mock_indexes.create.call_args
    assert kwargs["dimension"] == 1536
    assert kwargs["spec"] == {"serverless": {}}
