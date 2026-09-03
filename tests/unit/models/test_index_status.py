"""Unit tests for the IndexStatus model."""

from __future__ import annotations

import msgspec

from pinecone.models.indexes.index import IndexStatus


def test_index_status_fields() -> None:
    status = IndexStatus(ready=True, state="Ready")
    assert status.ready is True
    assert status.state == "Ready"


def test_index_status_decode() -> None:
    raw = b'{"ready": false, "state": "Initializing"}'
    status = msgspec.json.decode(raw, type=IndexStatus)
    assert status.ready is False
    assert status.state == "Initializing"


def test_index_status_decode_expanded_states() -> None:
    for state in (
        "Initializing",
        "InitializationFailed",
        "ScalingUp",
        "ScalingDown",
        "ScalingUpPodSize",
        "ScalingDownPodSize",
        "Terminating",
        "Ready",
        "Disabled",
    ):
        raw = msgspec.json.encode({"ready": False, "state": state})
        status = msgspec.json.decode(raw, type=IndexStatus)
        assert status.state == state
