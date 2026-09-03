"""Unit tests for read-capacity response models (2026-07)."""

from __future__ import annotations

import msgspec

from pinecone.models.indexes.read_capacity import (
    ReadCapacityDedicatedResponse,
    ReadCapacityOnDemandResponse,
    ReadCapacityResponse,
    ReadCapacityStatus,
)


def test_on_demand_response_decode() -> None:
    raw = b'{"mode": "OnDemand", "status": {"state": "Ready", "current_shards": null, "current_replicas": null}}'
    result = msgspec.json.decode(raw, type=ReadCapacityResponse)
    assert isinstance(result, ReadCapacityOnDemandResponse)
    assert result.status.state == "Ready"
    assert result.status.current_shards is None
    assert result.status.current_replicas is None


def test_on_demand_response_decode_without_current_counts() -> None:
    raw = b'{"mode": "OnDemand", "status": {"state": "Ready"}}'
    result = msgspec.json.decode(raw, type=ReadCapacityResponse)
    assert isinstance(result, ReadCapacityOnDemandResponse)
    assert result.status.current_shards is None
    assert result.status.current_replicas is None


def test_dedicated_response_decode_manual() -> None:
    raw = (
        b'{"mode": "Dedicated", "dedicated": {"node_type": "b1", "scaling": "Manual", '
        b'"manual": {"shards": 2, "replicas": 1}}, '
        b'"status": {"state": "Ready", "current_shards": 2, "current_replicas": 1}}'
    )
    result = msgspec.json.decode(raw, type=ReadCapacityResponse)
    assert isinstance(result, ReadCapacityDedicatedResponse)
    assert result.dedicated.node_type == "b1"
    assert result.dedicated.scaling == "Manual"
    assert result.dedicated.manual is not None
    assert result.dedicated.manual.shards == 2
    assert result.dedicated.manual.replicas == 1
    assert result.status.current_shards == 2
    assert result.status.current_replicas == 1


def test_read_capacity_union_decode_dispatches_on_mode() -> None:
    on_demand_raw = b'{"mode": "OnDemand", "status": {"state": "Ready"}}'
    dedicated_raw = (
        b'{"mode": "Dedicated", "dedicated": {"node_type": "b1", "scaling": "Manual", '
        b'"manual": {"shards": 1, "replicas": 1}}, "status": {"state": "Ready"}}'
    )

    on_demand = msgspec.json.decode(on_demand_raw, type=ReadCapacityResponse)
    dedicated = msgspec.json.decode(dedicated_raw, type=ReadCapacityResponse)

    assert isinstance(on_demand, ReadCapacityOnDemandResponse)
    assert isinstance(dedicated, ReadCapacityDedicatedResponse)


def test_read_capacity_status_error_message() -> None:
    raw = b'{"state":"Error","current_shards":null,"current_replicas":null,"error_message":"provisioning failed"}'
    status = msgspec.json.decode(raw, type=ReadCapacityStatus)
    assert status.state == "Error"
    assert status.error_message == "provisioning failed"


def test_read_capacity_status_no_error_message() -> None:
    raw = b'{"state":"Ready","current_shards":2,"current_replicas":1}'
    status = msgspec.json.decode(raw, type=ReadCapacityStatus)
    assert status.state == "Ready"
    assert status.error_message is None


def test_status_decode_with_scaling_and_migrating_states() -> None:
    scaling_raw = b'{"state": "Scaling", "current_shards": null, "current_replicas": null}'
    status = msgspec.json.decode(scaling_raw, type=ReadCapacityStatus)
    assert status.state == "Scaling"
    assert status.current_shards is None
    assert status.current_replicas is None

    migrating_raw = b'{"state": "Migrating", "current_shards": 2, "current_replicas": 1}'
    status2 = msgspec.json.decode(migrating_raw, type=ReadCapacityStatus)
    assert status2.state == "Migrating"
    assert status2.current_shards == 2
    assert status2.current_replicas == 1
