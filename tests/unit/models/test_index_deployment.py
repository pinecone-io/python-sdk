"""Unit tests for index deployment models (2026-07)."""

from __future__ import annotations

import msgspec
import pytest

from pinecone.models.indexes.deployment import (
    ByocDeployment,
    IndexDeployment,
    ManagedDeployment,
    PodDeployment,
)


def test_managed_deployment_decode() -> None:
    raw = b'{"deployment_type": "managed", "environment": "aped-1", "cloud": "aws", "region": "us-east-1"}'
    result = msgspec.json.decode(raw, type=IndexDeployment)
    assert isinstance(result, ManagedDeployment)
    assert result.environment == "aped-1"
    assert result.cloud == "aws"
    assert result.region == "us-east-1"


def test_managed_deployment_no_environment() -> None:
    raw = b'{"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"}'
    result = msgspec.json.decode(raw, type=ManagedDeployment)
    assert result.cloud == "aws"
    assert result.region == "us-east-1"
    assert result.environment is None


def test_pod_deployment_decode_full() -> None:
    raw = (
        b'{"deployment_type": "pod", "environment": "us-east1-gcp", '
        b'"pod_type": "p1.x1", "replicas": 2, "shards": 2}'
    )
    result = msgspec.json.decode(raw, type=IndexDeployment)
    assert isinstance(result, PodDeployment)
    assert result.environment == "us-east1-gcp"
    assert result.pod_type == "p1.x1"
    assert result.replicas == 2
    assert result.shards == 2


def test_pod_deployment_requires_replicas_and_shards() -> None:
    raw = b'{"deployment_type": "pod", "environment": "us-east1-gcp", "pod_type": "p1.x1"}'
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(raw, type=IndexDeployment)


def test_byoc_deployment_decode() -> None:
    raw = b'{"deployment_type": "byoc", "environment": "aws-us-east-1-b921"}'
    result = msgspec.json.decode(raw, type=IndexDeployment)
    assert isinstance(result, ByocDeployment)
    assert result.environment == "aws-us-east-1-b921"


def test_deployment_union_ignores_unknown_fields() -> None:
    raw = (
        b'{"deployment_type": "managed", "environment": "aped-1", "cloud": "aws", '
        b'"region": "us-east-1", "metadata_config": {"indexed": ["genre"]}}'
    )
    result = msgspec.json.decode(raw, type=IndexDeployment)
    assert isinstance(result, ManagedDeployment)
    assert result.environment == "aped-1"


def test_deployment_union_rejects_unknown_type() -> None:
    raw = b'{"deployment_type": "quantum", "environment": "e1"}'
    with pytest.raises(msgspec.ValidationError, match="deployment_type"):
        msgspec.json.decode(raw, type=IndexDeployment)


def test_encode_includes_discriminator() -> None:
    managed = ManagedDeployment(cloud="aws", region="us-east-1")
    built = msgspec.to_builtins(managed)
    assert built["deployment_type"] == "managed"

    pod = PodDeployment(environment="us-east1-gcp", pod_type="p1.x1", replicas=1, shards=1)
    assert msgspec.to_builtins(pod)["deployment_type"] == "pod"

    byoc = ByocDeployment(environment="e1")
    assert msgspec.to_builtins(byoc)["deployment_type"] == "byoc"
