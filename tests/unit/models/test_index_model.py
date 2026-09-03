"""Unit tests for the graduated IndexModel (2026-07)."""

from __future__ import annotations

from unittest.mock import MagicMock

import msgspec

from pinecone.models.indexes.deployment import (
    ByocDeployment,
    ManagedDeployment,
    PodDeployment,
)
from pinecone.models.indexes.index import IndexModel, IndexStatus
from pinecone.models.indexes.read_capacity import (
    ReadCapacityOnDemandResponse,
    ReadCapacityStatus,
)
from pinecone.models.indexes.schema import (
    DenseVectorField,
    FullTextSearchConfig,
    IndexSchema,
    IndexSchemaField,
    StringField,
)

_FULL_PAYLOAD = b"""
{
    "name": "my-index",
    "host": "my-index-abc123.svc.pinecone.io",
    "status": {"ready": true, "state": "Ready"},
    "schema": {
        "fields": {
            "title": {"type": "string", "full_text_search": {"language": "en"}},
            "embedding": {"type": "dense_vector", "dimension": 768, "metric": "cosine"}
        }
    },
    "deployment": {
        "deployment_type": "managed",
        "environment": "aped-4627-b74a",
        "cloud": "aws",
        "region": "us-east-1"
    },
    "deletion_protection": "disabled",
    "read_capacity": {"mode": "OnDemand", "status": {"state": "Ready"}},
    "tags": {"env": "test", "team": "ml"}
}
"""

_MINIMAL_PAYLOAD = b"""
{
    "name": "bare-index",
    "host": "bare-index-xyz.svc.pinecone.io",
    "status": {"ready": false, "state": "Initializing"},
    "schema": {"fields": {}},
    "deployment": {
        "deployment_type": "managed",
        "environment": "aped-0001",
        "cloud": "gcp",
        "region": "us-central1"
    },
    "deletion_protection": "enabled"
}
"""


def _make_model(
    *,
    name: str = "test-index",
    host: str | None = "test-index.svc.pinecone.io",
    state: str = "Ready",
    ready: bool = True,
    n_fields: int = 0,
    deployment_type: str = "managed",
    deletion_protection: str = "disabled",
    read_capacity: ReadCapacityOnDemandResponse | None = None,
    tags: dict[str, str] | None = None,
) -> IndexModel:
    fields: dict[str, IndexSchemaField] = {}
    for i in range(n_fields):
        fields[f"field_{i}"] = StringField(full_text_search=FullTextSearchConfig())

    if deployment_type == "pod":
        deployment: ManagedDeployment | PodDeployment = PodDeployment(
            environment="us-east1-gcp", pod_type="p1.x1", replicas=1, shards=1
        )
    else:
        deployment = ManagedDeployment(
            environment="aped-4627-b74a", cloud="aws", region="us-east-1"
        )

    return IndexModel(
        name=name,
        host=host,
        status=IndexStatus(ready=ready, state=state),
        schema=IndexSchema(fields=fields),
        deployment=deployment,
        deletion_protection=deletion_protection,
        read_capacity=read_capacity,
        tags=tags,
    )


def test_index_model_decode_full() -> None:
    m = msgspec.json.decode(_FULL_PAYLOAD, type=IndexModel)
    assert isinstance(m, IndexModel)
    assert m.name == "my-index"
    assert m.host == "https://my-index-abc123.svc.pinecone.io"
    assert isinstance(m.status, IndexStatus)
    assert m.status.ready is True
    assert m.status.state == "Ready"
    assert isinstance(m.schema, IndexSchema)
    assert len(m.schema.fields) == 2
    assert isinstance(m.schema.fields["title"], StringField)
    assert isinstance(m.schema.fields["embedding"], DenseVectorField)
    assert isinstance(m.deployment, ManagedDeployment)
    assert m.deployment.cloud == "aws"
    assert m.deployment.region == "us-east-1"
    assert m.deployment.environment == "aped-4627-b74a"
    assert m.deletion_protection == "disabled"
    assert isinstance(m.read_capacity, ReadCapacityOnDemandResponse)
    assert m.tags == {"env": "test", "team": "ml"}


def test_index_model_decode_minimal() -> None:
    m = msgspec.json.decode(_MINIMAL_PAYLOAD, type=IndexModel)
    assert m.name == "bare-index"
    assert m.read_capacity is None
    assert m.tags is None
    assert m.private_host is None
    assert m.source_collection is None
    assert m.source_backup_id is None
    assert m.cmek_id is None


def test_index_model_decodes_private_host_and_sources() -> None:
    payload = b"""
    {
        "name": "x",
        "host": "x.svc.pinecone.io",
        "status": {"ready": true, "state": "Ready"},
        "schema": {"fields": {}},
        "deployment": {
            "deployment_type": "managed",
            "environment": "e1",
            "cloud": "aws",
            "region": "us-east-1"
        },
        "deletion_protection": "disabled",
        "private_host": "p.svc.pinecone.io",
        "source_collection": "c1",
        "source_backup_id": "670e8400-e29b-41d4-a716-446655440000",
        "cmek_id": "arn:aws:kms:us-east-1:123456789012:key/mrk-abc123"
    }
    """
    m = msgspec.json.decode(payload, type=IndexModel)
    assert m.private_host == "https://p.svc.pinecone.io"
    assert m.source_collection == "c1"
    assert m.source_backup_id == "670e8400-e29b-41d4-a716-446655440000"
    assert m.cmek_id == "arn:aws:kms:us-east-1:123456789012:key/mrk-abc123"


def test_index_model_repr_single_line() -> None:
    m = _make_model()
    r = repr(m)
    assert r.startswith("IndexModel(")
    assert r.endswith(")")
    assert "\n" not in r
    assert "name=" in r
    assert "status=" in r
    assert "host=" in r


def test_index_model_repr_includes_schema_fields() -> None:
    m = _make_model(n_fields=3)
    assert "schema_fields=3" in repr(m)


def test_index_model_repr_omits_empty_tags() -> None:
    m = _make_model(tags=None)
    assert "tags=" not in repr(m)


def test_index_model_repr_pretty_cycle() -> None:
    m = _make_model()
    p = MagicMock()
    m._repr_pretty_(p, cycle=True)
    p.text.assert_called_with("IndexModel(...)")


def test_index_model_repr_html_contains_table() -> None:
    m = _make_model(n_fields=2)
    html = m._repr_html_()
    assert "IndexModel" in html
    assert "Name:" in html
    assert "Status:" in html
    assert "Deployment:" in html
    assert "Host:" in html
    assert "Schema fields:" in html


def test_index_model_repr_html_includes_tags_when_present() -> None:
    m = _make_model(tags={"env": "prod"})
    html = m._repr_html_()
    assert "env=prod" in html


def test_index_model_repr_html_pod_deployment_detail() -> None:
    m = _make_model(deployment_type="pod")
    html = m._repr_html_()
    assert "Pod" in html
    assert "(us-east1-gcp)" in html
    assert "Deployment:" in html


def test_index_model_repr_html_byoc_deployment_detail() -> None:
    m = IndexModel(
        name="byoc-index",
        host="byoc-index.svc.pinecone.io",
        status=IndexStatus(ready=True, state="Ready"),
        schema=IndexSchema(fields={}),
        deployment=ByocDeployment(environment="e1"),
        deletion_protection="disabled",
    )
    html = m._repr_html_()
    assert "Byoc" in html
    assert "(e1)" in html


def test_index_model_repr_html_includes_read_capacity_row() -> None:
    rc = ReadCapacityOnDemandResponse(
        status=ReadCapacityStatus(state="Ready"),
    )
    m = _make_model(read_capacity=rc)
    html = m._repr_html_()
    assert "Read capacity:" in html
    assert "OnDemand" in html


def test_index_model_repr_html_omits_read_capacity_when_none() -> None:
    m = _make_model()
    html = m._repr_html_()
    assert "Read capacity:" not in html


def test_index_model_repr_pretty_non_cycle_emits_core_fields() -> None:
    m = _make_model(n_fields=2)
    p = MagicMock()
    m._repr_pretty_(p, cycle=False)
    emitted = "".join(c.args[0] for c in p.text.call_args_list)
    assert "IndexModel(" in emitted
    assert "name='test-index'" in emitted
    assert "status='Ready'" in emitted
    assert "host='https://test-index.svc.pinecone.io'" in emitted
    assert "deletion_protection='disabled'" in emitted
    assert "schema=IndexSchema(fields=2 fields)" in emitted
    assert p.breakable.call_count >= 1


def test_index_model_repr_pretty_non_cycle_includes_read_capacity_when_present() -> None:
    rc = ReadCapacityOnDemandResponse(status=ReadCapacityStatus(state="Ready"))
    m = _make_model(read_capacity=rc)
    p = MagicMock()
    m._repr_pretty_(p, cycle=False)
    emitted = "".join(c.args[0] for c in p.text.call_args_list)
    assert "read_capacity=" in emitted


def test_index_model_repr_pretty_non_cycle_includes_tags_when_present() -> None:
    m = _make_model(tags={"env": "prod"})
    p = MagicMock()
    m._repr_pretty_(p, cycle=False)
    emitted = "".join(c.args[0] for c in p.text.call_args_list)
    assert "tags=" in emitted
    assert "'env'" in emitted


def test_index_model_repr_pretty_non_cycle_omits_optional_fields_when_none() -> None:
    m = _make_model()
    p = MagicMock()
    m._repr_pretty_(p, cycle=False)
    emitted = "".join(c.args[0] for c in p.text.call_args_list)
    assert "read_capacity=" not in emitted
    assert "tags=" not in emitted


def test_index_model_null_host() -> None:
    """IndexModel must decode null host without raising."""
    raw = b'{"name":"test","host":null,"status":{"state":"Initializing","ready":false},"schema":{"fields":{}},"deployment":{"deployment_type":"managed","cloud":"aws","region":"us-east-1"},"deletion_protection":"disabled"}'
    model = msgspec.json.decode(raw, type=IndexModel)
    assert model.host is None
    assert model.name == "test"


def test_index_model_missing_host_defaults_to_none() -> None:
    """IndexModel must decode a response that omits the host field entirely."""
    raw = b'{"name":"test","status":{"state":"Initializing","ready":false},"schema":{"fields":{}},"deployment":{"deployment_type":"managed","cloud":"aws","region":"us-east-1"},"deletion_protection":"disabled"}'
    model = msgspec.json.decode(raw, type=IndexModel)
    assert model.host is None
    assert model.name == "test"


def test_index_model_null_host_repr() -> None:
    """__repr__ must not raise when host is None."""
    m = _make_model(host=None)
    r = repr(m)
    assert "host=None" in r


def test_index_model_null_host_repr_html() -> None:
    """_repr_html_ must not raise when host is None and must emit 'not yet assigned'."""
    m = _make_model(host=None)
    html = m._repr_html_()
    assert "Host:" in html
    assert "not yet assigned" in html


def test_index_model_null_host_repr_pretty() -> None:
    """_repr_pretty_ must not raise when host is None."""
    m = _make_model(host=None)
    p = MagicMock()
    m._repr_pretty_(p, cycle=False)
    emitted = "".join(c.args[0] for c in p.text.call_args_list)
    assert "host=None" in emitted


def test_index_model_dir_lists_public_fields() -> None:
    m = _make_model()
    listing = dir(m)
    for expected in ("name", "host", "status", "schema", "deployment", "read_capacity"):
        assert expected in listing
    assert not any(name.startswith("_") for name in listing)
