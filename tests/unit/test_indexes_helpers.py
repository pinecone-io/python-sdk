"""Unit tests for index helper functions."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pinecone._internal.indexes_helpers import (
    _normalize_schema,
    async_poll_index_until_ready,
    build_byoc_body,
    build_integrated_body,
    poll_index_until_ready,
    validate_read_capacity,
)
from pinecone.errors.exceptions import IndexInitFailedError, IndexTerminatedError, ValidationError
from pinecone.models.indexes.deployment import ManagedDeployment
from pinecone.models.indexes.index import IndexModel, IndexStatus
from pinecone.models.indexes.schema import DenseVectorField, IndexSchema
from pinecone.models.indexes.specs import ByocSpec, EmbedConfig, IntegratedSpec


def test_validate_read_capacity_dedicated_partial_shards_only() -> None:
    """Partial patch with only shards specified must not raise."""
    read_capacity = {
        "mode": "Dedicated",
        "dedicated": {"scaling": "Manual", "manual": {"shards": 4}},
    }
    validate_read_capacity(read_capacity)


def test_validate_read_capacity_dedicated_partial_replicas_only() -> None:
    """Partial patch with only replicas specified must not raise."""
    read_capacity = {
        "mode": "Dedicated",
        "dedicated": {"scaling": "Manual", "manual": {"replicas": 2}},
    }
    validate_read_capacity(read_capacity)


def test_validate_read_capacity_dedicated_no_node_type() -> None:
    """Omitting node_type from a Dedicated patch must not raise."""
    read_capacity = {
        "mode": "Dedicated",
        "dedicated": {},
    }
    validate_read_capacity(read_capacity)


def test_validate_read_capacity_on_demand_no_raise() -> None:
    """OnDemand mode requires no further fields."""
    validate_read_capacity({"mode": "OnDemand"})


def test_validate_read_capacity_missing_mode_raises() -> None:
    """Omitting mode entirely must raise ValidationError."""
    with pytest.raises(ValidationError, match="mode"):
        validate_read_capacity({})


def test_validate_read_capacity_dedicated_wrong_type_raises() -> None:
    """Supplying a non-dict for 'dedicated' must raise ValidationError."""
    with pytest.raises(ValidationError, match="dedicated"):
        validate_read_capacity({"mode": "Dedicated", "dedicated": "string"})


def _make_integrated_spec() -> IntegratedSpec:
    return IntegratedSpec(
        cloud="aws",
        region="us-east-1",
        embed=EmbedConfig(model="multilingual-e5-large", field_map={"text": "body"}),
    )


def test_build_integrated_body_includes_schema() -> None:
    body = build_integrated_body(
        name="my-index",
        spec=_make_integrated_spec(),
        deletion_protection="disabled",
        tags=None,
        schema={"body": {"type": "str"}},
    )
    assert body["schema"] == {"fields": {"body": {"type": "str"}}}


def test_build_integrated_body_schema_absent_when_none() -> None:
    body = build_integrated_body(
        name="my-index",
        spec=_make_integrated_spec(),
        deletion_protection="disabled",
        tags=None,
        schema=None,
    )
    assert "schema" not in body


def test_build_integrated_body_includes_read_capacity() -> None:
    rc = {"mode": "OnDemand"}
    body = build_integrated_body(
        name="my-index",
        spec=_make_integrated_spec(),
        deletion_protection="disabled",
        tags=None,
        read_capacity=rc,
    )
    assert body["read_capacity"] == rc


def test_build_integrated_body_read_capacity_absent_when_none() -> None:
    body = build_integrated_body(
        name="my-index",
        spec=_make_integrated_spec(),
        deletion_protection="disabled",
        tags=None,
        read_capacity=None,
    )
    assert "read_capacity" not in body


def _make_byoc_body(**kwargs: object) -> dict[object, object]:
    return build_byoc_body(  # type: ignore[return-value]
        name="test-byoc",
        spec=kwargs.pop("spec", ByocSpec(environment="byoc-aws-abc123")),  # type: ignore[arg-type]
        dimension=128,
        metric="cosine",
        vector_type="dense",
        deletion_protection="disabled",
        tags=None,
        **kwargs,  # type: ignore[arg-type]
    )


def test_build_byoc_body_spec_schema_included() -> None:
    """ByocSpec.schema bare form must be wrapped before it reaches the wire."""
    spec = ByocSpec(environment="byoc-aws-abc123", schema={"genre": {"type": "str"}})
    body = _make_byoc_body(spec=spec)
    assert body["spec"]["byoc"]["schema"] == {"fields": {"genre": {"type": "str"}}}


def test_build_byoc_body_method_schema_included() -> None:
    """schema= method param must be wrapped and included in the byoc spec."""
    body = _make_byoc_body(schema={"genre": {"type": "str"}})
    assert body["spec"]["byoc"]["schema"] == {"fields": {"genre": {"type": "str"}}}


def test_build_byoc_body_schema_absent_when_none() -> None:
    """schema should not appear in the body when neither spec.schema nor schema= is set."""
    body = _make_byoc_body()
    assert "schema" not in body["spec"]["byoc"]


def test_build_integrated_body_embed_dimension_included() -> None:
    """EmbedConfig.dimension must be sent when set."""
    spec = IntegratedSpec(
        cloud="aws",
        region="us-east-1",
        embed=EmbedConfig(
            model="multilingual-e5-large",
            field_map={"text": "body"},
            dimension=768,
        ),
    )
    body = build_integrated_body(
        name="test-integrated",
        spec=spec,
        deletion_protection="disabled",
        tags=None,
    )
    assert body["embed"]["dimension"] == 768


def test_build_integrated_body_embed_dimension_absent_when_none() -> None:
    """embed.dimension must be absent when not set."""
    spec = IntegratedSpec(
        cloud="aws",
        region="us-east-1",
        embed=EmbedConfig(model="multilingual-e5-large", field_map={"text": "body"}),
    )
    body = build_integrated_body(
        name="test-integrated",
        spec=spec,
        deletion_protection="disabled",
        tags=None,
    )
    assert "dimension" not in body["embed"]


# _normalize_schema unit tests


def test_normalize_schema_wraps_bare_fields_map() -> None:
    result = _normalize_schema({"genre": {"filterable": True}})
    assert result == {"fields": {"genre": {"filterable": True}}}


def test_normalize_schema_passes_through_already_wrapped() -> None:
    wrapped = {"fields": {"genre": {"filterable": True}}}
    result = _normalize_schema(wrapped)
    assert result == {"fields": {"genre": {"filterable": True}}}


# ByocSpec schema round-trip tests


def test_build_byoc_body_spec_schema_wrapped_input_passes_through() -> None:
    spec = ByocSpec(
        environment="byoc-aws-abc123",
        schema={"fields": {"genre": {"filterable": True}}},
    )
    body = _make_byoc_body(spec=spec)
    assert body["spec"]["byoc"]["schema"] == {"fields": {"genre": {"filterable": True}}}


def test_build_byoc_body_spec_schema_bare_input_gets_wrapped() -> None:
    spec = ByocSpec(
        environment="byoc-aws-abc123",
        schema={"genre": {"filterable": True}},
    )
    body = _make_byoc_body(spec=spec)
    assert body["spec"]["byoc"]["schema"] == {"fields": {"genre": {"filterable": True}}}


def test_build_byoc_body_method_and_spec_schema_normalized_match() -> None:
    bare = {"genre": {"filterable": True}}
    wrapped = {"fields": {"genre": {"filterable": True}}}

    spec_via_spec = ByocSpec(environment="byoc-aws-abc123", schema=bare)
    body_via_spec = _make_byoc_body(spec=spec_via_spec)

    body_via_method = _make_byoc_body(schema=bare)

    assert (
        body_via_spec["spec"]["byoc"]["schema"]
        == body_via_method["spec"]["byoc"]["schema"]
        == wrapped
    )


# Integrated schema round-trip tests


def test_build_integrated_body_method_schema_bare_input_gets_wrapped() -> None:
    body = build_integrated_body(
        name="my-index",
        spec=_make_integrated_spec(),
        deletion_protection="disabled",
        tags=None,
        schema={"genre": {"filterable": True}},
    )
    assert body["schema"] == {"fields": {"genre": {"filterable": True}}}


# poll_index_until_ready terminal-state tests


def _make_index(state: str, ready: bool = False) -> IndexModel:
    return IndexModel(
        name="test-index",
        host="test-index.svc.pinecone.io",
        schema=IndexSchema(fields={"embedding": DenseVectorField(dimension=1536, metric="cosine")}),
        deployment=ManagedDeployment(cloud="aws", region="us-east-1"),
        status=IndexStatus(ready=ready, state=state),
        deletion_protection="disabled",
        tags=None,
    )


def test_poll_index_until_ready_raises_on_terminating() -> None:
    describe_fn = MagicMock(return_value=_make_index("Terminating"))
    with pytest.raises(IndexTerminatedError) as exc_info:
        poll_index_until_ready(describe_fn, "test-index", timeout=10)
    assert exc_info.value.state == "Terminating"
    assert exc_info.value.name == "test-index"


def test_poll_index_until_ready_raises_on_disabled() -> None:
    describe_fn = MagicMock(return_value=_make_index("Disabled"))
    with pytest.raises(IndexTerminatedError) as exc_info:
        poll_index_until_ready(describe_fn, "test-index", timeout=10)
    assert exc_info.value.state == "Disabled"
    assert exc_info.value.name == "test-index"


def test_poll_index_until_ready_still_raises_on_initialization_failed() -> None:
    describe_fn = MagicMock(return_value=_make_index("InitializationFailed"))
    with pytest.raises(IndexInitFailedError):
        poll_index_until_ready(describe_fn, "test-index", timeout=10)


def test_poll_index_until_ready_returns_on_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pinecone._internal.indexes_helpers.time.sleep", lambda *_: None)
    ready_index = _make_index("Ready", ready=True)
    describe_fn = MagicMock(return_value=ready_index)
    result = poll_index_until_ready(describe_fn, "test-index", timeout=10)
    assert result.status.ready is True


async def test_async_poll_index_until_ready_raises_on_terminating() -> None:
    describe_fn = AsyncMock(return_value=_make_index("Terminating"))
    with pytest.raises(IndexTerminatedError) as exc_info:
        await async_poll_index_until_ready(describe_fn, "test-index", timeout=10)
    assert exc_info.value.state == "Terminating"
    assert exc_info.value.name == "test-index"


async def test_async_poll_index_until_ready_raises_on_disabled() -> None:
    describe_fn = AsyncMock(return_value=_make_index("Disabled"))
    with pytest.raises(IndexTerminatedError) as exc_info:
        await async_poll_index_until_ready(describe_fn, "test-index", timeout=10)
    assert exc_info.value.state == "Disabled"
    assert exc_info.value.name == "test-index"


async def test_async_poll_index_until_ready_still_raises_on_initialization_failed() -> None:
    describe_fn = AsyncMock(return_value=_make_index("InitializationFailed"))
    with pytest.raises(IndexInitFailedError):
        await async_poll_index_until_ready(describe_fn, "test-index", timeout=10)
