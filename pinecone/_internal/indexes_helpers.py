"""Shared validation and body-building helpers for sync/async Indexes.

These are pure functions (no I/O) extracted from the duplicated private
methods on ``Indexes`` and ``AsyncIndexes``, plus polling helpers that
encapsulate the describe-until-ready loop.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, TypedDict

import msgspec
from typing_extensions import NotRequired

from pinecone._internal.validation import require_non_empty

if TYPE_CHECKING:
    from pinecone.models.indexes.index import IndexModel
from pinecone.errors.exceptions import PineconeTypeError, ValidationError
from pinecone.models.enums import DeletionProtection, Metric, VectorType
from pinecone.models.indexes.specs import ByocSpec, IntegratedSpec, PodSpec, ServerlessSpec

_VALID_METRICS = frozenset({"cosine", "euclidean", "dotproduct"})
_VALID_DELETION_PROTECTION = frozenset({"enabled", "disabled"})


def _validate_deletion_protection(deletion_protection: DeletionProtection | str) -> None:
    resolved = resolve_enum_value(deletion_protection)
    if resolved not in _VALID_DELETION_PROTECTION:
        raise ValidationError(
            f"deletion_protection must be one of {sorted(_VALID_DELETION_PROTECTION)}, "
            f"got {resolved!r}"
        )


class IndexKwargs(TypedDict):
    """Typed kwargs for constructing :class:`~pinecone.index.Index` or
    :class:`~pinecone.async_client.async_index.AsyncIndex`.
    """

    host: str
    api_key: str
    additional_headers: dict[str, str]
    timeout: float
    proxy_url: str
    proxy_headers: dict[str, str]
    ssl_ca_certs: str | None
    ssl_verify: bool
    source_tag: str
    connection_pool_maxsize: int


class _LegacyIndexKwargs(IndexKwargs):
    """IndexKwargs extended with the legacy pool_threads field (sync client only)."""

    pool_threads: NotRequired[int]


_INDEX_KWARG_KEYS = frozenset(IndexKwargs.__annotations__)


def apply_index_kwargs_overrides(
    base: IndexKwargs, overrides: Mapping[str, Any], *, caller: str
) -> IndexKwargs:
    """Apply explicit data-plane client kwargs to factory defaults."""
    allowed = _INDEX_KWARG_KEYS - {"host"}
    unexpected = sorted(set(overrides) - allowed)
    if unexpected:
        raise TypeError(f"{caller} got unexpected keyword arguments: {unexpected!r}")

    return IndexKwargs(
        host=base["host"],
        api_key=overrides.get("api_key", base["api_key"]),
        additional_headers=dict(overrides.get("additional_headers", base["additional_headers"])),
        timeout=overrides.get("timeout", base["timeout"]),
        proxy_url=overrides.get("proxy_url", base["proxy_url"]),
        proxy_headers=dict(overrides.get("proxy_headers", base["proxy_headers"])),
        ssl_ca_certs=overrides.get("ssl_ca_certs", base["ssl_ca_certs"]),
        ssl_verify=overrides.get("ssl_verify", base["ssl_verify"]),
        source_tag=overrides.get("source_tag", base["source_tag"]),
        connection_pool_maxsize=overrides.get(
            "connection_pool_maxsize", base["connection_pool_maxsize"]
        ),
    )


def resolve_enum_value(value: Any) -> Any:
    """Extract ``.value`` from enum-like objects, pass through otherwise."""
    return value.value if hasattr(value, "value") else value


def validate_read_capacity(read_capacity: dict[str, Any]) -> None:
    """Validate read_capacity structure for index configure."""
    if "mode" not in read_capacity:
        raise ValidationError("read_capacity must contain a 'mode' key")

    if read_capacity["mode"] == "Dedicated":
        dedicated = read_capacity.get("dedicated")
        if dedicated is not None and not isinstance(dedicated, dict):
            raise ValidationError(
                "read_capacity with mode 'Dedicated' must contain a 'dedicated' dict"
            )
        if dedicated is not None and "scaling" in dedicated:
            manual = dedicated.get("manual")
            if manual is not None and not isinstance(manual, dict):
                raise ValidationError("dedicated read_capacity manual must be a dict")


def validate_create_inputs(
    *,
    name: str,
    spec: ServerlessSpec | PodSpec | ByocSpec | dict[str, Any],
    dimension: int | None,
    metric: Metric | str,
    vector_type: VectorType | str,
    deletion_protection: DeletionProtection | str,
) -> None:
    """Client-side validation for create() arguments."""
    require_non_empty("name", name)
    if len(name) > 45:
        raise ValidationError("index name must not exceed 45 characters")
    if not re.fullmatch(r"[a-z0-9-]+", name):
        raise ValidationError("index name must contain only lowercase letters, digits, and hyphens")

    if spec is None:
        raise ValidationError("spec is required")

    resolved_metric = resolve_enum_value(metric)
    if resolved_metric not in _VALID_METRICS:
        raise ValidationError(
            f"metric must be one of {sorted(_VALID_METRICS)}, got {resolved_metric!r}"
        )

    _validate_deletion_protection(deletion_protection)

    if isinstance(spec, dict) and not ({"serverless", "pod", "byoc"} & spec.keys()):
        raise ValidationError("spec dict must contain a 'serverless', 'pod', or 'byoc' key")

    if dimension is not None and not isinstance(dimension, int):
        raise PineconeTypeError(f"dimension must be an integer, got {type(dimension).__name__!r}")

    resolved_vt = resolve_enum_value(vector_type)
    if resolved_vt == "sparse" and dimension is not None:
        raise ValidationError("dimension must not be provided for sparse indexes")
    if resolved_vt != "sparse" and dimension is None:
        raise ValidationError("dimension is required for dense indexes")


def build_create_body(
    *,
    name: str,
    spec: ServerlessSpec | PodSpec | ByocSpec | dict[str, Any],
    dimension: int | None,
    metric: Metric | str,
    vector_type: VectorType | str,
    deletion_protection: DeletionProtection | str,
    tags: Mapping[str, str] | None,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON body for POST /indexes."""
    body: dict[str, Any] = {
        "name": name,
        "metric": resolve_enum_value(metric),
        "vector_type": resolve_enum_value(vector_type),
        "deletion_protection": resolve_enum_value(deletion_protection),
    }
    if dimension is not None:
        body["dimension"] = dimension
    if tags is not None:
        body["tags"] = tags

    if isinstance(spec, ServerlessSpec):
        serverless_dict: dict[str, Any] = {"cloud": spec.cloud, "region": spec.region}
        if spec.read_capacity is not None:
            serverless_dict["read_capacity"] = spec.read_capacity
        if spec.schema is not None:
            serverless_dict["schema"] = _normalize_schema(spec.schema)
        body["spec"] = {"serverless": serverless_dict}
    elif isinstance(spec, PodSpec):
        body["spec"] = {"pod": msgspec.to_builtins(spec)}
    elif isinstance(spec, dict):
        body["spec"] = {k: dict(v) if isinstance(v, dict) else v for k, v in spec.items()}

    if schema is not None:
        normalized = _normalize_schema(schema)
        spec_dict = body["spec"]
        for key in spec_dict:
            spec_dict[key]["schema"] = normalized

    return body


def validate_byoc_inputs(
    *,
    name: str,
    spec: ByocSpec,
    dimension: int | None,
    deletion_protection: DeletionProtection | str,
) -> None:
    """Client-side validation for BYOC index creation."""
    require_non_empty("name", name)
    if len(name) > 45:
        raise ValidationError("index name must not exceed 45 characters")
    if not re.fullmatch(r"[a-z0-9-]+", name):
        raise ValidationError("index name must contain only lowercase letters, digits, and hyphens")
    require_non_empty("environment", spec.environment)
    if dimension is None:
        raise ValidationError("dimension is required for BYOC indexes")

    _validate_deletion_protection(deletion_protection)


def build_byoc_body(
    *,
    name: str,
    spec: ByocSpec,
    dimension: int | None,
    metric: Metric | str,
    vector_type: VectorType | str,
    deletion_protection: DeletionProtection | str,
    tags: Mapping[str, str] | None,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON body for POST /indexes (BYOC)."""
    body: dict[str, Any] = {
        "name": name,
        "metric": resolve_enum_value(metric),
        "vector_type": resolve_enum_value(vector_type),
        "deletion_protection": resolve_enum_value(deletion_protection),
    }
    if dimension is not None:
        body["dimension"] = dimension
    if tags is not None:
        body["tags"] = tags

    byoc_dict: dict[str, Any] = {"environment": spec.environment}
    if spec.read_capacity is not None:
        byoc_dict["read_capacity"] = spec.read_capacity
    if spec.schema is not None:
        byoc_dict["schema"] = _normalize_schema(spec.schema)
    if schema is not None:
        byoc_dict["schema"] = _normalize_schema(schema)
    body["spec"] = {"byoc": byoc_dict}

    return body


def validate_integrated_inputs(
    *,
    name: str,
    spec: IntegratedSpec,
    deletion_protection: DeletionProtection | str = "disabled",
) -> None:
    """Client-side validation for integrated index creation."""
    require_non_empty("name", name)
    if len(name) > 45:
        raise ValidationError("index name must not exceed 45 characters")
    if not re.fullmatch(r"[a-z0-9-]+", name):
        raise ValidationError("index name must contain only lowercase letters, digits, and hyphens")
    if not spec.cloud or not spec.cloud.strip():
        raise ValidationError("cloud is required for integrated indexes")
    if not spec.embed.model or not spec.embed.model.strip():
        raise ValidationError("embed model is required for integrated indexes")
    if not spec.embed.field_map:
        raise ValidationError("embed field_map is required for integrated indexes")
    _validate_deletion_protection(deletion_protection)


def build_integrated_body(
    *,
    name: str,
    spec: IntegratedSpec,
    deletion_protection: DeletionProtection | str,
    tags: Mapping[str, str] | None,
    schema: dict[str, Any] | None = None,
    read_capacity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON body for POST /indexes (integrated/model-backed)."""
    embed_body: dict[str, Any] = {
        "model": resolve_enum_value(spec.embed.model),
        "field_map": spec.embed.field_map,
    }
    if spec.embed.dimension is not None:
        embed_body["dimension"] = spec.embed.dimension
    if spec.embed.metric is not None:
        embed_body["metric"] = resolve_enum_value(spec.embed.metric)
    if spec.embed.read_parameters is not None:
        embed_body["read_parameters"] = spec.embed.read_parameters
    if spec.embed.write_parameters is not None:
        embed_body["write_parameters"] = spec.embed.write_parameters

    body: dict[str, Any] = {
        "name": name,
        "cloud": resolve_enum_value(spec.cloud),
        "region": spec.region,
        "embed": embed_body,
    }

    resolved_dp = resolve_enum_value(deletion_protection)
    if resolved_dp != "disabled":
        body["deletion_protection"] = resolved_dp
    if tags is not None:
        body["tags"] = tags
    if schema is not None:
        body["schema"] = _normalize_schema(schema)
    if read_capacity is not None:
        body["read_capacity"] = read_capacity

    return body


def _normalize_schema(raw: dict[str, Any]) -> dict[str, Any]:
    """Wrap a bare fields map in ``{"fields": ...}`` for backend compatibility.

    The backend's ``IndexMetadataMetadataSchema`` requires the top-level
    ``fields`` key. For ergonomic convenience, the SDK accepts either form:

    - ``{"fields": {"genre": {"filterable": True}}}`` (wrapped, per the OpenAPI spec)
    - ``{"genre": {"filterable": True}}`` (bare fields map)

    Both produce the wrapped wire format.
    """
    if list(raw.keys()) == ["fields"] and isinstance(raw.get("fields"), dict):
        return raw
    return {"fields": raw}


def poll_index_until_ready(
    describe_fn: Callable[[str], IndexModel],
    name: str,
    timeout: int | None,
    poll_interval: float = 5.0,
) -> IndexModel:
    """Poll ``describe_fn(name)`` until the index is ready or timeout is reached.

    Args:
        describe_fn: Synchronous callable that takes an index name and returns
            an :class:`IndexModel`.
        name: Name of the index to poll.
        timeout: Maximum seconds to wait. ``None`` means wait indefinitely.
        poll_interval: Seconds between successive polls.

    Returns:
        The :class:`IndexModel` once its status is ready.

    Raises:
        IndexInitFailedError: If the index enters ``InitializationFailed`` state.
        IndexTerminatedError: If the index enters ``Terminating`` or ``Disabled`` state.
        PineconeTimeoutError: If *timeout* seconds elapse without becoming ready.
    """
    from pinecone.errors.exceptions import (
        IndexInitFailedError,
        IndexTerminatedError,
        PineconeTimeoutError,
    )

    start = time.monotonic()
    while True:
        idx = describe_fn(name)
        if idx.status.ready:
            return idx
        if idx.status.state == "InitializationFailed":
            raise IndexInitFailedError(name)
        if idx.status.state in ("Terminating", "Disabled"):
            raise IndexTerminatedError(name, idx.status.state)
        if timeout is not None:
            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                raise PineconeTimeoutError(f"Index '{name}' not ready after {timeout}s")
        time.sleep(poll_interval)


async def async_poll_index_until_ready(
    describe_fn: Callable[[str], Awaitable[IndexModel]],
    name: str,
    timeout: int | None,
    poll_interval: float = 5.0,
) -> IndexModel:
    """Async variant of :func:`poll_index_until_ready`.

    Args:
        describe_fn: Async callable that takes an index name and returns
            an :class:`IndexModel`.
        name: Name of the index to poll.
        timeout: Maximum seconds to wait. ``None`` means wait indefinitely.
        poll_interval: Seconds between successive polls.

    Returns:
        The :class:`IndexModel` once its status is ready.

    Raises:
        IndexInitFailedError: If the index enters ``InitializationFailed`` state.
        IndexTerminatedError: If the index enters ``Terminating`` or ``Disabled`` state.
        PineconeTimeoutError: If *timeout* seconds elapse without becoming ready.
    """
    from pinecone.errors.exceptions import (
        IndexInitFailedError,
        IndexTerminatedError,
        PineconeTimeoutError,
    )

    start = time.monotonic()
    while True:
        idx = await describe_fn(name)
        if idx.status.ready:
            return idx
        if idx.status.state == "InitializationFailed":
            raise IndexInitFailedError(name)
        if idx.status.state in ("Terminating", "Disabled"):
            raise IndexTerminatedError(name, idx.status.state)
        if timeout is not None:
            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                raise PineconeTimeoutError(f"Index '{name}' not ready after {timeout}s")
        await asyncio.sleep(poll_interval)
