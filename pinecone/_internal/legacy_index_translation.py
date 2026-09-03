"""Pure translation of 2025-10 index kwargs into the 2026-07 request shape.

``pinecone/_internal/index_migration.py`` intercepts the legacy
``spec=``/``dimension=``/``metric=``/``vector_type=`` surface and raises,
on the grounds that translating ``dimension=1536`` would require inventing
a schema field name the caller's data-plane code has never heard of.
pinecone-db#18066 removed that obstacle: a schema whose fields are exactly
``_values`` (dense) and/or ``_sparse_values`` (sparse) and nothing else
persists no ``schema_def``, so the stored row is identical to a legacy
index and the vectors API serves it natively. Those two names are reserved
names the vectors API already addresses, so nothing is being invented.

Everything here is a pure function. The module imports nothing from
``pinecone.client``, ``pinecone.async_client``, or
``pinecone._internal.http_client``, which is what lets the sync and async
``Indexes`` surfaces share one implementation and lets the property tests
hammer it without fixtures. ``tests/unit/_internal/test_legacy_index_translation.py``
pins that constraint so it cannot rot.

``IntegratedSpec`` is deliberately not translated: integrated-embedding
indexes are created through ``create_for_model()``, so it stays in the
rejection path.
"""

from __future__ import annotations

import copy
from typing import Any

from pinecone._internal.index_migration import MIGRATION_GUIDE, reject_legacy_spec_fields
from pinecone._internal.indexes_helpers import resolve_enum_value
from pinecone.errors.exceptions import PineconeTypeError, PineconeValueError
from pinecone.models.enums import Metric, PodType, VectorType
from pinecone.models.indexes.specs import ByocSpec, IntegratedSpec, PodSpec, ServerlessSpec

#: Reserved schema field name the 2026-07 vectors API addresses dense values by.
DENSE_FIELD = "_values"

#: Reserved schema field name the 2026-07 vectors API addresses sparse values by.
SPARSE_FIELD = "_sparse_values"

_VALID_METRICS = frozenset({"cosine", "euclidean", "dotproduct"})

_VALID_VECTOR_TYPES = frozenset({"dense", "sparse"})

_DICT_SPEC_KEYS: tuple[tuple[str, type[ServerlessSpec | PodSpec | ByocSpec]], ...] = (
    ("serverless", ServerlessSpec),
    ("pod", PodSpec),
    ("byoc", ByocSpec),
)

_DEPLOYMENT_HINT = (
    "The 2026-07 API replaces spec= with deployment={'deployment_type': "
    "'managed'|'pod'|'byoc', ...}"
)


def _coerce_spec(spec: Any) -> ServerlessSpec | PodSpec | ByocSpec:
    """Normalize a legacy ``spec=`` value to the spec struct it stands for.

    A dict spec is rebuilt as its struct, so the dict form inherits the
    struct's defaults (a ``{"pod": {"environment": ...}}`` dict picks up
    ``pod_type="p1.x1"``, ``replicas=1``, ``shards=1``) instead of carrying a
    second copy of them.

    Raises:
        PineconeValueError: The value is not a spec the 2026-07 API has a
            deployment translation for.
        PineconeTypeError: The spec sets a field the 2026-07 create request
            has no destination for. See
            :func:`~pinecone._internal.index_migration.reject_legacy_spec_fields`.
    """
    if isinstance(spec, (ServerlessSpec, PodSpec, ByocSpec)):
        reject_legacy_spec_fields(spec)
        return spec

    if isinstance(spec, IntegratedSpec):
        raise PineconeValueError(
            "spec=IntegratedSpec(...) has no deployment= translation: the 2026-07 API "
            "creates integrated-embedding indexes through "
            "pc.indexes.create_for_model(name=..., cloud=..., region=..., embed=...). "
            f"See the migration guide: {MIGRATION_GUIDE}"
        )

    if isinstance(spec, dict):
        for key, struct in _DICT_SPEC_KEYS:
            inner = spec.get(key)
            if inner is None:
                continue
            try:
                resolved = struct(**inner)
            except TypeError as exc:
                raise PineconeValueError(
                    f"spec={{{key!r}: ...}} is not a valid {struct.__name__}: {exc}. "
                    f"{_DEPLOYMENT_HINT}. See the migration guide: {MIGRATION_GUIDE}"
                ) from exc
            reject_legacy_spec_fields(resolved)
            return resolved
        raise PineconeValueError(
            "spec dict must contain a 'serverless', 'pod', or 'byoc' key, got "
            f"{sorted(map(str, spec))}. {_DEPLOYMENT_HINT}. "
            f"See the migration guide: {MIGRATION_GUIDE}"
        )

    raise PineconeValueError(
        "spec must be a ServerlessSpec, PodSpec, ByocSpec, or an equivalent dict, got "
        f"{type(spec).__name__!r}. {_DEPLOYMENT_HINT}. "
        f"See the migration guide: {MIGRATION_GUIDE}"
    )


def spec_to_deployment(spec: Any) -> dict[str, Any]:
    """Translate a legacy ``spec=`` value into a 2026-07 ``deployment=`` dict.

    Args:
        spec: A :class:`~pinecone.models.indexes.specs.ServerlessSpec`,
            :class:`~pinecone.models.indexes.specs.PodSpec`,
            :class:`~pinecone.models.indexes.specs.ByocSpec`, or the equivalent
            ``{"serverless"|"pod"|"byoc": {...}}`` dict.

    Returns:
        The ``deployment`` object for the 2026-07 create request. Read capacity
        is *not* included — 2026-07 carries it at the top level of the request,
        so lift it out separately with :func:`spec_to_read_capacity`.

    Raises:
        PineconeValueError: *spec* is malformed or is not a spec at all, or
            (for a ``PodSpec``) *pods* is neither 1 (``PodSpec``'s own default,
            indistinguishable from "not set") nor *replicas x shards* — the
            2026-07 API has no independent *pods* field, and such a value
            can't be decomposed into the two fields it would replace.
        PineconeTypeError: *spec* sets ``metadata_config`` or
            ``source_collection``, neither of which the 2026-07 create
            request has a field to carry.

    Example:
        >>> spec_to_deployment(ServerlessSpec(cloud="aws", region="us-east-1"))
        {'deployment_type': 'managed', 'cloud': 'aws', 'region': 'us-east-1'}
    """
    resolved = _coerce_spec(spec)

    if isinstance(resolved, ServerlessSpec):
        return {
            "deployment_type": "managed",
            "cloud": resolved.cloud,
            "region": resolved.region,
        }

    if isinstance(resolved, PodSpec):
        expected_pods = resolved.replicas * resolved.shards
        if resolved.pods != 1 and resolved.pods != expected_pods:
            raise PineconeValueError(
                f"pods={resolved.pods} is inconsistent with replicas={resolved.replicas} "
                f"and shards={resolved.shards} (replicas x shards = {expected_pods}). "
                "The 2026-07 API has no pods= field — capacity is replicas x shards on "
                "the pod deployment. Set replicas= and shards= so their product equals "
                f"the pod count you want. See the migration guide: {MIGRATION_GUIDE}"
            )
        return {
            "deployment_type": "pod",
            "environment": resolved.environment,
            "pod_type": resolved.pod_type,
            "replicas": resolved.replicas,
            "shards": resolved.shards,
        }

    return {"deployment_type": "byoc", "environment": resolved.environment}


def spec_to_read_capacity(spec: Any) -> dict[str, Any] | None:
    """Lift read capacity out of a legacy ``spec=`` value.

    ``ServerlessSpec`` and ``ByocSpec`` nest read capacity inside the spec,
    while 2026-07 carries it at the top level of the request. It is the
    easiest part of the mapping to drop silently, so it gets its own function.

    Args:
        spec: The same values :func:`spec_to_deployment` accepts.

    Returns:
        A copy of the read-capacity object, or ``None`` when the spec sets
        none (always, for a ``PodSpec``). The copy keeps a caller who mutates
        the request body from reaching back into their own spec.

    Raises:
        PineconeValueError: *spec* is malformed or is not a spec at all.
        PineconeTypeError: *spec* sets ``metadata_config`` or
            ``source_collection``, neither of which the 2026-07 create
            request has a field to carry.
    """
    resolved = _coerce_spec(spec)
    if isinstance(resolved, PodSpec) or resolved.read_capacity is None:
        return None
    return copy.deepcopy(resolved.read_capacity)


def legacy_vector_schema(
    *,
    dimension: int | None,
    metric: Metric | str | None,
    vector_type: VectorType | str | None,
) -> dict[str, Any]:
    """Translate legacy vector kwargs into a 2026-07 ``schema=`` object.

    The result declares exactly one field, under the reserved name the vectors
    API addresses that vector type by, so the created index is byte-identical
    on the backend to a legacy one.

    Args:
        dimension: Dense vector width. Required for dense, rejected for sparse.
        metric: Similarity metric, as a :class:`~pinecone.models.enums.Metric`
            member or a plain string. ``None`` means ``"cosine"``. Dropped for
            sparse indexes — 9.x's ``metric="dotproduct"`` on a sparse index
            has nowhere to go in the 2026-07 schema.
        vector_type: ``"dense"`` (the default when ``None``) or ``"sparse"``,
            as a :class:`~pinecone.models.enums.VectorType` member or a string.

    Returns:
        ``{"fields": {"_values": {...}}}`` for dense, or
        ``{"fields": {"_sparse_values": {"type": "sparse_vector"}}}`` for sparse.

    Raises:
        PineconeValueError: The metric or vector_type is unknown, or
            ``dimension`` is missing for a dense index or supplied for a
            sparse one.
        PineconeTypeError: ``dimension`` is not an integer.

    The four messages raised here are the 9.x wording verbatim, because
    existing user code may match on them.
    """
    resolved_metric = "cosine" if metric is None else resolve_enum_value(metric)
    if resolved_metric not in _VALID_METRICS:
        raise PineconeValueError(
            f"metric must be one of {sorted(_VALID_METRICS)}, got {resolved_metric!r}"
        )

    if dimension is not None and not isinstance(dimension, int):
        raise PineconeTypeError(f"dimension must be an integer, got {type(dimension).__name__!r}")

    resolved_vector_type = resolve_enum_value(vector_type)
    if resolved_vector_type is not None and resolved_vector_type not in _VALID_VECTOR_TYPES:
        raise PineconeValueError(
            f"vector_type must be one of {sorted(_VALID_VECTOR_TYPES)}, "
            f"got {resolved_vector_type!r}"
        )

    if resolved_vector_type == "sparse":
        if dimension is not None:
            raise PineconeValueError("dimension must not be provided for sparse indexes")
        return {"fields": {SPARSE_FIELD: {"type": "sparse_vector"}}}

    if dimension is None:
        raise PineconeValueError("dimension is required for dense indexes")

    return {
        "fields": {
            DENSE_FIELD: {
                "type": "dense_vector",
                "dimension": dimension,
                "metric": resolved_metric,
            }
        }
    }


def legacy_pod_scaling(*, replicas: int | None, pod_type: PodType | str | None) -> dict[str, Any]:
    """Translate legacy ``configure()`` pod kwargs into a ``deployment=`` dict.

    Args:
        replicas: Replica count, or ``None`` when the caller did not pass one.
        pod_type: Pod type, as a :class:`~pinecone.models.enums.PodType` member
            or a plain string, or ``None`` when the caller did not pass one.

    Returns:
        Only the keys the caller actually supplied. ``configure()`` sends a
        sparse PATCH, so a ``None`` must not become an explicit key.
    """
    scaling: dict[str, Any] = {}
    if replicas is not None:
        scaling["replicas"] = replicas
    if pod_type is not None:
        scaling["pod_type"] = resolve_enum_value(pod_type)
    return scaling
