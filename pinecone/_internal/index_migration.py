"""Guided hard-break interception for 2025-10 index kwargs (2026-07 migration).

The 2026-07 control plane replaced the spec-based ``create``/``configure``
surface (``spec=``, ``dimension=``, ``metric=``, ``vector_type=``,
``replicas=``, ``pod_type=``, ``embed=``) with a schema/deployment surface.
Per the release decision (epoch #87), legacy keyword arguments are not
silently aliased — translating ``dimension=1536`` would require inventing a
schema field name the caller's data-plane code has never heard of, and the
old metadata ``schema=`` kwarg collides with the new search ``schema=``
kwarg with opposite semantics.

Instead, every legacy kwarg is intercepted before any HTTP request and
raises a :class:`~pinecone.errors.exceptions.PineconeTypeError` whose
message interpolates the caller's own values into a copy-pasteable
equivalent 2026-07 call wherever a faithful translation exists, and
explains why no translation exists where it does not.
"""

from __future__ import annotations

from typing import Any

from pinecone.errors.exceptions import PineconeTypeError

MIGRATION_GUIDE = "docs/migration/v10-2026-07-index-model.md"

#: Legacy Indexes.create() kwargs removed by the 2026-07 API.
LEGACY_CREATE_KWARGS = frozenset(
    {
        "spec",
        "dimension",
        "metric",
        "vector_type",
        "pods",
        "metadata_config",
        "source_collection",
        "source_backup_id",
    }
)

#: Legacy Indexes.configure() kwargs removed by the 2026-07 API.
LEGACY_CONFIGURE_KWARGS = frozenset(
    {"replicas", "pod_type", "embed", "spec", "serverless_read_capacity"}
)


def _fmt(value: Any) -> str:
    return repr(value)


def _spec_to_deployment_snippet(spec: Any) -> str | None:
    """Render the caller's legacy spec value as a 2026-07 ``deployment=`` dict."""
    type_name = type(spec).__name__
    if type_name == "ServerlessSpec":
        return (
            "deployment={'deployment_type': 'managed', "
            f"'cloud': {_fmt(spec.cloud)}, 'region': {_fmt(spec.region)}}}"
        )
    if type_name == "PodSpec":
        return (
            "deployment={'deployment_type': 'pod', "
            f"'environment': {_fmt(spec.environment)}, "
            f"'pod_type': {_fmt(spec.pod_type)}, "
            f"'replicas': {spec.replicas}, 'shards': {spec.shards}}}"
        )
    if type_name == "ByocSpec":
        return f"deployment={{'deployment_type': 'byoc', 'environment': {_fmt(spec.environment)}}}"
    if isinstance(spec, dict):
        if "serverless" in spec:
            inner = spec["serverless"]
            return (
                "deployment={'deployment_type': 'managed', "
                f"'cloud': {_fmt(inner.get('cloud'))}, 'region': {_fmt(inner.get('region'))}}}"
            )
        if "pod" in spec:
            inner = spec["pod"]
            return (
                "deployment={'deployment_type': 'pod', "
                f"'environment': {_fmt(inner.get('environment'))}, "
                f"'pod_type': {_fmt(inner.get('pod_type', 'p1.x1'))}, "
                f"'replicas': {inner.get('replicas', 1)}, 'shards': {inner.get('shards', 1)}}}"
            )
        if "byoc" in spec:
            inner = spec["byoc"]
            return (
                "deployment={'deployment_type': 'byoc', "
                f"'environment': {_fmt(inner.get('environment'))}}}"
            )
    return None


def _schema_field_snippet(legacy: dict[str, Any]) -> str:
    """Render dimension/metric/vector_type as a 2026-07 schema field example."""
    vector_type = legacy.get("vector_type", "dense")
    vector_type = vector_type.value if hasattr(vector_type, "value") else vector_type
    if vector_type == "sparse":
        return "schema={'fields': {'<field-name>': {'type': 'sparse_vector'}}}"
    dimension = legacy.get("dimension", "<dimension>")
    metric = legacy.get("metric", "cosine")
    metric = metric.value if hasattr(metric, "value") else metric
    return (
        "schema={'fields': {'<field-name>': "
        f"{{'type': 'dense_vector', 'dimension': {dimension}, 'metric': {_fmt(metric)}}}}}}}"
    )


def _integrated_spec_error(spec: Any, name: Any) -> PineconeTypeError:
    embed = spec.embed
    field_map = dict(embed.field_map) if embed.field_map else {}
    field = next(iter(field_map.values()), "<field-name>")
    lines = [
        "Indexes.create() no longer accepts spec=IntegratedSpec(...) — the 2026-07 API",
        "creates integrated-embedding indexes through create_for_model instead:",
        "",
        "    pc.indexes.create_for_model(",
        f"        name={_fmt(name) if name else '<name>'},",
        f"        cloud={_fmt(spec.cloud)},",
        f"        region={_fmt(spec.region)},",
        f"        embed={{'model': {_fmt(embed.model)}, 'field_map': {{'text': {_fmt(field)}}}}},",
        "    )",
        "",
        f"See the migration guide: {MIGRATION_GUIDE}",
    ]
    return PineconeTypeError("\n".join(lines))


def reject_legacy_create_kwargs(legacy: dict[str, Any], name: Any = None) -> None:
    """Raise a guided error for any legacy 2025-10 ``create()`` kwarg.

    No-op when *legacy* is empty. Unknown (non-legacy) kwargs raise a plain
    unexpected-keyword error so typos are still reported as such.
    """
    if not legacy:
        return

    unknown = set(legacy) - LEGACY_CREATE_KWARGS
    if unknown:
        raise PineconeTypeError(
            f"Indexes.create() got unexpected keyword argument(s): {sorted(unknown)}. "
            "Accepted keyword arguments: schema, name, deployment, read_capacity, "
            "deletion_protection, tags, cmek_id, timeout."
        )

    spec = legacy.get("spec")
    if spec is not None and type(spec).__name__ == "IntegratedSpec":
        raise _integrated_spec_error(spec, name)

    sources = [k for k in ("source_collection", "source_backup_id") if k in legacy]
    if sources and set(legacy) <= {"source_collection", "source_backup_id"}:
        raise PineconeTypeError(
            f"Indexes.create() does not accept {', '.join(sources)}: the 2026-07 API "
            "rejects index creation from a collection or backup with 400 'Creating an "
            "index from collection or backup is not yet supported'. To restore a backup, "
            "use pc.create_index_from_backup(backup_id=..., name=...). "
            f"(See question pinecone-io/python-sdk-internal#144 and {MIGRATION_GUIDE}.)"
        )

    removed = sorted(set(legacy) & LEGACY_CREATE_KWARGS)
    call_lines = ["    pc.indexes.create("]
    if name:
        call_lines.append(f"        name={_fmt(name)},")
    call_lines.append(f"        {_schema_field_snippet(legacy)},")
    deployment_snippet = _spec_to_deployment_snippet(spec) if spec is not None else None
    if deployment_snippet:
        call_lines.append(f"        {deployment_snippet},")
    call_lines.append("    )")

    lines = [
        "Indexes.create() no longer accepts legacy keyword argument(s) "
        f"{', '.join(removed)} — removed in the 2026-07 Pinecone API.",
        "Declare the index as a schema of named fields plus a deployment:",
        "",
        *call_lines,
        "",
        "Replace '<field-name>' with the field name your upsert/query code will "
        "address — the SDK cannot invent it for you because the 2026-07 data plane "
        "addresses vectors by field name.",
    ]
    if "pods" in legacy:
        lines.append(
            "Note: pods= has no 2026-07 equivalent; capacity is replicas x shards "
            "on the pod deployment."
        )
    if "metadata_config" in legacy or "schema" in legacy:
        lines.append(
            "Note: metadata fields are no longer declared at create time; they are "
            "indexed automatically at upsert."
        )
    if sources:
        lines.append(
            f"Note: {', '.join(sources)} is rejected by the 2026-07 API; use "
            "pc.create_index_from_backup(...) to restore a backup."
        )
    lines.append(f"See the migration guide: {MIGRATION_GUIDE}")
    raise PineconeTypeError("\n".join(lines))


def reject_legacy_configure_kwargs(legacy: dict[str, Any], name: str) -> None:
    """Raise a guided error for any legacy 2025-10 ``configure()`` kwarg."""
    if not legacy:
        return

    unknown = set(legacy) - LEGACY_CONFIGURE_KWARGS
    if unknown:
        raise PineconeTypeError(
            f"Indexes.configure() got unexpected keyword argument(s): {sorted(unknown)}. "
            "Accepted keyword arguments: deployment, schema, read_capacity, "
            "deletion_protection, tags."
        )

    if "embed" in legacy:
        raise PineconeTypeError(
            "Indexes.configure() no longer accepts embed= — the 2025-10 "
            "convert-to-integrated flow was removed in the 2026-07 Pinecone API "
            "(the server rejects unknown PATCH fields). Embedding configuration is "
            "set at creation time via pc.indexes.create_for_model(...). "
            f"See the migration guide: {MIGRATION_GUIDE}"
        )

    if "spec" in legacy:
        raise PineconeTypeError(
            "Indexes.configure() no longer accepts spec= — removed in the 2026-07 "
            "Pinecone API (the server rejects unknown PATCH fields). Use "
            "deployment={'replicas': ..., 'pod_type': ...} for pod scaling and a "
            "top-level read_capacity={...} for read-capacity changes. "
            f"See the migration guide: {MIGRATION_GUIDE}"
        )

    if "serverless_read_capacity" in legacy:
        value = legacy["serverless_read_capacity"]
        raise PineconeTypeError(
            "Indexes.configure() no longer accepts serverless_read_capacity= — the "
            "2026-07 API uses one top-level read_capacity for managed and BYOC "
            f"indexes. Call pc.indexes.configure({name!r}, read_capacity={value!r}) "
            f"instead. See the migration guide: {MIGRATION_GUIDE}"
        )

    pod_fields = {k: legacy[k] for k in ("replicas", "pod_type") if k in legacy}
    inner = ", ".join(f"{_fmt(k)}: {_fmt(v)}" for k, v in pod_fields.items())
    raise PineconeTypeError(
        f"Indexes.configure() no longer accepts {', '.join(sorted(pod_fields))}= — "
        "the 2026-07 Pinecone API nests pod scaling parameters under deployment. "
        f"Call pc.indexes.configure({name!r}, deployment={{{inner}}}) instead. "
        f"See the migration guide: {MIGRATION_GUIDE}"
    )
