"""Guided hard-break interception for index kwargs with no faithful translation.

``spec=IntegratedSpec(...)`` has no translation: integrated indexes are
created through ``create_for_model()``, so it is intercepted here. ``pods=``,
``metadata_config=``, ``source_collection=``, and ``source_backup_id=`` also
have no faithful translation — ``pods=`` has no 1:1 mapping onto
``replicas x shards``, ``metadata_config=`` has nothing to declare (metadata
is indexed automatically at upsert), and ``source_collection=``/
``source_backup_id=`` are rejected by the backend with a 400. Each one raises
a :class:`~pinecone.errors.exceptions.PineconeTypeError` before any HTTP
request, explaining why no translation exists.

The rest of the create() surface (``spec=`` for non-integrated specs,
``dimension=``, ``metric=``, ``vector_type=``) does have a faithful
translation and is handled directly by ``Indexes.create()`` /
``AsyncIndexes.create()`` via ``pinecone._internal.legacy_index_translation``.
"""

from __future__ import annotations

from typing import Any

from pinecone.errors.exceptions import PineconeTypeError

MIGRATION_GUIDE = "docs/migration/v10-2026-07-index-model.md"

#: create() kwargs with no faithful translation.
LEGACY_CREATE_KWARGS = frozenset(
    {"pods", "metadata_config", "source_collection", "source_backup_id"}
)

#: Legacy Indexes.configure() kwargs removed by the 2026-07 API.
LEGACY_CONFIGURE_KWARGS = frozenset(
    {"replicas", "pod_type", "embed", "spec", "serverless_read_capacity"}
)


def _fmt(value: Any) -> str:
    return repr(value)


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


def reject_integrated_spec_create(spec: Any, name: Any = None) -> None:
    """Raise a guided error for ``create(spec=IntegratedSpec(...))``.

    No-op when *spec* is not an ``IntegratedSpec``. Called directly by
    ``Indexes.create()``/``AsyncIndexes.create()`` since ``spec=`` is a
    named parameter there, not a captured ``**legacy_kwargs`` entry.
    """
    if spec is not None and type(spec).__name__ == "IntegratedSpec":
        raise _integrated_spec_error(spec, name)


def reject_legacy_create_kwargs(legacy: dict[str, Any], name: Any = None) -> None:
    """Raise a guided error for a ``create()`` kwarg with no translation.

    No-op when *legacy* is empty. Unknown (non-legacy) kwargs raise a plain
    unexpected-keyword error so typos are still reported as such, rather
    than being absorbed into the guided message below.
    """
    if not legacy:
        return

    unknown = set(legacy) - LEGACY_CREATE_KWARGS
    if unknown:
        raise PineconeTypeError(
            f"Indexes.create() got unexpected keyword argument(s): {sorted(unknown)}. "
            "Accepted keyword arguments: schema, name, deployment, read_capacity, "
            "deletion_protection, tags, cmek_id, timeout, spec, dimension, metric, "
            "vector_type."
        )

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
    lines = [
        "Indexes.create() no longer accepts keyword argument(s) "
        f"{', '.join(removed)} — none has a faithful 2026-07 equivalent:",
    ]
    if "pods" in legacy:
        lines.append(
            "  pods=: capacity is replicas x shards on the pod deployment. Use "
            "deployment={'deployment_type': 'pod', 'environment': ..., 'pod_type': ..., "
            "'replicas': ..., 'shards': ...} instead."
        )
    if "metadata_config" in legacy:
        lines.append(
            "  metadata_config=: metadata fields are indexed automatically at upsert; "
            "there is nothing to declare at create time."
        )
    if sources:
        lines.append(
            f"  {', '.join(sources)}: rejected with 400 'Creating an index from "
            "collection or backup is not yet supported'; use "
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
