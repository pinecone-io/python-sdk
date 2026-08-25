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

``configure()`` is narrower: ``replicas=``, ``pod_type=``, and
``serverless_read_capacity=`` are faithful 1:1 translations of a PATCH
sub-object, so :meth:`~pinecone.client.indexes.Indexes.configure` and
:meth:`~pinecone.async_client.indexes.AsyncIndexes.configure` accept them
directly as deprecated keyword-only sugar and translate them via
``pinecone/_internal/legacy_index_translation.py`` instead of routing them
through here. Only ``embed=`` and ``spec=`` have no destination in the
2026-07 PATCH body and still hard-break through
:func:`reject_legacy_configure_kwargs`.
"""

from __future__ import annotations

from typing import Any

from pinecone.errors.exceptions import PineconeTypeError

MIGRATION_GUIDE = "docs/migration/v10-2026-07-index-model.md"

#: create() kwargs with no faithful translation.
LEGACY_CREATE_KWARGS = frozenset(
    {"pods", "metadata_config", "source_collection", "source_backup_id"}
)

#: Indexes.configure() kwargs with no 2026-07 PATCH-body equivalent.
LEGACY_CONFIGURE_KWARGS = frozenset({"embed", "spec"})

#: create() kwargs valid only on Indexes.create(), not on Pinecone.create_index().
NEW_ONLY_CREATE_KWARGS = frozenset({"deployment", "read_capacity", "cmek_id", "schema"})

#: configure() kwargs valid only on Indexes.configure(), not on Pinecone.configure_index().
NEW_ONLY_CONFIGURE_KWARGS = frozenset({"deployment", "schema"})


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


def reject_new_only_create_kwargs(legacy: dict[str, Any]) -> None:
    """Raise for a ``create()`` kwarg the flat ``create_index()`` shim never accepted.

    No-op when *legacy* carries none of :data:`NEW_ONLY_CREATE_KWARGS`. Called
    by :meth:`Pinecone.create_index`/:meth:`AsyncPinecone.create_index` before
    delegating, so ``deployment=``/``read_capacity=``/``cmek_id=``/``schema=``
    cannot slip through the shim's own ``**legacy_kwargs`` (kept open for
    ``pods=``/``metadata_config=``/``source_collection=``/``source_backup_id=``,
    which still need to reach :func:`reject_legacy_create_kwargs`).
    """
    blocked = sorted(NEW_ONLY_CREATE_KWARGS & set(legacy))
    if blocked:
        raise PineconeTypeError(
            f"create_index() got unexpected keyword argument(s): {blocked}. "
            "create_index() is the legacy backwards-compatibility shim and only "
            "accepts: name, spec, dimension, metric, vector_type, "
            "deletion_protection, tags, timeout. Use pc.indexes.create() for the "
            "2026-07 schema=/deployment=/read_capacity=/cmek_id= surface."
        )


def reject_new_only_configure_kwargs(legacy: dict[str, Any]) -> None:
    """Raise for a ``configure()`` kwarg the flat ``configure_index()`` shim never accepted.

    No-op when *legacy* carries none of :data:`NEW_ONLY_CONFIGURE_KWARGS`.
    Called by :meth:`Pinecone.configure_index`/:meth:`AsyncPinecone.configure_index`
    before delegating, so ``deployment=``/``schema=`` cannot slip through the
    shim's own ``**legacy_kwargs`` (kept open for typo detection via
    :func:`reject_legacy_configure_kwargs`, which the unrecognized-kwarg path
    still reaches through :meth:`Indexes.configure`).
    """
    blocked = sorted(NEW_ONLY_CONFIGURE_KWARGS & set(legacy))
    if blocked:
        raise PineconeTypeError(
            f"configure_index() got unexpected keyword argument(s): {blocked}. "
            "configure_index() is the legacy backwards-compatibility shim and only "
            "accepts: name, replicas, pod_type, deletion_protection, tags, embed, "
            "read_capacity, serverless_read_capacity. Use pc.indexes.configure() "
            "for the 2026-07 deployment=/schema= surface."
        )


def reject_legacy_configure_kwargs(legacy: dict[str, Any]) -> None:
    """Raise a guided error for a ``configure()`` kwarg with no PATCH-body destination.

    A ``None`` value is treated the same as the key being absent, so a caller
    (or wrapper, such as :meth:`Pinecone.configure_index`) may pass
    ``embed=None`` unconditionally to mean "not requested" without tripping
    this check.
    """
    if not legacy:
        return

    unknown = set(legacy) - LEGACY_CONFIGURE_KWARGS
    if unknown:
        raise PineconeTypeError(
            f"Indexes.configure() got unexpected keyword argument(s): {sorted(unknown)}. "
            "Accepted keyword arguments: deployment, schema, read_capacity, "
            "deletion_protection, tags, replicas, pod_type, serverless_read_capacity."
        )

    given = {k: v for k, v in legacy.items() if v is not None}
    if not given:
        return

    if "embed" in given:
        raise PineconeTypeError(
            "Indexes.configure() no longer accepts embed= — the 2025-10 "
            "convert-to-integrated flow was removed in the 2026-07 Pinecone API "
            "(the server rejects unknown PATCH fields). Embedding configuration is "
            "set at creation time via pc.indexes.create_for_model(...). "
            f"See the migration guide: {MIGRATION_GUIDE}"
        )

    raise PineconeTypeError(
        "Indexes.configure() no longer accepts spec= — removed in the 2026-07 "
        "Pinecone API (the server rejects unknown PATCH fields). Use "
        "deployment={'replicas': ..., 'pod_type': ...} for pod scaling and a "
        "top-level read_capacity={...} for read-capacity changes. "
        f"See the migration guide: {MIGRATION_GUIDE}"
    )
