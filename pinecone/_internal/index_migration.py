"""Guided-error interception for index kwargs with no faithful 2026-07 translation.

``dimension=``, ``metric=``, ``vector_type=``, and ``spec=`` (for
non-integrated specs) on ``create()``, and ``replicas=``, ``pod_type=``, and
``serverless_read_capacity=`` on ``configure()``, are deprecated
keyword-only sugar handled directly by ``Indexes.create()`` /
``AsyncIndexes.create()`` and ``Indexes.configure()`` /
``AsyncIndexes.configure()`` via
``pinecone._internal.legacy_index_translation``; they never reach this
module.

What's left here is the kwargs that have no faithful translation and still
raise a :class:`~pinecone.errors.exceptions.PineconeTypeError` before any
HTTP request:

- ``create()``: ``pods=`` (no 1:1 mapping onto ``replicas x shards``),
  ``metadata_config=`` (metadata is indexed automatically at upsert, so
  there's nothing to declare), ``source_collection=``/``source_backup_id=``
  (rejected by the backend with a 400), and ``spec=IntegratedSpec(...)``
  (integrated-embedding indexes are created through ``create_for_model()``
  instead). ``metadata_config`` and ``source_collection`` are also
  rejected when they arrive as ``PodSpec`` attributes rather than as
  kwargs, so ``spec=`` gets the same answer as the kwarg spelling.
- ``configure()``: ``embed=`` and ``spec=`` (neither has a destination in
  the 2026-07 PATCH body; the 2025-10 convert-to-integrated flow is gone
  and the server rejects unknown PATCH fields).

:func:`reject_new_only_create_kwargs` and
:func:`reject_new_only_configure_kwargs` cover a different case: they guard
the flat ``Pinecone.create_index()``/``AsyncPinecone.create_index()`` and
``Pinecone.configure_index()``/``AsyncPinecone.configure_index()`` shims
against 2026-07-only arguments (``deployment=``, ``read_capacity=``,
``cmek_id=``, ``schema=``) that the shims never accepted, so those fail
before falling into the ``**legacy_kwargs`` bucket meant for the kwargs
above.

Each function's message interpolates the caller's own values into a
copy-pasteable equivalent 2026-07 call and explains why no translation
exists.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pinecone.errors.exceptions import PineconeTypeError

MIGRATION_GUIDE = "https://sdk.pinecone.io/python/migration/v10-migration.html"

#: create() kwargs with no faithful translation.
LEGACY_CREATE_KWARGS = frozenset(
    {"pods", "metadata_config", "source_collection", "source_backup_id"}
)

#: PodSpec attributes spelling the same values as the like-named create() kwargs.
LEGACY_SPEC_FIELDS = ("metadata_config", "source_collection")

#: Indexes.configure() kwargs with no 2026-07 PATCH-body equivalent.
LEGACY_CONFIGURE_KWARGS = frozenset({"embed", "spec"})

#: create() kwargs valid only on Indexes.create(), not on Pinecone.create_index().
NEW_ONLY_CREATE_KWARGS = frozenset({"deployment", "read_capacity", "cmek_id", "schema"})

#: configure() kwargs valid only on Indexes.configure(), not on Pinecone.configure_index().
NEW_ONLY_CONFIGURE_KWARGS = frozenset({"deployment", "schema"})


def _fmt(value: Any) -> str:
    return repr(value)


def _read(obj: Any, key: str, placeholder: Any) -> Any:
    """Read *key* off a struct or an equivalent mapping, or fall back.

    ``IntegratedSpec`` is a ``msgspec.Struct`` built by direct construction,
    which does not coerce its ``embed`` annotation, so ``embed`` is whatever
    the caller passed — commonly the 9.x plain dict, the same shape the
    guided message below prints. A message builder on a rejection path must
    not raise something other than the rejection.
    """
    value = obj.get(key) if isinstance(obj, Mapping) else getattr(obj, key, None)
    return placeholder if value is None else value


def _integrated_spec_error(spec: Any, name: Any) -> PineconeTypeError:
    embed = _read(spec, "embed", "")
    field_map = _read(embed, "field_map", {})
    field = (
        next(iter(field_map.values()), "<field-name>")
        if isinstance(field_map, Mapping)
        else "<field-name>"
    )
    lines = [
        "Indexes.create() no longer accepts spec=IntegratedSpec(...) — the 2026-07 API",
        "creates integrated-embedding indexes through create_for_model instead:",
        "",
        "    pc.indexes.create_for_model(",
        f"        name={_fmt(name) if name else '<name>'},",
        f"        cloud={_fmt(_read(spec, 'cloud', '<cloud>'))},",
        f"        region={_fmt(_read(spec, 'region', '<region>'))},",
        f"        embed={{'model': {_fmt(_read(embed, 'model', '<model-name>'))}, "
        f"'field_map': {{'text': {_fmt(field)}}}}},",
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


def reject_legacy_spec_fields(spec: Any) -> None:
    """Raise a guided error for a spec attribute with no ``create()`` destination.

    No-op unless *spec* sets one of :data:`LEGACY_SPEC_FIELDS`. ``PodSpec``
    still carries ``metadata_config`` and ``source_collection`` so that a
    9.x spec object round-trips, but the 2026-07 create request has nowhere
    to put either. Called by
    ``pinecone._internal.legacy_index_translation._coerce_spec`` so a value
    nested in ``spec=`` gets the same answer as the like-named top-level
    kwarg, which :func:`reject_legacy_create_kwargs` already rejects.
    """
    given = [field for field in LEGACY_SPEC_FIELDS if getattr(spec, field, None) is not None]
    if not given:
        return

    lines = [
        "Indexes.create() cannot translate spec=PodSpec(...) carrying "
        f"{', '.join(f'{field}=' for field in given)} — none has a destination in the "
        "2026-07 create request, and dropping a field you set would send a create call "
        "that differs from the one you wrote:"
    ]
    if "metadata_config" in given:
        lines.append(
            "  metadata_config=: metadata fields are indexed automatically at upsert; "
            "there is nothing to declare at create time."
        )
    if "source_collection" in given:
        lines.append(
            "  source_collection=: the 2026-07 API rejects index creation from a collection "
            "or backup with 400 'Creating an index from collection or backup is not yet "
            "supported'; use pc.create_index_from_backup(backup_id=..., name=...) to restore "
            "a backup."
        )
    lines.append(
        "2026-07 refuses pod index creation whatever the spec says, with 400 "
        "\"deployment_type 'pod' is not supported on this API version\", so no spelling of "
        "this call creates an index."
    )
    lines.append(f"See the migration guide: {MIGRATION_GUIDE}")
    raise PineconeTypeError("\n".join(lines))


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
            f"(See {MIGRATION_GUIDE}.)"
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
