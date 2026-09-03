"""Top-level export wiring for the 2026-07 graduation (#140, #317).

Asserts genuine reachability — each name is imported through the public
entry point and identity-checked against its defining module — for the
exports deferred to the #140 choke point: the #96 admin RBAC models, the
#101 assistant operation models, and the #106 SchemaBuilder, plus #192's
``TokenResponse``, wired by #317.

Reading ``_LAZY_IMPORTS`` is not enough: #96's wiring was deferred to
#107-#110 and silently dropped, and nothing caught it until these tests
imported the names for real. The parity tests below are deliberately
exhaustive with no carve-outs, so the next dropped hand-off fails here.
"""

from __future__ import annotations

import pytest

ADMIN_EXPORTS = {
    "InviteList": "pinecone.models.admin.invite",
    "InviteModel": "pinecone.models.admin.invite",
    "InviteStatus": "pinecone.models.admin.invite",
    "PaginationResponse": "pinecone.models.admin.pagination",
    "PrincipalType": "pinecone.models.admin.role_binding",
    "ResourceType": "pinecone.models.admin.role_binding",
    "RoleBindingInput": "pinecone.models.admin.role_binding",
    "RoleBindingList": "pinecone.models.admin.role_binding",
    "RoleBindingModel": "pinecone.models.admin.role_binding",
    "RoleName": "pinecone.models.admin.role_binding",
    "ServiceAccountList": "pinecone.models.admin.service_account",
    "ServiceAccountModel": "pinecone.models.admin.service_account",
    "ServiceAccountWithSecret": "pinecone.models.admin.service_account",
    "TokenResponse": "pinecone.models.admin.token",
    "UserList": "pinecone.models.admin.user",
    "UserModel": "pinecone.models.admin.user",
}

ASSISTANT_EXPORTS = {
    "ListOperationsResponse": "pinecone.models.assistant.list",
    "OperationModel": "pinecone.models.assistant.operation",
}

MODEL_EXPORTS = {**ADMIN_EXPORTS, **ASSISTANT_EXPORTS}

TOP_LEVEL_EXPORTS = {**MODEL_EXPORTS, "SchemaBuilder": "pinecone.schema_builder"}


@pytest.mark.parametrize(("name", "module_path"), sorted(TOP_LEVEL_EXPORTS.items()))
def test_top_level_export_is_defining_class(name: str, module_path: str) -> None:
    from importlib import import_module

    import pinecone

    assert name in pinecone.__all__
    assert pinecone._LAZY_IMPORTS[name] == (module_path, name)
    assert getattr(pinecone, name) is getattr(import_module(module_path), name)


@pytest.mark.parametrize(("name", "module_path"), sorted(MODEL_EXPORTS.items()))
def test_models_export_is_defining_class(name: str, module_path: str) -> None:
    from importlib import import_module

    import pinecone.models

    assert name in pinecone.models.__all__
    assert getattr(pinecone.models, name) is getattr(import_module(module_path), name)


def test_models_admin_export_parity() -> None:
    """Every public name in pinecone.models.admin is reachable from pinecone.models."""
    import pinecone.models
    import pinecone.models.admin

    missing = [n for n in pinecone.models.admin.__all__ if n not in pinecone.models.__all__]
    assert missing == [], (
        f"pinecone.models.admin exports missing from pinecone.models: {missing}. "
        "This test has no allowed-gap carve-out: wire the name up rather than "
        "exempting it (see #317)."
    )


def test_models_admin_top_level_parity() -> None:
    """Every public name in pinecone.models.admin is also reachable from pinecone."""
    from importlib import import_module

    import pinecone
    import pinecone.models.admin

    missing = [n for n in pinecone.models.admin.__all__ if n not in pinecone.__all__]
    assert missing == [], (
        f"pinecone.models.admin exports missing from the pinecone top level: {missing}."
    )

    for name in pinecone.models.admin.__all__:
        module_path, source_attr = pinecone._LAZY_IMPORTS[name]
        assert getattr(pinecone, name) is getattr(import_module(module_path), source_attr)


# ``ValidationError`` is a deliberately un-promoted deprecated alias for
# ``PineconeValueError``: touching it emits a DeprecationWarning, so it is
# reachable via ``pinecone.errors`` but intentionally absent from the top
# level. Every other name in ``pinecone.errors.__all__`` must be importable
# from ``pinecone`` — ``IndexTerminatedError`` was not, even though public
# ``Raises:`` docstrings tell callers to catch it (#317).
ERRORS_TOP_LEVEL_EXEMPT = frozenset({"ValidationError"})


def test_errors_top_level_parity() -> None:
    """Every public name in pinecone.errors is reachable from pinecone."""
    from importlib import import_module

    import pinecone
    import pinecone.errors

    expected = [n for n in pinecone.errors.__all__ if n not in ERRORS_TOP_LEVEL_EXEMPT]
    missing = [n for n in expected if n not in pinecone.__all__]
    assert missing == [], (
        f"pinecone.errors exports missing from the pinecone top level: {missing}. "
        "Add them to _LAZY_IMPORTS and __all__ rather than widening "
        "ERRORS_TOP_LEVEL_EXEMPT, which is only for deprecated aliases."
    )

    for name in expected:
        module_path, source_attr = pinecone._LAZY_IMPORTS[name]
        assert getattr(pinecone, name) is getattr(import_module(module_path), source_attr)


def test_errors_top_level_exemptions_are_still_real() -> None:
    """The exemption list must not outlive the names it excuses."""
    import pinecone
    import pinecone.errors

    stale = sorted(ERRORS_TOP_LEVEL_EXEMPT - set(pinecone.errors.__all__))
    assert stale == [], f"ERRORS_TOP_LEVEL_EXEMPT names no longer exported: {stale}"

    promoted = sorted(ERRORS_TOP_LEVEL_EXEMPT & set(pinecone.__all__))
    assert promoted == [], (
        f"ERRORS_TOP_LEVEL_EXEMPT names now exported at top level: {promoted}. "
        "Drop them from the exemption set."
    )


def test_top_level_all_and_lazy_imports_sorted() -> None:
    import pinecone

    assert pinecone.__all__ == sorted(pinecone.__all__)
    lazy_names = list(pinecone._LAZY_IMPORTS)
    assert lazy_names == sorted(lazy_names)


def test_schema_builder_constructs_and_builds() -> None:
    from pinecone import SchemaBuilder

    schema = (
        SchemaBuilder().add_dense_vector_field("embedding", dimension=8, metric="cosine").build()
    )
    assert schema["fields"]["embedding"]["dimension"] == 8


# ``describe_namespace`` hands callers nested ``IndexedFields`` and
# ``NamespaceSchema`` values, so a caller has to be able to name those types for
# an ``isinstance`` check or an annotation (#330).
NAMESPACE_EXPORTS = {
    "IndexedFields": "pinecone.models.namespaces.models",
    "ListNamespacesResponse": "pinecone.models.namespaces.models",
    "NamespaceDescription": "pinecone.models.namespaces.models",
    "NamespaceFieldConfig": "pinecone.models.namespaces.models",
    "NamespaceSchema": "pinecone.models.namespaces.models",
}


@pytest.mark.parametrize(("name", "module_path"), sorted(NAMESPACE_EXPORTS.items()))
def test_namespace_export_reachable_from_every_level(name: str, module_path: str) -> None:
    from importlib import import_module

    import pinecone
    import pinecone.models
    import pinecone.models.namespaces

    defining = getattr(import_module(module_path), name)

    assert name in pinecone.models.namespaces.__all__
    assert getattr(pinecone.models.namespaces, name) is defining

    assert name in pinecone.models.__all__
    assert getattr(pinecone.models, name) is defining

    assert name in pinecone.__all__
    assert pinecone._LAZY_IMPORTS[name] == (module_path, name)
    assert getattr(pinecone, name) is defining


def test_namespaces_all_matches_lazy_imports() -> None:
    """One derived list, because two hand-maintained lists drift."""
    import pinecone.models.namespaces as ns

    assert ns.__all__ == list(ns._LAZY_IMPORTS)


def test_models_namespaces_export_parity() -> None:
    """Every public name in pinecone.models.namespaces is reachable further up."""
    import pinecone
    import pinecone.models
    import pinecone.models.namespaces as ns

    assert [n for n in ns.__all__ if n not in pinecone.models.__all__] == []
    assert [n for n in ns.__all__ if n not in pinecone.__all__] == []
