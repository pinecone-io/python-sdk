"""Top-level export wiring for the 2026-07 graduation (#140).

Asserts genuine reachability — each name is imported through the public
entry point and identity-checked against its defining module — for the
exports deferred to the #140 choke point: the #96 admin RBAC models, the
#101 assistant operation models, and the #106 SchemaBuilder.
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
    assert missing == ["TokenResponse"], (
        f"pinecone.models.admin exports missing from pinecone.models: {missing}. "
        "TokenResponse (#192) is the one known, deliberate gap; anything else is a "
        "wiring regression."
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
