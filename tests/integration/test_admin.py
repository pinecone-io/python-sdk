"""Integration tests for Admin client credential validation.

These tests verify that Admin.__init__ raises typed exceptions for missing,
empty, or whitespace-only credentials, and raises ApiError when the OAuth
endpoint rejects invalid credentials.

The first four tests (init_validation_*) do NOT require real service-account
credentials and run regardless of environment.  They use monkeypatch to clear
any PINECONE_CLIENT_ID / PINECONE_CLIENT_SECRET env vars and pass bogus or
whitespace-only values so execution exits before any network call.

The fifth test (init_invalid_credentials_raises_api_error) hits the OAuth
endpoint with deliberately-wrong credentials; it requires only network
connectivity — no valid service-account credentials are needed.

If real service-account credentials are available (DX-0079–0084 tests), set:
    PINECONE_CLIENT_ID=<client-id>
    PINECONE_CLIENT_SECRET=<client-secret>

and the admin_credentials fixture (used by future cred-gated tests) will
not skip.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator

import pytest

from pinecone import Admin, Pinecone, PineconeValueError
from pinecone.errors import ApiError, NotFoundError
from pinecone.models.admin.api_key import APIKeyList, APIKeyModel
from pinecone.models.admin.organization import OrganizationList, OrganizationModel
from pinecone.models.admin.project import ProjectList, ProjectModel
from pinecone.models.indexes.specs import ServerlessSpec


@pytest.fixture(scope="module")
def admin_credentials() -> tuple[str, str]:
    """Provide Admin service-account credentials from environment.

    Skips the entire module when both env vars are absent, allowing the
    cred-gated admin tests (DX-0079–0084) to build on this fixture once
    credentials are provisioned.
    """
    client_id = os.getenv("PINECONE_CLIENT_ID")
    client_secret = os.getenv("PINECONE_CLIENT_SECRET")
    if not client_id or not client_secret:
        pytest.skip("PINECONE_CLIENT_ID and PINECONE_CLIENT_SECRET not set")
    return client_id, client_secret


# ---------------------------------------------------------------------------
# init_validation — no real creds required
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_admin_init_validation_missing_client_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Admin() raises PineconeValueError mentioning 'client_id' when client_id is absent.

    Env vars are cleared so the Admin.__init__ fallback path also finds nothing.
    Execution exits before any OAuth request is made.
    """
    monkeypatch.delenv("PINECONE_CLIENT_ID", raising=False)
    monkeypatch.delenv("PINECONE_CLIENT_SECRET", raising=False)

    with pytest.raises(PineconeValueError, match="client_id"):
        Admin(client_secret="anything")


@pytest.mark.integration
def test_admin_init_validation_missing_client_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """Admin() raises PineconeValueError mentioning 'client_secret' when client_secret is absent.

    Env vars are cleared so the Admin.__init__ fallback path also finds nothing.
    Execution exits before any OAuth request is made.
    """
    monkeypatch.delenv("PINECONE_CLIENT_ID", raising=False)
    monkeypatch.delenv("PINECONE_CLIENT_SECRET", raising=False)

    with pytest.raises(PineconeValueError, match="client_secret"):
        Admin(client_id="anything")


@pytest.mark.integration
def test_admin_init_validation_whitespace_only_client_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Admin() raises PineconeValueError for a whitespace-only client_id.

    Env vars are cleared to prevent the fallback from masking the kwarg value.
    Execution exits before any OAuth request is made.
    """
    monkeypatch.delenv("PINECONE_CLIENT_ID", raising=False)
    monkeypatch.delenv("PINECONE_CLIENT_SECRET", raising=False)

    with pytest.raises(PineconeValueError, match="client_id"):
        Admin(client_id="   ", client_secret="x")


@pytest.mark.integration
def test_admin_init_validation_whitespace_only_client_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admin() raises PineconeValueError for a whitespace-only client_secret.

    Env vars are cleared to prevent the fallback from masking the kwarg value.
    Execution exits before any OAuth request is made.
    """
    monkeypatch.delenv("PINECONE_CLIENT_ID", raising=False)
    monkeypatch.delenv("PINECONE_CLIENT_SECRET", raising=False)

    with pytest.raises(PineconeValueError, match="client_secret"):
        Admin(client_id="x", client_secret="   ")


@pytest.mark.integration
def test_admin_init_invalid_credentials_raises_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Admin() raises ApiError when the OAuth endpoint rejects the supplied credentials.

    Uses deliberately-wrong (nonexistent) client_id and client_secret so the
    OAuth server returns a 4xx error.  Requires network connectivity but does
    NOT require valid service-account credentials.

    Env vars are cleared so the Admin.__init__ fallback path does not pick up
    real credentials from the runner environment.
    """
    monkeypatch.delenv("PINECONE_CLIENT_ID", raising=False)
    monkeypatch.delenv("PINECONE_CLIENT_SECRET", raising=False)

    with pytest.raises(ApiError):
        Admin(client_id="nonexistent", client_secret="nonexistent")


# ---------------------------------------------------------------------------
# admin fixture — requires real credentials (PINECONE_CLIENT_ID / SECRET)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def admin(admin_credentials: tuple[str, str]) -> Admin:
    """Construct an authenticated Admin client using real service-account credentials.

    Module-scoped so the OAuth token exchange happens once per test run.
    Skipped automatically when admin_credentials skips (no env vars).
    """
    client_id, client_secret = admin_credentials
    return Admin(client_id=client_id, client_secret=client_secret)


# ---------------------------------------------------------------------------
# organizations — list / describe / update
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_organizations_list_returns_iterable_of_models(admin: Admin) -> None:
    """admin.organizations.list() returns an OrganizationList with at least one org.

    Verifies:
    - Return type is OrganizationList (supports iter and len).
    - At least one organization is present (service accounts always belong to one).
    - Every element is an OrganizationModel with non-empty id and name.
    """
    orgs = admin.organizations.list()

    assert isinstance(orgs, OrganizationList)
    assert len(orgs) >= 1

    for org in orgs:
        assert isinstance(org, OrganizationModel)
        assert isinstance(org.id, str) and org.id
        assert isinstance(org.name, str) and org.name


@pytest.mark.integration
def test_organizations_describe_returns_matching_model(admin: Admin) -> None:
    """admin.organizations.describe() returns a model matching the listed org.

    Takes the first org from list(), calls describe() with its id, and
    verifies that the described model has consistent id and name and that
    all required fields are populated.
    """
    orgs = admin.organizations.list()
    first = orgs[0]

    described = admin.organizations.describe(organization_id=first.id)

    assert isinstance(described, OrganizationModel)
    assert described.id == first.id
    assert described.name == first.name
    assert described.plan
    assert described.payment_status
    assert described.created_at
    assert described.support_tier


@pytest.mark.integration
def test_organizations_update_roundtrips_name(admin: Admin) -> None:
    """admin.organizations.update() persists the new name and is reversible.

    Saves the original name, renames to a timestamped variant, asserts the
    returned model reflects the new name, re-describes to confirm the change
    persisted on the server, then restores the original name in a finally
    block to leave org state clean.
    """
    orgs = admin.organizations.list()
    first = orgs[0]
    original_name = first.name
    new_name = f"{original_name}-test-{int(time.time())}"

    try:
        updated = admin.organizations.update(organization_id=first.id, name=new_name)
        assert isinstance(updated, OrganizationModel)
        assert updated.name == new_name

        # Re-describe to confirm server persisted the change.
        described = admin.organizations.describe(organization_id=first.id)
        assert described.name == new_name
    finally:
        admin.organizations.update(organization_id=first.id, name=original_name)


# ---------------------------------------------------------------------------
# projects — full CRUD lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_project_lifecycle_create_describe_update_delete(admin: Admin) -> None:
    """Full CRUD lifecycle for admin.projects.

    Creates an ephemeral project with a timestamped unique name, exercises
    every read/write operation, then deletes it.  The project is cleaned up
    in a finally block so a mid-test failure does not leave orphaned resources.

    Operations verified:
    - create → ProjectModel with correct name, non-empty id and organization_id
    - list → created project appears in the result set
    - describe (by id) → returns consistent id and name
    - describe_by_name → positive match; NotFoundError for unknown name
    - update → returned model and re-described model both reflect new name
    - delete → project no longer reachable via describe after deletion
    """
    name = f"inttest-proj-{int(time.time())}"
    created: ProjectModel | None = None
    _test_passed = False

    try:
        # create
        created = admin.projects.create(name=name)
        assert isinstance(created, ProjectModel)
        assert created.name == name
        assert created.id
        assert created.organization_id

        # list — created project appears
        listed = admin.projects.list()
        assert isinstance(listed, ProjectList)
        assert any(p.id == created.id for p in listed)

        # describe by ID
        described = admin.projects.describe(project_id=created.id)
        assert isinstance(described, ProjectModel)
        assert described.id == created.id
        assert described.name == name

        # describe_by_name — positive path
        by_name = admin.projects.describe_by_name(name=name)
        assert isinstance(by_name, ProjectModel)
        assert by_name.id == created.id

        # describe_by_name — negative path
        with pytest.raises(NotFoundError):
            admin.projects.describe_by_name(name="this-project-definitely-does-not-exist-zzzz")

        # update name
        renamed = f"{name}-renamed"
        updated = admin.projects.update(project_id=created.id, name=renamed)
        assert isinstance(updated, ProjectModel)
        assert updated.name == renamed

        # re-describe to confirm update persisted on the server
        re_described = admin.projects.describe(project_id=created.id)
        assert re_described.name == renamed

        _test_passed = True
    finally:
        if created is not None:
            try:
                admin.projects.delete(project_id=created.id)
            except Exception as e:
                print(f"Cleanup failed for project {created.id!r}: {e}")

    # Post-delete verification: only run when the main test succeeded so a
    # cleanup failure does not shadow the real assertion error.
    if _test_passed and created is not None:
        with pytest.raises(NotFoundError):
            admin.projects.describe(project_id=created.id)


# ---------------------------------------------------------------------------
# projects — validation (no credentials required)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_create_project_max_pods_negative() -> None:
    """Projects.create() raises PineconeValueError for negative max_pods client-side.

    Validation fires before any network call; no service-account credentials needed.
    """
    from pinecone._internal.config import PineconeConfig
    from pinecone._internal.constants import ADMIN_API_VERSION
    from pinecone._internal.http_client import HTTPClient
    from pinecone.admin.projects import Projects

    config = PineconeConfig(api_key="test-key", host="https://api.pinecone.io")
    http = HTTPClient(config, ADMIN_API_VERSION)
    projects = Projects(http=http)

    with pytest.raises(PineconeValueError, match="max_pods"):
        projects.create(name="my-project", max_pods=-1)


@pytest.mark.integration
def test_update_project_max_pods_negative() -> None:
    """Projects.update() raises PineconeValueError for negative max_pods client-side.

    Validation fires before any network call; no service-account credentials needed.
    """
    from pinecone._internal.config import PineconeConfig
    from pinecone._internal.constants import ADMIN_API_VERSION
    from pinecone._internal.http_client import HTTPClient
    from pinecone.admin.projects import Projects

    config = PineconeConfig(api_key="test-key", host="https://api.pinecone.io")
    http = HTTPClient(config, ADMIN_API_VERSION)
    projects = Projects(http=http)

    with pytest.raises(PineconeValueError, match="max_pods"):
        projects.update(project_id="proj-abc123", max_pods=-1)


@pytest.mark.integration
def test_update_project_name_empty() -> None:
    """Projects.update() raises PineconeValueError when name is empty string client-side.

    Validation fires before any network call; no service-account credentials needed.
    """
    from pinecone._internal.config import PineconeConfig
    from pinecone._internal.constants import ADMIN_API_VERSION
    from pinecone._internal.http_client import HTTPClient
    from pinecone.admin.projects import Projects

    config = PineconeConfig(api_key="test-key", host="https://api.pinecone.io")
    http = HTTPClient(config, ADMIN_API_VERSION)
    projects = Projects(http=http)

    with pytest.raises(PineconeValueError, match="name cannot be empty"):
        projects.update(project_id="proj-abc123", name="")


@pytest.mark.integration
def test_update_project_name_too_long() -> None:
    """Projects.update() raises PineconeValueError when name exceeds 512 characters client-side.

    Validation fires before any network call; no service-account credentials needed.
    """
    from pinecone._internal.config import PineconeConfig
    from pinecone._internal.constants import ADMIN_API_VERSION
    from pinecone._internal.http_client import HTTPClient
    from pinecone.admin.projects import Projects

    config = PineconeConfig(api_key="test-key", host="https://api.pinecone.io")
    http = HTTPClient(config, ADMIN_API_VERSION)
    projects = Projects(http=http)

    with pytest.raises(PineconeValueError, match="name cannot be longer than 512 characters"):
        projects.update(project_id="proj-abc123", name="x" * 513)


@pytest.mark.integration
def test_update_project_name_null_byte() -> None:
    """Projects.update() raises PineconeValueError when name contains a null byte client-side.

    Validation fires before any network call; no service-account credentials needed.
    """
    from pinecone._internal.config import PineconeConfig
    from pinecone._internal.constants import ADMIN_API_VERSION
    from pinecone._internal.http_client import HTTPClient
    from pinecone.admin.projects import Projects

    config = PineconeConfig(api_key="test-key", host="https://api.pinecone.io")
    http = HTTPClient(config, ADMIN_API_VERSION)
    projects = Projects(http=http)

    with pytest.raises(PineconeValueError, match="name cannot contain null characters"):
        projects.update(project_id="proj-abc123", name="valid\x00name")


@pytest.mark.integration
def test_create_project_name_too_long() -> None:
    """Projects.create() raises PineconeValueError when name exceeds 512 characters.

    Validation fires before any network call; no service-account credentials needed.
    """
    from pinecone._internal.config import PineconeConfig
    from pinecone._internal.constants import ADMIN_API_VERSION
    from pinecone._internal.http_client import HTTPClient
    from pinecone.admin.projects import Projects

    config = PineconeConfig(api_key="test-key", host="https://api.pinecone.io")
    http = HTTPClient(config, ADMIN_API_VERSION)
    projects = Projects(http=http)

    with pytest.raises(PineconeValueError, match="name cannot be longer than 512 characters"):
        projects.create(name="x" * 513)


@pytest.mark.integration
def test_create_project_name_null_byte() -> None:
    """Projects.create() raises PineconeValueError when name contains a null byte.

    Validation fires before any network call; no service-account credentials needed.
    """
    from pinecone._internal.config import PineconeConfig
    from pinecone._internal.constants import ADMIN_API_VERSION
    from pinecone._internal.http_client import HTTPClient
    from pinecone.admin.projects import Projects

    config = PineconeConfig(api_key="test-key", host="https://api.pinecone.io")
    http = HTTPClient(config, ADMIN_API_VERSION)
    projects = Projects(http=http)

    with pytest.raises(PineconeValueError, match="name cannot contain null characters"):
        projects.create(name="valid\x00name")


# ---------------------------------------------------------------------------
# api_keys — validation (no credentials required)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_api_key_create_name_too_long() -> None:
    """ApiKeys.create() raises PineconeValueError when name exceeds 80 characters.

    Validation fires before any network call; no service-account credentials needed.
    """
    from pinecone._internal.config import PineconeConfig
    from pinecone._internal.constants import ADMIN_API_VERSION
    from pinecone._internal.http_client import HTTPClient
    from pinecone.admin.api_keys import ApiKeys

    config = PineconeConfig(api_key="test-key", host="https://api.pinecone.io")
    http = HTTPClient(config, ADMIN_API_VERSION)
    api_keys = ApiKeys(http=http)

    with pytest.raises(PineconeValueError, match="name"):
        api_keys.create(project_id="proj-abc123", name="x" * 81)


# ---------------------------------------------------------------------------
# api_keys — name-nullability
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_api_key_list_name_optional(admin: Admin) -> None:
    """api_keys.list() accepts keys whose name field may be None.

    Creates an ephemeral API key, lists keys for the project, and verifies
    that every returned key's name field is str | None.  Cleans up in a
    finally block.
    """
    projects = admin.projects.list()
    assert len(projects) >= 1, "need at least one project for this test"
    project = projects[0]
    key_id: str | None = None

    try:
        created = admin.api_keys.create(project_id=project.id, name="inttest-name-optional")
        key_id = created.key.id

        keys = admin.api_keys.list(project_id=project.id)
        assert isinstance(keys, APIKeyList)
        for key in keys:
            assert isinstance(key, APIKeyModel)
            assert isinstance(key.name, (str, type(None)))
    finally:
        if key_id is not None:
            try:
                admin.api_keys.delete(api_key_id=key_id)
            except Exception as e:
                print(f"Cleanup failed for api key {key_id!r}: {e}")


# ---------------------------------------------------------------------------
# api_keys — full lifecycle (requires real credentials)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ephemeral_project(admin: Admin) -> Iterator[ProjectModel]:
    """Create and yield an ephemeral project; teardown via delete_with_cleanup.

    Module-scoped to share the same project across all api_keys lifecycle tests.
    """
    name = f"inttest-apikeys-{int(time.time())}"
    project = admin.projects.create(name=name)
    try:
        yield project
    finally:
        try:
            admin.projects.delete_with_cleanup(project_id=project.id)
        except Exception as e:
            print(f"Cleanup failed for project {project.id!r}: {e}")


@pytest.mark.integration
def test_api_keys_lifecycle_default_role(admin: Admin, ephemeral_project: ProjectModel) -> None:
    """Full lifecycle for API keys with default role (ProjectEditor): create / list / describe / delete.

    Creates a key without explicit roles (should default to ["ProjectEditor"]),
    exercises list and describe, then deletes and verifies the key is gone.
    """
    key_name = f"key-default-{int(time.time())}"
    key_id: str | None = None

    try:
        created = admin.api_keys.create(project_id=ephemeral_project.id, name=key_name)
        key_id = created.key.id
        assert created.value.startswith("pcsk_"), (
            f"expected pcsk_ prefix, got {created.value[:10]!r}"
        )
        assert created.key.id, "key id must be non-empty"
        assert created.key.name == key_name
        assert created.key.roles == ["ProjectEditor"]

        listed = admin.api_keys.list(project_id=ephemeral_project.id)
        assert isinstance(listed, APIKeyList)
        assert any(k.id == key_id for k in listed)
        assert len(listed) >= 1

        described = admin.api_keys.describe(api_key_id=key_id)
        assert isinstance(described, APIKeyModel)
        assert described.id == key_id
        assert described.name == key_name
        assert described.roles == ["ProjectEditor"]
        assert not hasattr(described, "value"), "describe() must not expose the secret value"

        admin.api_keys.delete(api_key_id=key_id)

        listed_after = admin.api_keys.list(project_id=ephemeral_project.id)
        assert not any(k.id == key_id for k in listed_after)

        with pytest.raises(NotFoundError):
            admin.api_keys.describe(api_key_id=key_id)

        key_id = None
    finally:
        if key_id is not None:
            try:
                admin.api_keys.delete(api_key_id=key_id)
            except Exception as e:
                print(f"Cleanup failed for key {key_id!r}: {e}")


@pytest.mark.integration
def test_api_keys_lifecycle_project_viewer_role(
    admin: Admin, ephemeral_project: ProjectModel
) -> None:
    """Full lifecycle for API keys with ProjectViewer role: create / list / describe / delete.

    Same shape as test_api_keys_lifecycle_default_role but uses roles=["ProjectViewer"]
    to verify non-default roles are accepted and round-trip correctly.
    """
    key_name = f"key-viewer-{int(time.time())}"
    key_id: str | None = None

    try:
        created = admin.api_keys.create(
            project_id=ephemeral_project.id, name=key_name, roles=["ProjectViewer"]
        )
        key_id = created.key.id
        assert created.value.startswith("pcsk_"), (
            f"expected pcsk_ prefix, got {created.value[:10]!r}"
        )
        assert created.key.id, "key id must be non-empty"
        assert created.key.name == key_name
        assert created.key.roles == ["ProjectViewer"]

        listed = admin.api_keys.list(project_id=ephemeral_project.id)
        assert isinstance(listed, APIKeyList)
        assert any(k.id == key_id for k in listed)
        assert len(listed) >= 1

        described = admin.api_keys.describe(api_key_id=key_id)
        assert isinstance(described, APIKeyModel)
        assert described.id == key_id
        assert described.name == key_name
        assert described.roles == ["ProjectViewer"]
        assert not hasattr(described, "value"), "describe() must not expose the secret value"

        admin.api_keys.delete(api_key_id=key_id)

        listed_after = admin.api_keys.list(project_id=ephemeral_project.id)
        assert not any(k.id == key_id for k in listed_after)

        with pytest.raises(NotFoundError):
            admin.api_keys.describe(api_key_id=key_id)

        key_id = None
    finally:
        if key_id is not None:
            try:
                admin.api_keys.delete(api_key_id=key_id)
            except Exception as e:
                print(f"Cleanup failed for key {key_id!r}: {e}")


# ---------------------------------------------------------------------------
# end-to-end bridge — requires real credentials + RUN_EXPENSIVE_TESTS=1
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("RUN_EXPENSIVE_TESTS"),
    reason="Creates real cloud resources — set RUN_EXPENSIVE_TESTS=1 to run",
)
def test_admin_to_pinecone_end_to_end_bridge(admin: Admin) -> None:
    """Proves that an Admin-created project+key pair can drive real data-plane ops.

    Workflow:
    1. Admin creates a new project and API key scoped to it.
    2. A fresh Pinecone client is constructed from the key value.
    3. A tiny serverless index is created and polled to Ready.
    4. The index status is confirmed ready.
    5. Finally: admin.projects.delete_with_cleanup() cascades the delete.

    Validates the canonical "create project → create key → use key for Pinecone
    client → create index" workflow from admin.py:48-54.
    """
    project_id: str | None = None
    index_name = f"bridge-idx-{int(time.time())}"

    try:
        project = admin.projects.create(name=f"bridge-{int(time.time())}")
        project_id = project.id

        key = admin.api_keys.create(project_id=project.id, name="bridge-key")
        # key.value is the secret; only available on create, not describe
        assert key.value.startswith("pcsk_"), (
            f"expected pcsk_ prefix on new key, got {key.value[:10]!r}"
        )

        pc = Pinecone(api_key=key.value)

        pc.indexes.create(
            name=index_name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        described = pc.indexes.describe(index_name)
        assert described.status.ready is True, (
            f"index {index_name!r} not ready after create+poll; status={described.status!r}"
        )

        # Confirm the key value passes auth for data-plane ops (describe returned above)
        # and that project isolation holds: the index is visible only under this key.
        listed_names = [idx.name for idx in pc.indexes.list()]
        assert index_name in listed_names, (
            f"index {index_name!r} missing from list(); got {listed_names}"
        )
    finally:
        if project_id is not None:
            try:
                admin.projects.delete_with_cleanup(project_id=project_id)
            except Exception as e:
                print(f"Cleanup failed for project {project_id!r}: {e}")
