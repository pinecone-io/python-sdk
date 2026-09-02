"""Integration tests for Pinecone client initialization."""

from __future__ import annotations

import json
import os

import httpx
import pytest
import respx

from pinecone import Pinecone
from pinecone.errors.exceptions import ValidationError
from pinecone.models.backups.model import CreateIndexFromBackupResponse
from tests.factories import make_index_response


@pytest.mark.integration
def test_validation_error_import_triggers_deprecation_warning() -> None:
    """Accessing pinecone.ValidationError triggers DeprecationWarning and returns PineconeValueError.

    Verifies unified-depr-0004 backward-compatibility contract: the ValidationError
    alias from v6 SDK remains accessible via the top-level module, BUT raises a
    DeprecationWarning directing callers to PineconeValueError.

    The shim in pinecone/__init__.__getattr__ fires warnings.warn(...,
    DeprecationWarning) on first access and then caches the result.  This test
    clears the cached entry before the assertion so the warning fires fresh even
    if a prior test or import already accessed pinecone.ValidationError.

    Assertions:
    - DeprecationWarning is raised with the expected message
    - The returned class is identical to PineconeValueError (same object)
    - ValidationError is a subclass of ValueError (for backward-compat catch blocks)
    - A PineconeValueError instance can be caught using the deprecated alias
    """
    import pinecone as pinecone_module
    from pinecone.errors.exceptions import PineconeValueError

    # Clear the cached entry so __getattr__ fires again and re-emits the warning.
    # (After the first access globals()["ValidationError"] is set, bypassing __getattr__.)
    pinecone_module.__dict__.pop("ValidationError", None)

    with pytest.warns(DeprecationWarning, match="ValidationError is deprecated"):
        deprecated_cls = pinecone_module.ValidationError

    # The alias must resolve to the same class object as PineconeValueError
    assert deprecated_cls is PineconeValueError, (
        "pinecone.ValidationError must be the same object as PineconeValueError"
    )

    # Must remain a proper ValueError subclass for old except-ValidationError blocks
    assert issubclass(deprecated_cls, ValueError), (
        "ValidationError must be a subclass of ValueError for backward-compatibility"
    )

    # Old code that catches ValidationError must catch PineconeValueError instances
    try:
        raise PineconeValueError("backward-compatibility catch test")
    except deprecated_cls:
        pass  # Expected — the alias catches PineconeValueError
    else:
        pytest.fail("ValidationError alias failed to catch a PineconeValueError instance")


BASE_URL = "https://api.pinecone.io"


@respx.mock
def test_create_index_from_backup_no_wait_returns_restore_job_id() -> None:
    """timeout=-1 returns CreateIndexFromBackupResponse with restore_job_id and index_id."""
    respx.post(f"{BASE_URL}/backups/bk-test/create-index").mock(
        return_value=httpx.Response(
            202,
            json={"restore_job_id": "rj-test-123", "index_id": "idx-test-456"},
        ),
    )

    pc = Pinecone(api_key="test-key", host=BASE_URL)
    result = pc.create_index_from_backup(
        name="test-restore-nowait",
        backup_id="bk-test",
        timeout=-1,
    )

    assert isinstance(result, CreateIndexFromBackupResponse)
    assert result.restore_job_id == "rj-test-123"
    assert result.index_id == "idx-test-456"


@pytest.mark.integration
def test_client_init_with_api_key() -> None:
    """Pinecone(api_key=...) creates a client with accessible control-plane namespaces."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        pytest.skip("PINECONE_API_KEY not set")

    pc = Pinecone(api_key=api_key)

    # Client should have the control-plane namespaces accessible
    assert pc.indexes is not None
    assert pc.inference is not None

    # version should be a non-empty string
    from pinecone import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0


_DENSE_SCHEMA = {
    "fields": {"embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}
}
_MANAGED_AWS = {"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"}


@respx.mock
def test_create_index_rejects_schema_kwarg() -> None:
    """create_index() rejects schema=/deployment= before making any request."""
    from pinecone.errors.exceptions import PineconeTypeError

    route = respx.post("https://api.pinecone.io/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response(name="test-schema-shim")),
    )

    pc = Pinecone(api_key="test-key", host=BASE_URL)
    with pytest.raises(PineconeTypeError, match=r"Use pc\.indexes\.create\(\)"):
        pc.create_index(
            name="test-schema-shim",
            schema=_DENSE_SCHEMA,
            deployment=_MANAGED_AWS,
            timeout=-1,
        )

    assert len(route.calls) == 0


@respx.mock
def test_configure_index_read_capacity() -> None:
    """configure_index shim forwards read_capacity as a top-level PATCH field.

    2026-07: the 2025-10 ``serverless_read_capacity=`` kwarg collapsed into one
    top-level ``read_capacity=`` covering managed and BYOC indexes.
    """
    route = respx.patch(f"{BASE_URL}/indexes/my-serverless-idx").mock(
        return_value=httpx.Response(202, json=make_index_response(name="my-serverless-idx")),
    )

    pc = Pinecone(api_key="test-key", host=BASE_URL)
    pc.configure_index("my-serverless-idx", read_capacity={"mode": "OnDemand"})

    sent_body = json.loads(route.calls[0].request.content)
    assert sent_body == {"read_capacity": {"mode": "OnDemand"}}


@respx.mock
def test_configure_index_tags_sparse_patch() -> None:
    """configure sends user tags directly without pre-fetching current state.

    The backend implements merge semantics (unmentioned tags preserved, empty-string
    value removes a tag), so the client must send only the user's dict — no describe
    round-trip and no client-side merge.
    """
    index_name = "my-tagged-idx"
    patch_route = respx.patch(f"{BASE_URL}/indexes/{index_name}").mock(
        return_value=httpx.Response(202, json=make_index_response(name=index_name)),
    )
    # Ensure no GET /indexes/{name} is registered — if the client calls describe,
    # respx will raise an error (no matching route) rather than silently passing.

    pc = Pinecone(api_key="test-key", host=BASE_URL)

    # Sparse add: send a new key; backend will preserve any pre-existing tags.
    pc.indexes.configure(index_name, tags={"new_key": "new_val"})
    sent_body = json.loads(patch_route.calls[0].request.content)
    assert sent_body["tags"] == {"new_key": "new_val"}, (
        "Client must send only the user-supplied tags dict, not a client-side merge"
    )
    assert len(patch_route.calls) == 1, "configure must issue exactly one PATCH (no describe GET)"

    patch_route.reset()

    # Sparse remove: empty-string value signals the backend to delete that tag.
    pc.indexes.configure(index_name, tags={"existing_key": ""})
    sent_body2 = json.loads(patch_route.calls[0].request.content)
    assert sent_body2["tags"] == {"existing_key": ""}, (
        "Client must forward the empty-string removal signal unchanged"
    )
    assert len(patch_route.calls) == 1, "configure must issue exactly one PATCH (no describe GET)"


@respx.mock
def test_create_index_read_capacity_forwarded() -> None:
    """read_capacity is a top-level create field and deserializes to a tagged response."""
    from pinecone.models.indexes.read_capacity import ReadCapacityOnDemandResponse

    response_body = make_index_response(name="rc-index")
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=response_body),
    )

    pc = Pinecone(api_key="test-key", host=BASE_URL)
    result = pc.indexes.create(
        name="rc-index",
        schema=_DENSE_SCHEMA,
        deployment=_MANAGED_AWS,
        read_capacity={"mode": "OnDemand"},
        timeout=-1,
    )

    assert result.name == "rc-index"
    sent_body = json.loads(route.calls[0].request.content)
    assert sent_body["read_capacity"] == {"mode": "OnDemand"}
    assert isinstance(result.read_capacity, ReadCapacityOnDemandResponse)
    assert result.read_capacity.status.state == "Ready"


@respx.mock
def test_create_byoc_index_schema_forwarded() -> None:
    """A byoc deployment and a top-level schema both reach the create body verbatim."""
    from pinecone.models.indexes.deployment import ByocDeployment

    byoc_response = make_index_response(
        name="byoc-schema-index",
        deployment={"deployment_type": "byoc", "environment": "byoc-aws-abc123"},
    )
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=byoc_response),
    )

    pc = Pinecone(api_key="test-key", host=BASE_URL)
    result = pc.indexes.create(
        name="byoc-schema-index",
        schema=_DENSE_SCHEMA,
        deployment={"deployment_type": "byoc", "environment": "byoc-aws-abc123"},
        timeout=-1,
    )

    sent_body = json.loads(route.calls[0].request.content)
    assert sent_body["schema"] == _DENSE_SCHEMA
    assert sent_body["deployment"] == {
        "deployment_type": "byoc",
        "environment": "byoc-aws-abc123",
    }
    assert isinstance(result.deployment, ByocDeployment)
    assert result.deployment.environment == "byoc-aws-abc123"


@respx.mock
def test_create_index_rejects_quarantined_pods_kwarg() -> None:
    """pods= has no 2026-07 deployment translation and raises before any request."""
    from pinecone.errors.exceptions import PineconeTypeError

    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response()),
    )

    pc = Pinecone(api_key="test-key", host=BASE_URL)
    with pytest.raises(PineconeTypeError, match="pods=: capacity is replicas x shards"):
        pc.indexes.create(
            name="legacy-index",
            dimension=128,
            metric="cosine",
            pods=1,
            timeout=-1,
        )

    assert len(route.calls) == 0


# ---------------------------------------------------------------------------
# Collection name validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "invalid_name",
    [
        "INVALID_NAME!",
        "MY_COLLECTION",
        "-leading",
        "trailing-",
        "a" * 46,
        "underscore_name",
        "test@name",
    ],
)
def test_create_collection_invalid_name(invalid_name: str) -> None:
    """Collections.create raises ValidationError for invalid names before any network call."""
    pc = Pinecone(api_key="test-key")
    with pytest.raises(ValidationError):
        pc.collections.create(name=invalid_name, source="fake-index")
