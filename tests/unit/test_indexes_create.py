"""Unit tests for Indexes.create() — serverless and pod-based index creation."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import HTTPClient
from pinecone.client.indexes import Indexes
from pinecone.errors.exceptions import IndexInitFailedError, PineconeValueError, ValidationError
from pinecone.models.enums import DeletionProtection, Metric, VectorType
from pinecone.models.indexes.index import IndexModel
from pinecone.models.indexes.specs import ByocSpec, PodSpec, ServerlessSpec
from tests.factories import make_index_response

BASE_URL = "https://api.test.pinecone.io"


@pytest.fixture
def http_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, CONTROL_PLANE_API_VERSION)


@pytest.fixture
def indexes(http_client: HTTPClient) -> Indexes:
    return Indexes(http=http_client)


# ---------------------------------------------------------------------------
# Success paths
# ---------------------------------------------------------------------------


@respx.mock
def test_create_serverless_index(indexes: Indexes) -> None:
    """Create with ServerlessSpec — verify POST body has correct shape."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response()),
    )

    result = indexes.create(
        name="test-index",
        dimension=1536,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        timeout=-1,
    )

    assert isinstance(result, IndexModel)
    assert result.name == "test-index"

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["name"] == "test-index"
    assert body["dimension"] == 1536
    assert body["metric"] == "cosine"
    assert body["vector_type"] == "dense"
    assert body["deletion_protection"] == "disabled"
    assert body["spec"] == {"serverless": {"cloud": "aws", "region": "us-east-1"}}


@respx.mock
def test_create_pod_index(indexes: Indexes) -> None:
    """Create with PodSpec — verify body includes pod spec."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201,
            json=make_index_response(
                spec={
                    "pod": {
                        "environment": "us-east1-gcp",
                        "pod_type": "p1.x1",
                        "replicas": 1,
                        "shards": 1,
                        "pods": 1,
                    }
                }
            ),
        ),
    )

    result = indexes.create(
        name="test-index",
        dimension=1536,
        spec=PodSpec(environment="us-east1-gcp"),
        timeout=-1,
    )

    assert isinstance(result, IndexModel)

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["spec"]["pod"]["environment"] == "us-east1-gcp"
    assert body["spec"]["pod"]["pod_type"] == "p1.x1"  # default


@respx.mock
def test_create_index_defaults(indexes: Indexes) -> None:
    """Omit optional params — verify defaults."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response()),
    )

    indexes.create(
        name="test-index",
        dimension=1536,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        timeout=-1,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["metric"] == "cosine"
    assert body["vector_type"] == "dense"
    assert body["deletion_protection"] == "disabled"
    assert "tags" not in body


@respx.mock
def test_create_index_with_tags(indexes: Indexes) -> None:
    """Verify tags are included in request body."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response()),
    )

    indexes.create(
        name="test-index",
        dimension=1536,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        tags={"env": "test", "team": "ml"},
        timeout=-1,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["tags"] == {"env": "test", "team": "ml"}


@respx.mock
def test_create_with_dict_spec(indexes: Indexes) -> None:
    """Pass raw dict spec — verify it's sent as-is."""
    raw_spec: dict[str, Any] = {"serverless": {"cloud": "gcp", "region": "us-central1"}}
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response()),
    )

    indexes.create(
        name="test-index",
        dimension=1536,
        spec=raw_spec,
        timeout=-1,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["spec"] == raw_spec


# ---------------------------------------------------------------------------
# Validation error paths
# ---------------------------------------------------------------------------


def test_create_missing_name_raises(indexes: Indexes) -> None:
    """Empty name raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        indexes.create(
            name="",
            dimension=1536,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    assert "name" in str(exc_info.value)


def test_create_missing_spec_raises(indexes: Indexes) -> None:
    """No spec (None) raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        indexes.create(
            name="test-index",
            dimension=1536,
            spec=None,  # type: ignore[arg-type]
        )
    assert "spec" in str(exc_info.value)


def test_create_dense_missing_dimension_raises(indexes: Indexes) -> None:
    """Dense index without dimension raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        indexes.create(
            name="test-index",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    assert "dimension" in str(exc_info.value)


def test_create_invalid_metric_raises(indexes: Indexes) -> None:
    """Invalid metric raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        indexes.create(
            name="test-index",
            dimension=1536,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            metric="hamming",
        )
    assert "metric" in str(exc_info.value)


def test_create_invalid_deletion_protection_raises(indexes: Indexes) -> None:
    """Invalid deletion protection value raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        indexes.create(
            name="test-index",
            dimension=1536,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            deletion_protection="maybe",
        )
    assert "deletion_protection" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Enum/string handling
# ---------------------------------------------------------------------------


@respx.mock
def test_create_with_metric_enum(indexes: Indexes) -> None:
    """Accept Metric enum for the metric parameter."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response(metric="euclidean")),
    )

    indexes.create(
        name="test-index",
        dimension=1536,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        metric=Metric.EUCLIDEAN,
        timeout=-1,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["metric"] == "euclidean"


@respx.mock
def test_create_with_vector_type_enum(indexes: Indexes) -> None:
    """Accept VectorType enum for the vector_type parameter."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201,
            json=make_index_response(vector_type="sparse", dimension=None),
        ),
    )

    indexes.create(
        name="sparse-enum-index",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        vector_type=VectorType.SPARSE,
        timeout=-1,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["vector_type"] == "sparse"


@respx.mock
def test_create_with_deletion_protection_enum(indexes: Indexes) -> None:
    """Accept DeletionProtection enum for the deletion_protection parameter."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response()),
    )

    indexes.create(
        name="test-index",
        dimension=1536,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        deletion_protection=DeletionProtection.ENABLED,
        timeout=-1,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["deletion_protection"] == "enabled"


@respx.mock
def test_create_with_all_enums(indexes: Indexes) -> None:
    """Accept all enum types together in create()."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201,
            json=make_index_response(metric="dotproduct"),
        ),
    )

    indexes.create(
        name="test-index",
        dimension=1536,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        metric=Metric.DOTPRODUCT,
        vector_type=VectorType.DENSE,
        deletion_protection=DeletionProtection.DISABLED,
        timeout=-1,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["metric"] == "dotproduct"
    assert body["vector_type"] == "dense"
    assert body["deletion_protection"] == "disabled"


# ---------------------------------------------------------------------------
# Timeout / polling
# ---------------------------------------------------------------------------


@respx.mock
def test_create_timeout_none_polls_indefinitely(indexes: Indexes) -> None:
    """With timeout=None (default), polls until index is ready."""
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201,
            json=make_index_response(status={"ready": False, "state": "Initializing"}),
        ),
    )
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        side_effect=[
            httpx.Response(
                200,
                json=make_index_response(status={"ready": False, "state": "Initializing"}),
            ),
            httpx.Response(
                200,
                json=make_index_response(status={"ready": True, "state": "Ready"}),
            ),
        ]
    )

    with patch("pinecone.client.indexes.time.sleep"):
        result = indexes.create(
            name="test-index",
            dimension=1536,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    assert result.status.ready is True
    # POST + 2 GET calls
    assert len(respx.calls) == 3


@respx.mock
def test_create_polls_until_ready(indexes: Indexes) -> None:
    """With timeout, poll describe until ready."""
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201,
            json=make_index_response(status={"ready": False, "state": "Initializing"}),
        ),
    )
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        side_effect=[
            httpx.Response(
                200,
                json=make_index_response(status={"ready": False, "state": "Initializing"}),
            ),
            httpx.Response(
                200,
                json=make_index_response(status={"ready": True, "state": "Ready"}),
            ),
        ]
    )

    with patch("pinecone.client.indexes.time.sleep"):
        result = indexes.create(
            name="test-index",
            dimension=1536,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

    assert result.status.ready is True


@respx.mock
def test_create_init_failed_raises(indexes: Indexes) -> None:
    """If index enters InitializationFailed, raise immediately."""
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201,
            json=make_index_response(status={"ready": False, "state": "Initializing"}),
        ),
    )
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(
            200,
            json=make_index_response(status={"ready": False, "state": "InitializationFailed"}),
        ),
    )

    with (
        patch("pinecone.client.indexes.time.sleep"),
        pytest.raises(IndexInitFailedError, match="InitializationFailed"),
    ):
        indexes.create(
            name="test-index",
            dimension=1536,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )


@respx.mock
def test_create_sparse_without_dimension(indexes: Indexes) -> None:
    """Sparse index can be created without specifying dimension."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201,
            json=make_index_response(vector_type="sparse", dimension=None),
        ),
    )

    result = indexes.create(
        name="sparse-index",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        vector_type="sparse",
        timeout=-1,
    )

    assert isinstance(result, IndexModel)
    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["vector_type"] == "sparse"
    assert "dimension" not in body


def test_create_name_too_long_raises(indexes: Indexes) -> None:
    """Name exceeding 45 characters raises ValidationError."""
    long_name = "a" * 46
    with pytest.raises(ValidationError) as exc_info:
        indexes.create(
            name=long_name,
            dimension=1536,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    assert "45 characters" in str(exc_info.value)


def test_create_name_invalid_chars_raises(indexes: Indexes) -> None:
    """Name with uppercase or special characters raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        indexes.create(
            name="My_Index!",
            dimension=1536,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    assert "lowercase" in str(exc_info.value)


@respx.mock
def test_create_name_valid_boundary(indexes: Indexes) -> None:
    """Name of exactly 45 lowercase chars succeeds."""
    valid_name = "a" * 45
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response(name=valid_name)),
    )

    result = indexes.create(
        name=valid_name,
        dimension=1536,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        timeout=-1,
    )
    assert isinstance(result, IndexModel)


@respx.mock
def test_create_sparse_with_dimension_allowed(indexes: Indexes) -> None:
    """Sparse index accepts an explicit dimension so Pinecone Local can be targeted.

    Pinecone Local requires a ``dimension`` field in the POST body even for sparse
    indexes.  The SDK no longer rejects this combination at the client-side validation
    layer; the field is forwarded to the server and the server decides whether it is
    valid for the target environment.
    """
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201, json=make_index_response(vector_type="sparse", dimension=1)
        ),
    )

    result = indexes.create(
        name="sparse-local",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        vector_type="sparse",
        dimension=1,
        timeout=-1,
    )

    assert result.vector_type == "sparse"

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["vector_type"] == "sparse"
    assert body["dimension"] == 1


def test_create_with_unrecognized_dict_spec_raises(indexes: Indexes) -> None:
    """Dict spec without a recognized key raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        indexes.create(
            name="test-index",
            dimension=1536,
            spec={"unknown": {"foo": "bar"}},
        )

    assert "serverless" in str(exc_info.value)
    assert "pod" in str(exc_info.value)


def test_create_with_empty_dict_spec_raises(indexes: Indexes) -> None:
    """Empty dict spec raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        indexes.create(
            name="test-index",
            dimension=1536,
            spec={},
        )

    assert "serverless" in str(exc_info.value)
    assert "pod" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Schema parameter
# ---------------------------------------------------------------------------


@respx.mock
def test_create_with_flat_schema(indexes: Indexes) -> None:
    """Flat schema dict is auto-wrapped under 'fields' before being placed in spec.serverless.schema."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response()),
    )

    schema: dict[str, Any] = {
        "genre": {"type": "str", "filterable": True},
        "year": {"type": "int", "filterable": True},
    }

    indexes.create(
        name="test-index",
        dimension=1536,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        schema=schema,
        timeout=-1,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["spec"]["serverless"]["schema"] == {"fields": schema}


@respx.mock
def test_create_with_nested_schema(indexes: Indexes) -> None:
    """Already-wrapped schema passes through unchanged on the wire."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response()),
    )

    nested_schema: dict[str, Any] = {
        "fields": {
            "genre": {"type": "str", "filterable": True},
        }
    }

    indexes.create(
        name="test-index",
        dimension=1536,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        schema=nested_schema,
        timeout=-1,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["spec"]["serverless"]["schema"] == nested_schema


@respx.mock
def test_create_without_schema(indexes: Indexes) -> None:
    """When schema is None, no schema key appears in the request body."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response()),
    )

    indexes.create(
        name="test-index",
        dimension=1536,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        timeout=-1,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert "schema" not in body["spec"]["serverless"]


# ---------------------------------------------------------------------------
# BYOC index creation
# ---------------------------------------------------------------------------


@respx.mock
def test_create_byoc_index(indexes: Indexes) -> None:
    """Create with ByocSpec — verify POST body has correct shape."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201,
            json=make_index_response(
                spec={"byoc": {"environment": "aws-us-east-1-b921"}},
            ),
        ),
    )

    result = indexes.create(
        name="byoc-idx",
        dimension=1536,
        spec=ByocSpec(environment="aws-us-east-1-b921"),
        timeout=-1,
    )

    assert isinstance(result, IndexModel)

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["name"] == "byoc-idx"
    assert body["dimension"] == 1536
    assert body["metric"] == "cosine"
    assert body["spec"] == {"byoc": {"environment": "aws-us-east-1-b921"}}


@respx.mock
def test_create_byoc_index_with_read_capacity(indexes: Indexes) -> None:
    """Create BYOC with read_capacity — verify it appears inside byoc spec."""
    read_capacity = {
        "mode": "Dedicated",
        "dedicated": {
            "node_type": "t1",
            "scaling": "Manual",
            "manual": {"replicas": 2, "shards": 1},
        },
    }
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201,
            json=make_index_response(
                spec={
                    "byoc": {
                        "environment": "aws-us-east-1-b921",
                        "read_capacity": read_capacity,
                    }
                },
            ),
        ),
    )

    result = indexes.create(
        name="byoc-drn",
        dimension=1536,
        spec=ByocSpec(
            environment="aws-us-east-1-b921",
            read_capacity=read_capacity,
        ),
        timeout=-1,
    )

    assert isinstance(result, IndexModel)

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["spec"]["byoc"]["environment"] == "aws-us-east-1-b921"
    assert body["spec"]["byoc"]["read_capacity"] == read_capacity


def test_create_byoc_missing_environment(indexes: Indexes) -> None:
    """ByocSpec with empty environment raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        indexes.create(
            name="byoc-idx",
            dimension=1536,
            spec=ByocSpec(environment=""),
        )
    assert "environment" in str(exc_info.value)


@respx.mock
def test_create_byoc_dict_spec(indexes: Indexes) -> None:
    """Pass raw dict with byoc key — verify it goes through dict path."""
    raw_spec: dict[str, Any] = {"byoc": {"environment": "aws-us-east-1-b921"}}
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201,
            json=make_index_response(
                spec=raw_spec,
            ),
        ),
    )

    indexes.create(
        name="byoc-idx",
        dimension=1536,
        spec=raw_spec,
        timeout=-1,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["spec"] == raw_spec


def test_create_byoc_missing_dimension(indexes: Indexes) -> None:
    """ByocSpec without dimension raises ValidationError."""
    with pytest.raises(ValidationError, match="dimension"):
        indexes.create(
            name="byoc-idx",
            spec=ByocSpec(environment="aws-us-east-1-b921"),
        )


# ---------------------------------------------------------------------------
# Legacy-parity tests (BC-0074): PineconeValueError for client-side validation
# These mirror the three legacy PineconeApiException scenarios that now raise
# PineconeValueError because validation happens client-side before any request.
# ---------------------------------------------------------------------------


def test_create_index_invalid_name_raises_value_error(indexes: Indexes) -> None:
    """Uppercase characters in index name are rejected client-side."""
    with pytest.raises(PineconeValueError, match="lowercase"):
        indexes.create(
            name="Invalid-name",
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


def test_create_index_invalid_metric_raises_value_error(indexes: Indexes) -> None:
    """Unsupported metric values are rejected client-side."""
    with pytest.raises(PineconeValueError, match="metric must be one of"):
        indexes.create(
            name="valid-name",
            dimension=2,
            metric="banana",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


def test_create_index_missing_dimension_raises_value_error(indexes: Indexes) -> None:
    """Dense indexes require an explicit dimension."""
    with pytest.raises(PineconeValueError, match="dimension is required"):
        indexes.create(
            name="valid-name",
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
