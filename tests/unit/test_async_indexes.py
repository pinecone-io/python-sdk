"""Unit tests for AsyncIndexes namespace — list, describe, exists, create, delete, configure."""

from __future__ import annotations

import itertools
import json
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import AsyncHTTPClient
from pinecone.async_client.indexes import _POLL_INTERVAL_SECONDS, AsyncIndexes
from pinecone.errors.exceptions import (
    ConflictError,
    IndexInitFailedError,
    NotFoundError,
    PineconeTimeoutError,
    PineconeValueError,
    ValidationError,
)
from pinecone.models.enums import DeletionProtection, EmbedModel, Metric, VectorType
from pinecone.models.indexes.index import IndexModel
from pinecone.models.indexes.list import IndexList
from pinecone.models.indexes.specs import (
    ByocSpec,
    EmbedConfig,
    IntegratedSpec,
    PodSpec,
    ServerlessSpec,
)
from tests.factories import (
    make_error_response,
    make_index_list_response,
    make_index_response,
)

BASE_URL = "https://api.test.pinecone.io"


@pytest.fixture
async def async_http_client() -> AsyncGenerator[AsyncHTTPClient]:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    client = AsyncHTTPClient(config, CONTROL_PLANE_API_VERSION)
    yield client
    await client.close()


@pytest.fixture
def async_indexes(async_http_client: AsyncHTTPClient) -> AsyncIndexes:
    return AsyncIndexes(http=async_http_client)


# ---------------------------------------------------------------------------
# list()
# ---------------------------------------------------------------------------


@respx.mock
async def test_list_indexes(async_indexes: AsyncIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(200, json=make_index_list_response()),
    )

    result = await async_indexes.list()

    assert isinstance(result, IndexList)
    assert len(result) == 1
    assert result[0].name == "test-index"
    assert result.names() == ["test-index"]

    # verify iteration
    names = [idx.name for idx in result]
    assert names == ["test-index"]


@respx.mock
async def test_list_indexes_empty(async_indexes: AsyncIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(200, json={"indexes": []}),
    )

    result = await async_indexes.list()

    assert isinstance(result, IndexList)
    assert len(result) == 0
    assert result.names() == []


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------


@respx.mock
async def test_describe_index(async_indexes: AsyncIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response()),
    )

    result = await async_indexes.describe("test-index")

    assert isinstance(result, IndexModel)
    assert result.name == "test-index"
    assert result.host == "https://test-index-abc1234.svc.us-east1-gcp.pinecone.io"
    assert result.deletion_protection == "disabled"
    assert result.status.ready is True
    assert result.status.state == "Ready"
    assert result.deployment.cloud == "aws"  # type: ignore[union-attr]
    embedding = result.schema.fields["embedding"]
    assert embedding.dimension == 1536  # type: ignore[union-attr]
    assert embedding.metric == "cosine"  # type: ignore[union-attr]
    assert result["name"] == "test-index"
    assert result["schema"] is result.schema


@respx.mock
async def test_describe_index_caches_host(async_indexes: AsyncIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response()),
    )

    await async_indexes.describe("test-index")

    assert "test-index" in async_indexes._host_cache
    expected_host = "https://test-index-abc1234.svc.us-east1-gcp.pinecone.io"
    assert async_indexes._host_cache["test-index"] == expected_host


@respx.mock
async def test_describe_nonexistent_index(async_indexes: AsyncIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes/no-such-index").mock(
        return_value=httpx.Response(
            404,
            json=make_error_response(404, "Index not found"),
        ),
    )

    with pytest.raises(NotFoundError):
        await async_indexes.describe("no-such-index")


async def test_describe_empty_name_raises(async_indexes: AsyncIndexes) -> None:
    with pytest.raises(ValidationError) as exc_info:
        await async_indexes.describe("")

    assert "name" in str(exc_info.value)
    assert "non-empty" in str(exc_info.value)


# ---------------------------------------------------------------------------
# exists()
# ---------------------------------------------------------------------------


@respx.mock
async def test_exists_true(async_indexes: AsyncIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response()),
    )

    assert await async_indexes.exists("test-index") is True


@respx.mock
async def test_exists_false(async_indexes: AsyncIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes/no-such-index").mock(
        return_value=httpx.Response(
            404,
            json=make_error_response(404, "Index not found"),
        ),
    )

    assert await async_indexes.exists("no-such-index") is False


async def test_exists_empty_name_returns_false(async_indexes: AsyncIndexes) -> None:
    assert await async_indexes.exists("") is False


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


@respx.mock
async def test_delete_index_default_polls(async_indexes: AsyncIndexes) -> None:
    """DELETE /indexes/test-index -> 202, then polls until gone (default)."""
    respx.delete(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202),
    )
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(404, json=make_error_response(404, "Not found")),
    )

    with patch("pinecone.async_client.indexes.asyncio.sleep", new_callable=AsyncMock):
        result = await async_indexes.delete("test-index")

    assert result is None


@respx.mock
async def test_delete_removes_host_cache(async_indexes: AsyncIndexes) -> None:
    """Deleting an index removes its cached host URL."""
    async_indexes._host_cache["my-index"] = "my-index-abc.svc.pinecone.io"

    respx.delete(f"{BASE_URL}/indexes/my-index").mock(
        return_value=httpx.Response(202),
    )
    respx.get(f"{BASE_URL}/indexes/my-index").mock(
        return_value=httpx.Response(404, json=make_error_response(404, "Not found")),
    )

    with patch("pinecone.async_client.indexes.asyncio.sleep", new_callable=AsyncMock):
        await async_indexes.delete("my-index")

    assert "my-index" not in async_indexes._host_cache


@respx.mock
async def test_delete_clears_stale_host_cache_after_polling(async_indexes: AsyncIndexes) -> None:
    """Stale host cache entry added by describe() polling is cleared when delete completes.

    Regression test for the bug where describe() re-adds the host to _host_cache during
    each successful poll iteration. After the polling loop exits via NotFoundError, the
    entry from the last successful describe() must be removed so that subsequent calls to
    pc.index("my-index") don't use a dead host.
    """
    async_indexes._host_cache["my-index"] = "my-index-abc.svc.pinecone.io"

    respx.delete(f"{BASE_URL}/indexes/my-index").mock(
        return_value=httpx.Response(202),
    )
    # First describe returns 200 (describe re-adds host to cache), then 404
    respx.get(f"{BASE_URL}/indexes/my-index").mock(
        side_effect=[
            httpx.Response(200, json=make_index_response()),
            httpx.Response(404, json=make_error_response(404, "Not found")),
        ],
    )

    with patch("pinecone.async_client.indexes.asyncio.sleep", new_callable=AsyncMock):
        await async_indexes.delete("my-index")

    assert "my-index" not in async_indexes._host_cache


@respx.mock
async def test_delete_polls_until_gone(async_indexes: AsyncIndexes) -> None:
    """With explicit timeout, poll describe until index disappears."""
    respx.delete(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202),
    )
    describe_route = respx.get(f"{BASE_URL}/indexes/test-index").mock(
        side_effect=[
            httpx.Response(200, json=make_index_response()),
            httpx.Response(404, json=make_error_response(404, "Not found")),
        ],
    )

    with patch("pinecone.async_client.indexes.asyncio.sleep"):
        await async_indexes.delete("test-index", timeout=300)

    assert describe_route.call_count == 2


@respx.mock
async def test_delete_timeout_negative_one_skips_polling(async_indexes: AsyncIndexes) -> None:
    """With timeout=-1, return immediately after API call — no polling."""
    respx.delete(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202),
    )
    describe_route = respx.get(f"{BASE_URL}/indexes/test-index")

    result = await async_indexes.delete("test-index", timeout=-1)

    assert result is None
    assert describe_route.call_count == 0


@respx.mock
async def test_delete_timeout_exceeded(async_indexes: AsyncIndexes) -> None:
    """If index still exists after timeout, raise PineconeTimeoutError."""
    respx.delete(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202),
    )
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response()),
    )

    with (
        patch("pinecone.async_client.indexes.asyncio.sleep"),
        patch(
            "pinecone.async_client.indexes.time.monotonic",
            side_effect=itertools.count(start=0.0, step=0.5).__next__,
        ),
        pytest.raises(PineconeTimeoutError, match=r"still exists after 1s"),
    ):
        await async_indexes.delete("test-index", timeout=1)


@respx.mock
async def test_delete_nonexistent_index(async_indexes: AsyncIndexes) -> None:
    """DELETE on non-existent index -> 404 -> NotFoundError."""
    respx.delete(f"{BASE_URL}/indexes/no-such-index").mock(
        return_value=httpx.Response(404, json=make_error_response(404, "Index not found")),
    )

    with pytest.raises(NotFoundError):
        await async_indexes.delete("no-such-index")


async def test_delete_empty_name_raises(async_indexes: AsyncIndexes) -> None:
    """Empty name raises ValidationError before any HTTP call."""
    with pytest.raises(ValidationError):
        await async_indexes.delete("")


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------


@respx.mock
async def test_create_serverless_index(async_indexes: AsyncIndexes) -> None:
    """Create with ServerlessSpec — verify POST body has correct shape."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response()),
    )

    result = await async_indexes.create(
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
async def test_create_pod_index(async_indexes: AsyncIndexes) -> None:
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

    result = await async_indexes.create(
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
async def test_create_index_defaults(async_indexes: AsyncIndexes) -> None:
    """Omit optional params — verify defaults."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response()),
    )

    await async_indexes.create(
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
async def test_create_index_with_tags(async_indexes: AsyncIndexes) -> None:
    """Verify tags are included in request body."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response()),
    )

    await async_indexes.create(
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
async def test_create_with_dict_spec(async_indexes: AsyncIndexes) -> None:
    """Pass raw dict spec — verify it's sent as-is."""
    raw_spec: dict[str, Any] = {"serverless": {"cloud": "gcp", "region": "us-central1"}}
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response()),
    )

    await async_indexes.create(
        name="test-index",
        dimension=1536,
        spec=raw_spec,
        timeout=-1,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["spec"] == raw_spec


async def test_create_missing_name_raises(async_indexes: AsyncIndexes) -> None:
    """Empty name raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        await async_indexes.create(
            name="",
            dimension=1536,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    assert "name" in str(exc_info.value)


async def test_create_missing_spec_raises(async_indexes: AsyncIndexes) -> None:
    """No spec (None) raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        await async_indexes.create(
            name="test-index",
            dimension=1536,
            spec=None,  # type: ignore[arg-type]
        )
    assert "spec" in str(exc_info.value)


async def test_create_dense_missing_dimension_raises(async_indexes: AsyncIndexes) -> None:
    """Dense index without dimension raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        await async_indexes.create(
            name="test-index",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    assert "dimension" in str(exc_info.value)


async def test_create_invalid_metric_raises(async_indexes: AsyncIndexes) -> None:
    """Invalid metric raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        await async_indexes.create(
            name="test-index",
            dimension=1536,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            metric="hamming",
        )
    assert "metric" in str(exc_info.value)


async def test_create_invalid_deletion_protection_raises(async_indexes: AsyncIndexes) -> None:
    """Invalid deletion protection value raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        await async_indexes.create(
            name="test-index",
            dimension=1536,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            deletion_protection="maybe",
        )
    assert "deletion_protection" in str(exc_info.value)


async def test_create_name_too_long_raises(async_indexes: AsyncIndexes) -> None:
    """Name exceeding 45 characters raises ValidationError."""
    long_name = "a" * 46
    with pytest.raises(ValidationError) as exc_info:
        await async_indexes.create(
            name=long_name,
            dimension=1536,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    assert "45 characters" in str(exc_info.value)


async def test_create_name_invalid_chars_raises(async_indexes: AsyncIndexes) -> None:
    """Name with uppercase or special characters raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        await async_indexes.create(
            name="My_Index!",
            dimension=1536,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    assert "lowercase" in str(exc_info.value)


@respx.mock
async def test_create_name_valid_boundary(async_indexes: AsyncIndexes) -> None:
    """Name of exactly 45 lowercase chars succeeds."""
    valid_name = "a" * 45
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response(name=valid_name)),
    )

    result = await async_indexes.create(
        name=valid_name,
        dimension=1536,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        timeout=-1,
    )
    assert isinstance(result, IndexModel)


async def test_create_sparse_with_dimension_raises(async_indexes: AsyncIndexes) -> None:
    """Sparse index with explicit dimension raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        await async_indexes.create(
            name="test",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            vector_type="sparse",
            dimension=384,
        )

    assert "dimension" in str(exc_info.value)
    assert "sparse" in str(exc_info.value)


async def test_create_with_unrecognized_dict_spec_raises(async_indexes: AsyncIndexes) -> None:
    """Dict spec without a recognized key raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        await async_indexes.create(
            name="test-index",
            dimension=1536,
            spec={"unknown": {"foo": "bar"}},
        )

    assert "serverless" in str(exc_info.value)
    assert "pod" in str(exc_info.value)


async def test_create_with_empty_dict_spec_raises(async_indexes: AsyncIndexes) -> None:
    """Empty dict spec raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        await async_indexes.create(
            name="test-index",
            dimension=1536,
            spec={},
        )

    assert "serverless" in str(exc_info.value)
    assert "pod" in str(exc_info.value)


@respx.mock
async def test_create_with_metric_enum(async_indexes: AsyncIndexes) -> None:
    """Accept Metric enum for the metric parameter."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response(metric="euclidean")),
    )

    await async_indexes.create(
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
async def test_create_with_vector_type_enum(async_indexes: AsyncIndexes) -> None:
    """Accept VectorType enum for the vector_type parameter."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201,
            json=make_index_response(vector_type="sparse", dimension=None),
        ),
    )

    await async_indexes.create(
        name="sparse-enum-index",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        vector_type=VectorType.SPARSE,
        timeout=-1,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["vector_type"] == "sparse"


@respx.mock
async def test_create_with_deletion_protection_enum(async_indexes: AsyncIndexes) -> None:
    """Accept DeletionProtection enum for the deletion_protection parameter."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response()),
    )

    await async_indexes.create(
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
async def test_create_timeout_none_polls_indefinitely(async_indexes: AsyncIndexes) -> None:
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

    with patch("pinecone.async_client.indexes.asyncio.sleep", new_callable=AsyncMock):
        result = await async_indexes.create(
            name="test-index",
            dimension=1536,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    assert result.status.ready is True
    assert len(respx.calls) == 3  # POST + 2 GET


@respx.mock
async def test_create_polls_until_ready(async_indexes: AsyncIndexes) -> None:
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

    with patch("pinecone.async_client.indexes.asyncio.sleep"):
        result = await async_indexes.create(
            name="test-index",
            dimension=1536,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

    assert result.status.ready is True


@respx.mock
async def test_create_init_failed_raises(async_indexes: AsyncIndexes) -> None:
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

    with patch("pinecone.async_client.indexes.asyncio.sleep"), pytest.raises(IndexInitFailedError):
        await async_indexes.create(
            name="test-index",
            dimension=1536,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )


@respx.mock
async def test_create_sparse_without_dimension(async_indexes: AsyncIndexes) -> None:
    """Sparse index can be created without specifying dimension."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201,
            json=make_index_response(vector_type="sparse", dimension=None),
        ),
    )

    result = await async_indexes.create(
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


# ---------------------------------------------------------------------------
# create() — schema
# ---------------------------------------------------------------------------


@respx.mock
async def test_create_with_flat_schema(async_indexes: AsyncIndexes) -> None:
    """Flat schema dict is auto-wrapped under 'fields' before being placed in spec.serverless.schema."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response()),
    )

    schema: dict[str, Any] = {
        "genre": {"type": "str", "filterable": True},
        "year": {"type": "int", "filterable": True},
    }

    await async_indexes.create(
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
async def test_create_with_nested_schema(async_indexes: AsyncIndexes) -> None:
    """Already-wrapped schema passes through unchanged on the wire."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response()),
    )

    nested_schema: dict[str, Any] = {
        "fields": {
            "genre": {"type": "str", "filterable": True},
        }
    }

    await async_indexes.create(
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
async def test_create_without_schema(async_indexes: AsyncIndexes) -> None:
    """When schema is None, no schema key appears in the request body."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response()),
    )

    await async_indexes.create(
        name="test-index",
        dimension=1536,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        timeout=-1,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert "schema" not in body["spec"]["serverless"]


# ---------------------------------------------------------------------------
# create() — BYOC
# ---------------------------------------------------------------------------


@respx.mock
async def test_create_byoc_index(async_indexes: AsyncIndexes) -> None:
    """Create with ByocSpec — verify POST body has correct shape."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201,
            json=make_index_response(
                spec={"byoc": {"environment": "aws-us-east-1-b921"}},
            ),
        ),
    )

    result = await async_indexes.create(
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
async def test_create_byoc_index_with_read_capacity(async_indexes: AsyncIndexes) -> None:
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

    result = await async_indexes.create(
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


async def test_create_byoc_missing_environment(async_indexes: AsyncIndexes) -> None:
    """ByocSpec with empty environment raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        await async_indexes.create(
            name="byoc-idx",
            dimension=1536,
            spec=ByocSpec(environment=""),
        )
    assert "environment" in str(exc_info.value)


@respx.mock
async def test_create_byoc_dict_spec(async_indexes: AsyncIndexes) -> None:
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

    await async_indexes.create(
        name="byoc-idx",
        dimension=1536,
        spec=raw_spec,
        timeout=-1,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["spec"] == raw_spec


async def test_create_byoc_missing_dimension(async_indexes: AsyncIndexes) -> None:
    """ByocSpec without dimension raises ValidationError."""
    with pytest.raises(ValidationError, match="dimension"):
        await async_indexes.create(
            name="byoc-idx",
            spec=ByocSpec(environment="aws-us-east-1-b921"),
        )


# ---------------------------------------------------------------------------
# configure()
# ---------------------------------------------------------------------------


@respx.mock
async def test_configure_replicas_only(async_indexes: AsyncIndexes) -> None:
    """PATCH body has spec.pod.replicas only; returns None."""
    route = respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    await async_indexes.configure("test-index", replicas=4)

    payload = _request_json(route)
    assert payload == {"spec": {"pod": {"replicas": 4}}}


@respx.mock
async def test_configure_pod_type_only(async_indexes: AsyncIndexes) -> None:
    """PATCH body has spec.pod.pod_type only."""
    route = respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    await async_indexes.configure("test-index", pod_type="p1.x2")

    payload = _request_json(route)
    assert payload == {"spec": {"pod": {"pod_type": "p1.x2"}}}


@respx.mock
async def test_configure_deletion_protection(async_indexes: AsyncIndexes) -> None:
    """PATCH body includes deletion_protection."""
    route = respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    await async_indexes.configure("test-index", deletion_protection="enabled")

    payload = _request_json(route)
    assert payload == {"deletion_protection": "enabled"}


@respx.mock
async def test_configure_tags_sent_directly(async_indexes: AsyncIndexes) -> None:
    """Tags are forwarded as-is; no describe pre-fetch is issued."""
    patch_route = respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    await async_indexes.configure(
        "test-index", tags={"new_tag": "hello", "existing": "overwritten"}
    )

    payload = _request_json(patch_route)
    assert payload == {"tags": {"new_tag": "hello", "existing": "overwritten"}}
    assert len(patch_route.calls) == 1, "exactly one PATCH; no GET for describe"


@respx.mock
async def test_configure_tag_removal_via_empty_string(async_indexes: AsyncIndexes) -> None:
    """Empty-string tag value is forwarded unchanged; backend handles removal."""
    patch_route = respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    await async_indexes.configure("test-index", tags={"remove_me": ""})

    payload = _request_json(patch_route)
    assert payload == {"tags": {"remove_me": ""}}
    assert len(patch_route.calls) == 1, "exactly one PATCH; no GET for describe"


@respx.mock
async def test_configure_deletion_protection_enum(async_indexes: AsyncIndexes) -> None:
    """Accept DeletionProtection enum for deletion_protection param."""
    route = respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    await async_indexes.configure("test-index", deletion_protection=DeletionProtection.ENABLED)

    payload = _request_json(route)
    assert payload == {"deletion_protection": "enabled"}


@respx.mock
async def test_configure_multiple_fields(async_indexes: AsyncIndexes) -> None:
    """Can set replicas, pod_type, and deletion_protection together."""
    route = respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    await async_indexes.configure(
        "test-index",
        replicas=4,
        pod_type="p1.x2",
        deletion_protection="disabled",
        tags={"env": "prod"},
    )

    payload = _request_json(route)
    assert payload["spec"]["pod"]["replicas"] == 4
    assert payload["spec"]["pod"]["pod_type"] == "p1.x2"
    assert payload["deletion_protection"] == "disabled"
    assert payload["tags"] == {"env": "prod"}


@respx.mock
async def test_configure_returns_none(async_indexes: AsyncIndexes) -> None:
    """configure() always returns None."""
    respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    assert await async_indexes.configure("test-index", replicas=1) is None  # type: ignore[func-returns-value]


async def test_configure_empty_name_raises(async_indexes: AsyncIndexes) -> None:
    """Empty name raises ValidationError before any HTTP call."""
    with pytest.raises(ValidationError):
        await async_indexes.configure("")


async def test_configure_invalid_deletion_protection_raises(async_indexes: AsyncIndexes) -> None:
    with pytest.raises(ValidationError, match="deletion_protection"):
        await async_indexes.configure("test-index", deletion_protection="invalid")


@respx.mock
async def test_configure_byoc_read_capacity_on_demand(
    async_indexes: AsyncIndexes,
) -> None:
    """PATCH body has spec.byoc.read_capacity with OnDemand mode."""
    route = respx.patch(f"{BASE_URL}/indexes/my-idx").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    await async_indexes.configure("my-idx", read_capacity={"mode": "OnDemand"})

    payload = _request_json(route)
    assert payload == {"spec": {"byoc": {"read_capacity": {"mode": "OnDemand"}}}}


@respx.mock
async def test_configure_byoc_read_capacity_dedicated(
    async_indexes: AsyncIndexes,
) -> None:
    """PATCH body has full dedicated read_capacity structure."""
    route = respx.patch(f"{BASE_URL}/indexes/my-idx").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    await async_indexes.configure(
        "my-idx",
        read_capacity={
            "mode": "Dedicated",
            "dedicated": {
                "node_type": "t1",
                "scaling": "Manual",
                "manual": {"replicas": 2, "shards": 1},
            },
        },
    )

    payload = _request_json(route)
    assert payload == {
        "spec": {
            "byoc": {
                "read_capacity": {
                    "mode": "Dedicated",
                    "dedicated": {
                        "node_type": "t1",
                        "scaling": "Manual",
                        "manual": {"replicas": 2, "shards": 1},
                    },
                }
            }
        }
    }


@respx.mock
async def test_configure_byoc_read_capacity_dedicated_partial_no_node_type(
    async_indexes: AsyncIndexes,
) -> None:
    """Partial Dedicated patch omitting node_type is valid and passed to the API."""
    route = respx.patch(f"{BASE_URL}/indexes/my-idx").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    rc = {"mode": "Dedicated", "dedicated": {"scaling": "Manual"}}
    await async_indexes.configure("my-idx", read_capacity=rc)

    payload = _request_json(route)
    assert payload == {"spec": {"byoc": {"read_capacity": rc}}}


@respx.mock
async def test_configure_byoc_read_capacity_dedicated_partial_no_scaling(
    async_indexes: AsyncIndexes,
) -> None:
    """Partial Dedicated patch omitting scaling is valid and passed to the API."""
    route = respx.patch(f"{BASE_URL}/indexes/my-idx").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    rc = {"mode": "Dedicated", "dedicated": {"node_type": "t1"}}
    await async_indexes.configure("my-idx", read_capacity=rc)

    payload = _request_json(route)
    assert payload == {"spec": {"byoc": {"read_capacity": rc}}}


async def test_configure_byoc_read_capacity_missing_mode(
    async_indexes: AsyncIndexes,
) -> None:
    """Missing mode key raises ValidationError."""
    with pytest.raises(ValidationError, match="mode"):
        await async_indexes.configure(
            "my-idx",
            read_capacity={"dedicated": {"node_type": "t1"}},
        )


async def test_configure_rejects_pod_fields_with_read_capacity(
    async_indexes: AsyncIndexes,
) -> None:
    """Passing both pod fields and read_capacity raises ValidationError."""
    with pytest.raises(ValidationError, match=r"pod.*read_capacity"):
        await async_indexes.configure(
            "my-idx",
            replicas=2,
            read_capacity={"mode": "OnDemand"},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _request_json(route: respx.Route) -> dict[str, Any]:
    """Extract the JSON body from the last request on a route."""
    import orjson

    request = route.calls.last.request
    return orjson.loads(request.content)  # type: ignore[no-any-return]


def _integrated_response(**overrides: object) -> dict[str, object]:
    """Return a realistic response for an integrated index."""
    return make_index_response(
        name="my-integrated-index",
        embed={
            "model": "multilingual-e5-large",
            "metric": "cosine",
            "dimension": 1024,
            "field_map": {"text": "my_text_field"},
        },
        **overrides,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# create() — IntegratedSpec
# ---------------------------------------------------------------------------


@respx.mock
async def test_create_integrated_index(async_indexes: AsyncIndexes) -> None:
    """Create with IntegratedSpec — verify correct wire format."""
    route = respx.post(f"{BASE_URL}/indexes/create-for-model").mock(
        return_value=httpx.Response(201, json=_integrated_response()),
    )

    result = await async_indexes.create(
        name="my-integrated-index",
        spec=IntegratedSpec(
            cloud="aws",
            region="us-east-1",
            embed=EmbedConfig(
                model="multilingual-e5-large",
                field_map={"text": "my_text_field"},
            ),
        ),
        timeout=-1,
    )

    assert isinstance(result, IndexModel)
    assert result.name == "my-integrated-index"

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["name"] == "my-integrated-index"
    assert body["cloud"] == "aws"
    assert body["region"] == "us-east-1"
    assert body["embed"]["model"] == "multilingual-e5-large"
    assert body["embed"]["field_map"] == {"text": "my_text_field"}
    # dimension and metric should NOT be in body (inferred by server)
    assert "dimension" not in body
    assert "metric" not in body
    # spec should NOT be in body (integrated uses flat structure)
    assert "spec" not in body


@respx.mock
async def test_create_integrated_with_metric(async_indexes: AsyncIndexes) -> None:
    """Metric override in embed config is included in request."""
    route = respx.post(f"{BASE_URL}/indexes/create-for-model").mock(
        return_value=httpx.Response(201, json=_integrated_response()),
    )

    await async_indexes.create(
        name="my-integrated-index",
        spec=IntegratedSpec(
            cloud="aws",
            region="us-east-1",
            embed=EmbedConfig(
                model="multilingual-e5-large",
                field_map={"text": "my_text_field"},
                metric="dotproduct",
            ),
        ),
        timeout=-1,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["embed"]["metric"] == "dotproduct"


@respx.mock
async def test_create_integrated_with_parameters(async_indexes: AsyncIndexes) -> None:
    """Read and write parameters are passed through."""
    route = respx.post(f"{BASE_URL}/indexes/create-for-model").mock(
        return_value=httpx.Response(201, json=_integrated_response()),
    )

    await async_indexes.create(
        name="my-integrated-index",
        spec=IntegratedSpec(
            cloud="aws",
            region="us-east-1",
            embed=EmbedConfig(
                model="multilingual-e5-large",
                field_map={"text": "my_text_field"},
                read_parameters={"input_type": "query", "truncate": "NONE"},
                write_parameters={"input_type": "passage"},
            ),
        ),
        timeout=-1,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["embed"]["read_parameters"] == {"input_type": "query", "truncate": "NONE"}
    assert body["embed"]["write_parameters"] == {"input_type": "passage"}


@respx.mock
async def test_create_integrated_with_tags(async_indexes: AsyncIndexes) -> None:
    """Tags are included in the request body."""
    route = respx.post(f"{BASE_URL}/indexes/create-for-model").mock(
        return_value=httpx.Response(201, json=_integrated_response()),
    )

    await async_indexes.create(
        name="my-integrated-index",
        spec=IntegratedSpec(
            cloud="aws",
            region="us-east-1",
            embed=EmbedConfig(
                model="multilingual-e5-large",
                field_map={"text": "my_text_field"},
            ),
        ),
        tags={"env": "test"},
        timeout=-1,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["tags"] == {"env": "test"}


@respx.mock
async def test_create_integrated_with_schema(async_indexes: AsyncIndexes) -> None:
    """schema= is forwarded in the request body for IntegratedSpec creates."""
    route = respx.post(f"{BASE_URL}/indexes/create-for-model").mock(
        return_value=httpx.Response(201, json=_integrated_response()),
    )

    await async_indexes.create(
        name="my-integrated-index",
        spec=IntegratedSpec(
            cloud="aws",
            region="us-east-1",
            embed=EmbedConfig(
                model="multilingual-e5-large",
                field_map={"text": "my_text_field"},
            ),
        ),
        schema={"genre": {"type": "str"}},
        timeout=-1,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["schema"] == {"fields": {"genre": {"type": "str"}}}


@respx.mock
async def test_create_integrated_without_schema(async_indexes: AsyncIndexes) -> None:
    """schema key is absent from the request body when not set for IntegratedSpec."""
    route = respx.post(f"{BASE_URL}/indexes/create-for-model").mock(
        return_value=httpx.Response(201, json=_integrated_response()),
    )

    await async_indexes.create(
        name="my-integrated-index",
        spec=IntegratedSpec(
            cloud="aws",
            region="us-east-1",
            embed=EmbedConfig(
                model="multilingual-e5-large",
                field_map={"text": "my_text_field"},
            ),
        ),
        timeout=-1,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert "schema" not in body


@respx.mock
async def test_create_integrated_with_read_capacity(async_indexes: AsyncIndexes) -> None:
    """read_capacity= is forwarded in the request body for IntegratedSpec creates."""
    route = respx.post(f"{BASE_URL}/indexes/create-for-model").mock(
        return_value=httpx.Response(201, json=_integrated_response()),
    )

    await async_indexes.create(
        name="my-integrated-index",
        spec=IntegratedSpec(
            cloud="aws",
            region="us-east-1",
            embed=EmbedConfig(
                model="multilingual-e5-large",
                field_map={"text": "my_text_field"},
            ),
        ),
        read_capacity={"mode": "OnDemand"},
        timeout=-1,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["read_capacity"] == {"mode": "OnDemand"}


@respx.mock
async def test_async_create_index_integrated_schema_and_read_capacity(
    async_indexes: AsyncIndexes,
) -> None:
    """schema= and read_capacity= together are both forwarded for IntegratedSpec."""
    route = respx.post(f"{BASE_URL}/indexes/create-for-model").mock(
        return_value=httpx.Response(201, json=_integrated_response()),
    )

    await async_indexes.create(
        name="my-integrated-index",
        spec=IntegratedSpec(
            cloud="aws",
            region="us-east-1",
            embed=EmbedConfig(
                model="multilingual-e5-large",
                field_map={"text": "my_text_field"},
            ),
        ),
        schema={"genre": {"type": "str"}},
        read_capacity={"mode": "OnDemand"},
        timeout=-1,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["schema"] == {"fields": {"genre": {"type": "str"}}}
    assert body["read_capacity"] == {"mode": "OnDemand"}


@respx.mock
async def test_create_integrated_with_embed_model_enum(async_indexes: AsyncIndexes) -> None:
    """EmbedModel enum values are accepted for model parameter."""
    route = respx.post(f"{BASE_URL}/indexes/create-for-model").mock(
        return_value=httpx.Response(201, json=_integrated_response()),
    )

    await async_indexes.create(
        name="my-integrated-index",
        spec=IntegratedSpec(
            cloud="aws",
            region="us-east-1",
            embed=EmbedConfig(
                model=EmbedModel.Multilingual_E5_Large,
                field_map={"text": "my_text_field"},
            ),
        ),
        timeout=-1,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["embed"]["model"] == "multilingual-e5-large"


async def test_create_integrated_missing_cloud_raises(async_indexes: AsyncIndexes) -> None:
    """Empty cloud raises ValidationError."""
    with pytest.raises(ValidationError, match="cloud"):
        await async_indexes.create(
            name="my-integrated-index",
            spec=IntegratedSpec(
                cloud="",
                region="us-east-1",
                embed=EmbedConfig(
                    model="multilingual-e5-large",
                    field_map={"text": "my_text_field"},
                ),
            ),
        )


async def test_create_integrated_missing_model_raises(async_indexes: AsyncIndexes) -> None:
    """Empty model raises ValidationError."""
    with pytest.raises(ValidationError, match="model"):
        await async_indexes.create(
            name="my-integrated-index",
            spec=IntegratedSpec(
                cloud="aws",
                region="us-east-1",
                embed=EmbedConfig(
                    model="",
                    field_map={"text": "my_text_field"},
                ),
            ),
        )


async def test_create_integrated_missing_field_map_raises(async_indexes: AsyncIndexes) -> None:
    """Empty field_map raises ValidationError."""
    with pytest.raises(ValidationError, match="field_map"):
        await async_indexes.create(
            name="my-integrated-index",
            spec=IntegratedSpec(
                cloud="aws",
                region="us-east-1",
                embed=EmbedConfig(
                    model="multilingual-e5-large",
                    field_map={},
                ),
            ),
        )


async def test_create_integrated_missing_name_raises(async_indexes: AsyncIndexes) -> None:
    """Empty name raises ValidationError."""
    with pytest.raises(ValidationError, match="name"):
        await async_indexes.create(
            name="",
            spec=IntegratedSpec(
                cloud="aws",
                region="us-east-1",
                embed=EmbedConfig(
                    model="multilingual-e5-large",
                    field_map={"text": "my_text_field"},
                ),
            ),
        )


# ---------------------------------------------------------------------------
# create() — timeout=-1 sentinel (no polling)
# ---------------------------------------------------------------------------


@respx.mock
async def test_create_timeout_minus_one_no_polling(async_indexes: AsyncIndexes) -> None:
    """timeout=-1 should skip polling, same as timeout=None."""
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201,
            json=make_index_response(status={"ready": False, "state": "Initializing"}),
        ),
    )

    result = await async_indexes.create(
        name="test-index",
        dimension=1536,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        timeout=-1,
    )

    assert result.status.ready is False
    assert len(respx.calls) == 1  # only the POST, no describe polling


@respx.mock
async def test_create_integrated_polls_until_ready(async_indexes: AsyncIndexes) -> None:
    """Integrated indexes use the same readiness polling."""
    respx.post(f"{BASE_URL}/indexes/create-for-model").mock(
        return_value=httpx.Response(
            201,
            json=_integrated_response(status={"ready": False, "state": "Initializing"}),
        ),
    )
    respx.get(f"{BASE_URL}/indexes/my-integrated-index").mock(
        side_effect=[
            httpx.Response(
                200,
                json=_integrated_response(status={"ready": False, "state": "Initializing"}),
            ),
            httpx.Response(
                200,
                json=_integrated_response(status={"ready": True, "state": "Ready"}),
            ),
        ]
    )

    with patch("pinecone.async_client.indexes.asyncio.sleep"):
        result = await async_indexes.create(
            name="my-integrated-index",
            spec=IntegratedSpec(
                cloud="aws",
                region="us-east-1",
                embed=EmbedConfig(
                    model="multilingual-e5-large",
                    field_map={"text": "my_text_field"},
                ),
            ),
            timeout=300,
        )

    assert result.status.ready is True


@respx.mock
async def test_create_integrated_no_polling_with_neg1(async_indexes: AsyncIndexes) -> None:
    """With timeout=-1, integrated create returns immediately without polling."""
    respx.post(f"{BASE_URL}/indexes/create-for-model").mock(
        return_value=httpx.Response(
            201,
            json=_integrated_response(status={"ready": False, "state": "Initializing"}),
        ),
    )

    result = await async_indexes.create(
        name="my-integrated-index",
        spec=IntegratedSpec(
            cloud="aws",
            region="us-east-1",
            embed=EmbedConfig(
                model="multilingual-e5-large",
                field_map={"text": "my_text_field"},
            ),
        ),
        timeout=-1,
    )

    assert result.status.ready is False
    assert len(respx.calls) == 1  # only the POST


# ---------------------------------------------------------------------------
# Polling edge cases
# ---------------------------------------------------------------------------


@respx.mock
async def test_create_duplicate_raises_conflict(async_indexes: AsyncIndexes) -> None:
    """409 response raises ConflictError."""
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            409,
            json=make_error_response(409, "Index already exists"),
        ),
    )

    with pytest.raises(ConflictError):
        await async_indexes.create(
            name="existing-index",
            dimension=1536,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=-1,
        )


@respx.mock
async def test_polling_sleep_interval(async_indexes: AsyncIndexes) -> None:
    """Verify asyncio.sleep is called with exactly _POLL_INTERVAL_SECONDS (5)."""
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

    with patch("pinecone.async_client.indexes.asyncio.sleep") as mock_sleep:
        await async_indexes.create(
            name="test-index",
            dimension=1536,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

    assert mock_sleep.call_count == 1
    mock_sleep.assert_called_with(_POLL_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# Legacy-parity tests (BC-0074): PineconeValueError for client-side validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_index_invalid_name_raises_value_error(
    async_indexes: AsyncIndexes,
) -> None:
    """Uppercase characters in index name are rejected client-side."""
    with pytest.raises(PineconeValueError, match="lowercase"):
        await async_indexes.create(
            name="Invalid-name",
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


@pytest.mark.asyncio
async def test_create_index_invalid_metric_raises_value_error(
    async_indexes: AsyncIndexes,
) -> None:
    """Unsupported metric values are rejected client-side."""
    with pytest.raises(PineconeValueError, match="metric must be one of"):
        await async_indexes.create(
            name="valid-name",
            dimension=2,
            metric="banana",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


@pytest.mark.asyncio
async def test_create_index_missing_dimension_raises_value_error(
    async_indexes: AsyncIndexes,
) -> None:
    """Dense indexes require an explicit dimension."""
    with pytest.raises(PineconeValueError, match="dimension is required"):
        await async_indexes.create(
            name="valid-name",
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
