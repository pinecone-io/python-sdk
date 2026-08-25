"""Unit tests for AsyncIndexes.create() — 2026-07 schema-based index creation.

Async mirror of tests/unit/client/test_indexes_create.py: body serialization
against the db_control 2026-07 spec examples, client-side validation quality,
the guided hard-break interception of 2025-10 kwargs, readiness polling, and
verbatim surfacing of server 400s.
"""

from __future__ import annotations

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
from pinecone.async_client.indexes import AsyncIndexes
from pinecone.errors.exceptions import (
    ApiError,
    PineconeTypeError,
    PineconeValueError,
)
from pinecone.models.indexes.index import IndexModel
from pinecone.models.indexes.specs import (
    ByocSpec,
    EmbedConfig,
    IntegratedSpec,
    PodSpec,
    ServerlessSpec,
)
from tests.factories import make_error_response, make_index_response

BASE_URL = "https://api.test.pinecone.io"

DENSE_SCHEMA: dict[str, Any] = {
    "fields": {"embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}
}
MANAGED_AWS: dict[str, Any] = {
    "deployment_type": "managed",
    "cloud": "aws",
    "region": "us-east-1",
}


@pytest.fixture
async def async_http_client() -> AsyncGenerator[AsyncHTTPClient]:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    client = AsyncHTTPClient(config, CONTROL_PLANE_API_VERSION)
    yield client
    await client.close()


@pytest.fixture
def indexes(async_http_client: AsyncHTTPClient) -> AsyncIndexes:
    return AsyncIndexes(http=async_http_client)


def _mock_created(name: str = "test-index") -> httpx.Response:
    return httpx.Response(201, json=make_index_response(name=name))


# ---------------------------------------------------------------------------
# Wire basics
# ---------------------------------------------------------------------------


@respx.mock
async def test_create_sends_post_with_configured_api_version(indexes: AsyncIndexes) -> None:
    """POST /indexes carries the SDK's control-plane version header."""
    route = respx.post(f"{BASE_URL}/indexes").mock(return_value=_mock_created())

    await indexes.create(name="test-index", schema=DENSE_SCHEMA, timeout=-1)

    request = route.calls.last.request
    assert request.url.path == "/indexes"
    assert request.headers.get("X-Pinecone-Api-Version") == CONTROL_PLANE_API_VERSION
    assert request.headers.get("Content-Type") == "application/json"


@respx.mock
async def test_create_minimal_body_has_only_schema(indexes: AsyncIndexes) -> None:
    """Only provided kwargs appear in the body; no null-valued keys."""
    route = respx.post(f"{BASE_URL}/indexes").mock(return_value=_mock_created())

    await indexes.create(schema=DENSE_SCHEMA, timeout=-1)

    body = json.loads(route.calls.last.request.content)
    assert body == {"schema": DENSE_SCHEMA}


@respx.mock
async def test_create_returns_index_model(indexes: AsyncIndexes) -> None:
    route = respx.post(f"{BASE_URL}/indexes").mock(return_value=_mock_created())

    result = await indexes.create(name="test-index", schema=DENSE_SCHEMA, timeout=-1)

    assert route.called
    assert isinstance(result, IndexModel)
    assert result.name == "test-index"


@respx.mock
async def test_create_full_body(indexes: AsyncIndexes) -> None:
    """All optional parameters serialize into the request body."""
    route = respx.post(f"{BASE_URL}/indexes").mock(return_value=_mock_created())

    await indexes.create(
        schema=DENSE_SCHEMA,
        name="my-index",
        deployment=MANAGED_AWS,
        read_capacity={"mode": "OnDemand"},
        deletion_protection="enabled",
        tags={"env": "prod"},
        cmek_id="arn:aws:kms:us-east-1:123456789012:key/mrk-abc123",
        timeout=-1,
    )

    body = json.loads(route.calls.last.request.content)
    assert body == {
        "schema": DENSE_SCHEMA,
        "name": "my-index",
        "deployment": MANAGED_AWS,
        "read_capacity": {"mode": "OnDemand"},
        "deletion_protection": "enabled",
        "tags": {"env": "prod"},
        "cmek_id": "arn:aws:kms:us-east-1:123456789012:key/mrk-abc123",
    }


# ---------------------------------------------------------------------------
# Spec-example bodies (db_control_2026-07.oas.yaml, create_index examples).
# The seventh example (serverless-restore-from-backup) is deliberately not
# expressible: source_backup_id is rejected client-side per question #144.
# ---------------------------------------------------------------------------

_SPEC_EXAMPLES: dict[str, dict[str, Any]] = {
    "dense-serverless": {
        "deletion_protection": "enabled",
        "deployment": {"cloud": "aws", "deployment_type": "managed", "region": "us-east-1"},
        "name": "movie-recommendations",
        "schema": {
            "fields": {"embedding": {"dimension": 1536, "metric": "cosine", "type": "dense_vector"}}
        },
    },
    "sparse-serverless": {
        "deployment": {"cloud": "aws", "deployment_type": "managed", "region": "us-east-1"},
        "name": "sparse-index",
        "schema": {"fields": {"sparse_embedding": {"type": "sparse_vector"}}},
    },
    "hybrid-serverless": {
        "deployment": {"cloud": "gcp", "deployment_type": "managed", "region": "us-central1"},
        "name": "hybrid-index",
        "schema": {
            "fields": {
                "embedding": {"dimension": 768, "metric": "dotproduct", "type": "dense_vector"},
                "sparse_embedding": {"type": "sparse_vector"},
            }
        },
    },
    "fts-serverless": {
        "deployment": {"cloud": "aws", "deployment_type": "managed", "region": "us-east-1"},
        "name": "fts-index",
        "schema": {
            "fields": {
                "body": {
                    "full_text_search": {"language": "en", "stemming": True, "stop_words": True},
                    "type": "string",
                }
            }
        },
    },
    "semantic-text-serverless": {
        "deployment": {"cloud": "aws", "deployment_type": "managed", "region": "us-east-1"},
        "name": "semantic-index",
        "schema": {
            "fields": {"content": {"model": "multilingual-e5-large", "type": "semantic_text"}}
        },
    },
    "dedicated-read-capacity": {
        "deployment": {"cloud": "aws", "deployment_type": "managed", "region": "us-east-1"},
        "name": "dedicated-index",
        "read_capacity": {
            "dedicated": {
                "manual": {"replicas": 2, "shards": 2},
                "node_type": "t1",
                "scaling": "Manual",
            },
            "mode": "Dedicated",
        },
        "schema": {
            "fields": {"embedding": {"dimension": 1536, "metric": "cosine", "type": "dense_vector"}}
        },
    },
}


@respx.mock
@pytest.mark.parametrize("example_name", sorted(_SPEC_EXAMPLES))
async def test_create_emits_spec_example_body(indexes: AsyncIndexes, example_name: str) -> None:
    """create() reproduces each create_index spec example body exactly."""
    example = _SPEC_EXAMPLES[example_name]
    route = respx.post(f"{BASE_URL}/indexes").mock(return_value=_mock_created(name=example["name"]))

    kwargs: dict[str, Any] = {
        "name": example["name"],
        "schema": example["schema"],
        "deployment": example["deployment"],
        "timeout": -1,
    }
    if "deletion_protection" in example:
        kwargs["deletion_protection"] = example["deletion_protection"]
    if "read_capacity" in example:
        kwargs["read_capacity"] = example["read_capacity"]

    await indexes.create(**kwargs)

    body = json.loads(route.calls.last.request.content)
    assert body == example


async def test_create_restore_from_backup_example_is_guided_away(indexes: AsyncIndexes) -> None:
    """The seventh spec example (source_backup_id) is rejected client-side.

    The 2026-07 backend rejects any source_backup_id/source_collection with
    400 'Creating an index from collection or backup is not yet supported'
    (question #144); the SDK omits the kwargs and points at
    create_index_from_backup instead.
    """
    with pytest.raises(PineconeTypeError, match="create_index_from_backup"):
        await indexes.create(
            name="xr-restore-test",
            schema=DENSE_SCHEMA,
            deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-west-2"},
            source_backup_id="670e8400-e29b-41d4-a716-446655440000",
        )


async def test_create_source_collection_is_guided_away(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeTypeError, match="not yet supported"):
        await indexes.create(schema=DENSE_SCHEMA, source_collection="movie-embeddings")


# ---------------------------------------------------------------------------
# Deprecated sugar (#500): spec=/dimension=/metric=/vector_type= translate
# into schema=/deployment=/read_capacity=.
# ---------------------------------------------------------------------------


@respx.mock
async def test_create_legacy_dimension_metric_spec_translates_to_dense_schema_and_deployment(
    indexes: AsyncIndexes,
) -> None:
    route = respx.post(f"{BASE_URL}/indexes").mock(return_value=_mock_created(name="movies"))

    await indexes.create(
        name="movies",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        timeout=-1,
    )

    body = json.loads(route.calls.last.request.content)
    assert body == {
        "name": "movies",
        "schema": {
            "fields": {"_values": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}
        },
        "deployment": {"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
    }


@respx.mock
async def test_create_legacy_metric_defaults_to_cosine_when_omitted(
    indexes: AsyncIndexes,
) -> None:
    route = respx.post(f"{BASE_URL}/indexes").mock(return_value=_mock_created())

    await indexes.create(
        dimension=8, spec=ServerlessSpec(cloud="aws", region="us-east-1"), timeout=-1
    )

    body = json.loads(route.calls.last.request.content)
    assert body["schema"]["fields"]["_values"]["metric"] == "cosine"


@respx.mock
async def test_create_legacy_sparse_vector_type_translates_to_sparse_schema(
    indexes: AsyncIndexes,
) -> None:
    route = respx.post(f"{BASE_URL}/indexes").mock(return_value=_mock_created(name="sparse"))

    await indexes.create(
        name="sparse",
        vector_type="sparse",
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        timeout=-1,
    )

    body = json.loads(route.calls.last.request.content)
    assert body["schema"] == {"fields": {"_sparse_values": {"type": "sparse_vector"}}}


@respx.mock
async def test_create_legacy_pod_spec_translates_to_pod_deployment(
    indexes: AsyncIndexes,
) -> None:
    route = respx.post(f"{BASE_URL}/indexes").mock(return_value=_mock_created(name="pods"))

    await indexes.create(
        name="pods",
        dimension=8,
        spec=PodSpec(environment="us-east1-gcp", pod_type="p1.x2", replicas=2, shards=3),
        timeout=-1,
    )

    body = json.loads(route.calls.last.request.content)
    assert body["deployment"] == {
        "deployment_type": "pod",
        "environment": "us-east1-gcp",
        "pod_type": "p1.x2",
        "replicas": 2,
        "shards": 3,
    }


@respx.mock
async def test_create_legacy_byoc_spec_translates_to_byoc_deployment(
    indexes: AsyncIndexes,
) -> None:
    route = respx.post(f"{BASE_URL}/indexes").mock(return_value=_mock_created(name="byoc"))

    await indexes.create(
        name="byoc", dimension=8, spec=ByocSpec(environment="aws-us-east-1-b921"), timeout=-1
    )

    body = json.loads(route.calls.last.request.content)
    assert body["deployment"] == {"deployment_type": "byoc", "environment": "aws-us-east-1-b921"}


@respx.mock
async def test_create_legacy_dict_spec_translates_to_deployment(indexes: AsyncIndexes) -> None:
    route = respx.post(f"{BASE_URL}/indexes").mock(return_value=_mock_created(name="movies"))

    await indexes.create(
        name="movies",
        dimension=3,
        spec={"serverless": {"cloud": "aws", "region": "eu-west-1"}},
        timeout=-1,
    )

    body = json.loads(route.calls.last.request.content)
    assert body["deployment"] == {
        "deployment_type": "managed",
        "cloud": "aws",
        "region": "eu-west-1",
    }


@respx.mock
async def test_create_legacy_spec_read_capacity_lands_at_top_level(
    indexes: AsyncIndexes,
) -> None:
    route = respx.post(f"{BASE_URL}/indexes").mock(return_value=_mock_created())

    await indexes.create(
        dimension=8,
        spec=ServerlessSpec(cloud="aws", region="us-east-1", read_capacity={"mode": "OnDemand"}),
        timeout=-1,
    )

    body = json.loads(route.calls.last.request.content)
    assert body["read_capacity"] == {"mode": "OnDemand"}
    assert "read_capacity" not in body["deployment"]


@respx.mock
async def test_create_legacy_read_capacity_kwarg_overrides_spec_read_capacity(
    indexes: AsyncIndexes,
) -> None:
    route = respx.post(f"{BASE_URL}/indexes").mock(return_value=_mock_created())

    await indexes.create(
        dimension=8,
        spec=ServerlessSpec(cloud="aws", region="us-east-1", read_capacity={"mode": "OnDemand"}),
        read_capacity={"mode": "Dedicated", "dedicated": {"node_type": "t1"}},
        timeout=-1,
    )

    body = json.loads(route.calls.last.request.content)
    assert body["read_capacity"] == {"mode": "Dedicated", "dedicated": {"node_type": "t1"}}


# ---------------------------------------------------------------------------
# Guided hard break: kwargs with no faithful translation
# ---------------------------------------------------------------------------


async def test_create_legacy_integrated_spec_redirects_to_create_for_model(
    indexes: AsyncIndexes,
) -> None:
    with pytest.raises(PineconeTypeError) as exc_info:
        await indexes.create(
            name="semantic",
            spec=IntegratedSpec(
                cloud="aws",
                region="us-east-1",
                embed=EmbedConfig(model="multilingual-e5-large", field_map={"text": "chunk_text"}),
            ),
        )

    message = str(exc_info.value)
    assert "create_for_model" in message
    assert "'multilingual-e5-large'" in message
    assert "'chunk_text'" in message


async def test_create_legacy_pods_kwarg_notes_no_equivalent(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeTypeError, match="pods=: capacity is replicas x shards"):
        await indexes.create(  # type: ignore[call-arg]
            name="pods", dimension=3, spec=PodSpec(environment="us-east1-gcp"), pods=4
        )


async def test_create_legacy_metadata_config_kwarg_notes_no_equivalent(
    indexes: AsyncIndexes,
) -> None:
    with pytest.raises(PineconeTypeError, match="metadata_config=: metadata fields"):
        await indexes.create(  # type: ignore[call-arg]
            schema=DENSE_SCHEMA, metadata_config={"indexed": ["genre"]}
        )


async def test_create_unknown_kwarg_lists_accepted_arguments(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeTypeError, match="unexpected keyword argument"):
        await indexes.create(schema=DENSE_SCHEMA, shcema_typo=True)  # type: ignore[call-arg]


async def test_create_unknown_kwarg_not_absorbed_by_legacy_sugar(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeTypeError, match=r"\['dimenson'\]"):
        await indexes.create(name="x", dimenson=3)  # type: ignore[call-arg]


async def test_create_quarantined_kwargs_rejected_before_any_request(
    indexes: AsyncIndexes,
) -> None:
    with respx.mock:
        route = respx.post(f"{BASE_URL}/indexes")
        with pytest.raises(PineconeTypeError):
            await indexes.create(name="x", dimension=3, pods=4)  # type: ignore[call-arg]
        assert route.call_count == 0


# ---------------------------------------------------------------------------
# Conflicts and collisions between the deprecated sugar and the 2026-07 args
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "legacy_kwargs",
    [
        {"dimension": 3},
        {"metric": "cosine"},
        {"vector_type": "dense"},
        {"dimension": 3, "metric": "cosine", "vector_type": "dense"},
    ],
)
async def test_create_schema_with_legacy_vector_kwarg_raises_naming_both(
    indexes: AsyncIndexes, legacy_kwargs: dict[str, Any]
) -> None:
    with pytest.raises(PineconeValueError, match="schema=") as exc_info:
        await indexes.create(schema=DENSE_SCHEMA, **legacy_kwargs)
    message = str(exc_info.value)
    for kwarg in legacy_kwargs:
        assert f"{kwarg}=" in message


async def test_create_deployment_with_spec_raises_naming_both(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeValueError, match="deployment=") as exc_info:
        await indexes.create(
            deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    assert "spec=" in str(exc_info.value)


async def test_create_neither_schema_nor_legacy_kwargs_extends_schema_required_message(
    indexes: AsyncIndexes,
) -> None:
    with pytest.raises(PineconeValueError, match="schema is required") as exc_info:
        await indexes.create(name="x")
    assert "dimension=" in str(exc_info.value)


@respx.mock
async def test_create_metric_never_injected_into_schema_call(indexes: AsyncIndexes) -> None:
    route = respx.post(f"{BASE_URL}/indexes").mock(return_value=_mock_created())

    await indexes.create(schema=DENSE_SCHEMA, timeout=-1)

    body = json.loads(route.calls.last.request.content)
    assert body == {"schema": DENSE_SCHEMA}


async def test_create_old_style_metadata_schema_raises_guided_message_not_a_400(
    indexes: AsyncIndexes,
) -> None:
    with pytest.raises(PineconeValueError, match=r"9\.x metadata schema") as exc_info:
        await indexes.create(schema={"fields": {"genre": {"filterable": True}}})
    message = str(exc_info.value)
    assert "indexed automatically at upsert" in message
    assert "docs/migration/v10-2026-07-index-model.md" in message


@respx.mock
async def test_create_spec_alone_without_dimension_still_requires_schema(
    indexes: AsyncIndexes,
) -> None:
    route = respx.post(f"{BASE_URL}/indexes")
    with pytest.raises(PineconeValueError, match="schema is required"):
        await indexes.create(spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    assert route.call_count == 0


# ---------------------------------------------------------------------------
# Validation quality: field name and limit in every message
# ---------------------------------------------------------------------------


async def test_create_missing_schema_raises(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeValueError, match="schema is required"):
        await indexes.create(name="x")


async def test_create_empty_schema_raises(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeValueError, match="schema cannot be an empty dict"):
        await indexes.create(schema={})


async def test_create_bare_field_map_schema_raises_with_fields_hint(indexes: AsyncIndexes) -> None:
    """The old flat metadata-schema shape gets a specific error naming 'fields'."""
    with pytest.raises(PineconeValueError, match="'fields'"):
        await indexes.create(schema={"genre": {"filterable": True}})


async def test_create_empty_fields_raises(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeValueError, match="at least one searched field"):
        await indexes.create(schema={"fields": {}})


@pytest.mark.parametrize(
    ("bad_name", "match"),
    [
        ("", "non-empty"),
        ("a" * 46, "too long"),
        ("UpperCase", "invalid characters"),
        ("-leading", "must not start with a hyphen"),
        ("trailing-", "must not end with a hyphen"),
    ],
)
async def test_create_invalid_name_raises_with_rule(
    indexes: AsyncIndexes, bad_name: str, match: str
) -> None:
    with pytest.raises(PineconeValueError, match=match):
        await indexes.create(name=bad_name, schema=DENSE_SCHEMA)


@respx.mock
async def test_create_underscore_prefixed_schema_field_name_reaches_server_unmodified(
    indexes: AsyncIndexes,
) -> None:
    route = respx.post(f"{BASE_URL}/indexes").mock(return_value=_mock_created())

    await indexes.create(
        schema={"fields": {"_values": {"type": "dense_vector", "dimension": 3}}}, timeout=-1
    )

    body = json.loads(route.calls.last.request.content)
    assert body["schema"]["fields"]["_values"] == {"type": "dense_vector", "dimension": 3}


async def test_create_empty_deployment_raises(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeValueError, match="deployment cannot be an empty dict"):
        await indexes.create(schema=DENSE_SCHEMA, deployment={})


async def test_create_empty_read_capacity_raises(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeValueError, match="read_capacity cannot be an empty dict"):
        await indexes.create(schema=DENSE_SCHEMA, read_capacity={})


async def test_create_empty_tags_raises(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeValueError, match="tags cannot be an empty dict"):
        await indexes.create(schema=DENSE_SCHEMA, tags={})


async def test_create_more_than_twenty_tags_raises(indexes: AsyncIndexes) -> None:
    tags = {f"key{i}": "v" for i in range(21)}
    with pytest.raises(PineconeValueError, match="maximum of 20"):
        await indexes.create(schema=DENSE_SCHEMA, tags=tags)


async def test_create_tag_key_too_long_raises(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeValueError, match="80-character limit"):
        await indexes.create(schema=DENSE_SCHEMA, tags={"k" * 81: "v"})


async def test_create_tag_value_non_ascii_raises(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeValueError, match="printable ASCII"):
        await indexes.create(schema=DENSE_SCHEMA, tags={"key": "café"})


async def test_create_invalid_deletion_protection_raises(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeValueError, match="deletion_protection"):
        await indexes.create(schema=DENSE_SCHEMA, deletion_protection="on")


@pytest.mark.parametrize("deployment_type", ["serverless", "MANAGED", "Pod"])
async def test_create_invalid_deployment_type_raises(
    indexes: AsyncIndexes, deployment_type: str
) -> None:
    with pytest.raises(PineconeValueError, match="deployment_type") as exc_info:
        await indexes.create(schema=DENSE_SCHEMA, deployment={"deployment_type": deployment_type})
    assert type(exc_info.value) is PineconeValueError


@pytest.mark.parametrize(
    "field_name",
    ["", "f" * 65],
    ids=["empty", "over-64-chars"],
)
async def test_create_schema_field_name_rule_raises_pinecone_value_error(
    indexes: AsyncIndexes, field_name: str
) -> None:
    with pytest.raises(PineconeValueError, match="Invalid schema field name") as exc_info:
        await indexes.create(
            schema={"fields": {field_name: {"type": "dense_vector", "dimension": 3}}}
        )
    assert type(exc_info.value) is PineconeValueError


# ---------------------------------------------------------------------------
# No client-side schema normalization: server 400s are surfaced verbatim
# ---------------------------------------------------------------------------


@respx.mock
async def test_create_bare_string_field_reaches_server_unmodified(indexes: AsyncIndexes) -> None:
    """No filterable:true is injected; the server's 400 text is surfaced."""
    server_message = (
        "The schema only accepts fields used for search: dense_vector, "
        "sparse_vector, or string with full_text_search"
    )
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(400, json=make_error_response(400, server_message))
    )

    with pytest.raises(ApiError, match="only accepts fields used for search"):
        await indexes.create(schema={"fields": {"title": {"type": "string"}}})

    body = json.loads(route.calls.last.request.content)
    assert body["schema"]["fields"]["title"] == {"type": "string"}


@respx.mock
async def test_create_semantic_text_rejection_surfaced_verbatim(indexes: AsyncIndexes) -> None:
    """semantic_text fields pass through and the backend rejection is surfaced (#145)."""
    server_message = "semantic_text fields are not supported in schema"
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(400, json=make_error_response(400, server_message))
    )

    with pytest.raises(ApiError, match="semantic_text fields are not supported"):
        await indexes.create(
            schema={
                "fields": {"content": {"type": "semantic_text", "model": "multilingual-e5-large"}}
            }
        )


# ---------------------------------------------------------------------------
# Readiness polling: poll-until-ready stays the default
# ---------------------------------------------------------------------------


@respx.mock
async def test_create_polls_until_ready_by_default(indexes: AsyncIndexes) -> None:
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201, json=make_index_response(status={"ready": False, "state": "Initializing"})
        )
    )
    describe_route = respx.get(f"{BASE_URL}/indexes/test-index").mock(
        side_effect=[
            httpx.Response(
                200, json=make_index_response(status={"ready": False, "state": "Initializing"})
            ),
            httpx.Response(200, json=make_index_response(status={"ready": True, "state": "Ready"})),
        ]
    )

    with patch("pinecone._internal.indexes_helpers.asyncio.sleep", new_callable=AsyncMock):
        result = await indexes.create(name="test-index", schema=DENSE_SCHEMA)

    assert describe_route.call_count == 2
    assert result.status.ready is True


@respx.mock
async def test_create_polls_by_server_assigned_name_when_name_omitted(
    indexes: AsyncIndexes,
) -> None:
    """When the server assigns the name, polling uses the name from the response."""
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201,
            json=make_index_response(
                name="srv-assigned", status={"ready": False, "state": "Initializing"}
            ),
        )
    )
    describe_route = respx.get(f"{BASE_URL}/indexes/srv-assigned").mock(
        return_value=httpx.Response(
            200,
            json=make_index_response(name="srv-assigned", status={"ready": True, "state": "Ready"}),
        )
    )

    with patch("pinecone._internal.indexes_helpers.asyncio.sleep", new_callable=AsyncMock):
        result = await indexes.create(schema=DENSE_SCHEMA)

    assert describe_route.called
    assert result.name == "srv-assigned"


@respx.mock
async def test_create_timeout_negative_one_skips_polling(indexes: AsyncIndexes) -> None:
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201, json=make_index_response(status={"ready": False, "state": "Initializing"})
        )
    )
    describe_route = respx.get(f"{BASE_URL}/indexes/test-index")

    result = await indexes.create(name="test-index", schema=DENSE_SCHEMA, timeout=-1)

    assert result.status.ready is False
    assert describe_route.call_count == 0
