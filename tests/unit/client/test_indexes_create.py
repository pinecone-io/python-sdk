"""Unit tests for Indexes.create() — 2026-07 schema-based index creation.

Covers body serialization against the db_control 2026-07 spec examples,
client-side validation quality, the guided hard-break interception of
2025-10 kwargs, readiness polling, and verbatim surfacing of server 400s.
"""

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
def http_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, CONTROL_PLANE_API_VERSION)


@pytest.fixture
def indexes(http_client: HTTPClient) -> Indexes:
    return Indexes(http=http_client)


def _mock_created(name: str = "test-index") -> httpx.Response:
    return httpx.Response(201, json=make_index_response(name=name))


# ---------------------------------------------------------------------------
# Wire basics
# ---------------------------------------------------------------------------


@respx.mock
def test_create_sends_post_with_configured_api_version(indexes: Indexes) -> None:
    """POST /indexes carries the SDK's control-plane version header."""
    route = respx.post(f"{BASE_URL}/indexes").mock(return_value=_mock_created())

    indexes.create(name="test-index", schema=DENSE_SCHEMA, timeout=-1)

    request = route.calls.last.request
    assert request.url.path == "/indexes"
    assert request.headers.get("X-Pinecone-Api-Version") == CONTROL_PLANE_API_VERSION
    assert request.headers.get("Content-Type") == "application/json"


@respx.mock
def test_create_minimal_body_has_only_schema(indexes: Indexes) -> None:
    """Only provided kwargs appear in the body; no null-valued keys."""
    route = respx.post(f"{BASE_URL}/indexes").mock(return_value=_mock_created())

    indexes.create(schema=DENSE_SCHEMA, timeout=-1)

    body = json.loads(route.calls.last.request.content)
    assert body == {"schema": DENSE_SCHEMA}


@respx.mock
def test_create_returns_index_model(indexes: Indexes) -> None:
    route = respx.post(f"{BASE_URL}/indexes").mock(return_value=_mock_created())

    result = indexes.create(name="test-index", schema=DENSE_SCHEMA, timeout=-1)

    assert route.called
    assert isinstance(result, IndexModel)
    assert result.name == "test-index"


@respx.mock
def test_create_full_body(indexes: Indexes) -> None:
    """All optional parameters serialize into the request body."""
    route = respx.post(f"{BASE_URL}/indexes").mock(return_value=_mock_created())

    indexes.create(
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
def test_create_emits_spec_example_body(indexes: Indexes, example_name: str) -> None:
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

    indexes.create(**kwargs)

    body = json.loads(route.calls.last.request.content)
    assert body == example


def test_create_restore_from_backup_example_is_guided_away(indexes: Indexes) -> None:
    """The seventh spec example (source_backup_id) is rejected client-side.

    The 2026-07 backend rejects any source_backup_id/source_collection with
    400 'Creating an index from collection or backup is not yet supported'
    (question #144); the SDK omits the kwargs and points at
    create_index_from_backup instead.
    """
    with pytest.raises(PineconeTypeError, match="create_index_from_backup"):
        indexes.create(
            name="xr-restore-test",
            schema=DENSE_SCHEMA,
            deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-west-2"},
            source_backup_id="670e8400-e29b-41d4-a716-446655440000",
        )


def test_create_source_collection_is_guided_away(indexes: Indexes) -> None:
    with pytest.raises(PineconeTypeError, match="not yet supported"):
        indexes.create(schema=DENSE_SCHEMA, source_collection="movie-embeddings")


# ---------------------------------------------------------------------------
# Guided hard break: 2025-10 kwargs raise with the equivalent 2026-07 call
# ---------------------------------------------------------------------------


def test_create_legacy_serverless_spec_raises_with_translation(indexes: Indexes) -> None:
    with pytest.raises(PineconeTypeError) as exc_info:
        indexes.create(  # type: ignore[call-arg]
            name="movies",
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    message = str(exc_info.value)
    assert "'type': 'dense_vector', 'dimension': 1536, 'metric': 'cosine'" in message
    assert "'deployment_type': 'managed', 'cloud': 'aws', 'region': 'us-east-1'" in message
    assert "docs/migration/v10-2026-07-index-model.md" in message


def test_create_legacy_pod_spec_raises_with_translation(indexes: Indexes) -> None:
    with pytest.raises(PineconeTypeError) as exc_info:
        indexes.create(  # type: ignore[call-arg]
            name="pods",
            dimension=8,
            spec=PodSpec(environment="us-east1-gcp", pod_type="p1.x2", replicas=2, shards=3),
        )

    message = str(exc_info.value)
    assert "'deployment_type': 'pod'" in message
    assert "'environment': 'us-east1-gcp'" in message
    assert "'replicas': 2, 'shards': 3" in message


def test_create_legacy_byoc_spec_raises_with_translation(indexes: Indexes) -> None:
    with pytest.raises(PineconeTypeError, match="'deployment_type': 'byoc'"):
        indexes.create(  # type: ignore[call-arg]
            name="byoc", dimension=8, spec=ByocSpec(environment="aws-us-east-1-b921")
        )


def test_create_legacy_dict_spec_raises_with_translation(indexes: Indexes) -> None:
    with pytest.raises(PineconeTypeError, match="'region': 'eu-west-1'"):
        indexes.create(  # type: ignore[call-arg]
            name="movies",
            dimension=3,
            spec={"serverless": {"cloud": "aws", "region": "eu-west-1"}},
        )


def test_create_legacy_sparse_vector_type_raises_with_sparse_snippet(indexes: Indexes) -> None:
    with pytest.raises(PineconeTypeError, match="sparse_vector"):
        indexes.create(  # type: ignore[call-arg]
            name="sparse",
            vector_type="sparse",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


def test_create_legacy_integrated_spec_redirects_to_create_for_model(indexes: Indexes) -> None:
    with pytest.raises(PineconeTypeError) as exc_info:
        indexes.create(  # type: ignore[call-arg]
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


def test_create_legacy_pods_kwarg_notes_no_equivalent(indexes: Indexes) -> None:
    with pytest.raises(PineconeTypeError, match="pods= has no 2026-07 equivalent"):
        indexes.create(  # type: ignore[call-arg]
            name="pods", dimension=3, spec=PodSpec(environment="us-east1-gcp"), pods=4
        )


def test_create_unknown_kwarg_lists_accepted_arguments(indexes: Indexes) -> None:
    with pytest.raises(PineconeTypeError, match="unexpected keyword argument"):
        indexes.create(schema=DENSE_SCHEMA, shcema_typo=True)  # type: ignore[call-arg]


def test_create_legacy_kwargs_rejected_before_any_request(indexes: Indexes) -> None:
    """Interception happens client-side: no HTTP request is made."""
    with respx.mock:
        route = respx.post(f"{BASE_URL}/indexes")
        with pytest.raises(PineconeTypeError):
            indexes.create(name="x", dimension=3, spec={"serverless": {}})  # type: ignore[call-arg]
        assert route.call_count == 0


# ---------------------------------------------------------------------------
# Validation quality: field name and limit in every message
# ---------------------------------------------------------------------------


def test_create_missing_schema_raises(indexes: Indexes) -> None:
    with pytest.raises(PineconeValueError, match="schema is required"):
        indexes.create(name="x")


def test_create_empty_schema_raises(indexes: Indexes) -> None:
    with pytest.raises(PineconeValueError, match="schema cannot be an empty dict"):
        indexes.create(schema={})


def test_create_bare_field_map_schema_raises_with_fields_hint(indexes: Indexes) -> None:
    """The old flat metadata-schema shape gets a specific error naming 'fields'."""
    with pytest.raises(PineconeValueError, match="'fields'"):
        indexes.create(schema={"genre": {"filterable": True}})


def test_create_empty_fields_raises(indexes: Indexes) -> None:
    with pytest.raises(PineconeValueError, match="at least one searched field"):
        indexes.create(schema={"fields": {}})


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
def test_create_invalid_name_raises_with_rule(indexes: Indexes, bad_name: str, match: str) -> None:
    with pytest.raises(PineconeValueError, match=match):
        indexes.create(name=bad_name, schema=DENSE_SCHEMA)


def test_create_invalid_schema_field_name_raises(indexes: Indexes) -> None:
    with pytest.raises(ValueError, match="_values"):
        indexes.create(schema={"fields": {"_values": {"type": "dense_vector", "dimension": 3}}})


def test_create_empty_deployment_raises(indexes: Indexes) -> None:
    with pytest.raises(PineconeValueError, match="deployment cannot be an empty dict"):
        indexes.create(schema=DENSE_SCHEMA, deployment={})


def test_create_empty_read_capacity_raises(indexes: Indexes) -> None:
    with pytest.raises(PineconeValueError, match="read_capacity cannot be an empty dict"):
        indexes.create(schema=DENSE_SCHEMA, read_capacity={})


def test_create_empty_tags_raises(indexes: Indexes) -> None:
    with pytest.raises(PineconeValueError, match="tags cannot be an empty dict"):
        indexes.create(schema=DENSE_SCHEMA, tags={})


def test_create_more_than_twenty_tags_raises(indexes: Indexes) -> None:
    tags = {f"key{i}": "v" for i in range(21)}
    with pytest.raises(PineconeValueError, match="maximum of 20"):
        indexes.create(schema=DENSE_SCHEMA, tags=tags)


def test_create_tag_key_too_long_raises(indexes: Indexes) -> None:
    with pytest.raises(PineconeValueError, match="80-character limit"):
        indexes.create(schema=DENSE_SCHEMA, tags={"k" * 81: "v"})


def test_create_tag_value_non_ascii_raises(indexes: Indexes) -> None:
    with pytest.raises(PineconeValueError, match="printable ASCII"):
        indexes.create(schema=DENSE_SCHEMA, tags={"key": "café"})


def test_create_invalid_deletion_protection_raises(indexes: Indexes) -> None:
    with pytest.raises(PineconeValueError, match="deletion_protection"):
        indexes.create(schema=DENSE_SCHEMA, deletion_protection="on")


def test_create_invalid_deployment_type_raises(indexes: Indexes) -> None:
    with pytest.raises(ValueError, match="deployment_type"):
        indexes.create(schema=DENSE_SCHEMA, deployment={"deployment_type": "serverless"})


# ---------------------------------------------------------------------------
# No client-side schema normalization: server 400s are surfaced verbatim
# ---------------------------------------------------------------------------


@respx.mock
def test_create_bare_string_field_reaches_server_unmodified(indexes: Indexes) -> None:
    """No filterable:true is injected; the server's 400 text is surfaced."""
    server_message = (
        "The schema only accepts fields used for search: dense_vector, "
        "sparse_vector, or string with full_text_search"
    )
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(400, json=make_error_response(400, server_message))
    )

    with pytest.raises(ApiError, match="only accepts fields used for search"):
        indexes.create(schema={"fields": {"title": {"type": "string"}}})

    body = json.loads(route.calls.last.request.content)
    assert body["schema"]["fields"]["title"] == {"type": "string"}


@respx.mock
def test_create_semantic_text_rejection_surfaced_verbatim(indexes: Indexes) -> None:
    """semantic_text fields pass through and the backend rejection is surfaced (#145)."""
    server_message = "semantic_text fields are not supported in schema"
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(400, json=make_error_response(400, server_message))
    )

    with pytest.raises(ApiError, match="semantic_text fields are not supported"):
        indexes.create(
            schema={
                "fields": {"content": {"type": "semantic_text", "model": "multilingual-e5-large"}}
            }
        )


# ---------------------------------------------------------------------------
# Readiness polling: poll-until-ready stays the default
# ---------------------------------------------------------------------------


@respx.mock
def test_create_polls_until_ready_by_default(indexes: Indexes) -> None:
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

    with patch("pinecone._internal.indexes_helpers.time.sleep"):
        result = indexes.create(name="test-index", schema=DENSE_SCHEMA)

    assert describe_route.call_count == 2
    assert result.status.ready is True


@respx.mock
def test_create_polls_by_server_assigned_name_when_name_omitted(indexes: Indexes) -> None:
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

    with patch("pinecone._internal.indexes_helpers.time.sleep"):
        result = indexes.create(schema=DENSE_SCHEMA)

    assert describe_route.called
    assert result.name == "srv-assigned"


@respx.mock
def test_create_timeout_negative_one_skips_polling(indexes: Indexes) -> None:
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201, json=make_index_response(status={"ready": False, "state": "Initializing"})
        )
    )
    describe_route = respx.get(f"{BASE_URL}/indexes/test-index")

    result = indexes.create(name="test-index", schema=DENSE_SCHEMA, timeout=-1)

    assert result.status.ready is False
    assert describe_route.call_count == 0
