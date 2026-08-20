"""Unit tests for AsyncIndexes.configure() — 2026-07 PATCH /indexes/{name}.

Async mirror of tests/unit/client/test_indexes_configure.py: the six
configure_index spec example bodies, the guided hard-break interception of
2025-10 kwargs (replicas/pod_type/embed/spec/serverless_read_capacity),
validation quality, and the IndexModel return.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import AsyncHTTPClient
from pinecone.async_client.indexes import AsyncIndexes
from pinecone.errors.exceptions import ApiError, PineconeTypeError, PineconeValueError
from pinecone.models.indexes.index import IndexModel
from tests.factories import make_error_response, make_index_response

BASE_URL = "https://api.test.pinecone.io"


@pytest.fixture
async def async_http_client() -> AsyncGenerator[AsyncHTTPClient]:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    client = AsyncHTTPClient(config, CONTROL_PLANE_API_VERSION)
    yield client
    await client.close()


@pytest.fixture
def indexes(async_http_client: AsyncHTTPClient) -> AsyncIndexes:
    return AsyncIndexes(http=async_http_client)


def _mock_patch() -> respx.Route:
    return respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response())
    )


# ---------------------------------------------------------------------------
# Wire basics
# ---------------------------------------------------------------------------


@respx.mock
async def test_configure_sends_patch_with_configured_api_version(indexes: AsyncIndexes) -> None:
    route = _mock_patch()

    result = await indexes.configure("test-index", deletion_protection="enabled")

    request = route.calls.last.request
    assert request.method == "PATCH"
    assert request.url.path == "/indexes/test-index"
    assert request.headers.get("X-Pinecone-Api-Version") == CONTROL_PLANE_API_VERSION
    assert isinstance(result, IndexModel)


@respx.mock
async def test_configure_returns_updated_index_model(indexes: AsyncIndexes) -> None:
    """configure() returns the updated IndexModel (2025-10 returned None)."""
    respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response(deletion_protection="enabled"))
    )

    result = await indexes.configure("test-index", deletion_protection="enabled")

    assert isinstance(result, IndexModel)
    assert result.deletion_protection == "enabled"


# ---------------------------------------------------------------------------
# Spec-example bodies (db_control_2026-07.oas.yaml, configure_index examples)
# ---------------------------------------------------------------------------

_SPEC_EXAMPLES: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    "scale-replicas": (
        {"deployment": {"replicas": 4}},
        {"deployment": {"replicas": 4}},
    ),
    "upgrade-pod-type": (
        {"deployment": {"pod_type": "p1.x2"}},
        {"deployment": {"pod_type": "p1.x2"}},
    ),
    "update-read-capacity": (
        {
            "read_capacity": {
                "dedicated": {
                    "manual": {"replicas": 2, "shards": 2},
                    "node_type": "t1",
                    "scaling": "Manual",
                },
                "mode": "Dedicated",
            }
        },
        {
            "read_capacity": {
                "dedicated": {
                    "manual": {"replicas": 2, "shards": 2},
                    "node_type": "t1",
                    "scaling": "Manual",
                },
                "mode": "Dedicated",
            }
        },
    ),
    "update-semantic-text-params": (
        {
            "schema": {
                "fields": {
                    "content": {
                        "read_parameters": {"input_type": "query", "truncate": "NONE"},
                        "type": "semantic_text",
                        "write_parameters": {"input_type": "passage"},
                    }
                }
            }
        },
        {
            "schema": {
                "fields": {
                    "content": {
                        "read_parameters": {"input_type": "query", "truncate": "NONE"},
                        "type": "semantic_text",
                        "write_parameters": {"input_type": "passage"},
                    }
                }
            }
        },
    ),
    "disable-deletion-protection": (
        {"deletion_protection": "disabled"},
        {"deletion_protection": "disabled"},
    ),
    "update-tags": (
        {"tags": {"tag0": "new-val", "tag1": ""}},
        {"tags": {"tag0": "new-val", "tag1": ""}},
    ),
}


@respx.mock
@pytest.mark.parametrize("example_name", sorted(_SPEC_EXAMPLES))
async def test_configure_emits_spec_example_body(indexes: AsyncIndexes, example_name: str) -> None:
    """configure() reproduces each configure_index spec example body exactly."""
    kwargs, expected_body = _SPEC_EXAMPLES[example_name]
    route = _mock_patch()

    await indexes.configure("test-index", **kwargs)

    body = json.loads(route.calls.last.request.content)
    assert body == expected_body


@respx.mock
async def test_configure_sparse_body_only_carries_provided_fields(indexes: AsyncIndexes) -> None:
    """Unset parameters stay off the wire entirely (no null keys)."""
    route = _mock_patch()

    await indexes.configure("test-index", tags={"env": "prod"})

    body = json.loads(route.calls.last.request.content)
    assert body == {"tags": {"env": "prod"}}


# ---------------------------------------------------------------------------
# Guided hard break: 2025-10 kwargs raise with the equivalent 2026-07 call
# ---------------------------------------------------------------------------


async def test_configure_legacy_replicas_raises_with_translation(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeTypeError) as exc_info:
        await indexes.configure("test-index", replicas=4)  # type: ignore[call-arg]

    message = str(exc_info.value)
    assert "deployment={'replicas': 4}" in message
    assert "docs/migration/v10-2026-07-index-model.md" in message


async def test_configure_legacy_pod_type_raises_with_translation(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeTypeError, match=r"deployment=\{'pod_type': 'p1.x2'\}"):
        await indexes.configure("test-index", pod_type="p1.x2")  # type: ignore[call-arg]


async def test_configure_legacy_replicas_and_pod_type_combined(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeTypeError, match="'replicas': 2"):
        await indexes.configure("test-index", replicas=2, pod_type="p1.x2")  # type: ignore[call-arg]


async def test_configure_legacy_embed_raises_with_guidance(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeTypeError, match="no longer accepts embed="):
        await indexes.configure("test-index", embed={"model": "multilingual-e5-large"})  # type: ignore[call-arg]


async def test_configure_legacy_spec_raises_with_guidance(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeTypeError, match="no longer accepts spec="):
        await indexes.configure("test-index", spec={"pod": {"replicas": 2}})  # type: ignore[call-arg]


async def test_configure_legacy_serverless_read_capacity_raises_with_translation(
    indexes: AsyncIndexes,
) -> None:
    with pytest.raises(PineconeTypeError, match="read_capacity=\\{'mode': 'OnDemand'\\}"):
        await indexes.configure(  # type: ignore[call-arg]
            "test-index", serverless_read_capacity={"mode": "OnDemand"}
        )


async def test_configure_unknown_kwarg_lists_accepted_arguments(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeTypeError, match="unexpected keyword argument"):
        await indexes.configure("test-index", replica_count=2)  # type: ignore[call-arg]


async def test_configure_legacy_kwargs_rejected_before_any_request(indexes: AsyncIndexes) -> None:
    with respx.mock:
        route = respx.patch(f"{BASE_URL}/indexes/test-index")
        with pytest.raises(PineconeTypeError):
            await indexes.configure("test-index", replicas=4)  # type: ignore[call-arg]
        assert route.call_count == 0


# ---------------------------------------------------------------------------
# Validation quality
# ---------------------------------------------------------------------------


async def test_configure_empty_name_raises(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeValueError, match="name"):
        await indexes.configure("", deletion_protection="enabled")


async def test_configure_no_parameters_raises(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeValueError, match="at least one configuration parameter"):
        await indexes.configure("test-index")


@pytest.mark.parametrize("param", ["deployment", "schema", "read_capacity", "tags"])
async def test_configure_empty_dict_kwarg_raises(indexes: AsyncIndexes, param: str) -> None:
    with pytest.raises(PineconeValueError, match=f"{param} cannot be an empty dict"):
        await indexes.configure("test-index", **{param: {}})


async def test_configure_deployment_type_in_deployment_raises(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeValueError, match="must not include 'deployment_type'"):
        await indexes.configure("test-index", deployment={"deployment_type": "pod", "replicas": 2})


async def test_configure_invalid_deletion_protection_raises(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeValueError, match="deletion_protection"):
        await indexes.configure("test-index", deletion_protection="yes")


async def test_configure_invalid_tag_value_raises(indexes: AsyncIndexes) -> None:
    with pytest.raises(PineconeValueError, match="120-character limit"):
        await indexes.configure("test-index", tags={"key": "v" * 121})


# ---------------------------------------------------------------------------
# The semantic_text-only client restriction was removed: any schema shape
# passes through and the server's policy error is surfaced verbatim.
# ---------------------------------------------------------------------------


@respx.mock
async def test_configure_non_semantic_text_schema_passes_through_to_server(
    indexes: AsyncIndexes,
) -> None:
    route = _mock_patch()

    await indexes.configure(
        "test-index",
        schema={"fields": {"extra": {"type": "dense_vector", "dimension": 3}}},
    )

    body = json.loads(route.calls.last.request.content)
    assert body["schema"]["fields"]["extra"]["type"] == "dense_vector"


@respx.mock
async def test_configure_schema_server_rejection_surfaced_verbatim(indexes: AsyncIndexes) -> None:
    server_message = "Only fields of type 'semantic_text' may be specified"
    respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(422, json=make_error_response(422, server_message))
    )

    with pytest.raises(ApiError, match="semantic_text"):
        await indexes.configure(
            "test-index",
            schema={"fields": {"extra": {"type": "dense_vector", "dimension": 3}}},
        )
