"""Unit tests for Indexes.configure() — 2026-07 PATCH /indexes/{name}.

Covers the six configure_index spec example bodies, the guided hard-break
interception of embed=/spec= (no 2026-07 translation exists), the deprecated
replicas=/pod_type=/serverless_read_capacity= sugar (translated, not
rejected), validation quality, and the IndexModel return.
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import HTTPClient
from pinecone.client.indexes import Indexes
from pinecone.errors.exceptions import ApiError, PineconeTypeError, PineconeValueError
from pinecone.models.indexes.index import IndexModel
from tests.factories import make_error_response, make_index_response

BASE_URL = "https://api.test.pinecone.io"


@pytest.fixture
def http_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, CONTROL_PLANE_API_VERSION)


@pytest.fixture
def indexes(http_client: HTTPClient) -> Indexes:
    return Indexes(http=http_client)


def _mock_patch() -> respx.Route:
    return respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response())
    )


# ---------------------------------------------------------------------------
# Wire basics
# ---------------------------------------------------------------------------


@respx.mock
def test_configure_sends_patch_with_configured_api_version(indexes: Indexes) -> None:
    route = _mock_patch()

    result = indexes.configure("test-index", deletion_protection="enabled")

    request = route.calls.last.request
    assert request.method == "PATCH"
    assert request.url.path == "/indexes/test-index"
    assert request.headers.get("X-Pinecone-Api-Version") == CONTROL_PLANE_API_VERSION
    assert isinstance(result, IndexModel)


@respx.mock
def test_configure_returns_updated_index_model(indexes: Indexes) -> None:
    """configure() returns the updated IndexModel (2025-10 returned None)."""
    respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response(deletion_protection="enabled"))
    )

    result = indexes.configure("test-index", deletion_protection="enabled")

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
def test_configure_emits_spec_example_body(indexes: Indexes, example_name: str) -> None:
    """configure() reproduces each configure_index spec example body exactly."""
    kwargs, expected_body = _SPEC_EXAMPLES[example_name]
    route = _mock_patch()

    indexes.configure("test-index", **kwargs)

    body = json.loads(route.calls.last.request.content)
    assert body == expected_body


@respx.mock
def test_configure_sparse_body_only_carries_provided_fields(indexes: Indexes) -> None:
    """Unset parameters stay off the wire entirely (no null keys)."""
    route = _mock_patch()

    indexes.configure("test-index", tags={"env": "prod"})

    body = json.loads(route.calls.last.request.content)
    assert body == {"tags": {"env": "prod"}}


def test_configure_legacy_embed_raises_with_guidance(indexes: Indexes) -> None:
    with pytest.raises(PineconeTypeError, match="no longer accepts embed="):
        indexes.configure("test-index", embed={"model": "multilingual-e5-large"})  # type: ignore[call-arg]


def test_configure_legacy_spec_raises_with_guidance(indexes: Indexes) -> None:
    with pytest.raises(PineconeTypeError, match="no longer accepts spec="):
        indexes.configure("test-index", spec={"pod": {"replicas": 2}})  # type: ignore[call-arg]


def test_configure_unknown_kwarg_lists_accepted_arguments(indexes: Indexes) -> None:
    with pytest.raises(PineconeTypeError, match="unexpected keyword argument"):
        indexes.configure("test-index", replica_count=2)  # type: ignore[call-arg]


def test_configure_legacy_hard_break_kwargs_rejected_before_any_request(indexes: Indexes) -> None:
    with respx.mock:
        route = respx.patch(f"{BASE_URL}/indexes/test-index")
        with pytest.raises(PineconeTypeError):
            indexes.configure("test-index", spec={"pod": {"replicas": 2}})  # type: ignore[call-arg]
        assert route.call_count == 0


def test_configure_replicas_sends_deployment_body(indexes: Indexes) -> None:
    with respx.mock:
        route = _mock_patch()

        indexes.configure("test-index", replicas=3)

        assert json.loads(route.calls.last.request.content) == {"deployment": {"replicas": 3}}


def test_configure_replicas_and_pod_type_send_one_deployment_object(indexes: Indexes) -> None:
    with respx.mock:
        route = _mock_patch()

        indexes.configure("test-index", replicas=3, pod_type="p1.x2")

        assert route.call_count == 1
        assert json.loads(route.calls.last.request.content) == {
            "deployment": {"replicas": 3, "pod_type": "p1.x2"}
        }


def test_configure_pod_type_alone_sends_only_pod_type_key(indexes: Indexes) -> None:
    with respx.mock:
        route = _mock_patch()

        indexes.configure("test-index", pod_type="p1.x2")

        assert json.loads(route.calls.last.request.content) == {"deployment": {"pod_type": "p1.x2"}}


def test_configure_serverless_read_capacity_sends_top_level_read_capacity(
    indexes: Indexes,
) -> None:
    with respx.mock:
        route = _mock_patch()

        indexes.configure("test-index", serverless_read_capacity={"mode": "OnDemand"})

        assert json.loads(route.calls.last.request.content) == {
            "read_capacity": {"mode": "OnDemand"}
        }


def test_configure_deployment_and_replicas_conflict_raises(indexes: Indexes) -> None:
    with pytest.raises(PineconeValueError) as exc_info:
        indexes.configure("test-index", deployment={"replicas": 2}, replicas=3)

    message = str(exc_info.value)
    assert "deployment=" in message
    assert "replicas=" in message


def test_configure_deployment_and_pod_type_conflict_raises(indexes: Indexes) -> None:
    with pytest.raises(PineconeValueError) as exc_info:
        indexes.configure("test-index", deployment={"pod_type": "p1.x2"}, pod_type="p1.x4")

    message = str(exc_info.value)
    assert "deployment=" in message
    assert "pod_type=" in message


def test_configure_read_capacity_and_serverless_read_capacity_conflict_raises(
    indexes: Indexes,
) -> None:
    with pytest.raises(PineconeValueError) as exc_info:
        indexes.configure(
            "test-index",
            read_capacity={"mode": "OnDemand"},
            serverless_read_capacity={"mode": "OnDemand"},
        )

    message = str(exc_info.value)
    assert "read_capacity=" in message
    assert "serverless_read_capacity=" in message


def test_configure_conflicts_raise_before_any_request(indexes: Indexes) -> None:
    with respx.mock:
        route = respx.patch(f"{BASE_URL}/indexes/test-index")
        with pytest.raises(PineconeValueError):
            indexes.configure("test-index", deployment={"replicas": 2}, replicas=3)
        assert route.call_count == 0


def test_configure_replicas_only_does_not_raise_no_parameters_error(indexes: Indexes) -> None:
    with respx.mock:
        _mock_patch()
        indexes.configure("test-index", replicas=3)


def test_configure_source_has_no_inline_pod_scaling_dict_building() -> None:
    source = inspect.getsource(Indexes.configure)
    assert "legacy_pod_scaling(" in source

    tree = ast.parse(textwrap.dedent(source))
    dict_keys = {
        key.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert not dict_keys & {"replicas", "pod_type"}


def test_configure_delegates_replicas_translation_to_pod_scaling_helper(
    indexes: Indexes,
) -> None:
    with respx.mock:
        route = _mock_patch()

        with patch(
            "pinecone.client.indexes.legacy_pod_scaling", return_value={"replicas": 99}
        ) as mock_translate:
            indexes.configure("test-index", replicas=3, pod_type="p1.x2")

        mock_translate.assert_called_once_with(replicas=3, pod_type="p1.x2")
        assert json.loads(route.calls.last.request.content) == {"deployment": {"replicas": 99}}


# ---------------------------------------------------------------------------
# Validation quality
# ---------------------------------------------------------------------------


def test_configure_empty_name_raises(indexes: Indexes) -> None:
    with pytest.raises(PineconeValueError, match="name"):
        indexes.configure("", deletion_protection="enabled")


def test_configure_no_parameters_raises(indexes: Indexes) -> None:
    with pytest.raises(PineconeValueError, match="at least one configuration parameter"):
        indexes.configure("test-index")


@pytest.mark.parametrize("param", ["deployment", "schema", "read_capacity", "tags"])
def test_configure_empty_dict_kwarg_raises(indexes: Indexes, param: str) -> None:
    with pytest.raises(PineconeValueError, match=f"{param} cannot be an empty dict"):
        indexes.configure("test-index", **{param: {}})


def test_configure_deployment_type_in_deployment_raises(indexes: Indexes) -> None:
    with pytest.raises(PineconeValueError, match="must not include 'deployment_type'"):
        indexes.configure("test-index", deployment={"deployment_type": "pod", "replicas": 2})


def test_configure_invalid_deletion_protection_raises(indexes: Indexes) -> None:
    with pytest.raises(PineconeValueError, match="deletion_protection"):
        indexes.configure("test-index", deletion_protection="yes")


def test_configure_invalid_tag_value_raises(indexes: Indexes) -> None:
    with pytest.raises(PineconeValueError, match="120-character limit"):
        indexes.configure("test-index", tags={"key": "v" * 121})


# ---------------------------------------------------------------------------
# The semantic_text-only client restriction was removed: any schema shape
# passes through and the server's policy error is surfaced verbatim.
# ---------------------------------------------------------------------------


@respx.mock
def test_configure_non_semantic_text_schema_passes_through_to_server(indexes: Indexes) -> None:
    route = _mock_patch()

    indexes.configure(
        "test-index",
        schema={"fields": {"extra": {"type": "dense_vector", "dimension": 3}}},
    )

    body = json.loads(route.calls.last.request.content)
    assert body["schema"]["fields"]["extra"]["type"] == "dense_vector"


@respx.mock
def test_configure_schema_server_rejection_surfaced_verbatim(indexes: Indexes) -> None:
    server_message = "Only fields of type 'semantic_text' may be specified"
    respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(422, json=make_error_response(422, server_message))
    )

    with pytest.raises(ApiError, match="semantic_text"):
        indexes.configure(
            "test-index",
            schema={"fields": {"extra": {"type": "dense_vector", "dimension": 3}}},
        )
