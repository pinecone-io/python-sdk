"""Unit tests for PreviewIndexes.exists()."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig, RetryConfig
from pinecone.errors.exceptions import ApiError, PineconeValueError
from pinecone.preview.indexes import PreviewIndexes

BASE_URL = "https://api.test.pinecone.io"

_PREVIEW_INDEX_RESPONSE: dict[str, Any] = {
    "name": "my-index",
    "host": "my-index-xyz.svc.pinecone.io",
    "status": {"ready": True, "state": "Ready"},
    "schema": {"fields": {"e": {"type": "dense_vector", "dimension": 4}}},
    "deployment": {
        "deployment_type": "managed",
        "environment": "us-east-1-aws",
        "cloud": "aws",
        "region": "us-east-1",
    },
    "deletion_protection": "disabled",
}

_NOT_FOUND_RESPONSE: dict[str, Any] = {
    "error": {"code": "NOT_FOUND", "message": "Index not found."},
    "status": 404,
}

_SERVER_ERROR_RESPONSE: dict[str, Any] = {
    "error": {"code": "INTERNAL", "message": "Internal server error."},
    "status": 500,
}


@pytest.fixture
def indexes() -> PreviewIndexes:
    config = PineconeConfig(
        api_key="test-key",
        host=BASE_URL,
        retry_config=RetryConfig(max_retries=0),
    )
    return PreviewIndexes(config=config)


@respx.mock
def test_exists_returns_true_when_describe_succeeds(indexes: PreviewIndexes) -> None:
    """exists() returns True when the index is found."""
    respx.get(f"{BASE_URL}/indexes/my-index").mock(
        return_value=httpx.Response(200, json=_PREVIEW_INDEX_RESPONSE)
    )

    assert indexes.exists("my-index") is True


@respx.mock
def test_exists_returns_false_on_not_found(indexes: PreviewIndexes) -> None:
    """exists() returns False when the server responds with 404."""
    respx.get(f"{BASE_URL}/indexes/missing").mock(
        return_value=httpx.Response(404, json=_NOT_FOUND_RESPONSE)
    )

    assert indexes.exists("missing") is False


def test_exists_empty_name_raises(indexes: PreviewIndexes) -> None:
    """exists("") raises PineconeValueError without issuing an HTTP call."""
    with pytest.raises(PineconeValueError):
        indexes.exists("")


@respx.mock
def test_exists_propagates_other_errors(indexes: PreviewIndexes) -> None:
    """exists() re-raises non-404 API errors instead of swallowing them."""
    route = respx.get(f"{BASE_URL}/indexes/my-index").mock(
        return_value=httpx.Response(500, json=_SERVER_ERROR_RESPONSE)
    )

    with pytest.raises(ApiError):
        indexes.exists("my-index")

    assert route.call_count == 1
