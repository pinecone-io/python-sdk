"""Unit tests for Preview and AsyncPreview namespace classes."""

from __future__ import annotations

import sys
from typing import cast
from unittest.mock import MagicMock

import httpx
import respx

import pinecone.schema_builder
from pinecone import Pinecone
from pinecone._internal.config import PineconeConfig
from pinecone._internal.http_client import AsyncHTTPClient, HTTPClient
from pinecone.preview import AsyncPreview, Preview, SchemaBuilder
from pinecone.preview._internal.constants import INDEXES_API_VERSION
from pinecone.preview.indexes import PreviewIndexes
from pinecone.preview.models.indexes import PreviewIndexModel

BASE_URL = "https://api.test.pinecone.io"

_PREVIEW_INDEX_RESPONSE: dict = {
    "name": "test-index",
    "host": "test-index-xyz.svc.pinecone.io",
    "status": {"ready": False, "state": "Initializing"},
    "schema": {"fields": {"e": {"type": "dense_vector", "dimension": 4}}},
    "deployment": {
        "deployment_type": "managed",
        "environment": "us-east-1-aws",
        "cloud": "aws",
        "region": "us-east-1",
    },
    "deletion_protection": "disabled",
}


def _cfg() -> PineconeConfig:
    return PineconeConfig(api_key="test")


def _http() -> HTTPClient:
    return cast(HTTPClient, MagicMock())


def _async_http() -> AsyncHTTPClient:
    return cast(AsyncHTTPClient, MagicMock())


def test_preview_constructor_stores_config() -> None:
    cfg = _cfg()
    preview = Preview(http=_http(), config=cfg)
    assert preview._config is cfg


def test_async_preview_constructor_stores_config() -> None:
    cfg = _cfg()
    preview = AsyncPreview(http=_async_http(), config=cfg)
    assert preview._config is cfg


def test_preview_repr() -> None:
    assert repr(Preview(http=_http(), config=_cfg())) == "Preview()"


def test_async_preview_repr() -> None:
    assert repr(AsyncPreview(http=_async_http(), config=_cfg())) == "AsyncPreview()"


def test_preview_and_async_preview_are_distinct_types() -> None:
    assert Preview is not AsyncPreview  # type: ignore[comparison-overlap]
    assert not issubclass(Preview, AsyncPreview)
    assert not issubclass(AsyncPreview, Preview)


def test_schema_builder_reexported_from_preview_module() -> None:
    assert SchemaBuilder is pinecone.schema_builder.SchemaBuilder


def test_preview_indexes_property_returns_preview_indexes_instance() -> None:
    pc = Pinecone(api_key="test")
    assert isinstance(pc.preview.indexes, PreviewIndexes)


def test_preview_indexes_property_is_cached() -> None:
    pc = Pinecone(api_key="test")
    first = pc.preview.indexes
    second = pc.preview.indexes
    assert first is second


def test_preview_property_is_cached() -> None:
    pc = Pinecone(api_key="test")
    first = pc.preview
    second = pc.preview
    assert first is second


def test_preview_indexes_lazy_until_accessed() -> None:
    pc = Pinecone(api_key="test")
    # Access .preview but not .indexes — _indexes must still be None
    preview = pc.preview
    assert preview._indexes is None
    # After accessing .indexes, _indexes is populated
    _ = preview.indexes
    assert preview._indexes is not None


@respx.mock
def test_preview_indexes_end_to_end_create() -> None:
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=_PREVIEW_INDEX_RESPONSE)
    )
    pc = Pinecone(api_key="test-key", host=BASE_URL)
    result = pc.preview.indexes.create(
        schema={"fields": {"e": {"type": "dense_vector", "dimension": 4}}}
    )
    assert route.called
    assert route.calls.last.request.headers.get("X-Pinecone-Api-Version") == INDEXES_API_VERSION
    assert isinstance(result, PreviewIndexModel)


def test_pinecone_init_does_not_import_preview_indexes() -> None:
    # Remove pinecone.preview.indexes from sys.modules to simulate a fresh import state
    saved = sys.modules.pop("pinecone.preview.indexes", None)
    try:
        pc = Pinecone(api_key="test")
        # Accessing .preview must NOT trigger import of pinecone.preview.indexes
        _ = pc.preview
        assert "pinecone.preview.indexes" not in sys.modules
        # Accessing .indexes MUST trigger the import
        _ = pc.preview.indexes
        assert "pinecone.preview.indexes" in sys.modules
    finally:
        if saved is not None:
            sys.modules["pinecone.preview.indexes"] = saved
