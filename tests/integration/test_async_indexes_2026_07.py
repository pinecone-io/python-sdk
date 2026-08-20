"""Integration tests for the graduated 2026-07 async index lifecycle (#133).

Async mirror of tests/integration/test_indexes_2026_07.py. Runs against
whatever PINECONE_API_KEY / PINECONE_CONTROLLER_HOST point to — a real
project or a local minicone (`PINECONE_API_KEY=mockkey
PINECONE_CONTROLLER_HOST=http://127.0.0.1:5080`). Until #112 flips
CONTROL_PLANE_API_VERSION to 2026-07, version-dispatching servers will
reject these requests; that skew is the planned intermediate state.

create_for_model has no lifecycle test here, matching the sync file: minicone
@ b5764e9 implements the stale flat built-spec shape for that operation
(minicone#49), so a faithful lifecycle run is not possible against the
simulator.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from pinecone import AsyncPinecone
from pinecone.errors import ApiError, NotFoundError, PineconeError
from pinecone.models.indexes.index import IndexModel

pytestmark = [pytest.mark.integration]

_DENSE_SCHEMA: dict[str, Any] = {
    "fields": {"embedding": {"type": "dense_vector", "dimension": 4, "metric": "cosine"}}
}


@pytest.fixture
async def index_name(async_client: AsyncPinecone) -> AsyncGenerator[str, None]:
    name = f"g133-{uuid.uuid4().hex[:10]}"
    yield name
    with contextlib.suppress(NotFoundError, PineconeError):
        await async_client.indexes.configure(name, deletion_protection="disabled")
    with contextlib.suppress(NotFoundError, PineconeError):
        await async_client.indexes.delete(name, timeout=120)


async def test_lifecycle_create_describe_list_configure_delete(
    async_client: AsyncPinecone, index_name: str
) -> None:
    created = await async_client.indexes.create(
        name=index_name,
        schema=_DENSE_SCHEMA,
        deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
        tags={"suite": "g133"},
        timeout=300,
    )
    assert isinstance(created, IndexModel)
    assert created.name == index_name
    assert created.status.ready is True
    assert "embedding" in created.schema.fields

    described = await async_client.indexes.describe(index_name)
    assert described.host
    assert await async_client.indexes.exists(index_name) is True

    names = [idx.name async for idx in async_client.indexes.list()]
    assert index_name in names

    updated = await async_client.indexes.configure(
        index_name, tags={"suite": "g133", "phase": "two"}
    )
    assert isinstance(updated, IndexModel)

    protected = await async_client.indexes.configure(index_name, deletion_protection="enabled")
    assert protected.deletion_protection == "enabled"
    unprotected = await async_client.indexes.configure(index_name, deletion_protection="disabled")
    assert unprotected.deletion_protection == "disabled"

    await async_client.indexes.delete(index_name, timeout=120)
    assert await async_client.indexes.exists(index_name) is False


async def test_create_fts_index(async_client: AsyncPinecone, index_name: str) -> None:
    created = await async_client.indexes.create(
        name=index_name,
        schema={
            "fields": {
                "body": {"type": "string", "full_text_search": {"language": "en"}},
            }
        },
        timeout=300,
    )
    assert created.status.ready is True
    assert "body" in created.schema.fields


async def test_create_without_name_assigns_one(async_client: AsyncPinecone) -> None:
    created = await async_client.indexes.create(schema=_DENSE_SCHEMA, timeout=300)
    try:
        assert created.name
        assert await async_client.indexes.exists(created.name) is True
    finally:
        with contextlib.suppress(NotFoundError, PineconeError):
            await async_client.indexes.delete(created.name, timeout=120)


async def test_bare_string_field_rejected_by_server(
    async_client: AsyncPinecone, index_name: str
) -> None:
    """A string field is accepted only with ``full_text_search``; the two
    unaccepted shapes are refused at different layers.

    ``{"type": "string"}`` matches neither variant of the server's untagged
    ``CreateStringSchemaField`` union — ``full_text_search`` is missing for the
    text-search variant, ``filterable`` is a required bool on the metadata
    variant — so it fails deserialization with 422 and never reaches schema
    validation, which is where the documented "fields used for search" 400
    lives. The two rejections are each other's positive control: a blanket
    refusal of index creation would answer both with one identical status.
    """
    with pytest.raises(ApiError) as bare:
        await async_client.indexes.create(
            name=index_name, schema={"fields": {"title": {"type": "string"}}}, timeout=-1
        )
    assert bare.value.status_code == 422

    with pytest.raises(ApiError) as filter_only:
        await async_client.indexes.create(
            name=index_name,
            schema={
                "fields": {
                    "embedding": {"type": "dense_vector", "dimension": 4, "metric": "cosine"},
                    "title": {"type": "string", "filterable": True},
                }
            },
            timeout=-1,
        )
    assert filter_only.value.status_code == 400
    assert "only accepts fields used for search" in str(filter_only.value)
