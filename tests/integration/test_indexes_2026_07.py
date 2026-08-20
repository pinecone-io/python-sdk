"""Integration tests for the graduated 2026-07 sync index lifecycle (#131).

Runs against whatever PINECONE_API_KEY / PINECONE_CONTROLLER_HOST point to —
a real project or a local minicone (`PINECONE_API_KEY=mockkey
PINECONE_CONTROLLER_HOST=http://127.0.0.1:5080`). Until #112 flips
CONTROL_PLANE_API_VERSION to 2026-07, version-dispatching servers will
reject these requests; that skew is the planned intermediate state.

Migrated from tests/integration/preview/test_indexes_lifecycle.py onto the
graduated names; the remaining spec-shape migration of
tests/integration/test_indexes.py belongs to #174.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Generator
from typing import Any

import pytest

from pinecone import Pinecone
from pinecone.errors import NotFoundError, PineconeError
from pinecone.models.indexes.index import IndexModel

pytestmark = [pytest.mark.integration]

_DENSE_SCHEMA: dict[str, Any] = {
    "fields": {"embedding": {"type": "dense_vector", "dimension": 4, "metric": "cosine"}}
}


@pytest.fixture
def index_name(client: Pinecone) -> Generator[str, None, None]:
    name = f"g131-{uuid.uuid4().hex[:10]}"
    yield name
    with contextlib.suppress(NotFoundError, PineconeError):
        client.indexes.configure(name, deletion_protection="disabled")
    with contextlib.suppress(NotFoundError, PineconeError):
        client.indexes.delete(name, timeout=120)


def test_lifecycle_create_describe_list_configure_delete(client: Pinecone, index_name: str) -> None:
    created = client.indexes.create(
        name=index_name,
        schema=_DENSE_SCHEMA,
        deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
        tags={"suite": "g131"},
        timeout=300,
    )
    assert isinstance(created, IndexModel)
    assert created.name == index_name
    assert created.status.ready is True
    assert "embedding" in created.schema.fields

    described = client.indexes.describe(index_name)
    assert described.host
    assert client.indexes.exists(index_name) is True

    names = [idx.name for idx in client.indexes.list()]
    assert index_name in names

    updated = client.indexes.configure(index_name, tags={"suite": "g131", "phase": "two"})
    assert isinstance(updated, IndexModel)

    protected = client.indexes.configure(index_name, deletion_protection="enabled")
    assert protected.deletion_protection == "enabled"
    unprotected = client.indexes.configure(index_name, deletion_protection="disabled")
    assert unprotected.deletion_protection == "disabled"

    client.indexes.delete(index_name, timeout=120)
    assert client.indexes.exists(index_name) is False


def test_create_fts_index(client: Pinecone, index_name: str) -> None:
    created = client.indexes.create(
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


def test_create_without_name_assigns_one(client: Pinecone) -> None:
    created = client.indexes.create(schema=_DENSE_SCHEMA, timeout=300)
    try:
        assert created.name
        assert client.indexes.exists(created.name) is True
    finally:
        with contextlib.suppress(NotFoundError, PineconeError):
            client.indexes.delete(created.name, timeout=120)


def test_bare_string_field_rejected_by_server(client: Pinecone, index_name: str) -> None:
    with pytest.raises(PineconeError, match="search"):
        client.indexes.create(
            name=index_name, schema={"fields": {"title": {"type": "string"}}}, timeout=-1
        )
