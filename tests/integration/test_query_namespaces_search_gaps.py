"""Gap-fill integration tests: namespaces CRUD + namespaces in stats.

The core query_namespaces/search suites cover query fan-out and search with
integrated inference. They do NOT exercise the per-namespace management
surface (create_namespace / describe_namespace / delete_namespace /
list_namespaces_paginated / list_namespaces) nor how namespaces show up in
describe_index_stats. These tests fill that gap for sync (REST) and async.

Run with:
    .venv/bin/python -m pytest -p no:cacheprovider -p no:timeout <this file>
"""

from __future__ import annotations

import pytest

from pinecone import AsyncPinecone, Pinecone
from pinecone.models.indexes.specs import ServerlessSpec
from pinecone.models.namespaces.models import NamespaceDescription
from tests.integration.conftest import (
    async_cleanup_resource,
    cleanup_resource,
    unique_name,
    wait_for_ready,
)


def _create_index(client: Pinecone, name: str) -> None:
    client.indexes.create(
        name=name,
        dimension=2,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        timeout=300,
    )
    wait_for_ready(
        lambda: client.indexes.describe(name).status.ready,
        timeout=300,
        description=f"index {name!r} ready",
    )


# ---------------------------------------------------------------------------
# namespaces CRUD — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_namespace_crud_rest(client: Pinecone) -> None:
    """create_namespace / list_namespaces / describe_namespace / delete_namespace (sync)."""
    name = unique_name("idx")
    try:
        _create_index(client, name)
        index = client.index(name=name)

        # Initially: no namespaces
        page = index.list_namespaces_paginated()
        assert page.total_count == 0, f"Expected 0 namespaces initially, got {page.total_count}"

        # create_namespace returns a NamespaceDescription
        desc = index.create_namespace(name="crud-ns1")
        assert isinstance(desc, NamespaceDescription)
        assert desc.name == "crud-ns1"

        index.create_namespace(name="crud-ns2")

        # list_namespaces_paginated shows both
        page = index.list_namespaces_paginated()
        names = {ns.name for ns in page.namespaces}
        assert "crud-ns1" in names and "crud-ns2" in names, f"got {names}"

        # describe_namespace returns record count (0 before upsert)
        desc = index.describe_namespace(name="crud-ns1")
        assert isinstance(desc, NamespaceDescription)
        assert desc.name == "crud-ns1"
        assert desc.record_count == 0

        # delete_namespace removes it — after delete, it should be gone from listing
        index.delete_namespace(name="crud-ns1")
        page = index.list_namespaces_paginated()
        names = {ns.name for ns in page.namespaces}
        assert "crud-ns1" not in names, f"namespace not deleted: {names}"

        # delete_namespace is idempotent-ish on re-delete (no exception for now-existing)
        # crud-ns2 still there
        assert "crud-ns2" in names
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


@pytest.mark.integration
def test_namespace_in_stats_after_upsert_rest(client: Pinecone) -> None:
    """describe_index_stats reports namespaces and vector counts after upsert (sync)."""
    name = unique_name("idx")
    try:
        _create_index(client, name)
        index = client.index(name=name)

        index.upsert(
            vectors=[
                {"id": "s1-v1", "values": [0.1, 0.9]},
                {"id": "s1-v2", "values": [0.2, 0.8]},
            ],
            namespace="stats-ns1",
        )
        index.upsert(
            vectors=[{"id": "s2-v1", "values": [0.5, 0.5]}],
            namespace="stats-ns2",
        )

        wait_for_ready(
            lambda: index.describe_index_stats().total_vector_count == 3,
            timeout=120,
            description="vectors visible in stats",
        )
        stats = index.describe_index_stats()
        ns = stats.namespaces
        assert "stats-ns1" in ns and "stats-ns2" in ns, f"namespaces missing in stats: {ns}"
        assert ns["stats-ns1"].vector_count == 2, f"stats-ns1 count: {ns['stats-ns1'].vector_count}"
        assert ns["stats-ns2"].vector_count == 1, f"stats-ns2 count: {ns['stats-ns2'].vector_count}"
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# namespaces CRUD — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_namespace_crud_rest_async(async_client: AsyncPinecone) -> None:
    """create_namespace / list_namespaces / describe_namespace / delete_namespace (async)."""
    name = unique_name("idx")
    try:
        desc = await async_client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = await async_client.index(host=desc.host)

        page = await index.list_namespaces_paginated()
        assert page.total_count == 0

        d = await index.create_namespace(name="acrud-ns1")
        assert isinstance(d, NamespaceDescription)
        assert d.name == "acrud-ns1"
        await index.create_namespace(name="acrud-ns2")

        page = await index.list_namespaces_paginated()
        names = {ns.name for ns in page.namespaces}
        assert "acrud-ns1" in names and "acrud-ns2" in names

        d = await index.describe_namespace(name="acrud-ns1")
        assert d.name == "acrud-ns1"
        assert d.record_count == 0

        await index.delete_namespace(name="acrud-ns1")
        page = await index.list_namespaces_paginated()
        names = {ns.name for ns in page.namespaces}
        assert "acrud-ns1" not in names
    finally:
        await async_cleanup_resource(lambda: async_client.indexes.delete(name), name, "index")


@pytest.mark.integration
@pytest.mark.anyio
async def test_namespace_in_stats_after_upsert_async(async_client: AsyncPinecone) -> None:
    """describe_index_stats reports namespaces and vector counts after upsert (async)."""
    name = unique_name("idx")
    try:
        desc = await async_client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = await async_client.index(host=desc.host)

        await index.upsert(
            vectors=[
                {"id": "s1-v1", "values": [0.1, 0.9]},
                {"id": "s1-v2", "values": [0.2, 0.8]},
            ],
            namespace="astats-ns1",
        )
        await index.upsert(
            vectors=[{"id": "s2-v1", "values": [0.5, 0.5]}],
            namespace="astats-ns2",
        )

        # async API has no wait_for_ready for stats; poll manually
        import asyncio

        for _ in range(40):
            stats = await index.describe_index_stats()
            if stats.total_vector_count >= 3:
                break
            await asyncio.sleep(3)

        ns = stats.namespaces
        assert "astats-ns1" in ns and "astats-ns2" in ns, f"namespaces missing in stats: {ns}"
        assert ns["astats-ns1"].vector_count == 2
        assert ns["astats-ns2"].vector_count == 1
    finally:
        await async_cleanup_resource(lambda: async_client.indexes.delete(name), name, "index")
