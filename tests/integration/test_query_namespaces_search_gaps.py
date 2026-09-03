"""Gap-fill integration tests: namespaces CRUD + namespaces in stats.

The core query_namespaces/search suites cover query fan-out and search with
integrated inference. They do NOT exercise the per-namespace management
surface (create_namespace / describe_namespace / delete_namespace /
list_namespaces_paginated / list_namespaces) nor how namespaces show up in
describe_index_stats. These tests fill that gap for sync (REST) and async.

Indexes come from :mod:`tests.integration.legacy_index`, not from
``pc.indexes.create``: 2026-07 has no way to create an index the vectors API
will serve, and the namespace surface exercised here is a vectors-API surface.

Unlike the other modules in this package these tests take a **dedicated**
index each rather than sharing one from ``legacy_index_factory``: they assert
on whole-index state — ``list_namespaces_paginated().total_count == 0`` before
any namespace exists, and an exact ``total_vector_count`` — which a shared
index's other tests would perturb. Namespace isolation cannot substitute for
a pristine index here.
"""

from __future__ import annotations

import os
from collections.abc import Generator

import pytest

from pinecone import AsyncPinecone, Pinecone
from pinecone._internal.constants import DEFAULT_BASE_URL
from pinecone.models.namespaces.models import NamespaceDescription
from tests.integration.conftest import async_poll_until, wait_for_ready
from tests.integration.legacy_index import (
    LegacyIndex,
    assert_serves_vectors_api,
    create_legacy_index,
    delete_legacy_index,
)


@pytest.fixture
def fresh_legacy_index(client: Pinecone, api_key: str) -> Generator[LegacyIndex, None, None]:
    """A dedicated, empty dim-2 legacy index for one test.

    Reads ``PINECONE_CONTROLLER_HOST`` for the same reason
    ``legacy_index_factory`` does: a run pointed at a local simulator must
    create its index there, not against production.
    """
    base_url = os.environ.get("PINECONE_CONTROLLER_HOST", DEFAULT_BASE_URL)
    index = create_legacy_index(api_key, dimension=2, base_url=base_url)
    assert_serves_vectors_api(client, index)
    try:
        yield index
    finally:
        delete_legacy_index(api_key, index.name, base_url=base_url)


# ---------------------------------------------------------------------------
# namespaces CRUD — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_namespace_crud_rest(client: Pinecone, fresh_legacy_index: LegacyIndex) -> None:
    """create_namespace / list_namespaces / describe_namespace / delete_namespace (sync)."""
    index = client.index(name=fresh_legacy_index.name)

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


@pytest.mark.integration
def test_namespace_in_stats_after_upsert_rest(
    client: Pinecone, fresh_legacy_index: LegacyIndex
) -> None:
    """describe_index_stats reports namespaces and vector counts after upsert (sync)."""
    index = client.index(name=fresh_legacy_index.name)

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


# ---------------------------------------------------------------------------
# namespaces CRUD — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_namespace_crud_rest_async(
    async_client: AsyncPinecone, fresh_legacy_index: LegacyIndex
) -> None:
    """create_namespace / list_namespaces / describe_namespace / delete_namespace (async)."""
    index = await async_client.index(host=fresh_legacy_index.host)
    try:
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
        await index.close()


@pytest.mark.integration
@pytest.mark.anyio
async def test_namespace_in_stats_after_upsert_async(
    async_client: AsyncPinecone, fresh_legacy_index: LegacyIndex
) -> None:
    """describe_index_stats reports namespaces and vector counts after upsert (async)."""
    index = await async_client.index(host=fresh_legacy_index.host)
    try:
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

        stats = await async_poll_until(
            query_fn=lambda: index.describe_index_stats(),
            check_fn=lambda r: r.total_vector_count >= 3,
            timeout=120,
            description="3 vectors visible in stats",
        )

        ns = stats.namespaces
        assert "astats-ns1" in ns and "astats-ns2" in ns, f"namespaces missing in stats: {ns}"
        assert ns["astats-ns1"].vector_count == 2
        assert ns["astats-ns2"].vector_count == 1
    finally:
        await index.close()
