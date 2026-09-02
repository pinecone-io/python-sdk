"""Integration tests for the document operations on AsyncIndex (#134).

The async mirror of ``tests/integration/test_documents.py``, gated on the same
``PINECONE_DOCUMENTS_INDEX_HOST`` variable and expecting the same index schema:
a ``dense_vector`` field named ``embedding`` (dimension 4) and a
full-text-searchable string field named ``content``. Setup instructions,
including the minicone recipe, are in that module's docstring::

    PINECONE_DOCUMENTS_INDEX_HOST=http://127.0.0.1:5081 PINECONE_API_KEY=mockkey \\
      uv run pytest tests/integration/test_async_documents.py -v

Known simulator gaps (tests fail against minicone, pass against production):
text scoring with ``fields:[...]`` (pinecone-io/minicone#47 — production
accepts ``fields``, tracked upstream as #147), ``query_string`` scoring
(explicitly unimplemented in the simulator), and the ``include_fields``
semantics pinned by
``test_include_fields_semantics_differ_between_search_and_fetch``
(pinecone-io/minicone#325 and #327 — the simulator returns every field when
search omits ``include_fields``, and treats ``["*"]`` as a literal field name).
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections.abc import AsyncIterator

import pytest

from pinecone.async_client.async_index import AsyncIndex
from pinecone.models.documents.responses import FetchDocumentsResponse
from pinecone.models.documents.score_by import DenseVectorQuery, TextQuery

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("PINECONE_DOCUMENTS_INDEX_HOST"),
        reason="set PINECONE_DOCUMENTS_INDEX_HOST to a schema-based index data-plane host",
    ),
]

DOCS = [
    {
        "_id": "doc-1",
        "content": "Machine learning is a subset of artificial intelligence.",
        "embedding": [1.0, 0.0, 0.0, 0.0],
        "category": "tech",
    },
    {
        "_id": "doc-2",
        "content": "Deep learning uses neural networks with many layers.",
        "embedding": [0.0, 1.0, 0.0, 0.0],
        "category": "research",
    },
    {
        "_id": "doc-3",
        "content": "Rome was not built in a day.",
        "embedding": [0.0, 0.0, 1.0, 0.0],
        "category": "history",
    },
]


@pytest.fixture
async def index() -> AsyncIterator[AsyncIndex]:
    client = AsyncIndex(host=os.environ["PINECONE_DOCUMENTS_INDEX_HOST"])
    yield client
    await client.close()


@pytest.fixture
async def namespace(index: AsyncIndex) -> AsyncIterator[str]:
    name = f"it async docs/{uuid.uuid4().hex[:8]}"
    yield name
    await index.documents.delete(namespace=name, delete_all=True)


async def _wait_for_ids(
    index: AsyncIndex, namespace: str, expected: set[str], timeout: float = 60.0
) -> FetchDocumentsResponse:
    deadline = time.time() + timeout
    while True:
        response = await index.documents.fetch(namespace=namespace, ids=sorted(expected))
        if set(response.documents) == expected:
            return response
        if time.time() > deadline:
            raise TimeoutError(f"documents {expected} not fetchable in {namespace!r}")
        await asyncio.sleep(1)


async def test_upsert_fetch_by_ids_and_filter(index: AsyncIndex, namespace: str) -> None:
    result = await index.documents.upsert(namespace=namespace, documents=DOCS)
    assert result.upserted_count == 3

    fetched = await _wait_for_ids(index, namespace, {"doc-1", "doc-2", "doc-3"})
    assert fetched.documents["doc-1"].category == "tech"
    assert fetched.usage is not None

    missing_tolerated = await index.documents.fetch(
        namespace=namespace, ids=["doc-1", "definitely-missing"]
    )
    assert set(missing_tolerated.documents) == {"doc-1"}

    filtered = await index.documents.fetch(
        namespace=namespace,
        filter={"category": {"$eq": "tech"}},
        include_fields=["category"],
    )
    assert set(filtered.documents) == {"doc-1"}


async def test_search_documents_dense(index: AsyncIndex, namespace: str) -> None:
    await index.documents.upsert(namespace=namespace, documents=DOCS)
    await _wait_for_ids(index, namespace, {"doc-1", "doc-2", "doc-3"})

    response = await index.documents.search(
        namespace=namespace,
        top_k=2,
        score_by=[DenseVectorQuery(field="embedding", values=[0.0, 1.0, 0.0, 0.0])],
        include_fields=["category"],
        filter={"category": {"$in": ["research", "history"]}},
    )
    assert response.matches
    assert response.matches[0]._id == "doc-2"
    assert response.matches[0].category == "research"
    assert response.usage is not None


async def test_include_fields_semantics_differ_between_search_and_fetch(
    index: AsyncIndex, namespace: str
) -> None:
    """Pins the search/fetch asymmetry that pinecone-io/python-sdk-internal#544 hit.

    The server resolves an empty field list per-operation rather than globally
    (``EmptySemantics`` in ``query-router``): search drops every field, fetch
    keeps every field. ``["*"]`` means all fields on both. Omitting the key and
    sending ``[]`` are indistinguishable server-side, so both are asserted.
    """
    await index.documents.upsert(namespace=namespace, documents=DOCS)
    await _wait_for_ids(index, namespace, {"doc-1", "doc-2", "doc-3"})

    score_by = [DenseVectorQuery(field="embedding", values=[1.0, 0.0, 0.0, 0.0])]

    omitted = await index.documents.search(namespace=namespace, top_k=1, score_by=score_by)
    assert set(omitted.matches[0].to_dict()) == {"_id", "_score"}

    empty = await index.documents.search(
        namespace=namespace, top_k=1, score_by=score_by, include_fields=[]
    )
    assert set(empty.matches[0].to_dict()) == {"_id", "_score"}

    wildcard = await index.documents.search(
        namespace=namespace, top_k=1, score_by=score_by, include_fields=["*"]
    )
    assert {"content", "category"} <= set(wildcard.matches[0].to_dict())

    for include_fields in (None, [], ["*"]):
        fetched = await index.documents.fetch(
            namespace=namespace, ids=["doc-1"], include_fields=include_fields
        )
        assert {"content", "category"} <= set(fetched.documents["doc-1"].to_dict()), (
            f"fetch with include_fields={include_fields!r} should return every field"
        )


async def test_search_documents_text(index: AsyncIndex, namespace: str) -> None:
    await index.documents.upsert(namespace=namespace, documents=DOCS)
    await _wait_for_ids(index, namespace, {"doc-1", "doc-2", "doc-3"})

    response = await index.documents.search(
        namespace=namespace,
        top_k=3,
        score_by=[TextQuery(query="machine learning", fields=["content"])],
    )
    assert response.matches
    assert response.matches[0]._id == "doc-1"


async def test_delete_documents_by_ids_filter_and_all(index: AsyncIndex, namespace: str) -> None:
    await index.documents.upsert(namespace=namespace, documents=DOCS)
    await _wait_for_ids(index, namespace, {"doc-1", "doc-2", "doc-3"})

    by_ids = await index.documents.delete(namespace=namespace, ids=["doc-3"])
    assert by_ids.matched_records is None

    by_filter = await index.documents.delete(
        namespace=namespace, filter={"category": {"$eq": "research"}}
    )
    assert by_filter.matched_records == 1

    deadline = time.time() + 60
    while True:
        remaining = await index.documents.fetch(
            namespace=namespace, ids=["doc-1", "doc-2", "doc-3"]
        )
        if set(remaining.documents) == {"doc-1"} or time.time() > deadline:
            break
        await asyncio.sleep(1)
    assert set(remaining.documents) == {"doc-1"}

    await index.documents.delete(namespace=namespace, delete_all=True)


async def test_update_documents_per_id_and_by_filter(index: AsyncIndex, namespace: str) -> None:
    await index.documents.upsert(namespace=namespace, documents=DOCS)
    await _wait_for_ids(index, namespace, {"doc-1", "doc-2", "doc-3"})

    per_id = await index.documents.update(
        namespace=namespace,
        documents=[
            {"_id": "doc-1", "category": "tech-updated"},
            {"_id": "doc-2", "_remove_fields": ["category"]},
            {"_id": "definitely-missing", "category": "no-op"},
        ],
    )
    assert per_id.matched_records is None

    deadline = time.time() + 60
    while True:
        patched = await index.documents.fetch(namespace=namespace, ids=["doc-1", "doc-2"])
        settled = patched.documents["doc-1"].get("category") == "tech-updated" and (
            patched.documents["doc-2"].get("category") is None
        )
        if settled or time.time() > deadline:
            break
        await asyncio.sleep(1)
    assert patched.documents["doc-1"].get("category") == "tech-updated"
    assert patched.documents["doc-2"].get("category") is None

    by_filter = await index.documents.update(
        namespace=namespace,
        filter={"category": {"$eq": "history"}},
        set_fields={"category": "archive"},
    )
    assert by_filter.matched_records == 1


async def test_list_documents_pages_and_prefix(index: AsyncIndex, namespace: str) -> None:
    docs = [{"_id": f"list-{i:03d}", "content": f"document {i}"} for i in range(25)]
    await index.documents.upsert(namespace=namespace, documents=docs)
    await _wait_for_ids(index, namespace, {"list-000", "list-012", "list-024"})

    listed = await index.documents.list(namespace=namespace, prefix="list-").to_list()
    assert [record.id for record in listed] == sorted(str(doc["_id"]) for doc in docs)

    paginator = index.documents.list(namespace=namespace, prefix="list-", limit=10)
    pages = [page async for page in paginator.pages()]
    assert len(pages) >= 3
    assert len(pages[0].items) == 10
    assert paginator.pagination_token is None

    narrowed = await index.documents.list(namespace=namespace, prefix="list-01").to_list()
    assert [record.id for record in narrowed] == [f"list-01{i}" for i in range(10)]

    absent = await index.documents.list(namespace=namespace, prefix="no-such-prefix-").to_list()
    assert absent == []


async def test_batch_upsert_documents(index: AsyncIndex, namespace: str) -> None:
    docs = [
        {
            "_id": f"bulk-{i}",
            "content": f"bulk document {i}",
            "embedding": [0.1, 0.2, 0.3, (i % 7) / 7.0],
        }
        for i in range(120)
    ]
    result = await index.documents.batch_upsert(
        namespace=namespace, documents=docs, batch_size=50, show_progress=False
    )
    assert result.total_item_count == 120
    assert result.successful_item_count == 120
    assert result.failed_item_count == 0
    assert result.total_batch_count == 3

    fetched = await _wait_for_ids(index, namespace, {"bulk-0", "bulk-59", "bulk-119"})
    assert len(fetched.documents) == 3
