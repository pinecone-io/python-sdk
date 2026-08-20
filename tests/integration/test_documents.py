"""Integration tests for the document operations on the sync Index (#132).

These run against any live 2026-07 documents data plane. Point
``PINECONE_DOCUMENTS_INDEX_HOST`` at the data-plane host of an index whose
schema has a ``dense_vector`` field named ``embedding`` (dimension 4) and a
full-text-searchable string field named ``content``; the suite skips when the
variable is unset. Works against the minicone simulator too::

    curl -s -X POST http://127.0.0.1:5080/indexes \\
      -H 'Api-Key: mockkey' -H 'X-Pinecone-Api-Version: 2026-07' \\
      -H 'Content-Type: application/json' \\
      -d '{"name":"docs-it","deployment":{"cloud":"aws","region":"us-east-1","deployment_type":"managed"},
           "schema":{"fields":{"content":{"type":"string","full_text_search":{}},
                               "embedding":{"type":"dense_vector","dimension":4,"metric":"cosine"}}}}'
    PINECONE_DOCUMENTS_INDEX_HOST=http://127.0.0.1:5081 PINECONE_API_KEY=mockkey \\
      uv run pytest tests/integration/test_documents.py -v

Known simulator gaps (tests fail against minicone, pass against production):
text scoring with ``fields:[...]`` (pinecone-io/minicone#47 — production
accepts ``fields``, tracked upstream as #147) and ``query_string`` scoring
(explicitly unimplemented in the simulator).
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterator

import pytest

from pinecone.index import Index
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
def index() -> Iterator[Index]:
    client = Index(host=os.environ["PINECONE_DOCUMENTS_INDEX_HOST"])
    yield client
    client.close()


@pytest.fixture
def namespace(index: Index) -> Iterator[str]:
    name = f"it docs/{uuid.uuid4().hex[:8]}"
    yield name
    index.delete_documents(namespace=name, delete_all=True)


def _wait_for_ids(
    index: Index, namespace: str, expected: set[str], timeout: float = 60.0
) -> FetchDocumentsResponse:
    deadline = time.time() + timeout
    while True:
        response = index.fetch_documents(namespace=namespace, ids=sorted(expected))
        if set(response.documents) == expected:
            return response
        if time.time() > deadline:
            raise TimeoutError(f"documents {expected} not fetchable in {namespace!r}")
        time.sleep(1)


def test_upsert_fetch_by_ids_and_filter(index: Index, namespace: str) -> None:
    result = index.upsert_documents(namespace=namespace, documents=DOCS)
    assert result.upserted_count == 3

    fetched = _wait_for_ids(index, namespace, {"doc-1", "doc-2", "doc-3"})
    assert fetched.documents["doc-1"].category == "tech"
    assert fetched.usage is not None

    missing_tolerated = index.fetch_documents(
        namespace=namespace, ids=["doc-1", "definitely-missing"]
    )
    assert set(missing_tolerated.documents) == {"doc-1"}

    filtered = index.fetch_documents(
        namespace=namespace,
        filter={"category": {"$eq": "tech"}},
        include_fields=["category"],
    )
    assert set(filtered.documents) == {"doc-1"}


def test_search_documents_dense(index: Index, namespace: str) -> None:
    index.upsert_documents(namespace=namespace, documents=DOCS)
    _wait_for_ids(index, namespace, {"doc-1", "doc-2", "doc-3"})

    response = index.search_documents(
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


def test_search_documents_text(index: Index, namespace: str) -> None:
    index.upsert_documents(namespace=namespace, documents=DOCS)
    _wait_for_ids(index, namespace, {"doc-1", "doc-2", "doc-3"})

    response = index.search_documents(
        namespace=namespace,
        top_k=3,
        score_by=[TextQuery(query="machine learning", fields=["content"])],
    )
    assert response.matches
    assert response.matches[0]._id == "doc-1"


def test_delete_documents_by_ids_filter_and_all(index: Index, namespace: str) -> None:
    index.upsert_documents(namespace=namespace, documents=DOCS)
    _wait_for_ids(index, namespace, {"doc-1", "doc-2", "doc-3"})

    by_ids = index.delete_documents(namespace=namespace, ids=["doc-3"])
    assert by_ids.matched_records is None

    by_filter = index.delete_documents(
        namespace=namespace, filter={"category": {"$eq": "research"}}
    )
    assert by_filter.matched_records == 1

    deadline = time.time() + 60
    while True:
        remaining = index.fetch_documents(namespace=namespace, ids=["doc-1", "doc-2", "doc-3"])
        if set(remaining.documents) == {"doc-1"} or time.time() > deadline:
            break
        time.sleep(1)
    assert set(remaining.documents) == {"doc-1"}

    index.delete_documents(namespace=namespace, delete_all=True)


def test_batch_upsert_documents(index: Index, namespace: str) -> None:
    docs = [
        {
            "_id": f"bulk-{i}",
            "content": f"bulk document {i}",
            "embedding": [0.1, 0.2, 0.3, (i % 7) / 7.0],
        }
        for i in range(120)
    ]
    result = index.batch_upsert_documents(
        namespace=namespace, documents=docs, batch_size=50, show_progress=False
    )
    assert result.total_item_count == 120
    assert result.successful_item_count == 120
    assert result.failed_item_count == 0
    assert result.total_batch_count == 3

    fetched = _wait_for_ids(index, namespace, {"bulk-0", "bulk-59", "bulk-119"})
    assert len(fetched.documents) == 3
