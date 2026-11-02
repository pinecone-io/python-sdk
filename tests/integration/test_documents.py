"""Integration tests for the document operations on the sync Index (#132).

These run against any live 2026-07 documents data plane. By default the
module-scoped ``documents_index_host`` fixture (``conftest.py``) creates a
throwaway index whose schema has a ``dense_vector`` field named ``embedding``
(dimension 4) and a full-text-searchable string field named ``content``, and
deletes it afterwards; that needs ``PINECONE_API_KEY``, and the suite skips
without it. To reuse an existing index with that schema instead, point
``PINECONE_DOCUMENTS_INDEX_HOST`` at its data-plane host. That is how to run
against the minicone simulator::

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
accepts ``fields``, tracked upstream as #147), ``query_string`` scoring
(explicitly unimplemented in the simulator), and the ``include_fields``
semantics pinned by
``test_include_fields_semantics_differ_between_search_and_fetch``
(pinecone-io/minicone#325 and #327 — the simulator returns every field when
search omits ``include_fields``, and treats ``["*"]`` as a literal field name).
"""

from __future__ import annotations

import contextlib
import time
import uuid
from collections.abc import Iterator
from typing import Any, cast

import pytest

from pinecone.errors import NotFoundError
from pinecone.index import Index
from pinecone.models.documents.responses import (
    DeleteDocumentsResponse,
    FetchDocumentsResponse,
    SearchDocumentsResponse,
)
from pinecone.models.documents.score_by import DenseVectorQuery, TextQuery
from tests.integration.conftest import poll_until

pytestmark = [pytest.mark.integration]

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
def index(documents_index_host: str) -> Iterator[Index]:
    client = Index(host=documents_index_host)
    yield client
    client.close()


@pytest.fixture
def namespace(index: Index) -> Iterator[str]:
    name = f"it docs/{uuid.uuid4().hex[:8]}"
    yield name
    with contextlib.suppress(NotFoundError):
        index.documents.delete(namespace=name, delete_all=True)


def _wait_for_ids(
    index: Index, namespace: str, expected: set[str], timeout: float = 60.0
) -> FetchDocumentsResponse:
    deadline = time.time() + timeout
    while True:
        response = index.documents.fetch(namespace=namespace, ids=sorted(expected))
        if set(response.documents) == expected:
            return response
        if time.time() > deadline:
            raise TimeoutError(f"documents {expected} not fetchable in {namespace!r}")
        time.sleep(1)


def _wait_until_searchable(
    index: Index, namespace: str, expected: set[str], timeout: float = 120.0
) -> None:
    """Wait for search to see ``expected``, the namespace's full contents.

    Fetch by id and search see a write within about a second, but the first
    few searches to see the documents can alternate with ones that miss them.
    Wait for three consecutive complete results. A filtered fetch lags further;
    see ``_fetch_by_filter_until``.
    """
    deadline = time.time() + timeout
    streak = 0
    while True:
        response = index.documents.search(
            namespace=namespace,
            top_k=len(expected),
            score_by=[DenseVectorQuery(field="embedding", values=[1.0, 0.0, 0.0, 0.0])],
        )
        streak = streak + 1 if {match._id for match in response.matches} == expected else 0
        if streak == 3:
            return
        if time.time() > deadline:
            raise TimeoutError(f"documents {expected} not searchable in {namespace!r}")
        time.sleep(1)


def _search_until(
    index: Index, namespace: str, expected: set[str], timeout: float = 120.0, **search: Any
) -> SearchDocumentsResponse:
    """Run a search until it returns exactly the ``expected`` ids, and return that response.

    For a few seconds after documents become searchable, any single search,
    filtered or not, can still come back empty, even after
    ``_wait_until_searchable`` has seen three complete results. A test that
    asserts on a search result polls that same search instead.
    """
    deadline = time.time() + timeout
    while True:
        response = index.documents.search(namespace=namespace, **search)
        if {match._id for match in response.matches} == expected:
            return response
        if time.time() > deadline:
            raise TimeoutError(f"search {search} never returned {expected} in {namespace!r}")
        time.sleep(1)


def _fetch_by_filter_until(
    index: Index, namespace: str, filter: dict[str, Any], expected: set[str]
) -> FetchDocumentsResponse:
    """Run a filtered fetch until it returns exactly the ``expected`` ids, and return that response.

    A filtered fetch reads only flushed data, about 7 s after a write. A
    filtered delete or update sent before the flush applies but reports
    ``matched_records=0``, so tests gate filter writes on this wait too.
    """
    return cast(
        FetchDocumentsResponse,
        poll_until(
            query_fn=lambda: index.documents.fetch(
                namespace=namespace, filter=filter, include_fields=["category"]
            ),
            check_fn=lambda r: set(r.documents) == expected,
            timeout=60,
            interval=1,
            description=f"fetch by filter {filter} returns {expected} in {namespace!r}",
        ),
    )


def test_upsert_fetch_by_ids_and_filter(index: Index, namespace: str) -> None:
    result = index.documents.upsert(namespace=namespace, documents=DOCS)
    assert result.upserted_count == 3

    fetched = _wait_for_ids(index, namespace, {"doc-1", "doc-2", "doc-3"})
    _wait_until_searchable(index, namespace, {"doc-1", "doc-2", "doc-3"})
    assert fetched.documents["doc-1"].category == "tech"
    assert fetched.usage is not None

    missing_tolerated = index.documents.fetch(
        namespace=namespace, ids=["doc-1", "definitely-missing"]
    )
    assert set(missing_tolerated.documents) == {"doc-1"}

    filtered = _fetch_by_filter_until(index, namespace, {"category": {"$eq": "tech"}}, {"doc-1"})
    assert set(filtered.documents) == {"doc-1"}


def test_search_documents_dense(index: Index, namespace: str) -> None:
    index.documents.upsert(namespace=namespace, documents=DOCS)
    response = _search_until(
        index,
        namespace,
        {"doc-2", "doc-3"},
        top_k=2,
        score_by=[DenseVectorQuery(field="embedding", values=[0.0, 1.0, 0.0, 0.0])],
        include_fields=["category"],
        filter={"category": {"$in": ["research", "history"]}},
    )
    assert response.matches
    assert response.matches[0]._id == "doc-2"
    assert response.matches[0].category == "research"
    assert response.usage is not None


def test_include_fields_semantics_differ_between_search_and_fetch(
    index: Index, namespace: str
) -> None:
    """Pins the search/fetch asymmetry that pinecone-io/python-sdk-internal#544 hit.

    The server resolves an empty field list per-operation rather than globally
    (``EmptySemantics`` in ``query-router``): search drops every field, fetch
    keeps every field. ``["*"]`` means all fields on both. Omitting the key and
    sending ``[]`` are indistinguishable server-side, so both are asserted.
    """
    index.documents.upsert(namespace=namespace, documents=DOCS)
    score_by = [DenseVectorQuery(field="embedding", values=[1.0, 0.0, 0.0, 0.0])]

    omitted = _search_until(index, namespace, {"doc-1"}, top_k=1, score_by=score_by)
    assert set(omitted.matches[0].to_dict()) == {"_id", "_score"}

    empty = _search_until(
        index, namespace, {"doc-1"}, top_k=1, score_by=score_by, include_fields=[]
    )
    assert set(empty.matches[0].to_dict()) == {"_id", "_score"}

    wildcard = _search_until(
        index, namespace, {"doc-1"}, top_k=1, score_by=score_by, include_fields=["*"]
    )
    assert {"content", "category"} <= set(wildcard.matches[0].to_dict())

    for include_fields in (None, [], ["*"]):
        fetched = index.documents.fetch(
            namespace=namespace, ids=["doc-1"], include_fields=include_fields
        )
        assert {"content", "category"} <= set(fetched.documents["doc-1"].to_dict()), (
            f"fetch with include_fields={include_fields!r} should return every field"
        )


def test_search_documents_text(index: Index, namespace: str) -> None:
    index.documents.upsert(namespace=namespace, documents=DOCS)
    response = _search_until(
        index,
        namespace,
        {"doc-1", "doc-2"},
        top_k=3,
        score_by=[TextQuery(query="machine learning", fields=["content"])],
    )
    assert response.matches
    assert response.matches[0]._id == "doc-1"


def test_delete_documents_by_ids_filter_and_all(index: Index, namespace: str) -> None:
    index.documents.upsert(namespace=namespace, documents=DOCS)
    _wait_until_searchable(index, namespace, {"doc-1", "doc-2", "doc-3"})

    by_ids = index.documents.delete(namespace=namespace, ids=["doc-3"])
    assert by_ids.matched_records is None

    _fetch_by_filter_until(index, namespace, {"category": {"$eq": "research"}}, {"doc-2"})
    by_filter = index.documents.delete(
        namespace=namespace, filter={"category": {"$eq": "research"}}
    )
    assert by_filter.matched_records == 1

    deadline = time.time() + 60
    while True:
        remaining = index.documents.fetch(namespace=namespace, ids=["doc-1", "doc-2", "doc-3"])
        if set(remaining.documents) == {"doc-1"} or time.time() > deadline:
            break
        time.sleep(1)
    assert set(remaining.documents) == {"doc-1"}

    all_docs = index.documents.delete(namespace=namespace, delete_all=True)
    assert isinstance(all_docs, DeleteDocumentsResponse)
    assert all_docs.matched_records is None, (
        "matched_records is populated only for a filtered delete; delete_all reports no count"
    )

    deadline = time.time() + 60
    while True:
        emptied = index.documents.fetch(namespace=namespace, ids=["doc-1", "doc-2", "doc-3"])
        if not emptied.documents or time.time() > deadline:
            break
        time.sleep(1)
    assert emptied.documents == {}, "delete_all left documents behind"


def test_update_documents_per_id_and_by_filter(index: Index, namespace: str) -> None:
    index.documents.upsert(namespace=namespace, documents=DOCS)
    _wait_until_searchable(index, namespace, {"doc-1", "doc-2", "doc-3"})

    per_id = index.documents.update(
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
        patched = index.documents.fetch(namespace=namespace, ids=["doc-1", "doc-2"])
        settled = patched.documents["doc-1"].get("category") == "tech-updated" and (
            patched.documents["doc-2"].get("category") is None
        )
        if settled or time.time() > deadline:
            break
        time.sleep(1)
    assert patched.documents["doc-1"].get("category") == "tech-updated"
    assert patched.documents["doc-2"].get("category") is None

    _fetch_by_filter_until(index, namespace, {"category": {"$eq": "history"}}, {"doc-3"})
    by_filter = index.documents.update(
        namespace=namespace,
        filter={"category": {"$eq": "history"}},
        set_fields={"category": "archive"},
    )
    assert by_filter.matched_records == 1


def test_list_documents_pages_and_prefix(index: Index, namespace: str) -> None:
    docs = [
        {"_id": f"list-{i:03d}", "content": f"document {i}", "embedding": [1.0, 0.0, 0.0, 0.0]}
        for i in range(25)
    ]
    index.documents.upsert(namespace=namespace, documents=docs)
    _wait_for_ids(index, namespace, {"list-000", "list-012", "list-024"})

    listed = index.documents.list(namespace=namespace, prefix="list-").to_list()
    assert [record.id for record in listed] == sorted(doc["_id"] for doc in docs)

    paginator = index.documents.list(namespace=namespace, prefix="list-", limit=10)
    pages = list(paginator.pages())
    assert len(pages) >= 3
    assert len(pages[0].items) == 10
    assert paginator.pagination_token is None

    narrowed = index.documents.list(namespace=namespace, prefix="list-01").to_list()
    assert [record.id for record in narrowed] == [f"list-01{i}" for i in range(10)]

    absent = index.documents.list(namespace=namespace, prefix="no-such-prefix-").to_list()
    assert absent == []


def test_batch_upsert_documents(index: Index, namespace: str) -> None:
    docs = [
        {
            "_id": f"bulk-{i}",
            "content": f"bulk document {i}",
            "embedding": [0.1, 0.2, 0.3, (i % 7) / 7.0],
        }
        for i in range(120)
    ]
    result = index.documents.batch_upsert(
        namespace=namespace, documents=docs, batch_size=50, show_progress=False
    )
    assert result.total_item_count == 120
    assert result.successful_item_count == 120
    assert result.failed_item_count == 0
    assert result.total_batch_count == 3

    fetched = _wait_for_ids(index, namespace, {"bulk-0", "bulk-59", "bulk-119"})
    assert len(fetched.documents) == 3
