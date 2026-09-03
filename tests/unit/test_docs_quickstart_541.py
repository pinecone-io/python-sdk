"""The quickstart's data-plane examples reach the document plane (#541).

``docs/getting-started/quickstart.md`` used to create an index with a schema
naming a custom dense-vector field and then drive it with ``index.upsert()`` /
``index.query()``. pinecone-db refuses that pairing: any user-supplied schema
containing a vector field makes the index documents-only, because the legacy
vector fields are named ``_values``/``_sparse_values`` and a leading underscore
is rejected at create time, so no schema a user can write keeps the vectors API.

Following ``test_docs_migration_db_control_137.py``: the examples are read out
of the published page and executed, never transcribed here, so a reader copying
the page runs what this test ran. Only the document routes are registered for
the data plane, so a block that falls back to ``/vectors/upsert`` or ``/query``
raises on an unmocked route rather than quietly passing.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from pinecone import DenseVectorQuery, Pinecone
from tests.factories import make_index_response

GUIDE = Path(__file__).resolve().parents[2] / "docs/getting-started/quickstart.md"

CONTROL_URL = "https://api.test.pinecone.io"
DATA_HOST = "quickstart-abc123.svc.us-east-1-aws.pinecone.io"
DATA_URL = f"https://{DATA_HOST}"

VECTOR_ROUTES = ("/vectors/upsert", "/query")


def _blocks() -> list[str]:
    text = GUIDE.read_text()
    sources = [m.group(1) for m in re.finditer(r"```python\n(.*?)```", text, re.DOTALL)]
    assert sources, f"no python blocks found in {GUIDE}"
    return sources


BLOCKS = _blocks()
DATA_BLOCKS = [
    (i, s)
    for i, s in enumerate(BLOCKS)
    if "documents." in s or "index.upsert(" in s or "index.query(" in s
]


def _index_response() -> dict[str, Any]:
    return make_index_response(
        name="quickstart",
        host=DATA_HOST,
        schema={
            "fields": {"embedding": {"type": "dense_vector", "dimension": 3, "metric": "cosine"}}
        },
    )


def _stub() -> None:
    body = _index_response()
    deleted: list[bool] = []

    def describe(request: httpx.Request) -> httpx.Response:
        """``indexes.delete`` polls describe until it 404s, so the stub has to stop existing."""
        if deleted:
            return httpx.Response(404, json={"error": {"code": "NOT_FOUND", "message": "gone"}})
        return httpx.Response(200, json=body)

    def delete(request: httpx.Request) -> httpx.Response:
        deleted.append(True)
        return httpx.Response(202, json={})

    respx.post(f"{CONTROL_URL}/indexes").mock(return_value=httpx.Response(201, json=body))
    respx.get(url__regex=rf"{CONTROL_URL}/indexes/[^/]+$").mock(side_effect=describe)
    respx.delete(url__regex=rf"{CONTROL_URL}/indexes/[^/]+$").mock(side_effect=delete)
    respx.post(url__regex=rf"{DATA_URL}/namespaces/[^/]+/documents/upsert$").mock(
        return_value=httpx.Response(200, json={"upserted_count": 3})
    )
    respx.post(url__regex=rf"{DATA_URL}/namespaces/[^/]+/documents/search$").mock(
        return_value=httpx.Response(
            200,
            json={
                "matches": [{"_id": "movie-001", "_score": 0.99, "title": "Arrival"}],
                "namespace": "movies",
                "usage": {"read_units": 1},
            },
        )
    )


@pytest.fixture(autouse=True)
def _api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The complete-example block builds its own ``Pinecone()`` from the environment."""
    monkeypatch.setenv("PINECONE_API_KEY", "key")
    monkeypatch.setenv("PINECONE_CONTROLLER_HOST", CONTROL_URL)


def _run(source: str) -> None:
    pc = Pinecone(api_key="key", host=CONTROL_URL)
    namespace: dict[str, Any] = {
        "Pinecone": Pinecone,
        "DenseVectorQuery": DenseVectorQuery,
        "pc": pc,
        "index": pc.index("quickstart"),
    }
    code = compile(source, str(GUIDE), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    eval(code, namespace)  # noqa: S307


@pytest.mark.parametrize(("index", "source"), DATA_BLOCKS, ids=[str(i) for i, _ in DATA_BLOCKS])
@respx.mock
def test_data_plane_block_runs_against_document_routes_only(index: int, source: str) -> None:
    _stub()
    _run(source)
    paths = [call.request.url.path for call in respx.calls if call.request.method == "POST"]
    assert paths, f"block {index} issued no POST"
    assert not [p for p in paths if p.endswith(VECTOR_ROUTES)], (
        f"block {index} still calls the vector interface: {paths}"
    )


def test_page_never_pairs_a_schema_index_with_the_vector_methods() -> None:
    text = GUIDE.read_text()
    assert "schema=" in text, "the quickstart no longer creates a schema-based index"
    for bad in ("index.upsert(vectors=", "index.query(vector="):
        assert bad not in text, f"quickstart still shows {bad!r} against a schema-based index"


def test_page_shows_the_document_interface() -> None:
    text = GUIDE.read_text()
    for needed in ("index.documents.upsert(", "index.documents.search(", "DenseVectorQuery("):
        assert needed in text, f"quickstart never shows {needed!r}"


@respx.mock
def test_dense_score_by_clause_names_the_schema_field_on_the_wire() -> None:
    _stub()
    source = next(s for _, s in DATA_BLOCKS if "documents.search(" in s)
    _run(source)
    search = [c for c in respx.calls if c.request.url.path.endswith("/documents/search")]
    assert search, "no document search reached the wire"
    body = search[-1].request.read().decode().replace(" ", "")
    assert '"type":"dense_vector"' in body
    assert '"embedding"' in body
