"""Every ``TypeError`` these call sites raise is named in a ``Raises:`` block (#438).

The #372 audit found three places where the SDK raises ``TypeError`` and no
docstring anywhere in the chain mentions it, so a caller cannot learn what to
catch short of reading source. gRPC's ``describe_namespace``/``delete_namespace``
already document theirs, which made it a three-surface parity gap rather than a
blanket omission.

Finding 1 is the namespace methods, finding 2 is ``Field.gt/gte/lt/lte``, and
finding 3 is ``_build_search_records_body``; the tests below run in that order.

Each finding is pinned from both directions: raise the real exception, then
assert the rendered ``Raises:`` block covers it. Prose alone would let the docs
outlive a raise that was removed; a raise alone is the hole #465 is about.

``_build_search_records_body`` is private, so its two conditions are pinned on
the public ``search()`` docstrings that reach it, plus the helper's own
docstring for anyone who lands there from a traceback.
"""

from __future__ import annotations

import inspect
import re
from typing import Any

import pytest

from pinecone._internal.data_plane_helpers import _build_search_records_body
from pinecone.async_client.async_index import AsyncIndex
from pinecone.async_client.documents import AsyncDocuments
from pinecone.client.documents import Documents
from pinecone.index import Index
from pinecone.utils.filter_builder import Field

NUMERIC_OPS = ("gt", "gte", "lt", "lte")

NAMESPACE_METHODS = [
    (Index, "describe_namespace"),
    (Index, "delete_namespace"),
    (AsyncIndex, "describe_namespace"),
    (AsyncIndex, "delete_namespace"),
]

SEARCH_METHODS = [(Index, "search"), (AsyncIndex, "search")]

TYPE_ERROR_ON_KWARGS = ":exc:`TypeError`: If unexpected keyword arguments are passed."

DOCUMENT_METHODS = [
    (cls, method)
    for cls in (Documents, AsyncDocuments)
    for method in ("batch_upsert", "delete", "fetch", "list", "search", "update", "upsert")
]

KEYWORD_ONLY_ON_CLASS = "misspelled keyword raises :exc:`TypeError`"


def _raises_block(obj: Any) -> str:
    """Return the ``Raises:`` section of *obj*'s docstring, whitespace-normalized.

    Napoleon renders sections in source order, so the block runs from ``Raises:``
    to the next line starting in column zero.
    """
    doc = inspect.cleandoc(obj.__doc__ or "")
    assert "Raises:" in doc, f"{obj.__qualname__} has no Raises: block at all"
    tail = doc.split("Raises:", 1)[1]
    block = re.split(r"\n(?=\S)", tail, maxsplit=1)[0]
    return " ".join(block.split())


def _index() -> Index:
    return Index(api_key="key", host="https://idx-abc123.svc.us-east-1-aws.pinecone.io")


def _async_index() -> AsyncIndex:
    return AsyncIndex(api_key="key", host="https://idx-abc123.svc.us-east-1-aws.pinecone.io")


@pytest.mark.parametrize(
    ("cls", "method"), NAMESPACE_METHODS, ids=[f"{c.__name__}.{m}" for c, m in NAMESPACE_METHODS]
)
@pytest.mark.asyncio
async def test_namespace_method_raises_type_error_for_an_unexpected_keyword(
    cls: type, method: str
) -> None:
    client = _index() if cls is Index else _async_index()

    async def call() -> None:
        result = getattr(client, method)(name="movies-en", nmaespace="typo")
        if inspect.isawaitable(result):
            await result

    with pytest.raises(TypeError) as excinfo:
        await call()
    assert "unexpected keyword arguments" in str(excinfo.value)


@pytest.mark.parametrize(
    ("cls", "method"), NAMESPACE_METHODS, ids=[f"{c.__name__}.{m}" for c, m in NAMESPACE_METHODS]
)
def test_namespace_method_documents_that_type_error(cls: type, method: str) -> None:
    assert TYPE_ERROR_ON_KWARGS in _raises_block(getattr(cls, method))


def test_namespace_type_error_wording_matches_the_grpc_twin() -> None:
    grpc = pytest.importorskip("pinecone.grpc")
    for method in ("describe_namespace", "delete_namespace"):
        assert TYPE_ERROR_ON_KWARGS in _raises_block(getattr(grpc.GrpcIndex, method))


@pytest.mark.parametrize("op", NUMERIC_OPS)
@pytest.mark.parametrize("value", ["0.5", None, True, [1]], ids=["str", "none", "bool", "list"])
def test_field_comparison_raises_type_error_for_a_non_numeric_value(op: str, value: Any) -> None:
    with pytest.raises(TypeError) as excinfo:
        getattr(Field("score"), op)(value)
    assert "numeric" in str(excinfo.value)


@pytest.mark.parametrize("op", NUMERIC_OPS)
def test_field_comparison_documents_that_type_error(op: str) -> None:
    block = _raises_block(getattr(Field, op))
    assert ":exc:`TypeError`:" in block
    assert "bool" in block, f"Field.{op} does not say that a bool is rejected"


def _body_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "top_k": None,
        "inputs": None,
        "vector": None,
        "id": None,
        "filter": None,
        "fields": None,
        "rerank": None,
        "match_terms": None,
        "query": None,
    }
    base.update(overrides)
    return base


def test_search_body_raises_type_error_when_query_and_flat_kwargs_conflict() -> None:
    with pytest.raises(TypeError) as excinfo:
        _build_search_records_body(**_body_kwargs(query={"top_k": 5}, top_k=5))
    assert "not both" in str(excinfo.value)


def test_search_body_raises_type_error_for_a_wrong_query_type() -> None:
    with pytest.raises(TypeError) as excinfo:
        _build_search_records_body(**_body_kwargs(query=["top_k", 5]))
    assert "must be a SearchQuery or Mapping" in str(excinfo.value)


def test_search_body_helper_has_a_docstring_naming_both_conditions() -> None:
    block = _raises_block(_build_search_records_body)
    assert ":exc:`TypeError`:" in block
    assert "query" in block


@pytest.mark.parametrize(
    ("cls", "method"), SEARCH_METHODS, ids=[f"{c.__name__}.{m}" for c, m in SEARCH_METHODS]
)
def test_search_documents_both_type_error_conditions(cls: type, method: str) -> None:
    block = _raises_block(getattr(cls, method))
    assert ":exc:`TypeError`:" in block, f"{cls.__name__}.{method} never mentions TypeError"
    assert "``query``" in block
    assert "both" in block, "the mutually-exclusive-argument condition is not described"
    assert "Mapping" in block or "mapping" in block, "the wrong-query-type condition is missing"


@pytest.mark.parametrize(
    ("cls", "method"), DOCUMENT_METHODS, ids=[f"{c.__name__}.{m}" for c, m in DOCUMENT_METHODS]
)
@pytest.mark.asyncio
async def test_document_method_raises_type_error_for_an_unexpected_keyword(
    cls: type, method: str
) -> None:
    namespace = _index().documents if cls is Documents else _async_index().documents

    async def call() -> None:
        result = getattr(namespace, method)(nmaespace="typo")
        if inspect.isawaitable(result):
            await result

    with pytest.raises(TypeError) as excinfo:
        await call()
    assert "unexpected keyword argument" in str(excinfo.value)


@pytest.mark.parametrize("cls", [Documents, AsyncDocuments], ids=lambda c: c.__name__)
def test_document_namespace_documents_that_type_error_on_the_class(cls: type) -> None:
    """Stated once on the class, not 7 times per lane.

    The condition is true of every method on these namespaces, so the
    per-method ``Raises:`` entry the ``NAMESPACE_METHODS`` convention uses
    would be 14 copies of one sentence. The guarantee is the same: a caller
    who misspells a keyword can learn what to catch without reading source.
    """
    doc = " ".join(inspect.cleandoc(cls.__doc__ or "").split())
    assert "keyword-only" in doc, f"{cls.__name__} never says its methods are keyword-only"
    assert KEYWORD_ONLY_ON_CLASS in doc, f"{cls.__name__} does not name the TypeError"
    assert ":exc:`PineconeValueError`" in doc, "the positional-misuse error is not named"
