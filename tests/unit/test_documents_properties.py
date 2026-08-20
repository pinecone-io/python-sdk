"""Property-based tests for document-operation validation (#132, #135).

Four properties are pinned here:

* ``_validate_documents`` accepts a document list iff every element carries a
  unique, non-empty, ASCII-only (``\\x01``-``\\x7F``) string ``_id`` of at most
  512 characters — exercised over a corpus that includes unicode, empty,
  overlong, duplicate, and non-string ids.
* The fetch and delete body builders accept an argument shape iff it satisfies
  the spec's selector tables (``db_data_2026-07.oas.yaml`` FetchDocumentsRequest
  :3116 — exactly one of ids|filter, pagination_token filter-only;
  DeleteDocumentsRequest:3203 — exactly one of ids|filter|delete_all), and an
  accepted shape serializes exactly the provided selectors.
* ``_remove_fields`` stays a control key in every accepted per-ID update
  payload: it serializes as a list of field names, no name it holds also
  carries a value on the wire, and the list itself is never a field's value.
* ``_build_list_documents_body`` accepts an argument shape iff ``prefix`` is at
  most 512 ASCII (``\\x01``-``\\x7F``) characters and ``limit`` falls in
  [1, 100] (``ListDocumentsRequest``), and serializes exactly what was given.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pinecone._internal.documents_helpers import (
    _build_delete_documents_body,
    _build_fetch_documents_body,
    _build_list_documents_body,
    _build_update_documents_body,
    _validate_documents,
)
from pinecone.errors.exceptions import PineconeValueError

# ---------------------------------------------------------------------------
# Property 1: _validate_documents accept-iff contract
# ---------------------------------------------------------------------------

_ascii_id = st.text(
    alphabet=st.characters(min_codepoint=0x01, max_codepoint=0x7F), min_size=1, max_size=512
)
_bad_id = st.one_of(
    st.just(""),
    st.text(min_size=1, max_size=8).filter(lambda s: any(not ("\x01" <= ch <= "\x7f") for ch in s)),
    st.text(
        alphabet=st.characters(min_codepoint=0x01, max_codepoint=0x7F), min_size=513, max_size=520
    ),
    st.integers(),
    st.none(),
    st.booleans(),
)
_id_value = st.one_of(_ascii_id, _bad_id)

_documents = st.lists(
    st.one_of(
        st.builds(lambda i: {"_id": i, "field": "value"}, _id_value),
        st.just({"field": "no id at all"}),
    ),
    min_size=0,
    max_size=8,
)


def _id_is_valid(value: Any) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value) <= 512
        and all("\x01" <= ch <= "\x7f" for ch in value)
    )


def _list_is_valid(docs: list[dict[str, Any]]) -> bool:
    if not docs:
        return False
    ids = [doc.get("_id") for doc in docs]
    if not all(_id_is_valid(i) for i in ids):
        return False
    return len(set(ids)) == len(ids)


@given(docs=_documents)
def test_validate_documents_accepts_iff_ids_unique_nonempty_ascii_512(
    docs: list[dict[str, Any]],
) -> None:
    if _list_is_valid(docs):
        normalized = _validate_documents(docs)
        assert normalized == docs
        assert normalized is not docs
    else:
        with pytest.raises(PineconeValueError):
            _validate_documents(docs)


@given(doc_id=_ascii_id, count=st.integers(min_value=2, max_value=5))
def test_validate_documents_rejects_any_duplicate(doc_id: str, count: int) -> None:
    docs = [{"_id": doc_id} for _ in range(count)]
    with pytest.raises(PineconeValueError, match="duplicate"):
        _validate_documents(docs)


# ---------------------------------------------------------------------------
# Property 2: fetch/delete argument shapes accept-iff the spec's selector tables
# ---------------------------------------------------------------------------

_fetch_shape = st.fixed_dictionaries(
    {
        "ids": st.one_of(st.none(), st.just(["doc-1"])),
        "filter": st.one_of(st.none(), st.just({}), st.just({"genre": {"$eq": "news"}})),
        "pagination_token": st.one_of(st.none(), st.just("tok-1")),
        "include_fields": st.one_of(st.none(), st.just(["title"])),
    }
)


@given(shape=_fetch_shape)
def test_fetch_shape_accept_reject_matches_spec(shape: dict[str, Any]) -> None:
    has_ids = shape["ids"] is not None
    has_filter = shape["filter"] is not None
    filter_nonempty = has_filter and len(shape["filter"]) > 0
    valid = (
        (has_ids != has_filter)
        and (not has_filter or filter_nonempty)
        and (shape["pagination_token"] is None or has_filter)
    )
    if valid:
        body = _build_fetch_documents_body(**shape)
        assert body == {k: v for k, v in shape.items() if v is not None}
    else:
        with pytest.raises(PineconeValueError):
            _build_fetch_documents_body(**shape)


_delete_shape = st.fixed_dictionaries(
    {
        "ids": st.one_of(st.none(), st.just(["doc-1"])),
        "filter": st.one_of(st.none(), st.just({}), st.just({"genre": {"$eq": "news"}})),
        "delete_all": st.booleans(),
    }
)


@given(shape=_delete_shape)
def test_delete_shape_accept_reject_matches_spec(shape: dict[str, Any]) -> None:
    has_ids = shape["ids"] is not None
    has_filter = shape["filter"] is not None
    filter_nonempty = has_filter and len(shape["filter"]) > 0
    selector_count = sum([has_ids, has_filter, shape["delete_all"]])
    valid = selector_count == 1 and (not has_filter or filter_nonempty)
    if valid:
        body = _build_delete_documents_body(**shape)
        expected: dict[str, Any] = {}
        if has_ids:
            expected["ids"] = shape["ids"]
        if has_filter:
            expected["filter"] = shape["filter"]
        if shape["delete_all"]:
            expected["delete_all"] = True
        assert body == expected
    else:
        with pytest.raises(PineconeValueError):
            _build_delete_documents_body(**shape)


# ---------------------------------------------------------------------------
# Property 3: _remove_fields is a control key, never a document field value
# ---------------------------------------------------------------------------

_FIELD_NAMES = ["title", "content", "category", "_remove_fields", "year"]
_field_name = st.sampled_from(_FIELD_NAMES)
_field_value = st.one_of(
    st.text(max_size=6),
    st.integers(),
    st.booleans(),
    st.lists(st.text(max_size=4), max_size=3),
)

_patch = st.builds(
    lambda set_fields, remove, drop_id: {
        **({} if drop_id else {"_id": "doc-1"}),
        **set_fields,
        **({} if remove is None else {"_remove_fields": remove}),
    },
    set_fields=st.dictionaries(_field_name, _field_value, max_size=3),
    remove=st.one_of(
        st.none(),
        st.lists(_field_name, max_size=3),
        st.just("title"),
        st.just([1, 2]),
    ),
    drop_id=st.booleans(),
)


def _patch_is_valid(patch: dict[str, Any]) -> bool:
    if not _id_is_valid(patch.get("_id")):
        return False
    remove = patch.get("_remove_fields")
    if remove is None:
        return True
    if not isinstance(remove, list) or not all(isinstance(name, str) for name in remove):
        return False
    set_names = {key for key in patch if key not in ("_id", "_remove_fields")}
    return not (set(remove) & set_names)


@given(patch=_patch)
def test_remove_fields_never_serializes_as_a_document_field_value(patch: dict[str, Any]) -> None:
    if not _patch_is_valid(patch):
        with pytest.raises(PineconeValueError):
            _build_update_documents_body(
                documents=[patch], filter=None, set_fields=None, remove_fields=None
            )
        return

    body = _build_update_documents_body(
        documents=[patch], filter=None, set_fields=None, remove_fields=None
    )
    (serialized,) = body["documents"]
    removed = serialized.get("_remove_fields")
    set_names = {key for key in serialized if key not in ("_id", "_remove_fields")}

    if removed is None:
        assert "_remove_fields" not in serialized
    else:
        assert isinstance(removed, list)
        assert all(isinstance(name, str) for name in removed)
        assert not (set(removed) & set_names), (
            "a removed field name must not also carry a value on the wire"
        )
    for name in set_names:
        assert serialized[name] is not removed, (
            "the _remove_fields list must never be a field's value"
        )
    assert serialized == patch


# ---------------------------------------------------------------------------
# Property 4: list body accept-iff prefix/limit satisfy the documented bounds
# ---------------------------------------------------------------------------

_list_shape = st.fixed_dictionaries(
    {
        "prefix": st.one_of(
            st.none(),
            st.text(alphabet=st.characters(min_codepoint=0x01, max_codepoint=0x7F), max_size=4),
            st.just("x" * 513),
            st.just("ünïcode"),
        ),
        "limit": st.one_of(st.none(), st.integers(min_value=-5, max_value=105)),
        "pagination_token": st.one_of(st.none(), st.just("tok-1")),
    }
)


@given(shape=_list_shape)
def test_list_shape_accept_reject_matches_spec(shape: dict[str, Any]) -> None:
    prefix = shape["prefix"]
    limit = shape["limit"]
    prefix_ok = prefix is None or (
        len(prefix) <= 512 and all("\x01" <= ch <= "\x7f" for ch in prefix)
    )
    limit_ok = limit is None or 1 <= limit <= 100
    if prefix_ok and limit_ok:
        body = _build_list_documents_body(**shape)
        assert body == {k: v for k, v in shape.items() if v is not None}
    else:
        with pytest.raises(PineconeValueError):
            _build_list_documents_body(**shape)
