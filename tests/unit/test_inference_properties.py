"""Property-based characterization of inference request serialization.

Two invariants the 2026-07 bump must not disturb, checked over the input space
rather than a handful of examples:

*Serialization is total and stable.* Whatever the caller passes — empty strings,
unicode, tuples instead of lists, arbitrary extra document keys — the body on
the wire is the documented normalization of it and nothing else. The 2026-07
inference spec is byte-identical in meaning to 2025-10, so any drift here would
be the SDK's, not the API's.

*The caller's dicts are never touched.* ``normalize_embed_inputs`` and
``normalize_rerank_documents`` copy each mapping with ``dict(...)``. Callers
reuse document lists across calls, so a normalizer that mutated or aliased them
would corrupt the caller's own data, silently and at a distance. This pins both
halves: value-equality after the call, and the body matching a pre-call copy.

Integer values are bounded to int64 on purpose: these properties are about
normalization of encodable bodies, and anything wider is rejected before the
wire by the encoder itself. That rejection — a Pinecone error naming the
offending field, since issue #187 — is pinned in
``tests/unit/test_json_encode_properties.py``.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx
import orjson
import respx
from hypothesis import given, settings
from hypothesis import strategies as st

from pinecone._internal.config import PineconeConfig
from pinecone.client.inference import Inference
from tests.factories import make_embed_response, make_rerank_response

BASE_URL = "https://api.test.pinecone.io"

_TEXT = st.text(max_size=40)
_INT64 = st.integers(min_value=-(2**63), max_value=2**63 - 1)
_SCALARS = st.one_of(st.booleans(), _INT64, _TEXT)
_KEYS = st.text(min_size=1, max_size=12)
_DOCUMENTS = st.lists(
    st.dictionaries(_KEYS, _SCALARS, max_size=4),
    min_size=1,
    max_size=5,
)


@contextmanager
def _inference() -> Iterator[Inference]:
    client = Inference(config=PineconeConfig(api_key="test-key", host=BASE_URL))
    try:
        yield client
    finally:
        client.close()


@given(inputs=st.lists(_TEXT, min_size=1, max_size=6))
@settings(max_examples=50, deadline=None)
def test_embed_string_inputs_serialize_as_text_objects(inputs: list[str]) -> None:
    with respx.mock(assert_all_called=False) as router, _inference() as inference:
        route = router.post(f"{BASE_URL}/embed").mock(
            return_value=httpx.Response(200, json=make_embed_response())
        )
        inference.embed("multilingual-e5-large", inputs)

        body = orjson.loads(route.calls.last.request.content)

    assert body == {
        "model": "multilingual-e5-large",
        "inputs": [{"text": text} for text in inputs],
    }


@given(
    inputs=st.lists(st.dictionaries(_KEYS, _SCALARS, max_size=4), min_size=1, max_size=5),
    parameters=st.dictionaries(_KEYS, _SCALARS, max_size=3),
)
@settings(max_examples=50, deadline=None)
def test_embed_mapping_inputs_are_passed_through_without_mutation(
    inputs: list[dict[str, Any]], parameters: dict[str, Any]
) -> None:
    before = copy.deepcopy(inputs)
    parameters_before = copy.deepcopy(parameters)

    with respx.mock(assert_all_called=False) as router, _inference() as inference:
        route = router.post(f"{BASE_URL}/embed").mock(
            return_value=httpx.Response(200, json=make_embed_response())
        )
        inference.embed("multilingual-e5-large", inputs, parameters=parameters)

        body = orjson.loads(route.calls.last.request.content)

    assert body["inputs"] == before
    assert body["parameters"] == parameters_before
    assert inputs == before
    assert parameters == parameters_before


@given(documents=_DOCUMENTS, rank_fields=st.lists(_KEYS, max_size=3))
@settings(max_examples=50, deadline=None)
def test_rerank_documents_survive_serialization_unmutated(
    documents: list[dict[str, Any]], rank_fields: list[str]
) -> None:
    before = copy.deepcopy(documents)

    with respx.mock(assert_all_called=False) as router, _inference() as inference:
        route = router.post(f"{BASE_URL}/rerank").mock(
            return_value=httpx.Response(200, json=make_rerank_response())
        )
        inference.rerank(
            "bge-reranker-v2-m3",
            "query",
            documents,
            rank_fields=rank_fields,
            top_n=len(documents),
        )

        body = orjson.loads(route.calls.last.request.content)

    assert body == {
        "model": "bge-reranker-v2-m3",
        "query": "query",
        "documents": before,
        "rank_fields": rank_fields,
        "return_documents": True,
        "top_n": len(before),
    }
    assert documents == before


@given(documents=_DOCUMENTS)
@settings(max_examples=25, deadline=None)
def test_rerank_top_n_may_exceed_the_document_count(documents: list[dict[str, Any]]) -> None:
    with respx.mock(assert_all_called=False) as router, _inference() as inference:
        route = router.post(f"{BASE_URL}/rerank").mock(
            return_value=httpx.Response(200, json=make_rerank_response())
        )
        inference.rerank("bge-reranker-v2-m3", "query", documents, top_n=len(documents) + 10)

        body = orjson.loads(route.calls.last.request.content)

    assert body["top_n"] == len(documents) + 10
