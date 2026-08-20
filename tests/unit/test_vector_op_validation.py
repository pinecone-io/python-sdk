"""Client-side validation for the 2026-07 vector operations.

2026-07 tightens the selector algebra on the four vector ops. ``query`` now makes
``id`` exclusive with *both* vector forms, not just the dense one; ``update``
forbids pairing ``filter`` with vector values, because a by-filter update spans
many records and can only set metadata; and every selecting filter carries
``minProperties: 1``, so an empty filter is an error rather than a match-all.
``fetch`` and ``list`` gained per-ID and per-prefix charset/length rules and a
bounded page size. Each is rejected before any HTTP request goes out, so a caller
gets a message naming the argument instead of a raw 400.

Spec basis (``apis`` @ 5f808858, ``db_data_2026-07.oas.yaml``):

- ``:2230-2244`` — ``QueryRequest`` ``anyOf`` vector|id|sparseVector, with
  ``not anyOf`` [id+vector, id+sparseVector]
- ``:1882-1889`` — ``DeleteRequest`` ``anyOf`` ids|filter|deleteAll; filter
  ``minProperties: 1``
- ``:2415-2427`` — ``UpdateRequest`` ``anyOf`` id|filter, ``not anyOf``
  [filter+values, filter+sparseValues]; filter ``minProperties: 1``
- ``:472-481`` — fetch ``ids``: per-ID ``^[\\x01-\\x7F]+$``, 1-512, ``minItems: 1``
- ``:600-619`` — list ``prefix`` ``^[\\x01-\\x7F]*$`` / 512, ``limit`` 1-100
- ``:2725+`` — ``FetchByMetadataRequest.filter`` required, ``minProperties: 1``
- ``:1901`` — describe_index_stats: Serverless and Starter reject a non-empty filter

``QueryRequest.filter`` carries no ``minProperties``, and the backend's
``validate_filter`` does not reject an empty one, so ``query`` is deliberately
absent from the empty-filter rules below.

Backend basis (``pinecone-db`` @ f6fd0a40), authoritative on behavior and wording:

- ``pc-validation/src/data_plane/mod.rs:365-416`` — delete: ``delete_all`` rejects
  ids and filter; with a filter present ``ids`` are never inspected (#149); empty
  filter rejected
- ``pc-validation/src/data_plane/mod.rs:259-336`` — query: ``id`` XOR vector on a
  dense index, ``id`` XOR sparse_vector on a sparse one, so ``id`` plus literal
  vector data of either form is rejected whichever index type serves it
- ``pc-validation/src/data_plane/mod.rs:795-812`` — update: filter plus values or
  sparse_values rejected; empty filter rejected
- ``pc-validation/src/data_plane/mod.rs:85-104`` (``validate_vector_id``),
  ``pc-validation/src/lib.rs:68-89`` (``validate_prefix``), ``:162-176``
  (list limit 1..``max_list_limit``)
- ``admission-control/src/validation.rs:51-125`` — fetch requires ids; fetch by
  metadata requires a non-empty filter and a limit in 1..max
- ``config/default.toml:241,261,267`` — ``max_id_length`` 512, ``max_list_limit``
  100, ``max_vectors_per_fetch_by_metadata_request`` 10000
- ``pc-validation/src/error.rs`` — the server strings quoted verbatim in the
  client-side messages

Every rule here was probed against the ``minicone`` simulator @ ``b5764e9`` and
rejected there too, except the per-ID and per-prefix charset rules, which the
simulator accepts and production refuses — a simulator gap, filed as
``pinecone-io/minicone#55``. The backend and the OAS agree, so precedence puts the
rule in.

Two tables here are the cross-lane parity fixtures the async (#123) and gRPC
(#124) lanes bind their own callables to, following the precedent
``tests/unit/test_namespace_validation.py`` set: ``QUERY_TRUTH_TABLE`` is the
spec's ``anyOf``/``not`` truth table, and ``VECTOR_OP_VALIDATION_CASES`` pins the
exact rejection text. Identical inputs must produce identical ``ValidationError``
messages in every lane, so the expected text lives here once.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import orjson
import pytest
import respx
from hypothesis import given
from hypothesis import strategies as st

from pinecone import Index
from pinecone._internal.constants import DATA_PLANE_API_VERSION
from pinecone._internal.validation import (
    require_delete_selectors,
    require_query_selectors,
    require_update_selectors,
)
from pinecone.errors.exceptions import ApiError, ValidationError

INDEX_HOST = "vec-validation-abc123.svc.pinecone.io"
BASE_URL = f"https://{INDEX_HOST}"

ID_MAX = 512
PREFIX_MAX = 512
LIST_LIMIT_MAX = 100
FBM_LIMIT_MAX = 10_000

FILTER = {"genre": {"$eq": "comedy"}}
VECTOR = [0.1, 0.2]
SPARSE = {"indices": [1], "values": [0.5]}
VECTOR_ID = "vec-1"

OVERLONG = "a" * (ID_MAX + 1)
OVERLONG_ECHO = f"{'a' * 32!r}...{'a' * 32!r} ({ID_MAX + 1} characters)"
"""How an over-limit value is echoed back: elided in the middle, with its true length."""


@pytest.fixture
def index() -> Any:
    client = Index(host=INDEX_HOST, api_key="test-key")
    yield client
    client.close()


def _query_response() -> dict[str, Any]:
    return {"matches": [], "namespace": "", "usage": {"readUnits": 1}}


def _fetch_response() -> dict[str, Any]:
    return {"vectors": {}, "namespace": "", "usage": {"readUnits": 1}}


def _list_response() -> dict[str, Any]:
    return {"vectors": [], "namespace": "", "usage": {"readUnits": 1}}


# ---------------------------------------------------------------------------
# The query anyOf/not truth table (cross-lane fixture)
# ---------------------------------------------------------------------------

QUERY_TRUTH_TABLE: list[tuple[str, bool, bool, bool, bool]] = [
    ("none", False, False, False, False),
    ("vector", True, False, False, True),
    ("id", False, True, False, True),
    ("sparse_vector", False, False, True, True),
    ("vector+sparse_vector", True, False, True, True),
    ("id+vector", True, True, False, False),
    ("id+sparse_vector", False, True, True, False),
    ("id+vector+sparse_vector", True, True, True, False),
]
"""``(case_id, has_vector, has_id, has_sparse, accepted)`` per 2026-07's ``anyOf``/``not``.

``id`` names a stored vector, so it stands alone; the two literal forms combine
into a hybrid query. Everything else is a contradiction or an empty selector set.
"""


def query_kwargs(has_vector: bool, has_id: bool, has_sparse: bool) -> dict[str, Any]:
    """Build ``query`` kwargs for one row of :data:`QUERY_TRUTH_TABLE`."""
    kwargs: dict[str, Any] = {"top_k": 1}
    if has_vector:
        kwargs["vector"] = VECTOR
    if has_id:
        kwargs["id"] = VECTOR_ID
    if has_sparse:
        kwargs["sparse_vector"] = SPARSE
    return kwargs


@pytest.mark.parametrize(
    ("has_vector", "has_id", "has_sparse", "accepted"),
    [pytest.param(v, i, s, ok, id=case_id) for case_id, v, i, s, ok in QUERY_TRUTH_TABLE],
)
@respx.mock
def test_query_selector_truth_table(
    index: Index,
    has_vector: bool,
    has_id: bool,
    has_sparse: bool,
    accepted: bool,
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.post(f"{BASE_URL}/query").mock(
        return_value=httpx.Response(200, json=_query_response())
    )
    kwargs = query_kwargs(has_vector, has_id, has_sparse)

    if accepted:
        index.query(**kwargs)
        assert route.called
    else:
        with pytest.raises(ValidationError):
            index.query(**kwargs)
        assert not respx_mock.calls, "rejection must happen before the request goes out"


@respx.mock
def test_query_sends_exactly_the_selectors_it_was_given(
    index: Index, respx_mock: respx.MockRouter
) -> None:
    """A hybrid query carries both literal forms and no ``id``."""
    route = respx_mock.post(f"{BASE_URL}/query").mock(
        return_value=httpx.Response(200, json=_query_response())
    )
    index.query(top_k=3, vector=VECTOR, sparse_vector=SPARSE)

    body = orjson.loads(route.calls.last.request.content)
    assert body["vector"] == VECTOR
    assert body["sparseVector"] == SPARSE
    assert "id" not in body


# ---------------------------------------------------------------------------
# The cross-lane rejection table
# ---------------------------------------------------------------------------

Invoke = Callable[[Index], object]

VECTOR_OP_VALIDATION_CASES: list[tuple[str, Invoke, str]] = [
    (
        "query_id_and_vector",
        lambda idx: idx.query(top_k=1, id=VECTOR_ID, vector=VECTOR),
        "id is mutually exclusive with vector — a query uses a stored vector's id "
        "OR literal vector data, not both. "
        "Pass id alone to query by stored vector, or vector alone to query by value. "
        "Cannot provide both 'ID' and 'vector' at the same time",
    ),
    (
        "query_id_and_sparse_vector",
        lambda idx: idx.query(top_k=1, id=VECTOR_ID, sparse_vector=SPARSE),
        "id is mutually exclusive with sparse_vector — a query uses a stored vector's id "
        "OR literal vector data, not both. "
        "Pass id alone to query by stored vector, or sparse_vector alone to query by value. "
        "Cannot provide both 'ID' and 'sparse_vector' at the same time",
    ),
    (
        "query_no_selector",
        lambda idx: idx.query(top_k=1),
        "At least one of vector, id, or sparse_vector must be provided",
    ),
    (
        "query_non_ascii_id",
        lambda idx: idx.query(top_k=1, id="naïve"),
        "id must contain only ASCII characters, got: 'naïve'",
    ),
    (
        "query_overlong_id",
        lambda idx: idx.query(top_k=1, id=OVERLONG),
        f"id exceeds the maximum length of {ID_MAX} characters, got {ID_MAX + 1}: {OVERLONG_ECHO}",
    ),
    (
        "delete_no_selector",
        lambda idx: idx.delete(),
        "Must specify one of ids, delete_all, or filter",
    ),
    (
        "delete_ids_and_filter",
        lambda idx: idx.delete(ids=[VECTOR_ID], filter=FILTER),
        "Cannot combine ids and filter — specify exactly one. "
        "The server silently ignores ids when a filter is present, so a request "
        "carrying both would delete every record the filter matches, not the "
        "intersection. To delete the intersection, query with the filter first "
        "and delete the returned ids.",
    ),
    (
        "delete_ids_and_delete_all",
        lambda idx: idx.delete(ids=[VECTOR_ID], delete_all=True),
        "Cannot combine ids and delete_all — specify exactly one. "
        "delete_all=True already covers every record in the namespace. "
        "No explicit IDs allowed when delete_all=true",
    ),
    (
        "delete_all_and_filter",
        lambda idx: idx.delete(delete_all=True, filter=FILTER),
        "Cannot combine delete_all and filter — specify exactly one. "
        "delete_all=True already covers every record in the namespace. "
        "No filter allowed when delete_all=true",
    ),
    (
        "delete_empty_filter",
        lambda idx: idx.delete(filter={}),
        "filter must contain at least one condition, got {}. "
        "Delete with empty metadata filter is not allowed",
    ),
    (
        "delete_empty_ids",
        lambda idx: idx.delete(ids=[]),
        "ids must be a non-empty list",
    ),
    (
        "delete_nul_id",
        lambda idx: idx.delete(ids=["a\x00b"]),
        "ids[0] must not contain null characters, got: 'a\\x00b'",
    ),
    (
        "update_filter_and_values",
        lambda idx: idx.update(filter=FILTER, values=VECTOR),
        "filter is mutually exclusive with values — a by-filter update is "
        "metadata-only, because it spans every record the filter matches. "
        "Pass set_metadata to update metadata by filter, or id to update one "
        "record's vector values. "
        "Update by metadata request does not support updating vector values.",
    ),
    (
        "update_filter_and_sparse_values",
        lambda idx: idx.update(filter=FILTER, sparse_values=SPARSE),
        "filter is mutually exclusive with sparse_values — a by-filter update is "
        "metadata-only, because it spans every record the filter matches. "
        "Pass set_metadata to update metadata by filter, or id to update one "
        "record's sparse values. "
        "Update by metadata request does not support updating vector values.",
    ),
    (
        "update_empty_filter",
        lambda idx: idx.update(filter={}, set_metadata={"year": 2020}),
        "filter must contain at least one condition, got {}. "
        "Update with empty metadata filter is not allowed",
    ),
    (
        "update_id_and_filter",
        lambda idx: idx.update(id=VECTOR_ID, filter=FILTER),
        "Exactly one of id or filter must be provided, not both",
    ),
    (
        "update_no_target",
        lambda idx: idx.update(values=VECTOR),
        "Exactly one of id or filter must be provided, got neither",
    ),
    (
        "update_empty_id",
        lambda idx: idx.update(id="", values=VECTOR),
        f"id must not be empty; vector IDs are 1-{ID_MAX} characters",
    ),
    (
        "fetch_empty_ids",
        lambda idx: idx.fetch(ids=[]),
        "ids must be a non-empty list",
    ),
    (
        "fetch_non_ascii_id",
        lambda idx: idx.fetch(ids=["ok", "naïve"]),
        "ids[1] must contain only ASCII characters, got: 'naïve'",
    ),
    (
        "fetch_overlong_id",
        lambda idx: idx.fetch(ids=[OVERLONG]),
        f"ids[0] exceeds the maximum length of {ID_MAX} characters, "
        f"got {ID_MAX + 1}: {OVERLONG_ECHO}",
    ),
    (
        "fetch_empty_id",
        lambda idx: idx.fetch(ids=[""]),
        f"ids[0] must not be empty; vector IDs are 1-{ID_MAX} characters",
    ),
    (
        "fetch_nul_id",
        lambda idx: idx.fetch(ids=["a\x00b"]),
        "ids[0] must not contain null characters, got: 'a\\x00b'",
    ),
    (
        "fetch_by_metadata_empty_filter",
        lambda idx: idx.fetch_by_metadata(filter={}),
        "filter must contain at least one condition, got {}. "
        "Empty filter provided for fetch by metadata request",
    ),
    (
        "fetch_by_metadata_limit_zero",
        lambda idx: idx.fetch_by_metadata(filter=FILTER, limit=0),
        f"limit must be between 1 and {FBM_LIMIT_MAX}, got 0",
    ),
    (
        "fetch_by_metadata_limit_over_max",
        lambda idx: idx.fetch_by_metadata(filter=FILTER, limit=FBM_LIMIT_MAX + 1),
        f"limit must be between 1 and {FBM_LIMIT_MAX}, got {FBM_LIMIT_MAX + 1}",
    ),
    (
        "list_limit_zero",
        lambda idx: idx.list_paginated(limit=0),
        f"limit must be between 1 and {LIST_LIMIT_MAX}, got 0",
    ),
    (
        "list_limit_over_max",
        lambda idx: idx.list_paginated(limit=LIST_LIMIT_MAX + 1),
        f"limit must be between 1 and {LIST_LIMIT_MAX}, got {LIST_LIMIT_MAX + 1}",
    ),
    (
        "list_prefix_non_ascii",
        lambda idx: idx.list_paginated(prefix="naïve"),
        "prefix must contain only ASCII characters, got: 'naïve'",
    ),
    (
        "list_prefix_overlong",
        lambda idx: idx.list_paginated(prefix=OVERLONG),
        f"prefix must be at most {PREFIX_MAX} characters, got {PREFIX_MAX + 1}: {OVERLONG_ECHO}",
    ),
    (
        "list_prefix_nul",
        lambda idx: idx.list_paginated(prefix="a\x00b"),
        "prefix must not contain null characters, got: 'a\\x00b'",
    ),
    (
        "list_generator_limit_over_max",
        lambda idx: next(iter(idx.list(limit=LIST_LIMIT_MAX + 1))),
        f"limit must be between 1 and {LIST_LIMIT_MAX}, got {LIST_LIMIT_MAX + 1}",
    ),
]


@pytest.mark.parametrize(
    ("invoke", "expected"),
    [pytest.param(fn, msg, id=case_id) for case_id, fn, msg in VECTOR_OP_VALIDATION_CASES],
)
@respx.mock
def test_rejected_before_any_http_call(
    index: Index, invoke: Invoke, expected: str, respx_mock: respx.MockRouter
) -> None:
    with pytest.raises(ValidationError) as excinfo:
        invoke(index)

    assert str(excinfo.value) == expected
    assert not respx_mock.calls, "validation must reject before the request goes out"


def test_every_message_names_the_offending_argument() -> None:
    """Every rejection has to be actionable without reading the SDK source."""
    named = {"id", "ids", "filter", "limit", "prefix", "At", "Must", "Cannot", "Exactly"}
    for case_id, _fn, message in VECTOR_OP_VALIDATION_CASES:
        first = message.split()[0].split("[")[0]
        assert first in named, f"{case_id}: message does not open by naming the rule: {message!r}"


def test_exclusivity_messages_name_both_arguments() -> None:
    """A rejection for an illegal pair has to name both halves of the pair."""
    pairs = {
        "query_id_and_vector": ("id", "vector"),
        "query_id_and_sparse_vector": ("id", "sparse_vector"),
        "delete_ids_and_filter": ("ids", "filter"),
        "delete_ids_and_delete_all": ("ids", "delete_all"),
        "delete_all_and_filter": ("delete_all", "filter"),
        "update_filter_and_values": ("filter", "values"),
        "update_filter_and_sparse_values": ("filter", "sparse_values"),
    }
    by_id = {case_id: msg for case_id, _fn, msg in VECTOR_OP_VALIDATION_CASES}
    for case_id, (left, right) in pairs.items():
        message = by_id[case_id]
        assert left in message and right in message, f"{case_id}: {message!r}"


def test_delete_ids_and_filter_message_explains_why_it_is_blocked() -> None:
    """The SDK is stricter than the spec here (#149), so the message has to justify it."""
    by_id = {case_id: msg for case_id, _fn, msg in VECTOR_OP_VALIDATION_CASES}
    message = by_id["delete_ids_and_filter"]
    assert "silently ignores ids" in message
    assert "query with the filter first" in message


def test_update_by_filter_messages_say_metadata_only() -> None:
    by_id = {case_id: msg for case_id, _fn, msg in VECTOR_OP_VALIDATION_CASES}
    for case_id in ("update_filter_and_values", "update_filter_and_sparse_values"):
        assert "metadata-only" in by_id[case_id]
        assert "set_metadata" in by_id[case_id]


@pytest.mark.parametrize(
    ("invoke", "server_text"),
    [
        pytest.param(
            lambda idx: idx.delete(filter={}),
            "Delete with empty metadata filter is not allowed",
            id="delete",
        ),
        pytest.param(
            lambda idx: idx.update(filter={}, set_metadata={"a": 1}),
            "Update with empty metadata filter is not allowed",
            id="update",
        ),
        pytest.param(
            lambda idx: idx.fetch_by_metadata(filter={}),
            "Empty filter provided for fetch by metadata request",
            id="fetch_by_metadata",
        ),
    ],
)
def test_empty_filter_quotes_the_server_wording_verbatim(
    index: Index, invoke: Invoke, server_text: str
) -> None:
    """Same bytes locally as from the server, per the #100 precedent."""
    with pytest.raises(ValidationError) as excinfo:
        invoke(index)

    assert "filter must contain at least one condition" in str(excinfo.value)
    assert server_text in str(excinfo.value)


# ---------------------------------------------------------------------------
# The shared validators are the cross-lane contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("has_vector", "has_id", "has_sparse", "accepted"),
    [pytest.param(v, i, s, ok, id=case_id) for case_id, v, i, s, ok in QUERY_TRUTH_TABLE],
)
def test_shared_query_validator_matches_the_truth_table(
    has_vector: bool, has_id: bool, has_sparse: bool, accepted: bool
) -> None:
    """The rule lives in the shared module, not in ``Index``.

    Every transport calls this one function, so pinning it here is what makes
    the asyncio (#123) and gRPC (#124) lanes inherit identical text instead of
    re-deriving it. No client and no transport involved.
    """
    kwargs = {
        "vector": VECTOR if has_vector else None,
        "id": VECTOR_ID if has_id else None,
        "sparse_vector": SPARSE if has_sparse else None,
    }
    if accepted:
        assert require_query_selectors(**kwargs) is None
    else:
        with pytest.raises(ValidationError):
            require_query_selectors(**kwargs)


def test_shared_delete_validator_keeps_exactly_one_semantics() -> None:
    assert require_delete_selectors(ids=["v1"], delete_all=False, filter=None) is None
    assert require_delete_selectors(ids=None, delete_all=True, filter=None) is None
    assert require_delete_selectors(ids=None, delete_all=False, filter=FILTER) is None

    with pytest.raises(ValidationError, match="silently ignores ids"):
        require_delete_selectors(ids=["v1"], delete_all=False, filter=FILTER)
    with pytest.raises(ValidationError, match="Must specify one of"):
        require_delete_selectors(ids=None, delete_all=False, filter=None)


def test_shared_update_validator_keeps_by_filter_metadata_only() -> None:
    assert (
        require_update_selectors(id="v1", filter=None, values=VECTOR, sparse_values=SPARSE) is None
    )
    assert require_update_selectors(id=None, filter=FILTER, values=None, sparse_values=None) is None

    with pytest.raises(ValidationError, match="metadata-only"):
        require_update_selectors(id=None, filter=FILTER, values=VECTOR, sparse_values=None)
    with pytest.raises(ValidationError, match="metadata-only"):
        require_update_selectors(id=None, filter=FILTER, values=None, sparse_values=SPARSE)


def test_index_delegates_to_the_shared_validators(index: Index) -> None:
    """A lane that restated the rule locally would drift; this catches that.

    The client must raise the shared module's text verbatim, which holds only if
    it calls the shared function rather than reimplementing the check.
    """
    with pytest.raises(ValidationError) as from_client:
        index.query(top_k=1, id=VECTOR_ID, sparse_vector=SPARSE)
    with pytest.raises(ValidationError) as from_validator:
        require_query_selectors(vector=None, id=VECTOR_ID, sparse_vector=SPARSE)

    assert str(from_client.value) == str(from_validator.value)


# ---------------------------------------------------------------------------
# Accepted inputs — the rules must not over-reject
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vector_id",
    [
        pytest.param("a", id="single_char"),
        pytest.param("a" * ID_MAX, id="at_length_limit"),
        pytest.param("doc1#chunk2", id="hash_separator"),
        pytest.param(" ", id="single_space"),
        pytest.param("\x01", id="lowest_legal_code_point"),
        pytest.param("\x7f", id="highest_legal_code_point"),
        pytest.param("id/with/slashes", id="slashes_are_legal_in_a_query_param"),
    ],
)
@respx.mock
def test_fetch_accepts_legal_ids(index: Index, vector_id: str) -> None:
    route = respx.get(url__startswith=f"{BASE_URL}/vectors/fetch").mock(
        return_value=httpx.Response(200, json=_fetch_response())
    )
    index.fetch(ids=[vector_id])
    assert route.calls.last.request.url.params.get_list("ids") == [vector_id]


@respx.mock
def test_fetch_accepts_many_ids_and_sends_them_all(index: Index) -> None:
    route = respx.get(url__startswith=f"{BASE_URL}/vectors/fetch").mock(
        return_value=httpx.Response(200, json=_fetch_response())
    )
    ids = [f"vec-{n}" for n in range(50)]
    index.fetch(ids=ids)
    assert route.calls.last.request.url.params.get_list("ids") == ids


@respx.mock
def test_query_accepts_legal_id(index: Index) -> None:
    route = respx.post(f"{BASE_URL}/query").mock(
        return_value=httpx.Response(200, json=_query_response())
    )
    index.query(top_k=1, id="a" * ID_MAX)
    assert route.called


@respx.mock
def test_update_accepts_legal_id_with_values(index: Index) -> None:
    route = respx.post(f"{BASE_URL}/vectors/update").mock(return_value=httpx.Response(200, json={}))
    index.update(id="vec-1", values=VECTOR, sparse_values=SPARSE)

    body = orjson.loads(route.calls.last.request.content)
    assert body["id"] == "vec-1"
    assert body["values"] == VECTOR


@respx.mock
def test_update_by_filter_accepts_set_metadata_and_dry_run(index: Index) -> None:
    """The legal by-filter shape: metadata only."""
    route = respx.post(f"{BASE_URL}/vectors/update").mock(
        return_value=httpx.Response(200, json={"matchedRecords": 42})
    )
    result = index.update(filter=FILTER, set_metadata={"year": 2020}, dry_run=True)

    body = orjson.loads(route.calls.last.request.content)
    assert body["filter"] == FILTER
    assert body["setMetadata"] == {"year": 2020}
    assert body["dryRun"] is True
    assert "values" not in body and "sparseValues" not in body
    assert result.matched_records == 42


@respx.mock
def test_delete_accepts_each_mode_alone(index: Index) -> None:
    route = respx.post(f"{BASE_URL}/vectors/delete").mock(return_value=httpx.Response(200, json={}))
    index.delete(ids=["vec-1"])
    index.delete(delete_all=True)
    index.delete(filter=FILTER)

    bodies = [orjson.loads(call.request.content) for call in route.calls]
    assert bodies[0]["ids"] == ["vec-1"]
    assert bodies[1]["deleteAll"] is True
    assert bodies[2]["filter"] == FILTER


@respx.mock
def test_delete_all_false_is_not_a_selector(index: Index) -> None:
    """``delete_all=False`` is the default, not a third mode, so ids stay legal alongside it."""
    route = respx.post(f"{BASE_URL}/vectors/delete").mock(return_value=httpx.Response(200, json={}))
    index.delete(ids=["vec-1"], delete_all=False)
    assert route.called


@pytest.mark.parametrize(
    "limit",
    [pytest.param(1, id="lower_bound"), pytest.param(LIST_LIMIT_MAX, id="upper_bound")],
)
@respx.mock
def test_list_limit_bounds_are_inclusive(index: Index, limit: int) -> None:
    route = respx.get(url__startswith=f"{BASE_URL}/vectors/list").mock(
        return_value=httpx.Response(200, json=_list_response())
    )
    index.list_paginated(limit=limit)
    assert route.calls.last.request.url.params["limit"] == str(limit)


@pytest.mark.parametrize(
    "limit",
    [pytest.param(1, id="lower_bound"), pytest.param(FBM_LIMIT_MAX, id="upper_bound")],
)
@respx.mock
def test_fetch_by_metadata_limit_bounds_are_inclusive(index: Index, limit: int) -> None:
    route = respx.post(f"{BASE_URL}/vectors/fetch_by_metadata").mock(
        return_value=httpx.Response(200, json=_fetch_response())
    )
    index.fetch_by_metadata(filter=FILTER, limit=limit)

    body = orjson.loads(route.calls.last.request.content)
    assert body["limit"] == limit


@respx.mock
def test_empty_list_prefix_is_accepted_and_sent(index: Index) -> None:
    """``^[\\x01-\\x7F]*$`` — unlike an ID, the prefix may be empty; it matches everything."""
    route = respx.get(url__startswith=f"{BASE_URL}/vectors/list").mock(
        return_value=httpx.Response(200, json=_list_response())
    )
    index.list_paginated(prefix="")
    assert route.calls.last.request.url.params["prefix"] == ""


@respx.mock
def test_list_prefix_at_length_limit_is_accepted(index: Index) -> None:
    route = respx.get(url__startswith=f"{BASE_URL}/vectors/list").mock(
        return_value=httpx.Response(200, json=_list_response())
    )
    index.list_paginated(prefix="a" * PREFIX_MAX)
    assert route.called


# ---------------------------------------------------------------------------
# describe_index_stats — index-type policy stays server-side
# ---------------------------------------------------------------------------


@respx.mock
def test_describe_index_stats_forwards_a_non_empty_filter(index: Index) -> None:
    """Whether filtering is supported depends on the index, which the client cannot know."""
    route = respx.post(f"{BASE_URL}/describe_index_stats").mock(
        return_value=httpx.Response(200, json={"namespaces": {}, "totalVectorCount": 0})
    )
    index.describe_index_stats(filter=FILTER)

    assert orjson.loads(route.calls.last.request.content)["filter"] == FILTER


@respx.mock
def test_describe_index_stats_surfaces_the_server_rejection_text(index: Index) -> None:
    """Serverless and Starter 4xx a non-empty filter; the body text has to reach the caller."""
    message = (
        "Serverless and Starter indexes do not support describing index stats "
        "with metadata filtering."
    )
    respx.post(f"{BASE_URL}/describe_index_stats").mock(
        return_value=httpx.Response(400, json={"code": 3, "message": message, "details": []})
    )
    with pytest.raises(ApiError) as excinfo:
        index.describe_index_stats(filter=FILTER)

    assert excinfo.value.status_code == 400
    assert message in str(excinfo.value)


@respx.mock
def test_describe_index_stats_accepts_an_empty_filter(index: Index) -> None:
    """The backend only rejects a *non-empty* filter, so an empty one must not be blocked."""
    route = respx.post(f"{BASE_URL}/describe_index_stats").mock(
        return_value=httpx.Response(200, json={"namespaces": {}, "totalVectorCount": 0})
    )
    index.describe_index_stats(filter={})
    assert route.called


@respx.mock
def test_query_accepts_an_empty_filter(index: Index) -> None:
    """``QueryRequest.filter`` has no ``minProperties``; tightening it would exceed the server."""
    route = respx.post(f"{BASE_URL}/query").mock(
        return_value=httpx.Response(200, json=_query_response())
    )
    index.query(top_k=1, vector=VECTOR, filter={})
    assert route.called


# ---------------------------------------------------------------------------
# X-Pinecone-Api-Version per operation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path", "response", "invoke"),
    [
        pytest.param(
            "POST",
            "/query",
            _query_response(),
            lambda idx: idx.query(top_k=1, vector=VECTOR),
            id="queryVectors",
        ),
        pytest.param(
            "GET",
            "/vectors/fetch",
            _fetch_response(),
            lambda idx: idx.fetch(ids=["vec-1"]),
            id="fetchVectors",
        ),
        pytest.param(
            "POST",
            "/vectors/fetch_by_metadata",
            _fetch_response(),
            lambda idx: idx.fetch_by_metadata(filter=FILTER),
            id="fetch_vectors_by_metadata",
        ),
        pytest.param(
            "POST",
            "/vectors/delete",
            {},
            lambda idx: idx.delete(ids=["vec-1"]),
            id="deleteVectors",
        ),
        pytest.param(
            "POST",
            "/vectors/update",
            {},
            lambda idx: idx.update(id="vec-1", values=VECTOR),
            id="updateVector",
        ),
        pytest.param(
            "GET",
            "/vectors/list",
            _list_response(),
            lambda idx: idx.list_paginated(limit=10),
            id="listVectors",
        ),
        pytest.param(
            "POST",
            "/describe_index_stats",
            {"namespaces": {}, "totalVectorCount": 0},
            lambda idx: idx.describe_index_stats(),
            id="describeIndexStats",
        ),
    ],
)
@respx.mock
def test_api_version_header_on_every_vector_op(
    index: Index, method: str, path: str, response: dict[str, Any], invoke: Invoke
) -> None:
    route = respx.request(method, url__startswith=f"{BASE_URL}{path}").mock(
        return_value=httpx.Response(200, json=response)
    )
    invoke(index)

    assert route.calls.last.request.headers["X-Pinecone-Api-Version"] == DATA_PLANE_API_VERSION
    assert DATA_PLANE_API_VERSION == "2026-07"


# ---------------------------------------------------------------------------
# Properties: the client-side rules are exactly the spec's
# ---------------------------------------------------------------------------

_legal_chars = st.characters(min_codepoint=1, max_codepoint=127)
_legal_ids = st.text(alphabet=_legal_chars, min_size=1, max_size=ID_MAX)
_illegal_ids = st.one_of(
    st.just(""),
    st.text(alphabet=_legal_chars, min_size=ID_MAX + 1, max_size=ID_MAX + 40),
    st.text(alphabet=st.characters(min_codepoint=128), min_size=1, max_size=20),
    st.text(alphabet=_legal_chars, max_size=20).map(lambda s: s + "\x00"),
)


@given(vector_id=_legal_ids)
def test_every_legal_id_passes_fetch_validation(vector_id: str) -> None:
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(url__startswith=f"{BASE_URL}/vectors/fetch").mock(
            return_value=httpx.Response(200, json=_fetch_response())
        )
        client = Index(host=INDEX_HOST, api_key="test-key")
        try:
            client.fetch(ids=[vector_id])
        finally:
            client.close()
        assert route.called


@given(vector_id=_illegal_ids)
def test_every_illegal_id_is_rejected_before_http(vector_id: str) -> None:
    with respx.mock as mock:
        client = Index(host=INDEX_HOST, api_key="test-key")
        try:
            with pytest.raises(ValidationError):
                client.fetch(ids=[vector_id])
            with pytest.raises(ValidationError):
                client.delete(ids=[vector_id])
        finally:
            client.close()
        assert not mock.calls


@given(prefix=st.text(alphabet=_legal_chars, max_size=PREFIX_MAX))
def test_every_legal_prefix_passes_list_validation(prefix: str) -> None:
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(url__startswith=f"{BASE_URL}/vectors/list").mock(
            return_value=httpx.Response(200, json=_list_response())
        )
        client = Index(host=INDEX_HOST, api_key="test-key")
        try:
            client.list_paginated(prefix=prefix)
        finally:
            client.close()
        assert route.called


@given(limit=st.integers(min_value=1, max_value=LIST_LIMIT_MAX))
def test_every_legal_list_limit_passes_validation(limit: int) -> None:
    with respx.mock as mock:
        mock.get(url__startswith=f"{BASE_URL}/vectors/list").mock(
            return_value=httpx.Response(200, json=_list_response())
        )
        client = Index(host=INDEX_HOST, api_key="test-key")
        try:
            client.list_paginated(limit=limit)
        finally:
            client.close()


@given(
    limit=st.one_of(
        st.integers(max_value=0), st.integers(min_value=LIST_LIMIT_MAX + 1, max_value=10**6)
    )
)
def test_every_illegal_list_limit_is_rejected_before_http(limit: int) -> None:
    with respx.mock as mock:
        client = Index(host=INDEX_HOST, api_key="test-key")
        try:
            with pytest.raises(
                ValidationError, match=f"limit must be between 1 and {LIST_LIMIT_MAX}"
            ):
                client.list_paginated(limit=limit)
        finally:
            client.close()
        assert not mock.calls


@given(has_vector=st.booleans(), has_id=st.booleans(), has_sparse=st.booleans())
def test_query_selectors_match_the_spec_truth_table(
    has_vector: bool, has_id: bool, has_sparse: bool
) -> None:
    """Randomized presence combinations must land exactly on 2026-07's anyOf/not."""
    spec_allows = (has_vector or has_id or has_sparse) and not (
        has_id and (has_vector or has_sparse)
    )

    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(f"{BASE_URL}/query").mock(
            return_value=httpx.Response(200, json=_query_response())
        )
        client = Index(host=INDEX_HOST, api_key="test-key")
        try:
            kwargs = query_kwargs(has_vector, has_id, has_sparse)
            if spec_allows:
                client.query(**kwargs)
                assert route.called
            else:
                with pytest.raises(ValidationError):
                    client.query(**kwargs)
                assert not mock.calls
        finally:
            client.close()


@given(
    values=st.one_of(st.none(), st.just(VECTOR)),
    sparse_values=st.one_of(st.none(), st.just(SPARSE)),
)
def test_update_by_filter_rejects_any_vector_payload(
    values: list[float] | None, sparse_values: dict[str, Any] | None
) -> None:
    """A by-filter update accepts metadata and nothing else, whichever value form is passed."""
    carries_vector_data = values is not None or sparse_values is not None

    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(f"{BASE_URL}/vectors/update").mock(
            return_value=httpx.Response(200, json={"matchedRecords": 1})
        )
        client = Index(host=INDEX_HOST, api_key="test-key")
        try:
            if carries_vector_data:
                with pytest.raises(ValidationError, match="metadata-only"):
                    client.update(filter=FILTER, values=values, sparse_values=sparse_values)
                assert not mock.calls
            else:
                client.update(filter=FILTER, set_metadata={"year": 2020})
                assert route.called
        finally:
            client.close()
