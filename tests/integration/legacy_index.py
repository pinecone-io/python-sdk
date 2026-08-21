"""Create a **legacy** (vectors-API) index for the integration suite.

Why the SDK's own create path is bypassed
-----------------------------------------
2026-07 offers no way to create a vectors-API index. Its ``POST /indexes``
always persists an index schema, and a schema naming its own data fields is
served by the *documents* API, which refuses every vectors-API call (#322).
So a test that needs ``upsert`` / ``query`` / ``fetch`` to **succeed** cannot
get its index from ``pc.indexes.create`` — and ``pc.indexes.create`` no longer
accepts the ``dimension`` / ``metric`` / ``spec`` shape at all.

Vector operations nevertheless remain intended for indexes created under
earlier API versions, and a production workload upserting and querying such
an index must not break when it upgrades the SDK. Proving that is what these
helpers exist for.

The sanctioned workaround (decided on #322) is to create the index with a
bespoke REST call against a **previous API version**, then run the vector
operations against it at 2026-07. The version used is
:data:`LEGACY_CREATE_API_VERSION` — the newest one whose ``POST /indexes``
still takes ``dimension`` / ``metric`` / ``spec`` and persists no schema, so
the index it creates is served by the vectors API. The exact call is::

    POST https://api.pinecone.io/indexes
    Api-Key: <key>
    X-Pinecone-Api-Version: 2026-04
    Content-Type: application/json

    {"name": "...", "dimension": 3, "metric": "cosine",
     "vector_type": "dense",
     "spec": {"serverless": {"cloud": "aws", "region": "us-east-1"}}}

A sparse index omits ``dimension`` and passes ``vector_type="sparse"`` with
``metric="dotproduct"``.

``httpx`` is called directly rather than through any part of ``pinecone``:
the SDK is the thing under test, so its transport must not also be the
fixture that sets the test up.

Nothing here imports ``tests.integration.conftest``, at module scope or
inside a function, and it must stay that way. Importing it runs its
module-level ``load_env()``, which would put a real ``PINECONE_API_KEY`` into
``os.environ`` for the rest of the session — and this module is imported by a
unit test, where that silently changes what other unit tests observe. That is
also why the index name is built here rather than borrowed from
``conftest.unique_name``.

Confirm what you got
--------------------
Ask for a legacy index and you may still be handed a document-schema one — by
a backend change, a wrong version constant, or a typo. That failure is silent
and dangerous: on a document-schema index *every* vectors-API call is refused,
so any assertion of the form "this vectors call fails" starts passing for the
wrong reason, and a green run looks identical. #305 lost
``test_dimension_mismatch`` to exactly this.

:func:`assert_serves_vectors_api` is the guard. Describing a legacy index at
2026-07 reports a schema holding only the reserved ``_values`` /
``_sparse_values`` fields; those names cannot be reached from any create call,
so their presence is the signature of a vectors-API index. Every module built
on these helpers should assert it once.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from pinecone import Pinecone
from pinecone._internal.constants import API_VERSION_HEADER, DEFAULT_BASE_URL

LEGACY_CREATE_API_VERSION = "2026-04"
"""Previous API version whose index-create still yields a vectors-API index."""

LEGACY_DENSE_FIELD = "_values"
LEGACY_SPARSE_FIELD = "_sparse_values"

_LEGACY_SCHEMA_FIELDS = frozenset({LEGACY_DENSE_FIELD, LEGACY_SPARSE_FIELD})

_READY_TIMEOUT_SECONDS = 300
_READY_INTERVAL_SECONDS = 3
_DELETE_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class LegacyIndex:
    """A ready legacy index. ``host`` is what data-plane clients need."""

    name: str
    host: str
    dimension: int | None
    metric: str
    vector_type: str


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Api-Key": api_key,
        API_VERSION_HEADER: LEGACY_CREATE_API_VERSION,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def legacy_create_body(
    name: str,
    *,
    dimension: int | None,
    metric: str,
    vector_type: str,
    cloud: str = "aws",
    region: str = "us-east-1",
) -> dict[str, Any]:
    """Build the ``POST /indexes`` body for the legacy create shape.

    Split out from :func:`create_legacy_index` so the wire shape can be
    asserted without a live key (``tests/unit/test_legacy_index_helper.py``).
    """
    body: dict[str, Any] = {
        "name": name,
        "metric": metric,
        "vector_type": vector_type,
        "spec": {"serverless": {"cloud": cloud, "region": region}},
    }
    if dimension is not None:
        body["dimension"] = dimension
    return body


def create_legacy_index(
    api_key: str,
    *,
    dimension: int | None = None,
    metric: str = "cosine",
    vector_type: str = "dense",
    name: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = _READY_TIMEOUT_SECONDS,
) -> LegacyIndex:
    """Create a legacy index and poll until it is ready.

    Raises:
        httpx.HTTPStatusError: If the create call is refused — most likely
            because the API version this helper pins no longer accepts the
            legacy create shape, which would mean the guarantee these tests
            protect can no longer be set up at all. Do not paper over it.
        TimeoutError: If the index does not report ready within *timeout*.
            The index is deleted before this is raised — the caller never
            receives a name, so nothing else can clean it up.
    """
    index_name = name or f"idx-legacy-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    body = legacy_create_body(
        index_name, dimension=dimension, metric=metric, vector_type=vector_type
    )

    with httpx.Client(base_url=base_url, timeout=60.0) as http:
        response = http.post("/indexes", headers=_headers(api_key), json=body)
        response.raise_for_status()

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            described = http.get(f"/indexes/{index_name}", headers=_headers(api_key))
            if described.status_code == 200:
                payload = described.json()
                if payload.get("status", {}).get("ready"):
                    return LegacyIndex(
                        name=index_name,
                        host=payload["host"],
                        dimension=payload.get("dimension"),
                        metric=payload["metric"],
                        vector_type=payload["vector_type"],
                    )
            time.sleep(_READY_INTERVAL_SECONDS)

    delete_legacy_index(api_key, index_name, base_url=base_url)
    raise TimeoutError(f"legacy index {index_name} not ready after {timeout}s")


def delete_legacy_index(
    api_key: str,
    name: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = _DELETE_TIMEOUT_SECONDS,
) -> None:
    """Delete a legacy index and wait until it is gone.

    Best-effort: this runs in fixture teardown, where raising would mask the
    test result. A delete that does not land is reported loudly instead,
    because the index keeps costing real quota until someone notices.

    The wait is the point — the delete is asynchronous, so a helper that only
    fires the request and returns cannot tell a cleanup from a leak. Confirm
    by polling the index away rather than by iterating a listing (#346).
    """
    try:
        with httpx.Client(base_url=base_url, timeout=60.0) as http:
            response = http.delete(f"/indexes/{name}", headers=_headers(api_key))
            if response.status_code >= 400 and response.status_code != 404:
                print(f"  WARNING: delete of legacy index {name} returned {response.status_code}")
                return

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                described = http.get(f"/indexes/{name}", headers=_headers(api_key))
                if described.status_code == 404:
                    print(f"  Cleaned up legacy index: {name}")
                    return
                time.sleep(_READY_INTERVAL_SECONDS)

        print(f"  WARNING: legacy index {name} still present after {timeout}s — may leak quota")
    except Exception as exc:
        print(f"  WARNING: failed to clean up legacy index {name}: {exc}")


def assert_serves_vectors_api(client: Pinecone, index: LegacyIndex) -> None:
    """Assert *index* is served by the vectors API, not the documents API.

    Reads the index through the SDK at 2026-07, so it also covers the
    describe half of the upgrade path: a client on the current API version
    has to be able to read an index it did not create.
    """
    described = client.indexes.describe(index.name)
    schema = described.schema
    assert schema is not None, (
        f"{index.name}: 2026-07 describe returned no schema; cannot tell whether "
        "this index is served by the vectors API"
    )

    declared = set(schema.fields)
    unexpected = declared - _LEGACY_SCHEMA_FIELDS
    assert not unexpected, (
        f"{index.name} declares data fields {sorted(unexpected)}, so it is a "
        f"document-schema index and refuses the entire vectors API. Every "
        f"vectors-API assertion in this module would pass vacuously. Check that "
        f"the index was created at {LEGACY_CREATE_API_VERSION} and that that "
        f"version still persists no schema."
    )
    assert declared, f"{index.name} declares no schema fields at all"
