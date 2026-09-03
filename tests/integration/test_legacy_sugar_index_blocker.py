"""Tripwire for the minicone gap that blocks test_legacy_sugar_index.py.

test_legacy_sugar_index.py and its async twin mark themselves module-skip
because minicone does not yet implement pinecone-io/pinecone-db#18066 — the
change that routes an index whose schema is exactly the reserved _values/
_sparse_values field(s) to the vectors API. Without it, every index those
modules build through Indexes.create()'s legacy sugar lands on the documents
API instead, which refuses vectors-API writes outright (#322).

A plain module skip has no way to notice when that gap closes: nothing ever
un-skips it, so the ~840 lines of round-trip proof in those two modules would
stay silently unexecuted forever, even against a backend that has the fix.
This test is the tripwire. It is deliberately not skipped, and it does the
cheapest possible check — one index, one upsert, no polling — rather than
duplicating the full round trip:

* The write still being refused is today's expected state: this test passes.
* The write being accepted means #18066 has landed, so this test FAILS with a
  message naming exactly what to do next: remove the skip in
  test_legacy_sugar_index.py and test_legacy_sugar_index_async.py so the real
  proof starts running, then delete this file.
"""

from __future__ import annotations

import uuid

import pytest

from pinecone import Pinecone, ServerlessSpec
from pinecone.errors import ApiError
from tests.integration.conftest import ensure_index_deleted, unique_name

CLOUD = "aws"
REGION = "us-east-1"


@pytest.mark.integration
def test_blocker_still_present(client: Pinecone) -> None:
    """Fails the moment pinecone-io/pinecone-db#18066 lands, so the sibling skips can't rot."""
    name = unique_name("sugar-blocker-probe")
    model = client.indexes.create(
        name=name, dimension=1, metric="cosine", spec=ServerlessSpec(cloud=CLOUD, region=REGION)
    )
    try:
        index = client.index(host=model.host)
        try:
            index.upsert(
                vectors=[{"id": "probe", "values": [0.0]}],
                namespace=f"blocker-probe-{uuid.uuid4().hex[:8]}",
            )
        except ApiError as exc:
            assert exc.status_code == 400, (
                f"expected the documents API to refuse this write with a 400, "
                f"got {exc.status_code}: {exc}"
            )
            return

        pytest.fail(
            "BLOCKER RESOLVED: pinecone-io/pinecone-db#18066 appears fixed — a "
            "sugar-created index just accepted a vectors-API upsert instead of "
            "refusing it. Remove the module-level pytest.mark.skip in "
            "tests/integration/test_legacy_sugar_index.py and "
            "tests/integration/test_legacy_sugar_index_async.py so the real "
            "round-trip proof runs, then delete this tripwire test."
        )
    finally:
        ensure_index_deleted(client, name)
