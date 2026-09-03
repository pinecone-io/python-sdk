"""Integration tests for the collections surface (async REST).

Phase 4 area tag: collection-lifecycle
Transport: rest-async

NOTE: collections are snapshots of pod-based indexes, and API version
2026-07 refuses to create a pod-based index at all
(400 INVALID_ARGUMENT: deployment_type 'pod' is not supported on this API
version). There is therefore no reachable source index for
collections.create(), and no collection lifecycle to exercise here. What is
still reachable is the rejection path below, plus the client-side name
validation in test_client_async.py. See docs/migration/v10-migration.md,
"Pod deployments, and what that means for collections".
"""

from __future__ import annotations

import pytest

from pinecone import AsyncPinecone
from pinecone.errors.exceptions import ApiError
from tests.integration.conftest import async_cleanup_resource, unique_name

# ---------------------------------------------------------------------------
# collection-from-serverless — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_collection_from_serverless_raises_error_async(async_client: AsyncPinecone) -> None:
    """Creating a collection from a serverless index raises ApiError(400) — async REST.

    Collections are a pod-index-only feature (unified-col-0008): the API
    returns HTTP 400 when the caller attempts to snapshot a serverless index.
    This verifies that the async client reaches POST /collections and
    surfaces the refusal as ApiError with status_code=400 rather than as a
    transport or routing error.

    Scope note: on 2026-07 a serverless index is the only index that can be
    created, so this asserts only the 400 and not its reason — it cannot
    distinguish "wrong source type" from any other 400 the collections
    surface may return at this API version. It is a reachability and
    error-mapping check, not a proof of the pod-only rule.

    Area tag: collection-lifecycle
    Transport: rest-async
    Claim: unified-col-0008
    """
    index_name = unique_name("idx")
    col_name = unique_name("col")

    try:
        # Create a plain 2026-07 serverless index. This test does not touch
        # the vectors API (it only checks that collections.create rejects a
        # non-pod source), so it needs no legacy-index workaround.
        await async_client.indexes.create(
            name=index_name,
            schema={
                "fields": {
                    "embedding": {"type": "dense_vector", "dimension": 2, "metric": "cosine"}
                }
            },
            deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
            timeout=120,
        )

        # Attempt to create a collection from the serverless index.
        # The API must reject this with HTTP 400.
        with pytest.raises(ApiError) as exc_info:
            await async_client.collections.create(name=col_name, source=index_name)

        err = exc_info.value
        assert err.status_code == 400, (
            f"Expected HTTP 400 for serverless collection source, got {err.status_code}"
        )

    finally:
        # The collection was never created (error was raised), so only clean
        # up the source index.
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(index_name),
            index_name,
            "index",
        )
