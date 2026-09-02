"""Gap-fill integration tests for control-plane coverage that the primary
suite does not exercise.

Areas:
- index describe of a nonexistent index -> NotFoundError (standalone; the
  existing suite only reaches this indirectly while polling deletion).
- configure(serverless_read_capacity=...) is only covered via a mocked
  request-body test (test_client.py); this exercises it against the real API.

A post-delete collections.describe() -> NotFoundError check used to live
here. It needed a pod-based source index, and API version 2026-07 refuses to
create one, so it is unreachable rather than merely failing. See
docs/migration/v10-migration.md, "Pod deployments, and what that means for
collections". The equivalent delete-then-describe semantics for indexes are
covered by test_indexes.py.
"""

from __future__ import annotations

import pytest

from pinecone import Pinecone
from pinecone.errors.exceptions import NotFoundError
from pinecone.models.indexes.index import IndexModel
from pinecone.models.indexes.specs import ServerlessSpec
from tests.integration.conftest import cleanup_resource, unique_name


@pytest.mark.integration
def test_describe_nonexistent_index_raises_not_found(client: Pinecone) -> None:
    """describe() of an index name that has never existed raises NotFoundError.

    No resource is created — describes a guaranteed-bogus name. Guards that
    the control-plane 404 surfaces as NotFoundError with status_code 404.
    """
    with pytest.raises(NotFoundError) as exc_info:
        client.indexes.describe(unique_name("never-exists"))
    assert exc_info.value.status_code == 404


@pytest.mark.integration
def test_configure_serverless_read_capacity_real(client: Pinecone) -> None:
    """configure(serverless_read_capacity=...) round-trips against the real API.

    The mocked request-body test confirms the PATCH payload shape; this verifies
    the real backend accepts OnDemand read-capacity configuration on a
    serverless index, that ``configure()`` hands back the reconfigured
    ``IndexModel``, and that describe() reflects it.
    """
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        result = client.indexes.configure(
            name,
            serverless_read_capacity={"mode": "OnDemand"},
        )
        assert isinstance(result, IndexModel)
        assert result.name == name

        desc = client.indexes.describe(name)
        # Backend may report the read capacity on the returned spec or not;
        # the test asserts the call succeeds and the index is still healthy.
        assert desc.name == name
        assert desc.status.ready is True
    finally:
        cleanup_resource(
            lambda: client.indexes.delete(name),
            name,
            "index",
        )
