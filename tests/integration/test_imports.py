"""Integration tests for bulk import lifecycle (sync REST).

Phase 4 area tag: import-lifecycle
Transport: rest (sync), grpc: N/A

Tests Index.start_import(), describe_import(), list_imports(),
list_imports_paginated(), and cancel_import().  Uses a public S3 URI that
the API accepts and queues; the import is cancelled before it can run so
no actual data is transferred.
"""

from __future__ import annotations

import pytest

from pinecone import Pinecone
from pinecone.errors.exceptions import PineconeValueError
from pinecone.models.imports.list import ImportList
from pinecone.models.imports.model import ImportModel, StartImportResponse
from tests.integration.conftest import cleanup_resource, unique_name
from tests.integration.index_shapes import MANAGED_AWS, dense_schema

# A public S3 URI that the Pinecone import API accepts. The import will
# be cancelled before it runs, so it does not matter whether the bucket
# contains Parquet files.
_TEST_URI = "s3://pinecone-test-public/"


# ---------------------------------------------------------------------------
# import-lifecycle — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_import_lifecycle_rest(client: Pinecone) -> None:
    """Full import lifecycle via sync REST: create a serverless index, call
    start_import(), describe_import(), list_imports(), list_imports_paginated(),
    and cancel_import().  Verifies ImportModel and StartImportResponse structures.

    Area tag: import-lifecycle
    Transport: rest
    """
    index_name = unique_name("idx")
    import_id: str | None = None

    try:
        # 1. Create a serverless index (needed to get a data-plane host)
        client.indexes.create(
            name=index_name,
            schema=dense_schema(2, "cosine"),
            deployment=MANAGED_AWS,
            timeout=120,
        )

        # 2. Get a data-plane Index handle
        idx = client.index(name=index_name)

        # 3. start_import — verify StartImportResponse structure
        start_resp = idx.start_import(uri=_TEST_URI, error_mode="continue")
        assert isinstance(start_resp, StartImportResponse)
        assert isinstance(start_resp.id, str)
        assert start_resp.id != ""
        import_id = start_resp.id

        # 4. describe_import — verify ImportModel fields
        import_op = idx.describe_import(import_id)
        assert isinstance(import_op, ImportModel)
        assert import_op.id == import_id
        assert import_op.status in ("Pending", "InProgress", "Failed", "Completed", "Cancelled")
        assert isinstance(import_op.uri, str)
        assert import_op.uri != ""
        assert isinstance(import_op.created_at, str)
        assert import_op.created_at != ""
        # Optional numeric fields may be None or a number
        assert import_op.percent_complete is None or isinstance(import_op.percent_complete, float)
        assert import_op.records_imported is None or isinstance(import_op.records_imported, int)

        # 5. list_imports — import appears in the result iterator
        all_imports = list(idx.list_imports())
        ids_in_list = [imp.id for imp in all_imports]
        assert import_id in ids_in_list, (
            f"Import {import_id!r} not found in list_imports() result (got: {ids_in_list})"
        )
        for imp in all_imports:
            assert isinstance(imp, ImportModel)
            assert isinstance(imp.id, str)
            assert isinstance(imp.status, str)

        # 6. list_imports_paginated — returns an ImportList for a single page
        page = idx.list_imports_paginated(limit=10)
        assert isinstance(page, ImportList)
        assert hasattr(page, "__iter__")
        assert hasattr(page, "__len__")
        for imp in page:
            assert isinstance(imp, ImportModel)

        # 7. cancel_import — completes without error (returns None)
        result = idx.cancel_import(import_id)
        assert result is None

        # Verify cancelled status
        cancelled_op = idx.describe_import(import_id)
        assert cancelled_op.status == "Cancelled"
        import_id = None  # Successfully cancelled; skip cleanup

    finally:
        if import_id is not None:
            # Best-effort cancel if the test failed before reaching step 7
            try:
                idx.cancel_import(import_id)
                print(f"  Cleaned up import: {import_id}")
            except Exception as exc:
                print(f"  WARNING: Failed to cancel import {import_id}: {exc}")
        cleanup_resource(
            lambda: client.indexes.delete(index_name),
            index_name,
            "index",
        )


# ---------------------------------------------------------------------------
# import-validation — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_import_input_validation_rest(client: Pinecone) -> None:
    """Client-side validation for start_import(), describe_import(), cancel_import() fires
    before any HTTP call is made, so a fake host is sufficient — no real index needed.

    Verifies:
    - unified-imp-0006: error_mode is normalized to lowercase; CONTINUE/Continue/continue
      and ABORT/Abort/abort are all accepted (they do not raise PineconeValueError).
    - unified-imp-0009: integer import IDs are silently converted to strings; no
      PineconeValueError is raised for an integer ID (a network error may follow).
    - unified-imp-0010: an invalid error_mode string raises PineconeValueError before
      any network call.
    - unified-imp-0011: describe_import() and cancel_import() raise PineconeValueError
      when the ID is empty or exceeds 1000 characters.
    """
    # Fake host: "svc.pinecone.io" contains a dot so it passes host validation.
    # Validation fires synchronously before any HTTP request.
    index = client.index(host="fake-host.svc.pinecone.io")

    # --- unified-imp-0010: invalid error_mode raises PineconeValueError ---
    with pytest.raises(PineconeValueError):
        index.start_import(uri="s3://test/bucket/", error_mode="invalid")

    with pytest.raises(PineconeValueError):
        index.start_import(uri="s3://test/bucket/", error_mode="")

    with pytest.raises(PineconeValueError):
        index.start_import(uri="s3://test/bucket/", error_mode="skip")

    # --- unified-imp-0006: case-insensitive normalization — must NOT raise PineconeValueError ---
    for mode in ("CONTINUE", "Continue", "continue", "ABORT", "Abort", "abort"):
        try:
            index.start_import(uri="s3://test/bucket/", error_mode=mode)
            # Unreachable with fake host, but if it somehow succeeds, that's also fine
        except PineconeValueError:
            pytest.fail(
                f"error_mode={mode!r} should be accepted after case normalization, "
                f"but PineconeValueError was raised"
            )
        except Exception:
            pass  # Expected: connection/HTTP error after validation passes

    # --- unified-imp-0011: ID length validation in describe_import ---
    with pytest.raises(PineconeValueError):
        index.describe_import("")  # empty ID

    with pytest.raises(PineconeValueError):
        index.describe_import("x" * 1001)  # 1001 chars — over limit

    # 1000-char ID should pass validation (then fail at network, not at validation)
    try:
        index.describe_import("x" * 1000)
    except PineconeValueError:
        pytest.fail("1000-character import ID should pass length validation")
    except Exception:
        pass  # Expected: connection error after validation passes

    # --- unified-imp-0011: ID length validation in cancel_import ---
    with pytest.raises(PineconeValueError):
        index.cancel_import("")  # empty ID

    with pytest.raises(PineconeValueError):
        index.cancel_import("x" * 1001)  # over limit

    # --- unified-imp-0009: integer IDs are converted to strings silently ---
    try:
        index.describe_import(42)  # integer → "42" → valid 2-char string
    except PineconeValueError:
        pytest.fail("Integer import ID should be silently converted to string, not rejected")
    except Exception:
        pass  # Expected: connection error after conversion succeeds

    try:
        index.cancel_import(42)
    except PineconeValueError:
        pytest.fail("Integer import ID should be silently converted to string, not rejected")
    except Exception:
        pass  # Expected: connection error after conversion succeeds
