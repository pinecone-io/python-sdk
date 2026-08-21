"""Guards the legacy-index test helper without a live key (#363).

``tests/integration/legacy_index.py`` is the only place in the suite that can
produce an index the vectors API will serve, and it works by pinning a
*previous* API version. Two things about it can rot silently:

- the pinned version could drift to one whose create persists a schema, which
  would hand every caller a document-schema index instead — and then every
  "this vectors call fails" assertion downstream passes for the wrong reason;
- the vacuity guard itself could stop distinguishing the two shapes.

Neither failure is visible in an integration run without a live key, and the
integration suite is not a CI gate. These tests are, so the wire shape and
the guard are asserted here against fakes.
"""

from __future__ import annotations

import ast
import os
import pathlib

import httpx
import pytest
import respx

from pinecone.models.indexes.schema import DenseVectorField, IndexSchema, SparseVectorField
from tests.live_suite import ENV_FILE_OVERRIDE

_OVERRIDE_BEFORE = os.environ.get(ENV_FILE_OVERRIDE)
os.environ[ENV_FILE_OVERRIDE] = os.path.join(os.path.dirname(__file__), "no-such-.env")
try:
    from tests.integration import legacy_index
    from tests.integration.legacy_index import (
        LEGACY_CREATE_API_VERSION,
        LEGACY_DENSE_FIELD,
        LEGACY_SPARSE_FIELD,
        LegacyIndex,
        _headers,
        assert_serves_vectors_api,
        create_legacy_index,
        delete_legacy_index,
        legacy_create_body,
    )
finally:
    if _OVERRIDE_BEFORE is None:
        del os.environ[ENV_FILE_OVERRIDE]
    else:
        os.environ[ENV_FILE_OVERRIDE] = _OVERRIDE_BEFORE

pytestmark = pytest.mark.timeout(30)


@pytest.fixture
def instant_polls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the helpers' poll intervals free.

    Requested explicitly rather than relying on the autouse fixture in
    ``tests/unit/conftest.py``: that one patches ``time.sleep`` through
    ``pinecone._internal.http_client``, which happens to reach the same module
    object, so these tests would be fast for a reason unrelated to them and
    would start taking real seconds the moment that fixture is narrowed.
    """
    monkeypatch.setattr(legacy_index.time, "sleep", lambda *_a, **_kw: None)


_DENSE = LegacyIndex(
    name="idx-legacy-dense", host="h", dimension=3, metric="cosine", vector_type="dense"
)


class _FakeIndexes:
    def __init__(self, schema: IndexSchema | None) -> None:
        self._schema = schema

    def describe(self, name: str) -> object:
        return type("_Described", (), {"name": name, "schema": self._schema})()


class _FakeClient:
    def __init__(self, schema: IndexSchema | None) -> None:
        self.indexes = _FakeIndexes(schema)


def test_create_targets_a_version_that_predates_the_schema_api() -> None:
    """The pinned version must be older than the schema-based create.

    A 2026-07 create always persists a schema, so pinning it — or anything
    later — would silently produce the one index shape these tests cannot use.
    """
    from pinecone._internal.constants import CONTROL_PLANE_API_VERSION

    assert LEGACY_CREATE_API_VERSION < CONTROL_PLANE_API_VERSION


def test_create_headers_pin_the_previous_api_version() -> None:
    headers = _headers("sekrit")
    assert headers["X-Pinecone-Api-Version"] == LEGACY_CREATE_API_VERSION
    assert headers["Api-Key"] == "sekrit"
    assert headers["Content-Type"] == "application/json"


def test_dense_create_body_uses_the_legacy_shape() -> None:
    """dimension / metric / spec, and no ``schema`` key.

    A ``schema`` key would route the create to the 2026-07 handler and defeat
    the whole helper.
    """
    body = legacy_create_body("idx", dimension=3, metric="cosine", vector_type="dense")
    assert body == {
        "name": "idx",
        "dimension": 3,
        "metric": "cosine",
        "vector_type": "dense",
        "spec": {"serverless": {"cloud": "aws", "region": "us-east-1"}},
    }
    assert "schema" not in body
    assert "deployment" not in body


def test_sparse_create_body_omits_dimension() -> None:
    body = legacy_create_body("idx", dimension=None, metric="dotproduct", vector_type="sparse")
    assert "dimension" not in body
    assert body["vector_type"] == "sparse"
    assert body["metric"] == "dotproduct"


@respx.mock
def test_delete_waits_for_the_index_to_disappear(instant_polls: None) -> None:
    """Firing the delete is not the same as the index being gone.

    The delete is asynchronous, so a helper that returns after the request
    cannot distinguish a cleanup from a leak, and a mock asserting only that
    ``delete`` was called would pass either way.
    """
    respx.delete("https://api.pinecone.io/indexes/idx-legacy-going").mock(
        return_value=httpx.Response(202)
    )
    describe = respx.get("https://api.pinecone.io/indexes/idx-legacy-going").mock(
        side_effect=[httpx.Response(200, json={"name": "idx-legacy-going"}), httpx.Response(404)]
    )

    delete_legacy_index("k", "idx-legacy-going")

    assert describe.call_count == 2


@respx.mock
def test_delete_reports_a_leak_when_the_index_never_disappears(
    capsys: pytest.CaptureFixture[str],
    instant_polls: None,
) -> None:
    respx.delete("https://api.pinecone.io/indexes/idx-legacy-stuck").mock(
        return_value=httpx.Response(202)
    )
    respx.get("https://api.pinecone.io/indexes/idx-legacy-stuck").mock(
        return_value=httpx.Response(200, json={"name": "idx-legacy-stuck"})
    )

    delete_legacy_index("k", "idx-legacy-stuck", timeout=0)

    assert "may leak quota" in capsys.readouterr().out


def test_helper_never_imports_the_integration_conftest() -> None:
    """The helper must stay importable from a unit test without side effects.

    ``tests.integration.conftest`` calls ``load_env()`` at module scope, so
    importing it — at module scope or lazily inside a function — would put a
    real ``PINECONE_API_KEY`` into ``os.environ`` for the rest of the unit
    session and silently change what other unit tests observe.
    """
    tree = ast.parse(pathlib.Path(legacy_index.__file__).read_text())
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not {m for m in imported if m.startswith("tests.integration.conftest")}, imported


@respx.mock
def test_create_deletes_the_index_when_it_never_becomes_ready(
    capsys: pytest.CaptureFixture[str],
    instant_polls: None,
) -> None:
    """A readiness timeout must not leak a live index.

    The caller never receives a name in that case, so nothing downstream can
    clean it up — the helper has to do it before raising, and has to see the
    index actually go. Asserting only that ``DELETE`` was called would not
    show that: the helper swallows exceptions, so an unmocked describe would
    leave the wait-until-gone path unexercised and the test still green.
    """
    create = respx.post("https://api.pinecone.io/indexes").mock(
        return_value=httpx.Response(201, json={"name": "idx-legacy-timeout"})
    )
    delete = respx.delete("https://api.pinecone.io/indexes/idx-legacy-timeout").mock(
        return_value=httpx.Response(202)
    )
    describe = respx.get("https://api.pinecone.io/indexes/idx-legacy-timeout").mock(
        side_effect=[httpx.Response(200, json={"name": "idx-legacy-timeout"}), httpx.Response(404)]
    )

    with pytest.raises(TimeoutError, match="not ready"):
        create_legacy_index("k", dimension=3, name="idx-legacy-timeout", timeout=0)

    assert create.called
    assert delete.called
    assert describe.call_count == 2
    assert "Cleaned up legacy index: idx-legacy-timeout" in capsys.readouterr().out


def test_guard_accepts_a_schema_of_only_reserved_vector_fields() -> None:
    schema = IndexSchema(
        fields={
            LEGACY_DENSE_FIELD: DenseVectorField(dimension=3, metric="cosine"),
            LEGACY_SPARSE_FIELD: SparseVectorField(),
        }
    )
    assert_serves_vectors_api(_FakeClient(schema), _DENSE)


def test_guard_rejects_a_user_named_vector_field() -> None:
    """The failure mode the guard exists for.

    A user-named dense field is what ``pc.indexes.create`` produces, and it
    makes the index documents-API only. Every vectors-API assertion in the
    calling module would then pass vacuously.
    """
    schema = IndexSchema(fields={"embedding": DenseVectorField(dimension=3, metric="cosine")})
    with pytest.raises(AssertionError, match="document-schema index"):
        assert_serves_vectors_api(_FakeClient(schema), _DENSE)


def test_guard_rejects_a_missing_schema() -> None:
    with pytest.raises(AssertionError, match="no schema"):
        assert_serves_vectors_api(_FakeClient(None), _DENSE)


def test_guard_rejects_an_empty_schema() -> None:
    with pytest.raises(AssertionError, match="no schema fields"):
        assert_serves_vectors_api(_FakeClient(IndexSchema(fields={})), _DENSE)
