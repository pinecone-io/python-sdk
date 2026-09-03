"""Guards that legacy_index_factory reads PINECONE_CONTROLLER_HOST (#516).

``tests/integration/legacy_index.py``'s ``create_legacy_index`` and
``delete_legacy_index`` build a raw ``httpx.Client`` outside the SDK on
purpose — the SDK is the thing under test, so it cannot also be the fixture.
But ``tests.integration.conftest``'s ``legacy_index_factory`` hardcoded that
client's ``base_url``, so pointing the SDK itself at a non-default host (e.g.
a local simulator via ``PINECONE_CONTROLLER_HOST``) still left this fixture
hitting production and 401ing. These tests drive the fixture's generator
function directly, substituting fakes that record the ``base_url`` they were
given, so the regression fails without a live server.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from pinecone._internal.constants import DEFAULT_BASE_URL
from tests.live_suite import ENV_FILE_OVERRIDE

_OVERRIDE_BEFORE = os.environ.get(ENV_FILE_OVERRIDE)
os.environ[ENV_FILE_OVERRIDE] = os.path.join(os.path.dirname(__file__), "no-such-.env")
try:
    from tests.integration import conftest as integration_conftest
finally:
    if _OVERRIDE_BEFORE is None:
        del os.environ[ENV_FILE_OVERRIDE]
    else:
        os.environ[ENV_FILE_OVERRIDE] = _OVERRIDE_BEFORE

pytestmark = pytest.mark.timeout(30)


def _drive_factory(
    monkeypatch: pytest.MonkeyPatch, *, api_key: str = "k"
) -> tuple[list[str], list[str]]:
    created_base_urls: list[str] = []
    deleted_base_urls: list[str] = []

    def fake_create(
        api_key: str,
        *,
        dimension: int | None,
        metric: str,
        vector_type: str,
        base_url: str,
    ) -> Any:
        created_base_urls.append(base_url)
        return integration_conftest.LegacyIndex(
            name="idx-legacy-fake",
            host="h",
            dimension=dimension,
            metric=metric,
            vector_type=vector_type,
        )

    def fake_delete(api_key: str, name: str, *, base_url: str) -> None:
        deleted_base_urls.append(base_url)

    monkeypatch.setattr(integration_conftest, "create_legacy_index", fake_create)
    monkeypatch.setattr(integration_conftest, "delete_legacy_index", fake_delete)

    gen = integration_conftest.legacy_index_factory.__wrapped__(api_key)
    factory = next(gen)
    factory(dimension=3)
    with pytest.raises(StopIteration):
        next(gen)

    return created_base_urls, deleted_base_urls


def test_factory_uses_the_controller_host_env_var_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PINECONE_CONTROLLER_HOST", "http://127.0.0.1:5080")

    created, deleted = _drive_factory(monkeypatch)

    assert created == ["http://127.0.0.1:5080"]
    assert deleted == ["http://127.0.0.1:5080"]


def test_factory_defaults_to_production_when_env_var_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PINECONE_CONTROLLER_HOST", raising=False)

    created, deleted = _drive_factory(monkeypatch)

    assert created == [DEFAULT_BASE_URL]
    assert deleted == [DEFAULT_BASE_URL]
