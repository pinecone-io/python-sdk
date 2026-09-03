"""Guards that the live suites' index cleanup actually deletes (#346).

``indexes.list()`` returns a lazy :class:`Paginator` on 2026-07. The cleanup
helpers reached for a ``.indexes`` attribute it does not have, inside a
``try``/``except`` that printed a warning and moved on — so they raised on
every poll, deleted nothing, and leaked a real cloud index per smoke run for
as long as nobody read the logs.

A mock asserting ``delete`` *was called* would have passed throughout. These
tests therefore drive the helpers against a fake backend that models the
asynchronous delete, and assert on the **backend's remaining contents** — the
only observation that distinguishes "cleaned up" from "raised and gave up".
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from pinecone.models.pagination import AsyncPaginator, Page, Paginator
from tests.live_suite import ENV_FILE_OVERRIDE

# Importing the live suites' conftest runs its module-level load_env(), which
# would put the developer's real PINECONE_API_KEY into os.environ for the rest
# of the unit-test session — enough to flip tests/unit/test_config_repr.py.
# Point the documented override at a file that does not exist so the load is a
# no-op, and restore the variable so tests/unit/test_live_suite_env.py still
# sees an unset override.
_OVERRIDE_BEFORE = os.environ.get(ENV_FILE_OVERRIDE)
os.environ[ENV_FILE_OVERRIDE] = os.path.join(os.path.dirname(__file__), "no-such-.env")
try:
    from tests.integration.conftest import (
        async_ensure_index_deleted,
        ensure_index_deleted,
    )
    from tests.smoke.scripts.cleanup_orphans import cleanup
finally:
    if _OVERRIDE_BEFORE is None:
        del os.environ[ENV_FILE_OVERRIDE]
    else:
        os.environ[ENV_FILE_OVERRIDE] = _OVERRIDE_BEFORE

pytestmark = pytest.mark.timeout(30)


class _FakeIndex:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeBackend:
    """Index store whose deletes land after ``lag`` further list calls.

    The lag is what the helpers' polling loop exists for; without it a helper
    that never polls at all would still pass.
    """

    def __init__(self, names: list[str], *, lag: int = 2) -> None:
        self.names = list(names)
        self.lag = lag
        self._pending: dict[str, int] = {}
        self.delete_calls: list[str] = []
        self.list_calls = 0

    def _tick(self) -> None:
        for name in list(self._pending):
            self._pending[name] -= 1
            if self._pending[name] <= 0:
                del self._pending[name]
                if name in self.names:
                    self.names.remove(name)

    def delete(self, name: str, **kwargs: Any) -> None:
        self.delete_calls.append(name)
        if name not in self.names:
            raise RuntimeError(f"404 not found: {name}")
        if self.lag == 0:
            self.names.remove(name)
        else:
            self._pending[name] = self.lag

    def list(self, **kwargs: Any) -> Paginator[_FakeIndex]:
        self.list_calls += 1
        self._tick()
        snapshot = [_FakeIndex(n) for n in self.names]

        def fetch_page(token: str | None) -> Page[_FakeIndex]:
            return Page(items=snapshot, pagination_token=None)

        return Paginator(fetch_page=fetch_page)


class _AsyncFakeBackend(_FakeBackend):
    async def delete(self, name: str, **kwargs: Any) -> None:  # type: ignore[override]
        super().delete(name, **kwargs)

    def list(self, **kwargs: Any) -> AsyncPaginator[_FakeIndex]:  # type: ignore[override]
        self.list_calls += 1
        self._tick()
        snapshot = [_FakeIndex(n) for n in self.names]

        async def fetch_page(token: str | None) -> Page[_FakeIndex]:
            return Page(items=snapshot, pagination_token=None)

        return AsyncPaginator(fetch_page=fetch_page)


class _EmptyCollections:
    def list(self) -> Any:
        class _L:
            @staticmethod
            def names() -> list[str]:
                return []

        return _L()


class _EmptyAssistants:
    def list(self) -> Paginator[Any]:
        return Paginator(fetch_page=lambda token: Page(items=[], pagination_token=None))


class _FakeClient:
    def __init__(self, backend: _FakeBackend) -> None:
        self.indexes = backend
        self.collections = _EmptyCollections()
        self.assistants = _EmptyAssistants()
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_ensure_index_deleted_removes_the_index() -> None:
    backend = _FakeBackend(["smoke-keep", "smoke-doomed"])
    client = _FakeClient(backend)

    ensure_index_deleted(client, "smoke-doomed", timeout=30, interval=0)  # type: ignore[arg-type]

    assert backend.names == ["smoke-keep"], "the index was not actually deleted"
    assert backend.delete_calls == ["smoke-doomed"]


def test_ensure_index_deleted_polls_until_the_backend_catches_up() -> None:
    """The delete is asynchronous; returning before it lands would leak quota."""
    backend = _FakeBackend(["smoke-doomed"], lag=3)
    client = _FakeClient(backend)

    ensure_index_deleted(client, "smoke-doomed", timeout=30, interval=0)  # type: ignore[arg-type]

    assert backend.names == []
    assert backend.list_calls >= 3, "returned without waiting for the delete to land"


def test_ensure_index_deleted_reports_a_leak_it_cannot_confirm(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A cleanup that gives up must say so — silence is how #346 hid."""
    backend = _FakeBackend(["smoke-immortal"], lag=10**6)
    client = _FakeClient(backend)

    ensure_index_deleted(client, "smoke-immortal", timeout=0, interval=0)  # type: ignore[arg-type]

    assert "may leak quota" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_async_ensure_index_deleted_removes_the_index() -> None:
    backend = _AsyncFakeBackend(["smoke-keep", "smoke-doomed"])
    client = _FakeClient(backend)

    await async_ensure_index_deleted(client, "smoke-doomed", timeout=30, interval=0)  # type: ignore[arg-type]

    assert backend.names == ["smoke-keep"], "the index was not actually deleted"


def test_cleanup_orphans_deletes_prefixed_indexes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The orphan script must find the leaked indexes, not report zero of them.

    It fires the deletes without polling (``timeout=-1``), so the fake applies
    them immediately; the assertion that matters is that all three names were
    listed and only the prefixed two were deleted.
    """
    backend = _FakeBackend(["smoke-orphan-a", "smoke-orphan-b", "keep-me"], lag=0)
    client = _FakeClient(backend)
    monkeypatch.setattr("tests.smoke.scripts.cleanup_orphans.Pinecone", lambda **kwargs: client)
    monkeypatch.setenv("PINECONE_API_KEY", "unit-test-not-a-real-key")

    assert cleanup() == 0
    assert sorted(backend.delete_calls) == ["smoke-orphan-a", "smoke-orphan-b"]
    assert backend.names == ["keep-me"]
    assert client.closed


def test_cleanup_orphans_dry_run_deletes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _FakeBackend(["smoke-orphan-a"])
    client = _FakeClient(backend)
    monkeypatch.setattr("tests.smoke.scripts.cleanup_orphans.Pinecone", lambda **kwargs: client)
    monkeypatch.setenv("PINECONE_API_KEY", "unit-test-not-a-real-key")

    assert cleanup(dry_run=True) == 0
    assert backend.delete_calls == []
    assert backend.names == ["smoke-orphan-a"]


@pytest.mark.parametrize("attr", ["indexes", "names", "data"])
def test_paginator_has_no_list_model_attributes(attr: str) -> None:
    """The mistake itself, pinned: these are the attributes #346 reached for."""
    paginator: Paginator[Any] = Paginator(
        fetch_page=lambda token: Page(items=[], pagination_token=None)
    )
    with pytest.raises(AttributeError):
        getattr(paginator, attr)
