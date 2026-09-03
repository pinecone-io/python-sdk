"""Guards ``hermetic_pinecone_env_module`` in conftest.py (#345).

#353 made the unit suite hermetic with an autouse, **function-scoped** scrub of
every ``PINECONE_*`` variable. pytest sets higher-scoped fixtures up first, so a
module- or session-scoped fixture runs *before* that scrub and sees the
developer's ambient environment. #345 hit this by moving property-test clients
into module-scoped fixtures: ``PINECONE_ADDITIONAL_HEADERS`` from the ambient
environment landed in the shared client's header set, which is exactly the
"CI gate whose result depends on the machine" that #353 removed.

These tests pin both halves: that the hazard is real (so nobody deletes the
module-scoped scrub as redundant), and that depending on it closes the hole.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

_PROBE = "PINECONE_TEST_HERMETIC_PROBE"


@pytest.fixture(scope="module", autouse=True)
def _set_ambient_probe() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv(_PROBE, "ambient")
        yield


@pytest.fixture(scope="module")
def unscrubbed_view(_set_ambient_probe: None) -> str | None:
    return os.environ.get(_PROBE)


@pytest.fixture(scope="module")
def scrubbed_view(
    unscrubbed_view: str | None,
    hermetic_pinecone_env_module: None,
) -> str | None:
    """Depends on ``unscrubbed_view`` only to order it first.

    ``hermetic_pinecone_env_module`` holds its scrub for the whole module, so
    whichever of these two fixtures is built second sees a scrubbed
    environment. Naming it as a dependency fixes the order whichever test the
    randomised runner happens to start with.
    """
    return os.environ.get(_PROBE)


def test_a_module_scoped_fixture_runs_before_the_per_test_scrub(
    unscrubbed_view: str | None,
) -> None:
    assert unscrubbed_view == "ambient", (
        "a module-scoped fixture no longer sees ambient PINECONE_* variables. "
        "If pytest's setup order changed, hermetic_pinecone_env_module may be "
        "redundant — confirm that before removing it."
    )
    assert os.environ.get(_PROBE) is None, (
        "the per-test scrub did not hide the probe from the test body"
    )


def test_depending_on_the_module_scoped_scrub_hides_them(
    scrubbed_view: str | None,
) -> None:
    assert scrubbed_view is None, (
        f"hermetic_pinecone_env_module let {_PROBE}={scrubbed_view!r} through; a "
        "module-scoped fixture building an SDK object would bake it in"
    )
