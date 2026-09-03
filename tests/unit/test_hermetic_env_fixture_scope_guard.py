"""Guards the fixture-scope half of #426, reconstructing the #345 near-miss.

#353's scrub is function-scoped (``_hermetic_pinecone_env`` in conftest.py).
pytest instantiates higher-scoped fixtures first, so a module- or
session-scoped fixture that does not explicitly depend on
``hermetic_pinecone_env_module`` runs *before* that scrub and sees whatever
``PINECONE_*`` value the developer's shell already had exported — exactly what
happened to #345's shared property-test clients, which baked
``PINECONE_ADDITIONAL_HEADERS`` into every client in the module.
``tests/unit/test_hermetic_env_module_scope.py`` already proves the *opt-in*
fixture closes this when a fixture author remembers to ask for it; this file
proves the general case, where a fixture author does not.

The fix (``_hermetic_pinecone_env_session`` in conftest.py) is a session-scoped,
autouse scrub that runs before any module- or session-scoped fixture in the
whole ``tests/unit/`` session can — because it is autouse, it is in the very
first test's fixture closure, and pytest sets same-or-higher scope up first.
That makes the opt-in unnecessary for this specific hazard: even a fixture
that asks for nothing sees a clean environment.

To reconstruct #345 faithfully, the probe value below is set at **module
import time** — during collection, which finishes for every file under
``tests/unit/`` before any fixture of any scope is ever set up — rather than
by a fixture inside this test run. A fixture-set value would itself run after
session scope and prove nothing; a value already present when the whole
session's fixtures start is what a developer's exported shell variable
actually looks like from pytest's perspective, and collection-time is the
earliest this test file can put one there.
"""

from __future__ import annotations

import os

import pytest

_PROBE = "PINECONE_FIXTURE_SCOPE_GUARD_PROBE"

os.environ[_PROBE] = "present-before-any-fixture-ran"


@pytest.fixture(scope="module")
def naive_module_scoped_fixture() -> str | None:
    """The #345 shape: reads the environment directly, opts into no scrub."""
    return os.environ.get(_PROBE)


def test_a_naive_module_scoped_fixture_does_not_see_pre_session_pollution(
    naive_module_scoped_fixture: str | None,
) -> None:
    assert naive_module_scoped_fixture is None, (
        f"reconstructed #345: a module-scoped fixture that never asked for "
        f"hermetic_pinecone_env_module still saw {_PROBE}="
        f"{naive_module_scoped_fixture!r}. _hermetic_pinecone_env_session in "
        f"tests/unit/conftest.py should have removed this before any "
        f"module-scoped fixture in the session ran."
    )


def test_the_probe_variable_is_gone_by_the_time_a_test_body_runs() -> None:
    assert os.environ.get(_PROBE) is None
