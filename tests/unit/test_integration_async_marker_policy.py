"""Guards the integration suite's asyncio-marker ban from CI (#313).

The integration suite needs a live key and is not a CI gate, so its own
collection-time guard in ``tests/integration/conftest.py`` only fires when
somebody runs it. This is the gated copy, and it is a source scan rather than
an import so that the unit suite never loads the integration conftest — which
would read the real ``.env`` into this process at import time.

Why the ban: pytest-asyncio closes the event loop before running async fixture
teardown, so the ``await pc.close()`` in the ``async_client`` fixture raises
``RuntimeError: Event loop is closed`` and every test sharing that fixture
errors in teardown. ``@pytest.mark.anyio`` sequences finalization inside the
loop's lifetime instead. See commit bd074083.
"""

from __future__ import annotations

from pathlib import Path

_TESTS = Path(__file__).resolve().parent.parent
_INTEGRATION = _TESTS / "integration"


def test_no_asyncio_marker_under_tests_integration() -> None:
    offenders = [
        f"{path.relative_to(_TESTS)}:{lineno}"
        for path in sorted(_INTEGRATION.rglob("*.py"))
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if line.strip().startswith("@pytest.mark.asyncio")
    ]
    assert offenders == [], (
        f"pytest.mark.asyncio found under tests/integration: {offenders}. "
        "Use pytest.mark.anyio — see this module's docstring."
    )
