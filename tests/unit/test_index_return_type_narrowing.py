"""``Pinecone.index()`` narrows its return type per ``grpc=`` argument.

``index()`` returns ``Index | GrpcIndex``, and ``documents`` — the entry point
for the quickstart's headline flow — exists only on ``Index``. Without
overloads keyed on the ``grpc`` literal, a reader who copies
``docs/getting-started/quickstart.md`` into a type-checked project gets
``Item "GrpcIndex" of "Index | GrpcIndex" has no attribute "documents"``.

The page is type-checked as published rather than transcribed here, so the
guard tracks whatever the page says. Two things are pinned alongside it: the
revealed type for each ``grpc=`` form, and that a misspelled client attribute
is still reported. The second matters because ``Pinecone.__getattr__``, which
explains removed attributes at runtime, is deliberately hidden from type
checkers — a visible ``__getattr__`` makes every attribute name valid and would
silence the misspelling.

One mypy subprocess covers all three, sharing the repository's cache with the
``mypy --strict pinecone/`` gate.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
QUICKSTART = REPO_ROOT / "docs/getting-started/quickstart.md"

REVEAL_SOURCE = """\
from __future__ import annotations

from pinecone import Pinecone

pc = Pinecone(api_key="key")

reveal_type(pc.index("quickstart"))
reveal_type(pc.index(name="quickstart", grpc=False))
reveal_type(pc.index(name="quickstart", grpc=True))

use_grpc: bool = True
reveal_type(pc.index(name="quickstart", grpc=use_grpc))

reveal_type(pc.no_such_attribute)
"""

EXPECTED_REVEALS = [
    "pinecone.index.Index",
    "pinecone.index.Index",
    "pinecone.grpc.GrpcIndex",
    "pinecone.index.Index | pinecone.grpc.GrpcIndex",
    "Any",
]


def _quickstart_source() -> str:
    """Concatenate the quickstart's python blocks into one module.

    The blocks are sequential steps of a single script, so type-checking them
    together is what a reader following the page ends up with.
    """
    blocks = re.findall(r"```python\n(.*?)```", QUICKSTART.read_text(), re.DOTALL)
    assert blocks, f"no python blocks found in {QUICKSTART}"
    return "from __future__ import annotations\n\n" + "\n".join(blocks)


@pytest.fixture(scope="module")
def mypy_output(tmp_path_factory: pytest.TempPathFactory) -> str:
    if importlib.util.find_spec("mypy") is None:
        pytest.skip("mypy is not installed in this environment")
    scratch = tmp_path_factory.mktemp("index_narrowing")
    (scratch / "quickstart_page.py").write_text(_quickstart_source())
    (scratch / "reveal_types.py").write_text(REVEAL_SOURCE)
    result = subprocess.run(  # noqa: S603 — fixed argv, sys.executable is trusted
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--no-error-summary",
            "--hide-error-context",
            str(scratch / "quickstart_page.py"),
            str(scratch / "reveal_types.py"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if "Cannot find implementation" in result.stdout:
        pytest.skip(f"mypy cannot resolve the pinecone package:\n{result.stdout}")
    return result.stdout


def test_quickstart_page_type_checks_clean(mypy_output: str) -> None:
    """Every python block on the quickstart page passes ``mypy --strict``."""
    offenders = [line for line in mypy_output.splitlines() if "quickstart_page.py" in line]
    assert offenders == [], (
        "docs/getting-started/quickstart.md no longer type-checks; "
        "`index.documents` needs `Pinecone.index` to narrow to `Index`:\n" + "\n".join(offenders)
    )


def test_revealed_types_track_the_grpc_argument(mypy_output: str) -> None:
    """``grpc=False`` reveals ``Index``, ``grpc=True`` reveals ``GrpcIndex``."""
    revealed = re.findall(r'Revealed type is "(.*)"', mypy_output)
    assert revealed == EXPECTED_REVEALS


def test_misspelled_client_attribute_is_still_reported(mypy_output: str) -> None:
    """``__getattr__`` must stay invisible to type checkers.

    Were it visible, mypy would accept every attribute name on ``Pinecone``
    and this error would disappear.
    """
    assert 'has no attribute "no_such_attribute"  [attr-defined]' in mypy_output
