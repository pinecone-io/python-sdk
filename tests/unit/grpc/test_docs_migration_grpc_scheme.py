"""Executes the gRPC endpoint scheme section of ``docs/migration/v10-migration.md``.

Same discipline as ``test_docs_migration_ssl_config_421.py``: the examples are
read out of the published guide and run, never transcribed here, so a
transcription cannot drift from what a reader relies on. Each block is executed
against a stubbed channel, and the endpoint the channel was handed is checked
against what the surrounding prose promises.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pinecone.errors.exceptions import PineconeValueError

GUIDE = Path(__file__).resolve().parents[3] / "docs/migration/v10-migration.md"
SECTION_START = "(grpc-scheme)="
_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"


def _section() -> str:
    text = GUIDE.read_text()
    assert SECTION_START in text, f"{SECTION_START} missing from {GUIDE}"
    return text.split(SECTION_START, 1)[1]


def _python_blocks() -> list[str]:
    blocks = []
    current: list[str] | None = None
    for line in _section().splitlines():
        if line.strip() == "```python":
            current = []
        elif line.strip() == "```" and current is not None:
            blocks.append("\n".join(current))
            current = None
        elif current is not None:
            current.append(line)
    assert len(blocks) == 2, f"expected 2 python blocks in {GUIDE}, got {len(blocks)}"
    return blocks


BLOCKS = _python_blocks()


@pytest.fixture(autouse=True)
def _no_scheme_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PINECONE_GRPC_SCHEME", raising=False)


@pytest.fixture
def channel_module() -> Iterator[MagicMock]:
    module = MagicMock()
    module.GrpcChannel.return_value = MagicMock()
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: module}):
        yield module


@pytest.mark.parametrize("block", BLOCKS, ids=["client_kwarg", "index_kwarg"])
def test_the_guides_examples_dial_a_plaintext_endpoint(
    block: str, channel_module: MagicMock
) -> None:
    exec(block, {})  # noqa: S102

    endpoint = channel_module.GrpcChannel.call_args[0][0]
    assert endpoint == "http://10.0.0.7:50051"


def test_the_env_var_the_guide_names_is_the_one_that_is_read(
    channel_module: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert "`PINECONE_GRPC_SCHEME` sets it" in _section()
    monkeypatch.setenv("PINECONE_GRPC_SCHEME", "http")

    from pinecone.grpc import GrpcIndex

    GrpcIndex(host="http://10.0.0.7:50051", api_key="k")

    assert channel_module.GrpcChannel.call_args[0][0] == "http://10.0.0.7:50051"


def test_the_refused_combination_really_is_refused(channel_module: MagicMock) -> None:
    assert '`grpc_scheme="https"` with `secure=False`' in _section()

    from pinecone.grpc import GrpcIndex

    with pytest.raises(PineconeValueError):
        GrpcIndex(host="http://10.0.0.7:50051", api_key="k", secure=False, grpc_scheme="https")


@pytest.mark.parametrize(("secure", "expected"), [(True, "https"), (False, "http")])
def test_an_unset_scheme_follows_secure_as_the_guide_says(
    channel_module: MagicMock, secure: bool, expected: str
) -> None:
    assert "keeps the scheme following `secure`" in _section()

    from pinecone.grpc import GrpcIndex

    GrpcIndex(host="idx.svc.pinecone.io", api_key="k", secure=secure)

    endpoint = channel_module.GrpcChannel.call_args[0][0]
    assert endpoint == f"{expected}://idx.svc.pinecone.io"
