"""Smoke tests to verify the scaffold is correctly set up."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from pinecone import __version__
from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import (
    ADMIN_API_VERSION,
    CONTROL_PLANE_API_VERSION,
    DATA_PLANE_API_VERSION,
    INFERENCE_API_VERSION,
)
from pinecone._internal.validation import require_non_empty, require_positive
from pinecone.errors.exceptions import (
    ApiError,
    ConflictError,
    NotFoundError,
    PineconeError,
    UnauthorizedError,
    ValidationError,
)


def test_version() -> None:
    # Catch the "pyproject.toml and pinecone/__init__.py drifted" bug, but stay
    # version-agnostic so a release bump doesn't have to edit this test.
    # Use regex over tomllib so this test runs on Python 3.10 (tomllib is 3.11+).
    text = Path(__file__).parents[2].joinpath("pyproject.toml").read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match is not None, "could not find version line in pyproject.toml"
    assert __version__ == match.group(1)


def test_config_defaults() -> None:
    cfg = PineconeConfig(api_key="key")
    assert cfg.api_key == "key"
    assert cfg.timeout == 30.0
    assert cfg.additional_headers == {}


def test_config_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PINECONE_API_KEY", "env-key")
    cfg = PineconeConfig()
    assert cfg.api_key == "env-key"


def test_api_version_constants() -> None:
    assert CONTROL_PLANE_API_VERSION == "2025-10"
    assert DATA_PLANE_API_VERSION == "2025-10"
    assert INFERENCE_API_VERSION == "2025-10"
    assert ADMIN_API_VERSION == "2026-07"


def test_exception_hierarchy() -> None:
    assert issubclass(ApiError, PineconeError)
    assert issubclass(NotFoundError, ApiError)
    assert issubclass(ConflictError, ApiError)
    assert issubclass(UnauthorizedError, ApiError)
    assert issubclass(ValidationError, PineconeError)


def test_api_error_attributes() -> None:
    body: dict[str, Any] = {"error": "test"}
    err = ApiError(message="fail", status_code=500, body=body)
    assert err.status_code == 500
    assert err.body == body
    assert str(err) == "[500] fail"


def test_not_found_error_defaults() -> None:
    err = NotFoundError()
    assert err.status_code == 404


def test_conflict_error_defaults() -> None:
    err = ConflictError()
    assert err.status_code == 409


def test_unauthorized_error_defaults() -> None:
    err = UnauthorizedError()
    assert err.status_code == 401


def test_validation_error() -> None:
    err = ValidationError("bad input")
    assert err.message == "bad input"
    assert isinstance(err, PineconeError)


def test_require_non_empty() -> None:
    require_non_empty("name", "valid")
    with pytest.raises(ValidationError):
        require_non_empty("name", "")
    with pytest.raises(ValidationError):
        require_non_empty("name", "   ")


def test_require_positive() -> None:
    require_positive("dim", 1)
    with pytest.raises(ValidationError):
        require_positive("dim", 0)
    with pytest.raises(ValidationError):
        require_positive("dim", -1)
