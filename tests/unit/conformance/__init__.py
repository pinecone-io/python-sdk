"""2026-07 API conformance suite. Contract: see README.md in this package."""

from __future__ import annotations

from tests.unit.conformance._registry import (
    API_VERSION_HEADER,
    CLAIMS,
    EXPECTED_API_VERSION,
    MANIFEST_PATH,
    ClaimRecorder,
    ConformanceError,
    UnknownOperationError,
    api_op,
    manifest_operations,
)

__all__ = [
    "API_VERSION_HEADER",
    "CLAIMS",
    "EXPECTED_API_VERSION",
    "MANIFEST_PATH",
    "ClaimRecorder",
    "ConformanceError",
    "UnknownOperationError",
    "api_op",
    "manifest_operations",
]
