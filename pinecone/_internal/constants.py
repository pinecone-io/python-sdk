"""Shared constants for the Pinecone SDK."""

from __future__ import annotations

CONTROL_PLANE_API_VERSION: str = "2025-10"
DATA_PLANE_API_VERSION: str = "2025-10"
INFERENCE_API_VERSION: str = "2025-10"
ADMIN_API_VERSION: str = "2025-10"
ASSISTANT_API_VERSION: str = "2025-10"
ASSISTANT_API_VERSION_2026_04: str = "2026-04"

DEFAULT_BASE_URL: str = "https://api.pinecone.io"
DEFAULT_MAX_CONCURRENCY: int = 8
ASSISTANT_EVALUATION_BASE_URL: str = "https://prod-1-data.ke.pinecone.io/assistant"
API_VERSION_HEADER: str = "X-Pinecone-Api-Version"
