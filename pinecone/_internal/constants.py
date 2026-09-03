"""Shared constants for the Pinecone SDK."""

from __future__ import annotations

CONTROL_PLANE_API_VERSION: str = "2026-07"
DATA_PLANE_API_VERSION: str = "2026-07"
INFERENCE_API_VERSION: str = "2026-07"
ADMIN_API_VERSION: str = "2026-07"
ASSISTANT_API_VERSION: str = "2026-07"

DEFAULT_BASE_URL: str = "https://api.pinecone.io"
DEFAULT_MAX_CONCURRENCY: int = 8
ASSISTANT_EVALUATION_BASE_URL: str = "https://prod-1-data.ke.pinecone.io/assistant"
API_VERSION_HEADER: str = "X-Pinecone-Api-Version"
