"""Protocol stub for the Rust-backed GrpcChannel extension module.

Declares the expected interface so mypy can verify all call sites in
:class:`~pinecone.grpc.GrpcIndex` even when the native extension is not
built.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GrpcChannelProtocol(Protocol):
    """Structural type for the Rust-backed ``GrpcChannel``."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str,
        version: str,
        secure: bool = True,
        timeout_s: float | None = None,
        connect_timeout_s: float | None = None,
        max_retries: int | None = None,
        source_tag: str | None = None,
        proxy_url: str | None = None,
        on_throttle: Callable[[str], None] | None = None,
    ) -> None: ...

    def upsert(
        self,
        vectors: list[dict[str, Any]],
        namespace: str | None,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Upsert a batch of vectors."""
        ...

    def query(
        self,
        top_k: int,
        *,
        vector: Sequence[float] | None = None,
        id: str | None = None,
        namespace: str | None = None,
        filter: Mapping[str, Any] | None = None,
        include_values: bool = False,
        include_metadata: bool = False,
        sparse_vector: Mapping[str, Any] | None = None,
        scan_factor: float | None = None,
        max_candidates: int | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Query for nearest neighbors."""
        ...

    def fetch(
        self,
        ids: Sequence[str],
        *,
        namespace: str | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Fetch vectors by ID."""
        ...

    def delete(
        self,
        *,
        ids: Sequence[str] | None = None,
        delete_all: bool = False,
        namespace: str | None = None,
        filter: Mapping[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> None:
        """Delete vectors."""
        ...

    def update(
        self,
        id: str | None,
        *,
        values: Sequence[float] | None = None,
        sparse_values: Mapping[str, Any] | None = None,
        set_metadata: Mapping[str, Any] | None = None,
        namespace: str | None = None,
        filter: Mapping[str, Any] | None = None,
        dry_run: bool | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Update a vector."""
        ...

    def list(
        self,
        *,
        prefix: str | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
        namespace: str | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """List vector IDs."""
        ...

    def describe_index_stats(
        self,
        *,
        filter: Mapping[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Describe index statistics."""
        ...

    def create_namespace(
        self,
        name: str,
        schema: Mapping[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Create a namespace."""
        ...

    def describe_namespace(
        self,
        namespace: str,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Describe a namespace."""
        ...

    def list_namespaces(
        self,
        *,
        pagination_token: str | None = None,
        limit: int | None = None,
        prefix: str | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """List namespaces (single page)."""
        ...

    def delete_namespace(
        self,
        namespace: str,
        *,
        timeout_s: float | None = None,
    ) -> None:
        """Delete a namespace."""
        ...

    def close(self) -> None:
        """Close the channel and release resources."""
        ...
