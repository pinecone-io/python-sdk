"""Configuration for the Pinecone SDK."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal

from pinecone.errors.exceptions import PineconeValueError

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

GrpcScheme = Literal["http", "https"]

GRPC_SCHEMES: tuple[str, ...] = ("http", "https")


def resolve_grpc_scheme(scheme: str | None) -> GrpcScheme | None:
    """Resolve the gRPC endpoint scheme from an explicit value or the environment.

    Args:
        scheme: The caller's explicit choice, or ``None`` to fall back to the
            ``PINECONE_GRPC_SCHEME`` environment variable.

    Returns:
        ``"http"``, ``"https"``, or ``None`` when neither source names one, which
        leaves the choice to the caller's own default. A resolved ``"http"``
        scheme warns once per process when the gRPC endpoint is built, unless
        the host is loopback or RFC 1918 private.

    Raises:
        PineconeValueError: If either source names a scheme other than ``http``
            or ``https``.
    """
    resolved = scheme if scheme is not None else os.environ.get("PINECONE_GRPC_SCHEME", "").strip()
    if not resolved:
        return None
    if resolved == "http":
        return "http"
    if resolved == "https":
        return "https"
    raise PineconeValueError(
        f"Invalid gRPC scheme {resolved!r}. Must be one of: {', '.join(GRPC_SCHEMES)}."
    )


def normalize_host(host: str | None) -> str:
    """Normalize a host string by ensuring it has an https:// prefix.

    - If host is None or empty, return "".
    - If host doesn't start with http:// or https://, prepend https://.
    - Existing http:// or https:// prefixes are preserved as-is.
    - Stacked-scheme patterns (e.g. "https://https://foo.io") are repaired by
      stripping outer schemes and keeping the innermost one.

    Idempotent: normalizing an already-normalized host returns it unchanged.
    """
    if not host:
        return ""
    # Strip outer schemes until at most one remains, so any depth of stacking
    # collapses to a single scheme and the result is idempotent.
    while True:
        stripped = False
        for outer in ("https://", "http://"):
            if host.startswith(outer):
                remainder = host[len(outer) :]
                if remainder.startswith(("http://", "https://")):
                    host = remainder
                    stripped = True
                break
        if not stripped:
            break
    if not host.startswith(("http://", "https://")):
        return f"https://{host}"
    return host


def normalize_source_tag(tag: str | None) -> str:
    """Normalize a source tag string.

    - Lowercase the input.
    - Strip characters not in [a-z0-9_ :].
    - Replace spaces with underscores.
    """
    if not tag:
        return ""
    import re

    lowered = tag.lower()
    cleaned = re.sub(r"[^a-z0-9_ :]", "", lowered)
    return cleaned.replace(" ", "_")


def _parse_additional_headers_env() -> dict[str, str]:
    """Parse PINECONE_ADDITIONAL_HEADERS env var as JSON."""
    raw = os.environ.get("PINECONE_ADDITIONAL_HEADERS", "")
    if not raw:
        return {}
    import json

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    except Exception:
        logger.warning(
            "Failed to parse PINECONE_ADDITIONAL_HEADERS env var, ignoring: %s",
            raw,
        )
    return {}


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for HTTP retry behavior.

    Args:
        max_retries: Number of retries after the initial attempt. Defaults to 3 (4 total attempts).
        backoff_factor: Minimum delay floor in seconds between retries. The
            decorrelated-jitter algorithm samples from
            ``uniform(backoff_factor, prev_delay * 3)`` capped at ``max_wait``.
            Defaults to 0.25.
        max_wait: Maximum backoff delay in seconds. Defaults to 60.0.
        retryable_status_codes: HTTP status codes that trigger a retry. Defaults to
            ``{408, 429, 500, 502, 503, 504}``.
        on_throttle: Internal SDK callback invoked with the request URL host on every
            retryable response (including ones that will be retried). Used by the SDK
            to wire adaptive concurrency limiters; not intended for user configuration.
    """

    max_retries: int = 3
    backoff_factor: float = 0.25
    max_wait: float = 60.0
    retryable_status_codes: frozenset[int] = field(
        default_factory=lambda: frozenset({408, 429, 500, 502, 503, 504})
    )
    on_throttle: Callable[[str], None] | None = None


@dataclass(frozen=True)
class PineconeConfig:
    """SDK configuration with environment variable fallbacks.

    Args:
        api_key: Pinecone API key. Falls back to PINECONE_API_KEY env var.
        host: API host URL. Falls back to PINECONE_CONTROLLER_HOST env var.
        timeout: Request timeout in seconds. Defaults to 30.
        additional_headers: Extra headers to include in every request.
        source_tag: Source tag for User-Agent string.
        proxy_url: HTTP proxy URL.
        ssl_ca_certs: Path to CA certificate bundle.
        ssl_verify: Whether to verify SSL certificates.
        grpc_scheme: URL scheme used to dial the gRPC data plane, ``"http"`` or
            ``"https"``. Falls back to the PINECONE_GRPC_SCHEME env var, then to
            ``None``, which lets the gRPC client keep its own default. Dialling
            ``"http"`` against a host outside loopback and the RFC 1918 private
            ranges warns once per process, since the API key and every payload
            then cross a public network unencrypted.
    """

    api_key: str = ""
    host: str = ""
    timeout: float = 30.0
    additional_headers: dict[str, str] = field(default_factory=dict)
    source_tag: str = ""
    proxy_url: str = ""
    proxy_headers: dict[str, str] = field(default_factory=dict)
    ssl_ca_certs: str | None = None
    ssl_verify: bool = True
    grpc_scheme: GrpcScheme | None = None
    connection_pool_maxsize: int = 0
    retry_config: RetryConfig = field(default_factory=RetryConfig)

    _SENSITIVE_HEADER_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"authorization", "api-key", "proxy-authorization"}
    )

    def _redact_headers(self, headers: dict[str, str]) -> dict[str, str]:
        return {
            k: "***" if k.lower() in self._SENSITIVE_HEADER_KEYS else v for k, v in headers.items()
        }

    def __repr__(self) -> str:
        masked = f"...{self.api_key[-4:]}" if len(self.api_key) >= 4 else "***"
        return (
            f"PineconeConfig("
            f"api_key='{masked}', "
            f"host='{self.host}', "
            f"timeout={self.timeout}, "
            f"additional_headers={self._redact_headers(self.additional_headers)!r}, "
            f"source_tag='{self.source_tag}', "
            f"proxy_url='{self.proxy_url}', "
            f"proxy_headers={self._redact_headers(self.proxy_headers)!r}, "
            f"ssl_ca_certs={self.ssl_ca_certs!r}, "
            f"ssl_verify={self.ssl_verify}, "
            f"grpc_scheme={self.grpc_scheme!r}, "
            f"connection_pool_maxsize={self.connection_pool_maxsize}"
            f")"
        )

    def __post_init__(self) -> None:
        if not self.api_key:
            env_key = os.environ.get("PINECONE_API_KEY", "")
            object.__setattr__(self, "api_key", env_key)
        if not self.host:
            env_host = os.environ.get("PINECONE_CONTROLLER_HOST", "")
            object.__setattr__(self, "host", normalize_host(env_host))
        else:
            object.__setattr__(self, "host", normalize_host(self.host))
        if not self.additional_headers:
            env_headers = _parse_additional_headers_env()
            if env_headers:
                object.__setattr__(self, "additional_headers", env_headers)
        if self.source_tag:
            object.__setattr__(self, "source_tag", normalize_source_tag(self.source_tag))
        object.__setattr__(self, "grpc_scheme", resolve_grpc_scheme(self.grpc_scheme))
