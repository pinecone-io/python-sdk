"""Configuration for the Pinecone SDK."""

from __future__ import annotations

import functools
import logging
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal

from pinecone._internal.validation import validate_timeout
from pinecone.errors.exceptions import PineconeValueError

if TYPE_CHECKING:
    from collections.abc import Callable

    import httpx

logger = logging.getLogger(__name__)

GrpcScheme = Literal["http", "https"]

GRPC_SCHEMES: tuple[str, ...] = ("http", "https")


_VISIBLE = r"[\x21-\x7e]"
_HEADER_VALUE = re.compile(rf"{_VISIBLE}(?:[\t ]*{_VISIBLE})*")


def check_api_key_is_header_safe(api_key: str) -> None:
    """Raise if *api_key* cannot be sent as an HTTP header value.

    An empty key is left to the caller's own "no API key" error. The message never
    includes the key. Called by each client after it resolves its key, not by
    :class:`PineconeConfig`, so a clients-only check cannot reject an unrelated
    ``PINECONE_API_KEY`` for the Admin client, which never sends it.
    """
    if api_key and _HEADER_VALUE.fullmatch(api_key) is None:
        raise PineconeValueError(
            "The API key (api_key or PINECONE_API_KEY) is not a valid HTTP header value: it is "
            "blank or whitespace-only, has leading or "
            "trailing whitespace (such as a newline read from a file), or contains a "
            "control or non-ASCII character. Strip the key, for example api_key=key.strip()."
        )


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


def _redact_url_userinfo(url: str) -> str:
    """Return *url* with everything between the scheme and the last ``@`` replaced by ``***``.

    Masking up to the last ``@`` covers passwords containing an unencoded ``/``, ``?``
    or ``#``; over-redacting an ``@`` in a path or query is acceptable.
    """
    scheme = re.match(r"[A-Za-z][A-Za-z0-9+.-]*://", url)
    start = scheme.end() if scheme else 0
    at = url.rfind("@", start)
    if at == -1:
        return url
    return f"{url[:start]}***{url[at:]}"


_PROXY_SCHEMES = ("http", "https", "socks5", "socks5h")
_SCHEME = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*)://")
_PERCENT_ENCODE_HINT = (
    "percent-encode reserved characters in the username and password "
    "(e.g. '#' as %23, '/' as %2F, '?' as %3F, '@' as %40)"
)
_GRPC_PROXY_REMEDY = (
    "use an http:// proxy URL (traffic inside the tunnel is still TLS) or use REST "
    "(grpc=False), which also supports https:// and SOCKS proxies"
)


def _try_parse_url(url: str) -> httpx.URL | None:
    import httpx

    try:
        return httpx.URL(url)
    except Exception:
        return None


def _redact_proxy_url(proxy_url: str, parsed: httpx.URL | None) -> str:
    """Return *proxy_url* fit for an error message, with any credentials masked.

    A URL that does not parse may hide its password anywhere (httpx reads
    ``http://u:p%40ss@host`` with no ``@`` as host ``u`` and port ``p%40ss...``),
    so only its scheme is shown. So is a URL that parsed without userinfo but
    carries an encoded or look-alike ``@`` (``%40``, or any non-ASCII character).
    """
    scheme = _SCHEME.match(proxy_url)
    prefix = scheme.group(0) if scheme else ""
    if parsed is None:
        return f"{prefix}***"
    if parsed.userinfo:
        return f"{prefix}***@{parsed.netloc.decode('ascii')}"
    if "%40" in proxy_url or not proxy_url.isascii():
        return f"{prefix}***"
    return _redact_url_userinfo(proxy_url)


def _unparseable_reason(after_scheme: str) -> str:
    """Name what is wrong with a proxy URL httpx could not parse, without echoing it."""
    has_userinfo = "@" in after_scheme
    if not has_userinfo and ("%40" in after_scheme or not after_scheme.isascii()):
        return "it could not be parsed; separate the credentials from the host with a plain '@'"
    host_port = re.split(r"[/?#]", after_scheme.rsplit("@", 1)[-1], maxsplit=1)[0]
    if not host_port.startswith("[") and host_port.count(":") > 1:
        return "an IPv6 address must be in brackets, e.g. http://[::1]:3128"
    if ":" in host_port and not host_port.endswith("]"):
        port = host_port.rsplit(":", 1)[1]
        if not (port.isdigit() and 1 <= int(port) <= 65535):
            return "the port must be a number from 1 to 65535"
    if has_userinfo:
        return f"it could not be parsed; {_PERCENT_ENCODE_HINT}"
    return "it could not be parsed"


def validate_proxy_url(proxy_url: str, *, grpc: bool = False) -> None:
    """Raise :class:`PineconeValueError` unless *proxy_url* is a usable proxy URL.

    REST accepts ``http``, ``https``, ``socks5`` and ``socks5h``. With *grpc*, only
    ``http://`` is accepted, since the gRPC transport cannot speak TLS or SOCKS to
    the proxy. Every reason comes from our own checks: the parser's exception
    text can contain the password, so it is never shown or chained.
    """
    parsed = _try_parse_url(proxy_url)
    redacted = _redact_proxy_url(proxy_url, parsed)

    def invalid(reason: str) -> PineconeValueError:
        return PineconeValueError(f"Invalid proxy URL {redacted!r}: {reason}")

    if any(ord(c) < 0x20 or ord(c) == 0x7F for c in proxy_url.strip()):
        raise invalid("it contains a control character such as a newline; remove it")
    if proxy_url != proxy_url.strip():
        raise invalid("it has leading or trailing whitespace; remove it")
    scheme = _SCHEME.match(proxy_url)
    if grpc and (scheme is None or scheme.group(1).lower() != "http"):
        raise PineconeValueError(
            f"gRPC supports only http:// proxies; got {redacted!r}; {_GRPC_PROXY_REMEDY}"
        )
    if scheme is None:
        raise invalid("it has no scheme; start it with http:// or https://")
    if scheme.group(1).lower() not in _PROXY_SCHEMES:
        raise invalid(
            f"scheme must be 'http', 'https', 'socks5' or 'socks5h', got {scheme.group(1)!r}"
        )
    if parsed is None:
        raise invalid(_unparseable_reason(proxy_url[scheme.end() :]))
    if not parsed.host:
        raise invalid("it has no host")
    if parsed.port is not None and not 1 <= parsed.port <= 65535:
        raise invalid("the port must be a number from 1 to 65535")
    if b"@" in parsed.raw_path or "@" in parsed.fragment:
        raise invalid(f"it has an '@' after the host; {_PERCENT_ENCODE_HINT}")


def _parse_additional_headers_env() -> dict[str, str]:
    """Parse PINECONE_ADDITIONAL_HEADERS env var as JSON."""
    raw = os.environ.get("PINECONE_ADDITIONAL_HEADERS", "")
    if not raw:
        return {}
    return dict(_parse_additional_headers(raw))


@functools.lru_cache(maxsize=8)
def _parse_additional_headers(raw: str) -> dict[str, str]:
    # Cached so a malformed value warns once per process rather than on every
    # config built from the env. The value carries header secrets, so it is
    # never logged.
    import json

    try:
        parsed = json.loads(raw)
    except Exception as exc:
        logger.warning(
            "Failed to parse PINECONE_ADDITIONAL_HEADERS env var as JSON (%s), ignoring it",
            type(exc).__name__,
        )
        return {}
    if isinstance(parsed, dict):
        return {str(k): str(v) for k, v in parsed.items()}
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
            f"proxy_url='{_redact_url_userinfo(self.proxy_url)}', "
            f"proxy_headers={self._redact_headers(self.proxy_headers)!r}, "
            f"ssl_ca_certs={self.ssl_ca_certs!r}, "
            f"ssl_verify={self.ssl_verify}, "
            f"grpc_scheme={self.grpc_scheme!r}, "
            f"connection_pool_maxsize={self.connection_pool_maxsize}"
            f")"
        )

    def __post_init__(self) -> None:
        validate_timeout("timeout", self.timeout)
        if self.proxy_url:
            validate_proxy_url(self.proxy_url)
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
