"""Backwards-compatibility shim for :mod:`pinecone.config.config`.

Re-exports classes that used to live at :mod:`pinecone.config.config` before
the ``python-sdk2`` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

# XXX: no canonical equivalent; this shim is the only definition
from __future__ import annotations

from typing import NamedTuple

from pinecone._internal.config import _redact_url_userinfo
from pinecone._internal.http_client import _redact_headers

__all__ = ["Config"]

_EMPTY_HEADERS: dict[str, str] = {}


class Config(NamedTuple):
    """Legacy SDK configuration named tuple.

    Fields mirror the legacy :class:`pinecone.config.config.Config` from
    before the ``python-sdk2`` rewrite.
    """

    api_key: str = ""
    host: str = ""
    proxy_url: str | None = None
    proxy_headers: dict[str, str] | None = None
    ssl_ca_certs: str | None = None
    ssl_verify: bool | None = None
    additional_headers: dict[str, str] = _EMPTY_HEADERS
    source_tag: str | None = None

    def __repr__(self) -> str:
        masked_key = f"...{self.api_key[-4:]}" if len(self.api_key) >= 4 else "***"
        redacted_additional = _redact_headers(self.additional_headers)
        redacted_proxy = (
            _redact_headers(self.proxy_headers) if self.proxy_headers is not None else None
        )
        proxy_url = _redact_url_userinfo(self.proxy_url) if self.proxy_url is not None else None
        return (
            f"Config("
            f"api_key={masked_key!r}, "
            f"host={self.host!r}, "
            f"proxy_url={proxy_url!r}, "
            f"proxy_headers={redacted_proxy!r}, "
            f"ssl_ca_certs={self.ssl_ca_certs!r}, "
            f"ssl_verify={self.ssl_verify!r}, "
            f"additional_headers={redacted_additional!r}, "
            f"source_tag={self.source_tag!r}"
            f")"
        )
