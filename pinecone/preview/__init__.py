"""Preview namespace — pre-release API features not covered by SemVer.

Access via ``pc.preview``. See docs/conventions/preview-channel.md for
the full lifecycle (introduction, iteration, graduation, retirement).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pinecone.errors.exceptions import PineconeValueError
from pinecone.preview.schema_builder import PreviewSchemaBuilder as PreviewSchemaBuilder

SchemaBuilder = PreviewSchemaBuilder  # spec/preview.md §12 — entry-point alias

if TYPE_CHECKING:
    from pinecone._internal.config import PineconeConfig
    from pinecone._internal.http_client import AsyncHTTPClient, HTTPClient
    from pinecone.preview.async_index import AsyncPreviewIndex
    from pinecone.preview.async_indexes import AsyncPreviewIndexes
    from pinecone.preview.index import PreviewIndex
    from pinecone.preview.indexes import PreviewIndexes

__all__ = ["AsyncPreview", "Preview", "PreviewSchemaBuilder", "SchemaBuilder"]


class Preview:
    """Sync preview namespace — routes to per-area preview classes.

    .. admonition:: Preview
       :class: warning

       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Args:
        http: Shared HTTP client from the parent :class:`~pinecone.Pinecone` instance.
        config: SDK configuration shared with the parent client.

    Examples:
        >>> from pinecone import Pinecone
        >>> pc = Pinecone(api_key="your-api-key")
        >>> info = pc.preview.indexes.describe("articles-en-preview")
        >>> print(info.host)
    """

    def __init__(self, http: HTTPClient, config: PineconeConfig) -> None:
        self._http = http
        self._config = config
        self._indexes: PreviewIndexes | None = None
        self._host_cache: dict[str, str] = {}

    @property
    def indexes(self) -> PreviewIndexes:
        """Access the preview indexes control-plane namespace.

        Lazily instantiated on first access. Reuses the parent client's
        configuration and credentials.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Returns:
            :class:`~pinecone.preview.indexes.PreviewIndexes` instance.

        Examples:
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> info = pc.preview.indexes.describe("articles-en-preview")
            >>> print(info.host)
        """
        if self._indexes is None:
            from pinecone.preview.indexes import PreviewIndexes

            self._indexes = PreviewIndexes(config=self._config)
        return self._indexes

    def index(
        self,
        *,
        name: str | None = None,
        host: str | None = None,
    ) -> PreviewIndex:
        """Get a :class:`~pinecone.preview.index.PreviewIndex` for data-plane operations.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Exactly one of ``name`` or ``host`` must be provided.  When ``name``
        is given, the host is resolved via
        :meth:`~pinecone.preview.indexes.PreviewIndexes.describe` and the
        result is cached on this :class:`Preview` instance so subsequent calls
        with the same name avoid an extra control-plane round-trip.

        Args:
            name: Index name. The host is resolved and cached via
                ``preview.indexes.describe(name).host``.
            host: Data-plane host URL. Passed through directly without a
                control-plane call.

        Returns:
            :class:`~pinecone.preview.index.PreviewIndex` connected to the
            resolved host.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If neither
                or both of ``name`` and ``host`` are provided.
            :exc:`~pinecone.errors.exceptions.NotFoundError`: If ``name`` is
                given but the index does not exist.
            :exc:`~pinecone.errors.exceptions.ApiError`: If the describe call
                returns an error response.

        Examples:
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> index = pc.preview.index(name="articles-en-preview")

            Or pass a host directly to skip the control-plane call:

            >>> index = pc.preview.index(host="https://articles-en-preview-xyz.pinecone.io")
        """
        if name is None and host is None:
            raise PineconeValueError("Exactly one of 'name' or 'host' must be provided.")
        if name is not None and host is not None:
            raise PineconeValueError("Exactly one of 'name' or 'host' must be provided.")

        resolved_host: str
        if host is not None:
            resolved_host = host
        elif name is not None:
            if name not in self._host_cache:
                described_host = self.indexes.describe(name).host
                if described_host is None:
                    raise PineconeValueError(
                        f"Index {name!r} does not yet have a host assigned — "
                        "the index may still be initializing. "
                        "Wait until the index status is 'Ready' before connecting."
                    )
                if not self._config.ssl_verify and described_host.startswith("https://"):
                    described_host = "http://" + described_host[len("https://"):]
                self._host_cache[name] = described_host
            resolved_host = self._host_cache[name]
        else:
            raise PineconeValueError("Exactly one of 'name' or 'host' must be provided.")

        from pinecone.preview.index import PreviewIndex

        return PreviewIndex(host=resolved_host, config=self._config)

    def close(self) -> None:
        """Close preview sub-clients. Idempotent.

        .. admonition:: Preview
           :class: warning

           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Examples:
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> index = pc.preview.index(name="articles-en-preview")
            >>> pc.preview.close()

            Or use the parent client as a context manager, which closes preview automatically:

            >>> with Pinecone(api_key="your-api-key") as pc:
            ...     index = pc.preview.index(name="articles-en-preview")
        """
        if self._indexes is not None:
            self._indexes.close()

    def __repr__(self) -> str:
        return "Preview()"


class AsyncPreview:
    """Async preview namespace — routes to per-area async preview classes.

    .. admonition:: Preview
       :class: warning

       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Args:
        http: Shared async HTTP client from the parent :class:`~pinecone.AsyncPinecone` instance.
        config: SDK configuration shared with the parent client.

    Examples:
        >>> import asyncio
        >>> from pinecone import AsyncPinecone
        >>> async def main():
        ...     pc = AsyncPinecone(api_key="your-api-key")
        ...     info = await pc.preview.indexes.describe("articles-en-preview")
        ...     print(info.host)
        >>> asyncio.run(main())
    """

    def __init__(self, http: AsyncHTTPClient, config: PineconeConfig) -> None:
        self._http = http
        self._config = config
        self._indexes: AsyncPreviewIndexes | None = None
        self._host_cache: dict[str, str] = {}

    @property
    def indexes(self) -> AsyncPreviewIndexes:
        """Access the async preview indexes control-plane namespace.

        Lazily instantiated on first access. Reuses the parent client's
        configuration and credentials.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Returns:
            :class:`~pinecone.preview.async_indexes.AsyncPreviewIndexes` instance.

        Examples:
            >>> import asyncio
            >>> from pinecone import AsyncPinecone
            >>> async def main():
            ...     pc = AsyncPinecone(api_key="your-api-key")
            ...     info = await pc.preview.indexes.describe("articles-en-preview")
            ...     print(info.host)
            >>> asyncio.run(main())
        """
        if self._indexes is None:
            from pinecone.preview.async_indexes import AsyncPreviewIndexes

            self._indexes = AsyncPreviewIndexes(config=self._config)
        return self._indexes

    def index(
        self,
        *,
        name: str | None = None,
        host: str | None = None,
    ) -> AsyncPreviewIndex:
        """Get an :class:`~pinecone.preview.async_index.AsyncPreviewIndex` for data-plane ops.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Exactly one of ``name`` or ``host`` must be provided.  When ``name``
        is given, the host is resolved lazily on the first data-plane call via
        :meth:`~pinecone.preview.async_indexes.AsyncPreviewIndexes.describe`
        and the result is cached on this :class:`AsyncPreview` instance so
        subsequent data-plane calls with the same name avoid an extra
        control-plane round-trip.

        This method is synchronous (no ``await`` required). Host resolution
        only happens when the returned
        :class:`~pinecone.preview.async_index.AsyncPreviewIndex` performs its
        first I/O operation.

        Args:
            name: Index name. The host is resolved and cached lazily via
                ``await preview.indexes.describe(name)`` on first data-plane use.
            host: Data-plane host URL. Passed through directly without a
                control-plane call.

        Returns:
            :class:`~pinecone.preview.async_index.AsyncPreviewIndex` connected
            to the resolved host.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If neither
                or both of ``name`` and ``host`` are provided.
            :exc:`~pinecone.errors.exceptions.NotFoundError`: If ``name`` is
                given but the index does not exist (raised on first data-plane call).
            :exc:`~pinecone.errors.exceptions.ApiError`: If the describe call
                returns an error response (raised on first data-plane call).

        Examples:
            >>> import asyncio
            >>> from pinecone import AsyncPinecone
            >>> async def main():
            ...     pc = AsyncPinecone(api_key="your-api-key")
            ...     index = pc.preview.index(name="articles-en-preview")
            >>> asyncio.run(main())

            Or pass a host directly to skip the control-plane call:

            >>> async def main():
            ...     pc = AsyncPinecone(api_key="your-api-key")
            ...     index = pc.preview.index(host="https://articles-en-preview-xyz.pinecone.io")
            >>> asyncio.run(main())
        """
        if name is None and host is None:
            raise PineconeValueError("Exactly one of 'name' or 'host' must be provided.")
        if name is not None and host is not None:
            raise PineconeValueError("Exactly one of 'name' or 'host' must be provided.")

        from pinecone.preview.async_index import AsyncPreviewIndex

        if host is not None:
            return AsyncPreviewIndex(host=host, config=self._config)

        # name path: defer describe() to the first data-plane call.
        # The two validation checks above guarantee name is str here; this
        # guard is for mypy --strict type narrowing only.
        if name is None:
            raise PineconeValueError("Exactly one of 'name' or 'host' must be provided.")

        host_cache = self._host_cache
        indexes = self.indexes
        ssl_verify = self._config.ssl_verify

        async def _resolve() -> str:
            if name not in host_cache:
                desc = await indexes.describe(name)
                if desc.host is None:
                    raise PineconeValueError(
                        f"Index {name!r} does not yet have a host assigned — "
                        "the index may still be initializing. "
                        "Wait until the index status is 'Ready' before connecting."
                    )
                resolved = desc.host
                if not ssl_verify and resolved.startswith("https://"):
                    resolved = "http://" + resolved[len("https://"):]
                host_cache[name] = resolved
            return host_cache[name]

        return AsyncPreviewIndex(config=self._config, _host_provider=_resolve)

    async def close(self) -> None:
        """Close async preview sub-clients. Idempotent.

        .. admonition:: Preview
           :class: warning

           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Examples:
            >>> import asyncio
            >>> from pinecone import AsyncPinecone
            >>> async def main():
            ...     pc = AsyncPinecone(api_key="your-api-key")
            ...     index = pc.preview.index(name="articles-en-preview")
            ...     await pc.preview.close()
            >>> asyncio.run(main())
        """
        if self._indexes is not None:
            await self._indexes.close()

    def __repr__(self) -> str:
        return "AsyncPreview()"
