"""Backwards-compatibility method shims for :class:`AssistantModel`.

Legacy callers invoked data-plane operations directly on the assistant
object (``assistant.upload_file(...)``, ``assistant.chat(...)``).
In the new SDK these live on the :class:`Assistants` namespace and
take the assistant name as a parameter. Each method in this mixin
delegates to the namespace using ``self.name``.

Back-reference storage:
    msgspec Struct instances do not have a ``__dict__`` by default and
    their ``__setattr__`` only allows setting declared struct fields.
    ``AssistantModel`` is declared with ``dict=True`` which adds a
    ``__dict__`` to each instance. :meth:`Assistants._attach_ref` writes
    directly into ``model.__dict__["_assistants"]`` to store the reference
    without going through msgspec's ``__setattr__``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import IO, TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from pinecone.client.assistants import Assistants
    from pinecone.models.assistant.chat import ChatCompletionResponse, ChatResponse
    from pinecone.models.assistant.context import ContextResponse
    from pinecone.models.assistant.file_model import AssistantFileModel
    from pinecone.models.assistant.list import ListFilesResponse
    from pinecone.models.assistant.message import Message
    from pinecone.models.assistant.options import ContextOptions
    from pinecone.models.assistant.streaming import (
        ChatCompletionStream,
        ChatStream,
    )


class AssistantModelLegacyMethodsMixin:
    """Data-plane methods available directly on :class:`AssistantModel`.

    Each method here delegates to the matching method on the
    :class:`Assistants` namespace, passing ``self.name`` as
    ``assistant_name``. Call them on an assistant object returned by
    :meth:`Assistants.describe`, :meth:`Assistants.create`, or
    :meth:`Assistants.list`.

    Sync only: a model obtained from ``AsyncAssistants`` raises
    :exc:`TypeError` here.

    .. deprecated:: 9.0.0
        Call the :class:`Assistants` namespace method directly, passing
        ``assistant_name=``.
    """

    # Declared ClassVar so msgspec ignores it when reading __struct_fields__.
    _assistants_ref: ClassVar[Any | None] = None

    def _resolve_assistants(self) -> Assistants:
        """Return the owning sync :class:`Assistants` namespace.

        Raises:
            RuntimeError: If the model has no client reference at all.
            TypeError: If the back-reference is an :class:`AsyncAssistants`
                instance — legacy shims are sync-only; async callers must use
                the namespace method directly (e.g. ``await pc.assistants.chat(
                assistant_name=model.name, ...)``).
        """

        # AsyncAssistants is imported lazily to avoid a circular import at module level.
        ref: Assistants | None = getattr(self, "_assistants", None)
        if ref is None:
            raise RuntimeError(
                "This AssistantModel has no client reference, so legacy "
                "methods cannot delegate. Use pc.assistants.<method>(...) "
                "directly, or obtain the model via "
                "pc.assistants.describe(name=...)."
            )
        from pinecone.async_client.assistants import AsyncAssistants

        if isinstance(ref, AsyncAssistants):
            raise TypeError(
                "Legacy assistant methods on AssistantModel are sync-only "
                "and cannot be used on a model retrieved from AsyncAssistants. "
                "Use the async namespace directly: "
                "await pc.assistants.<method>(assistant_name=model.name, ...)."
            )
        return ref

    def describe_file(
        self,
        file_id: str,
        include_url: bool = False,
        **kwargs: Any,
    ) -> AssistantFileModel:
        """Get the status and metadata of a file uploaded to this assistant.

        .. deprecated:: 9.0.0
            Use :meth:`Assistants.describe_file` instead.

        Args:
            file_id: Unique identifier of the file to retrieve.
            include_url: If ``True``, include a signed download URL in the
                response.

        Returns:
            :class:`AssistantFileModel` with file metadata and status.

        Examples:
            >>> file = assistant.describe_file(file_id="file-abc123")  # doctest: +SKIP
            >>> file.status  # doctest: +SKIP
            'Available'
        """
        ns = self._resolve_assistants()
        return ns.describe_file(
            assistant_name=self.name,  # type: ignore[attr-defined]
            file_id=file_id,
            include_url=include_url,
            **kwargs,
        )

    def upload_bytes_stream(
        self,
        stream: IO[bytes],
        file_name: str,
        metadata: dict[str, Any] | None = None,
        multimodal: bool | None = None,
        timeout: int | None = None,
        file_id: str | None = None,
        **kwargs: Any,
    ) -> AssistantFileModel:
        """Upload an in-memory byte stream as a file to this assistant.

        .. deprecated:: 9.0.0
            Use :meth:`Assistants.upload_file` with ``file_stream=`` and
            ``file_name=`` instead.

        Args:
            stream: An open byte stream to upload.
            file_name: Filename to associate with the upload. Must include a
                supported extension (``.txt``, ``.pdf``, ``.json``, ``.md``,
                or ``.docx``), since the extension determines how the file
                is processed.
            metadata: Optional metadata to attach to the file, e.g.
                ``{"department": "research"}``.
            multimodal: Whether to enable multimodal processing for PDFs.
            timeout: Seconds to wait for processing to complete. ``None``
                (default) polls indefinitely. Use ``-1`` to return
                immediately after upload with one describe call.
            file_id: Optional identifier for the uploaded file. When given,
                any existing file with that id is replaced.

        Returns:
            :class:`AssistantFileModel` describing the uploaded file, once
            processing completes.

        Examples:
            >>> with open("report.pdf", "rb") as f:  # doctest: +SKIP
            ...     file = assistant.upload_bytes_stream(f, file_name="report.pdf")
        """
        ns = self._resolve_assistants()
        return ns.upload_file(
            assistant_name=self.name,  # type: ignore[attr-defined]
            file_stream=stream,
            file_name=file_name,
            metadata=metadata,
            multimodal=multimodal,
            timeout=timeout,
            file_id=file_id,
            **kwargs,
        )

    def list_files(
        self,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[AssistantFileModel]:
        """Return every file for this assistant as a plain list.

        .. deprecated:: 9.0.0
            Use :meth:`Assistants.list_files` instead, which returns a lazy
            paginator rather than a materialized list.

        Args:
            filter: Optional metadata filter restricting which files are
                returned.

        Returns:
            List of :class:`AssistantFileModel` objects.

        Examples:
            >>> files = assistant.list_files()  # doctest: +SKIP
            >>> [f.name for f in files]  # doctest: +SKIP
            ['report.pdf', 'notes.md']
        """
        ns = self._resolve_assistants()
        return list(
            ns.list_files(
                assistant_name=self.name,  # type: ignore[attr-defined]
                filter=filter,
            )
        )

    def list_files_paginated(
        self,
        filter: dict[str, Any] | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
        *,
        page_size: int | None = None,
        **kwargs: Any,
    ) -> ListFilesResponse:
        """Return a single page of files for this assistant.

        .. deprecated:: 9.0.0
            Use :meth:`Assistants.list_files_page` instead. ``limit`` and
            ``page_size`` are accepted here for backwards compatibility but
            are not forwarded to the request.

        Args:
            filter: Optional metadata filter restricting which files are
                returned.
            limit: Accepted but not forwarded to the request.
            pagination_token: Token from a previous response to fetch the
                next page.
            page_size: Accepted but not forwarded to the request.

        Returns:
            :class:`ListFilesResponse` with a ``files`` list and an optional
            ``next`` continuation token.

        Examples:
            >>> page = assistant.list_files_paginated()  # doctest: +SKIP
            >>> [f.name for f in page.files]  # doctest: +SKIP
            ['report.pdf']
        """
        ns = self._resolve_assistants()
        return ns.list_files_page(
            assistant_name=self.name,  # type: ignore[attr-defined]
            filter=filter,
            pagination_token=pagination_token,
        )

    def upload_file(
        self,
        file_path: str,
        metadata: dict[str, Any] | None = None,
        multimodal: bool | None = None,
        timeout: int | None = None,
        file_id: str | None = None,
        **kwargs: Any,
    ) -> AssistantFileModel:
        """Upload a local file to this assistant.

        .. deprecated:: 9.0.0
            Use :meth:`Assistants.upload_file` instead.

        Args:
            file_path: Path to a local file to upload.
            metadata: Optional metadata to attach to the file, e.g.
                ``{"department": "research"}``.
            multimodal: Whether to enable multimodal processing for PDFs.
            timeout: Seconds to wait for processing to complete. ``None``
                (default) polls indefinitely. Use ``-1`` to return
                immediately after upload with one describe call.
            file_id: Optional identifier for the uploaded file. When given,
                any existing file with that id is replaced.

        Returns:
            :class:`AssistantFileModel` describing the uploaded file, once
            processing completes.

        Examples:
            >>> file = assistant.upload_file(file_path="/data/report.pdf")  # doctest: +SKIP
            >>> file.status  # doctest: +SKIP
            'Available'
        """
        ns = self._resolve_assistants()
        return ns.upload_file(
            assistant_name=self.name,  # type: ignore[attr-defined]
            file_path=file_path,
            metadata=metadata,
            multimodal=multimodal,
            timeout=timeout,
            file_id=file_id,
            **kwargs,
        )

    def delete_file(
        self,
        file_id: str,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Delete a file from this assistant.

        .. deprecated:: 9.0.0
            Use :meth:`Assistants.delete_file` instead.

        Args:
            file_id: Unique identifier of the file to delete.
            timeout: Seconds to wait for the deletion to finish. ``None``
                (default) polls indefinitely. Use ``-1`` to return as soon
                as the request is accepted — the file may still exist when
                this returns.

        Raises:
            :exc:`PineconeTimeoutError`: If the deletion has not finished
                before *timeout* seconds elapse.

        Examples:
            >>> assistant.delete_file(file_id="file-abc123")  # doctest: +SKIP
        """
        ns = self._resolve_assistants()
        ns.delete_file(
            assistant_name=self.name,  # type: ignore[attr-defined]
            file_id=file_id,
            timeout=timeout,
            **kwargs,
        )

    def chat_completions(
        self,
        messages: list[Message] | list[dict[str, Any]],
        filter: dict[str, Any] | None = None,
        stream: bool = False,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> ChatCompletionResponse | ChatCompletionStream:
        """Chat with this assistant using an OpenAI-compatible interface.

        .. deprecated:: 9.0.0
            Use :meth:`Assistants.chat_completions` instead.

        Args:
            messages: Conversation messages. Dicts are converted to
                :class:`Message` objects; role defaults to ``"user"`` when
                not present.
            filter: Metadata filter restricting which documents are used as
                context.
            stream: If ``True``, return a :class:`ChatCompletionStream`.
            model: Large language model to use, e.g. ``"gpt-4o"``. If
                omitted, ``None`` is sent to the API rather than a default
                model name, so pass one explicitly.
            temperature: Controls randomness. Lower values produce more
                deterministic responses.

        Returns:
            :class:`ChatCompletionResponse` for non-streaming requests, or a
            :class:`ChatCompletionStream` for streaming requests.

        Examples:
            >>> response = assistant.chat_completions(  # doctest: +SKIP
            ...     messages=[{"content": "What is Pinecone?"}],
            ...     model="gpt-4o",
            ... )
            >>> response.choices[0].message.content  # doctest: +SKIP
        """
        ns = self._resolve_assistants()
        return ns.chat_completions(
            assistant_name=self.name,  # type: ignore[attr-defined]
            messages=messages,
            filter=filter,
            stream=stream,
            model=model,  # type: ignore[arg-type]
            temperature=temperature,
            **kwargs,
        )

    def context(
        self,
        query: str | None = None,
        messages: Sequence[Message | Mapping[str, str]] | None = None,
        filter: dict[str, Any] | None = None,
        top_k: int | None = None,
        snippet_size: int | None = None,
        multimodal: bool | None = None,
        include_binary_content: bool | None = None,
        context_options: ContextOptions | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ContextResponse:
        """Retrieve relevant context snippets from this assistant.

        .. deprecated:: 9.0.0
            Use :meth:`Assistants.context` instead. Exactly one of *query*
            or *messages* must be provided; the namespace method raises
            :exc:`PineconeValueError` otherwise.

        Args:
            query: Text query to use for context retrieval. Mutually
                exclusive with *messages*.
            messages: Conversation messages to use for context retrieval.
                Mutually exclusive with *query*.
            filter: Metadata filter restricting which documents contribute
                context.
            top_k: Maximum number of context snippets to return.
            snippet_size: Maximum snippet size in tokens.
            multimodal: Whether to include image-related context snippets.
            include_binary_content: Whether image snippets include base64
                image data. Only meaningful when *multimodal* is ``True``.
            context_options: Convenience bundle for *multimodal*,
                *include_binary_content*, *top_k*, and *snippet_size*. Any
                of those four passed explicitly override the matching value
                from *context_options*.

        Returns:
            :class:`ContextResponse` containing the matching context
            snippets.

        Examples:
            >>> response = assistant.context(query="What is Pinecone?")  # doctest: +SKIP
            >>> for snippet in response.snippets:  # doctest: +SKIP
            ...     print(snippet.content)
        """
        ns = self._resolve_assistants()
        # Unpack context_options. Explicit kwargs win over context_options values.
        if context_options is not None:
            if isinstance(context_options, dict):
                if multimodal is None:
                    multimodal = context_options.get("multimodal")
                if include_binary_content is None:
                    include_binary_content = context_options.get("include_binary_content")
                if top_k is None:
                    top_k = context_options.get("top_k")
                if snippet_size is None:
                    snippet_size = context_options.get("snippet_size")
            else:
                if multimodal is None:
                    multimodal = context_options.multimodal
                if include_binary_content is None:
                    include_binary_content = context_options.include_binary_content
                if top_k is None:
                    top_k = context_options.top_k
                if snippet_size is None:
                    snippet_size = context_options.snippet_size
        return ns.context(
            assistant_name=self.name,  # type: ignore[attr-defined]
            query=query,
            messages=messages,
            filter=filter,
            top_k=top_k,
            snippet_size=snippet_size,
            multimodal=multimodal,
            include_binary_content=include_binary_content,
        )

    def chat(
        self,
        messages: list[Message] | list[dict[str, Any]],
        filter: dict[str, Any] | None = None,
        stream: bool = False,
        model: str | None = None,
        temperature: float | None = None,
        json_response: bool = False,
        include_highlights: bool = False,
        context_options: ContextOptions | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChatResponse | ChatStream:
        """Chat with this assistant and receive citations in Pinecone-native format.

        .. deprecated:: 9.0.0
            Use :meth:`Assistants.chat` instead.

        Args:
            messages: Conversation messages. Dicts are converted to
                :class:`Message` objects; role defaults to ``"user"`` when
                not present.
            filter: Metadata filter restricting which documents are used as
                context.
            stream: If ``True``, return a :class:`ChatStream`.
            model: Large language model to use, e.g. ``"gpt-4o"``. If
                omitted, ``None`` is sent to the API rather than a default
                model name, so pass one explicitly.
            temperature: Controls randomness. Lower values produce more
                deterministic responses.
            json_response: If ``True``, instruct the assistant to return a
                JSON response. Cannot be combined with ``stream=True``.
            include_highlights: If ``True``, include highlight snippets from
                referenced documents in citations.
            context_options: Options controlling context retrieval.

        Returns:
            :class:`ChatResponse` for non-streaming requests, or a
            :class:`ChatStream` for streaming requests.

        Examples:
            >>> response = assistant.chat(  # doctest: +SKIP
            ...     messages=[{"content": "What is Pinecone?"}],
            ...     model="gpt-4o",
            ... )
        """
        ns = self._resolve_assistants()
        return ns.chat(
            assistant_name=self.name,  # type: ignore[attr-defined]
            messages=messages,
            filter=filter,
            stream=stream,
            model=model,  # type: ignore[arg-type]
            temperature=temperature,
            json_response=json_response,
            include_highlights=include_highlights,
            context_options=context_options,
            **kwargs,
        )
