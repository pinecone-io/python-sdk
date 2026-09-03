"""Streaming chunk models for the Assistant API.

Two stream shapes live here. The Pinecone-native chat stream yields the four
:data:`ChatStreamChunk` variants, dispatched on a ``type`` tag, and only
:class:`StreamContentChunk` carries response text. The OpenAI-compatible
completion stream yields :class:`ChatCompletionStreamChunk`, which carries its
text nested under ``choices``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any, TypeAlias

from msgspec import Struct

from pinecone.models._display import HtmlBuilder, safe_display, truncate_text
from pinecone.models.assistant._mixin import StructDictMixin
from pinecone.models.assistant.chat import ChatCitation, ChatUsage


class StreamMessageStart(
    StructDictMixin, Struct, kw_only=True, tag="message_start", tag_field="type"
):
    """The chunk that opens a chat stream, carrying no response text.

    Arrives once, before any content. Carries nothing you have to render, but
    ``context_snippet_count`` lets you detect "no relevant context found"
    before the answer starts arriving.

    Attributes:
        type: Discriminator value ``"message_start"``.
        model: Name of the model that generated the answer, which need not be
            the name you requested.
        role: The role of the message author (e.g. ``"assistant"``).
        id: Identifier of the chat response, the same on every chunk of the
            stream, or ``None`` if the server did not report it here.
        context_snippet_count: Number of retrieved context snippets that were
            provided to the model, or ``None`` if the server did not report it.
            ``0`` means no relevant context was found for the query.
        content_filter_results: Safety classifications reported by the LLM
            provider, or ``None`` when the provider returned none. Read
            ``spec`` for the provider's name and ``results`` for a payload
            whose shape that provider defines.

    .. seealso::
       :data:`ChatStreamChunk` — the four chunk types and the loop that
       consumes them.
    """

    model: str
    role: str
    id: str | None = None
    context_snippet_count: int | None = None
    content_filter_results: dict[str, Any] | None = None

    @property
    def type(self) -> str:
        """Discriminator value, always ``"message_start"``."""
        return str(self.__struct_config__.tag)

    @safe_display
    def __repr__(self) -> str:
        id_part = f"id={self.id!r}, " if self.id is not None else ""
        snippet_part = (
            f", context_snippet_count={self.context_snippet_count}"
            if self.context_snippet_count is not None
            else ""
        )
        return (
            f"StreamMessageStart({id_part}model={self.model!r}, role={self.role!r}{snippet_part})"
        )

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("StreamMessageStart(...)")
            return
        with p.group(2, "StreamMessageStart(", ")"):
            if self.id is not None:
                p.breakable()
                p.text(f"id={self.id!r},")
            p.breakable()
            p.text(f"model={self.model!r},")
            p.breakable()
            p.text(f"role={self.role!r},")
            if self.context_snippet_count is not None:
                p.breakable()
                p.text(f"context_snippet_count={self.context_snippet_count},")
            if self.content_filter_results is not None:
                p.breakable()
                filter_text = truncate_text(str(self.content_filter_results), 200)
                p.text(f"content_filter_results={filter_text},")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("StreamMessageStart")
        builder.row("Type:", self.type)
        if self.id is not None:
            builder.row("Id:", self.id)
        builder.row("Model:", self.model)
        builder.row("Role:", self.role)
        if self.context_snippet_count is not None:
            builder.row("Context snippets:", self.context_snippet_count)
        if self.content_filter_results is not None:
            builder.row(
                "Content filter results:", truncate_text(str(self.content_filter_results), 500)
            )
        return builder.build()


class StreamContentDelta(StructDictMixin, Struct, kw_only=True):
    """The delta payload within a content chunk.

    Reached as ``chunk.delta`` on a :class:`StreamContentChunk`. This is where
    the response text lives in a Pinecone-native chat stream.

    Attributes:
        content: The text fragment for this chunk. Concatenate the fragments
            in arrival order to rebuild the full answer.
    """

    content: str

    @safe_display
    def __repr__(self) -> str:
        return f"StreamContentDelta(content={truncate_text(self.content, 80)!r})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("StreamContentDelta(...)")
            return
        with p.group(2, "StreamContentDelta(", ")"):
            p.breakable()
            p.text(f"content={truncate_text(self.content, 200)!r},")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("StreamContentDelta")
        builder.row("Content", truncate_text(self.content, 500))
        return builder.build()


class StreamContentChunk(
    StructDictMixin, Struct, kw_only=True, tag="content_chunk", tag_field="type"
):
    """The only chunk type that carries response text, at ``delta.content``.

    Arrives many times per response, each with one fragment of the answer.
    A caller that renders the answer as it streams needs this chunk and
    nothing else.

    Attributes:
        type: Discriminator value ``"content_chunk"``.
        id: Identifier of the chat response, the same on every chunk of the
            stream.
        delta: The :class:`StreamContentDelta` holding this fragment; the text
            is at ``delta.content``.
        model: Name of the model that generated the answer, or ``None`` if the
            server did not repeat it on this chunk.
        content_filter_results: Safety classifications reported by the LLM
            provider for this fragment, or ``None`` when the provider returned
            none. Read ``spec`` for the provider's name and ``results`` for a
            payload whose shape that provider defines.

    .. seealso::
       :data:`ChatStreamChunk` — the four chunk types and the loop that
       consumes them.
    """

    id: str
    delta: StreamContentDelta
    model: str | None = None
    content_filter_results: dict[str, Any] | None = None

    @property
    def type(self) -> str:
        """Discriminator value, always ``"content_chunk"``."""
        return str(self.__struct_config__.tag)

    @safe_display
    def __repr__(self) -> str:
        model_part = f", model={self.model!r}" if self.model is not None else ""
        return f"StreamContentChunk(id={self.id!r}, delta={self.delta!r}{model_part})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("StreamContentChunk(...)")
            return
        with p.group(2, "StreamContentChunk(", ")"):
            p.breakable()
            p.text(f"id={self.id!r},")
            if self.model is not None:
                p.breakable()
                p.text(f"model={self.model!r},")
            p.breakable()
            p.text(f"delta={self.delta!r},")
            if self.content_filter_results is not None:
                p.breakable()
                filter_text = truncate_text(str(self.content_filter_results), 200)
                p.text(f"content_filter_results={filter_text},")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("StreamContentChunk")
        builder.row("Type:", self.type)
        builder.row("Id:", self.id)
        if self.model is not None:
            builder.row("Model:", self.model)
        builder.row("Content:", truncate_text(self.delta.content, 500))
        if self.content_filter_results is not None:
            builder.row(
                "Content filter results:", truncate_text(str(self.content_filter_results), 500)
            )
        return builder.build()


class StreamCitationChunk(StructDictMixin, Struct, kw_only=True, tag="citation", tag_field="type"):
    """The chunk that links a position in the answer to its source documents.

    Arrives zero or more times, alongside the content chunks. This is the
    chunk a RAG caller needs to render sources: ``chunk.citation.position`` is
    the character position in the response text the citation annotates, and
    each entry of ``chunk.citation.references`` exposes ``reference.file``
    (an :class:`~pinecone.models.assistant.file_model.AssistantFileModel`,
    so ``reference.file.name`` and ``reference.file.metadata``),
    ``reference.pages``, and ``reference.highlight``. The highlight is
    ``None`` unless the chat request set ``include_highlights=True``.

    Attributes:
        type: Discriminator value ``"citation"``.
        id: Identifier of the chat response, the same on every chunk of the
            stream.
        citation: The :class:`~pinecone.models.assistant.chat.ChatCitation`
            holding ``position`` and ``references``.
        model: Name of the model that generated the answer, or ``None`` if the
            server did not repeat it on this chunk.

    .. seealso::
       :data:`ChatStreamChunk` — the four chunk types and the loop that
       consumes them.
    """

    id: str
    citation: ChatCitation
    model: str | None = None

    @property
    def type(self) -> str:
        """Discriminator value, always ``"citation"``."""
        return str(self.__struct_config__.tag)

    @safe_display
    def __repr__(self) -> str:
        model_part = f", model={self.model!r}" if self.model is not None else ""
        return f"StreamCitationChunk(id={self.id!r}, citation={self.citation!r}{model_part})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("StreamCitationChunk(...)")
            return
        with p.group(2, "StreamCitationChunk(", ")"):
            p.breakable()
            p.text(f"id={self.id!r},")
            if self.model is not None:
                p.breakable()
                p.text(f"model={self.model!r},")
            p.breakable()
            p.text(f"citation={self.citation!r},")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("StreamCitationChunk")
        builder.row("Type:", self.type)
        builder.row("Id:", self.id)
        if self.model is not None:
            builder.row("Model:", self.model)
        builder.row("Position:", self.citation.position)
        builder.row("References:", len(self.citation.references))
        return builder.build()


class StreamMessageEnd(StructDictMixin, Struct, kw_only=True, tag="message_end", tag_field="type"):
    """The chunk that closes a chat stream, carrying usage and finish reason.

    Arrives once, last, and carries no response text. Read ``finish_reason``
    here to tell a complete answer from one the model cut short.

    Attributes:
        type: Discriminator value ``"message_end"``.
        id: Identifier of the chat response, the same on every chunk of the
            stream.
        usage: :class:`~pinecone.models.assistant.chat.ChatUsage` token counts
            for the whole request, or ``None`` if the server did not report
            them.
        model: Name of the model that generated the answer, or ``None`` if the
            server did not repeat it on this chunk.
        finish_reason: Why generation stopped: ``"stop"`` (the model
            finished), ``"length"`` (the token limit was reached),
            ``"content_filter"`` (content filtering rules blocked the output),
            or ``"tool_calls"`` (a tool call was triggered). The literal
            string ``"null"`` also reaches callers and is a different value
            from Python ``None``, so treat this as an open set of strings
            rather than switching exhaustively on the four above.
        content_filter_results: Safety classifications reported by the LLM
            provider, or ``None`` when the provider returned none. Read
            ``spec`` for the provider's name and ``results`` for a payload
            whose shape that provider defines.

    .. seealso::
       :data:`ChatStreamChunk` — the four chunk types and the loop that
       consumes them.
    """

    id: str
    usage: ChatUsage | None = None
    model: str | None = None
    finish_reason: str | None = None
    content_filter_results: dict[str, Any] | None = None

    @property
    def type(self) -> str:
        """Discriminator value, always ``"message_end"``."""
        return str(self.__struct_config__.tag)

    @safe_display
    def __repr__(self) -> str:
        model_part = f", model={self.model!r}" if self.model is not None else ""
        usage_part = f", usage={self.usage!r}" if self.usage is not None else ""
        finish_part = (
            f", finish_reason={self.finish_reason!r}" if self.finish_reason is not None else ""
        )
        return f"StreamMessageEnd(id={self.id!r}{finish_part}{usage_part}{model_part})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("StreamMessageEnd(...)")
            return
        with p.group(2, "StreamMessageEnd(", ")"):
            p.breakable()
            p.text(f"id={self.id!r},")
            if self.model is not None:
                p.breakable()
                p.text(f"model={self.model!r},")
            if self.finish_reason is not None:
                p.breakable()
                p.text(f"finish_reason={self.finish_reason!r},")
            if self.usage is not None:
                p.breakable()
                p.text(f"usage={self.usage!r},")
            if self.content_filter_results is not None:
                p.breakable()
                filter_text = truncate_text(str(self.content_filter_results), 200)
                p.text(f"content_filter_results={filter_text},")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("StreamMessageEnd")
        builder.row("Type:", self.type)
        builder.row("Id:", self.id)
        if self.model is not None:
            builder.row("Model:", self.model)
        if self.finish_reason is not None:
            builder.row("Finish reason:", self.finish_reason)
        if self.usage is not None:
            builder.row("Prompt tokens:", self.usage.prompt_tokens)
            builder.row("Completion tokens:", self.usage.completion_tokens)
            builder.row("Total tokens:", self.usage.total_tokens)
        if self.content_filter_results is not None:
            builder.row(
                "Content filter results:", truncate_text(str(self.content_filter_results), 500)
            )
        return builder.build()


ChatStreamChunk: TypeAlias = (
    StreamMessageStart | StreamContentChunk | StreamCitationChunk | StreamMessageEnd
)
"""One chunk of a Pinecone-native chat stream, tagged by its ``type`` field.

Iterating a :class:`ChatStream` yields these four classes, and branching on
which one arrived is the whole contract. Each also exposes its tag as
``chunk.type``, so a caller can dispatch on ``isinstance`` or on the string.

:class:`StreamMessageStart` (``type == "message_start"``)
    Arrives once, first. No response text. ``model``, ``role``, and
    ``context_snippet_count`` — a ``0`` there means nothing relevant was
    retrieved, which you learn before the answer starts.

:class:`StreamContentChunk` (``type == "content_chunk"``)
    Arrives many times, and is **the only chunk carrying response text**, at
    ``chunk.delta.content``. Concatenate the fragments in arrival order.

:class:`StreamCitationChunk` (``type == "citation"``)
    Arrives zero or more times, alongside the content chunks.
    ``chunk.citation.position`` is a character position in the response text,
    and each of ``chunk.citation.references`` has ``reference.file.name``,
    ``reference.pages``, and ``reference.highlight``.

:class:`StreamMessageEnd` (``type == "message_end"``)
    Arrives once, last. No response text. ``usage`` token counts and
    ``finish_reason``.

Examples:
    .. code-block:: python

        from pinecone import (
            Pinecone,
            StreamCitationChunk,
            StreamContentChunk,
            StreamMessageEnd,
            StreamMessageStart,
        )

        pc = Pinecone(api_key="your-api-key")
        stream = pc.assistants.chat(
            assistant_name="acme-support-bot",
            messages=[{"content": "Which regions support BYOC?"}],
            stream=True,
        )

        answer: list[str] = []
        sources: list[str] = []
        for chunk in stream:
            if isinstance(chunk, StreamMessageStart):
                if chunk.context_snippet_count == 0:
                    print("no relevant context was retrieved")
            elif isinstance(chunk, StreamContentChunk):
                answer.append(chunk.delta.content)
                print(chunk.delta.content, end="", flush=True)
            elif isinstance(chunk, StreamCitationChunk):
                for reference in chunk.citation.references:
                    sources.append(reference.file.name)
            elif isinstance(chunk, StreamMessageEnd):
                print(f"\\nstopped because: {chunk.finish_reason}")

        print("".join(answer), sources)

    Use :meth:`ChatStream.text` instead when you only want the text and no
    citations.

.. seealso::
   :class:`ChatCompletionStreamChunk` — the chunk type of the
   OpenAI-compatible stream, whose text is nested under ``choices`` and whose
   citations are woven into the text rather than delivered as objects.
"""


class ChatStream:
    """A Pinecone-native chat stream, returned by ``chat(..., stream=True)``.

    Iterating it yields the :data:`ChatStreamChunk` variants, which is the
    only way to reach citations and token usage. :meth:`text` and
    :meth:`collect` skip the dispatch and hand you text alone. The stream is
    single-pass: iterating, :meth:`text`, and :meth:`collect` all consume the
    same underlying iterator, so pick one.

    Examples:
        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone(api_key="your-api-key")
            stream = pc.assistants.chat(
                assistant_name="acme-support-bot",
                messages=[{"content": "What can you help me with?"}],
                stream=True,
            )
            for fragment in stream.text():
                print(fragment, end="", flush=True)

    .. seealso::
       - :data:`ChatStreamChunk` — the four chunk types, and the loop to write
         when you need citations rather than text alone.
       - :class:`ChatCompletionStream` — the same request in the
         OpenAI-compatible shape, from
         :meth:`~pinecone.client.assistants.Assistants.chat_completions`.
       - :class:`AsyncChatStream` — the ``AsyncPinecone`` equivalent.
    """

    def __init__(self, stream: Iterator[ChatStreamChunk]) -> None:
        self._stream = stream

    @safe_display
    def __repr__(self) -> str:
        return (
            "ChatStream(single-pass, Pinecone-native chat stream"
            " — iterate with `for chunk in stream` or `stream.text()`)"
        )

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("ChatStream")
        builder.row("Type", "Pinecone-native chat stream")
        builder.row("Iteration", "single-pass")
        builder.row(
            "Usage hint",
            "Iterate with `for chunk in stream`, or call `.text()` for"
            " text-only fragments, or `.collect()` for the full message",
        )
        return builder.build()

    def __iter__(self) -> Iterator[ChatStreamChunk]:
        return self._stream

    def text(self) -> Iterator[str]:
        """Yield only the response text, dropping every non-content chunk.

        Returns:
            Iterator over ``delta.content`` of each
            :class:`StreamContentChunk`, in arrival order. The start,
            citation and end chunks are discarded, so citations and token
            usage are not reachable through this method — iterate the stream
            itself for those.

        Examples:
            .. code-block:: python

                stream = pc.assistants.chat(
                    assistant_name="acme-support-bot",
                    messages=[{"content": "Explain vector databases in one sentence."}],
                    stream=True,
                )
                for fragment in stream.text():
                    print(fragment, end="", flush=True)

        .. seealso::
           :meth:`collect` — the same fragments already joined into one
           string, for when you do not need to render as they arrive.
        """
        for chunk in self._stream:
            if isinstance(chunk, StreamContentChunk):
                yield chunk.delta.content

    def collect(self) -> str:
        """Drain the whole stream and return the answer as one string.

        Blocks until the server closes the stream, so nothing is rendered
        while the model is still generating.

        Returns:
            Every :class:`StreamContentChunk` fragment joined in arrival
            order. Citations and token usage are discarded along with the
            other chunk types — iterate the stream itself for those.

        Examples:
            .. code-block:: python

                stream = pc.assistants.chat(
                    assistant_name="acme-support-bot",
                    messages=[{"content": "Explain vector databases in one sentence."}],
                    stream=True,
                )
                print(stream.collect())

        .. seealso::
           :meth:`text` — the fragments one at a time, for rendering the
           answer as it arrives.
        """
        return "".join(
            chunk.delta.content for chunk in self._stream if isinstance(chunk, StreamContentChunk)
        )


class AsyncChatStream:
    """A Pinecone-native chat stream from ``AsyncPinecone``.

    Iterating it yields the same :data:`ChatStreamChunk` variants as
    :class:`ChatStream`, so the branching contract is identical; only the
    ``async for``/``await`` mechanics differ. The stream is single-pass:
    iterating, :meth:`text`, and :meth:`collect` all consume the same
    underlying async iterator, so pick one.

    Examples:
        .. code-block:: python

            import asyncio

            from pinecone import AsyncPinecone

            async def main() -> None:
                async with AsyncPinecone(api_key="your-api-key") as pc:
                    stream = await pc.assistants.chat(
                        assistant_name="acme-support-bot",
                        messages=[{"content": "What can you help me with?"}],
                        stream=True,
                    )
                    async for fragment in stream.text():
                        print(fragment, end="", flush=True)

            asyncio.run(main())

    .. seealso::
       - :data:`ChatStreamChunk` — the four chunk types, and the loop to write
         when you need citations rather than text alone.
       - :class:`AsyncChatCompletionStream` — the same request in the
         OpenAI-compatible shape.
    """

    def __init__(self, stream: AsyncIterator[ChatStreamChunk]) -> None:
        self._stream = stream

    @safe_display
    def __repr__(self) -> str:
        return (
            "AsyncChatStream(single-pass async, Pinecone-native chat stream"
            " — iterate with `async for chunk in stream` or `await stream.collect()`)"
        )

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("AsyncChatStream")
        builder.row("Type", "Pinecone-native chat stream")
        builder.row("Iteration", "single-pass async")
        builder.row(
            "Usage hint",
            "Iterate with `async for chunk in stream`, or call `.text()` for"
            " text-only fragments, or `await .collect()` for the full message",
        )
        return builder.build()

    def __aiter__(self) -> AsyncIterator[ChatStreamChunk]:
        return self._stream

    async def text(self) -> AsyncIterator[str]:
        """Yield only the response text, dropping every non-content chunk.

        Returns:
            Async iterator over ``delta.content`` of each
            :class:`StreamContentChunk`, in arrival order. The start,
            citation and end chunks are discarded, so citations and token
            usage are not reachable through this method — iterate the stream
            itself for those.

        Examples:
            .. code-block:: python

                async def main() -> None:
                    stream = await pc.assistants.chat(
                        assistant_name="acme-support-bot",
                        messages=[{"content": "Explain vector databases in one sentence."}],
                        stream=True,
                    )
                    async for fragment in stream.text():
                        print(fragment, end="", flush=True)

        .. seealso::
           :meth:`collect` — the same fragments already joined into one
           string, for when you do not need to render as they arrive.
        """
        async for chunk in self._stream:
            if isinstance(chunk, StreamContentChunk):
                yield chunk.delta.content

    async def collect(self) -> str:
        """Drain the whole stream and return the answer as one string.

        Awaits until the server closes the stream, so nothing is rendered
        while the model is still generating.

        Returns:
            Every :class:`StreamContentChunk` fragment joined in arrival
            order. Citations and token usage are discarded along with the
            other chunk types — iterate the stream itself for those.

        Examples:
            .. code-block:: python

                async def main() -> None:
                    stream = await pc.assistants.chat(
                        assistant_name="acme-support-bot",
                        messages=[{"content": "Explain vector databases in one sentence."}],
                        stream=True,
                    )
                    print(await stream.collect())

        .. seealso::
           :meth:`text` — the fragments one at a time, for rendering the
           answer as it arrives.
        """
        return "".join(
            [
                chunk.delta.content
                async for chunk in self._stream
                if isinstance(chunk, StreamContentChunk)
            ]
        )


class ChatCompletionStream:
    """An OpenAI-compatible stream, from ``chat_completions(..., stream=True)``.

    Iterating it yields :class:`ChatCompletionStreamChunk`, whose text sits at
    ``chunk.choices[0].delta.content`` and can be ``None`` or ``""`` on the
    role-only first chunk and the finish chunk; :meth:`text` and
    :meth:`collect` filter those out for you. Reach for this shape when you
    are pointing existing OpenAI client code at Pinecone. The stream is
    single-pass: iterating, :meth:`text`, and :meth:`collect` all consume the
    same underlying iterator, so pick one.

    Examples:
        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone(api_key="your-api-key")
            stream = pc.assistants.chat_completions(
                assistant_name="acme-support-bot",
                messages=[{"content": "What can you help me with?"}],
                stream=True,
            )
            for fragment in stream.text():
                print(fragment, end="", flush=True)

    .. seealso::
       - :class:`ChatStream` — the Pinecone-native shape, which delivers
         citations as objects you can render instead of weaving them into the
         text. Prefer it unless you need OpenAI compatibility.
       - :class:`AsyncChatCompletionStream` — the ``AsyncPinecone``
         equivalent.
    """

    def __init__(self, stream: Iterator[ChatCompletionStreamChunk]) -> None:
        self._stream = stream

    @safe_display
    def __repr__(self) -> str:
        return (
            "ChatCompletionStream(single-pass, OpenAI-compatible"
            " — iterate with `for chunk in stream` or `stream.text()`)"
        )

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("ChatCompletionStream")
        builder.row("Type", "OpenAI-compatible")
        builder.row("Iteration", "single-pass")
        builder.row(
            "Usage hint",
            "Iterate with `for chunk in stream`, or call `.text()` for"
            " text-only fragments, or `.collect()` for the full message",
        )
        return builder.build()

    def __iter__(self) -> Iterator[ChatCompletionStreamChunk]:
        return self._stream

    def text(self) -> Iterator[str]:
        """Yield the response text, skipping role-only and finish chunks.

        Returns:
            Iterator over ``choices[0].delta.content`` of each chunk that has
            one, in arrival order. Chunks whose content is ``None`` or ``""``,
            and chunks with an empty ``choices`` list, are skipped, as is
            ``usage`` on the final chunk — iterate the stream itself for that.

        Examples:
            .. code-block:: python

                stream = pc.assistants.chat_completions(
                    assistant_name="acme-support-bot",
                    messages=[{"content": "Explain vector databases in one sentence."}],
                    stream=True,
                )
                for fragment in stream.text():
                    print(fragment, end="", flush=True)

        .. seealso::
           :meth:`collect` — the same fragments already joined into one
           string, for when you do not need to render as they arrive.
        """
        for chunk in self._stream:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content is not None and content != "":
                    yield content

    def collect(self) -> str:
        """Drain the whole stream and return the answer as one string.

        Blocks until the server closes the stream, so nothing is rendered
        while the model is still generating.

        Returns:
            Every non-empty ``choices[0].delta.content`` fragment joined in
            arrival order. The final chunk's ``usage`` is discarded — iterate
            the stream itself for that.

        Examples:
            .. code-block:: python

                stream = pc.assistants.chat_completions(
                    assistant_name="acme-support-bot",
                    messages=[{"content": "Explain vector databases in one sentence."}],
                    stream=True,
                )
                print(stream.collect())

        .. seealso::
           :meth:`text` — the fragments one at a time, for rendering the
           answer as it arrives.
        """
        parts: list[str] = []
        for chunk in self._stream:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content is not None and content != "":
                    parts.append(content)
        return "".join(parts)


class AsyncChatCompletionStream:
    """An OpenAI-compatible stream from ``AsyncPinecone``.

    Iterating it yields the same :class:`ChatCompletionStreamChunk` objects as
    :class:`ChatCompletionStream`, with text at
    ``chunk.choices[0].delta.content``; only the ``async for``/``await``
    mechanics differ. The stream is single-pass: iterating, :meth:`text`, and
    :meth:`collect` all consume the same underlying async iterator, so pick
    one.

    Examples:
        .. code-block:: python

            import asyncio

            from pinecone import AsyncPinecone

            async def main() -> None:
                async with AsyncPinecone(api_key="your-api-key") as pc:
                    stream = await pc.assistants.chat_completions(
                        assistant_name="acme-support-bot",
                        messages=[{"content": "What can you help me with?"}],
                        stream=True,
                    )
                    async for fragment in stream.text():
                        print(fragment, end="", flush=True)

            asyncio.run(main())

    .. seealso::
       :class:`AsyncChatStream` — the Pinecone-native shape, which delivers
       citations as objects you can render instead of weaving them into the
       text. Prefer it unless you need OpenAI compatibility.
    """

    def __init__(self, stream: AsyncIterator[ChatCompletionStreamChunk]) -> None:
        self._stream = stream

    @safe_display
    def __repr__(self) -> str:
        return (
            "AsyncChatCompletionStream(single-pass async, OpenAI-compatible"
            " — iterate with `async for chunk in stream`)"
        )

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("AsyncChatCompletionStream")
        builder.row("Type", "OpenAI-compatible")
        builder.row("Iteration", "single-pass async")
        builder.row(
            "Usage hint",
            "Iterate with `async for chunk in stream`, or call `.text()` for"
            " text-only fragments, or `await .collect()` for the full message",
        )
        return builder.build()

    def __aiter__(self) -> AsyncIterator[ChatCompletionStreamChunk]:
        return self._stream

    async def text(self) -> AsyncIterator[str]:
        """Yield the response text, skipping role-only and finish chunks.

        Returns:
            Async iterator over ``choices[0].delta.content`` of each chunk
            that has one, in arrival order. Chunks whose content is ``None``
            or ``""``, and chunks with an empty ``choices`` list, are skipped,
            as is ``usage`` on the final chunk — iterate the stream itself for
            that.

        Examples:
            .. code-block:: python

                async def main() -> None:
                    stream = await pc.assistants.chat_completions(
                        assistant_name="acme-support-bot",
                        messages=[{"content": "Explain vector databases in one sentence."}],
                        stream=True,
                    )
                    async for fragment in stream.text():
                        print(fragment, end="", flush=True)

        .. seealso::
           :meth:`collect` — the same fragments already joined into one
           string, for when you do not need to render as they arrive.
        """
        async for chunk in self._stream:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content is not None and content != "":
                    yield content

    async def collect(self) -> str:
        """Drain the whole stream and return the answer as one string.

        Awaits until the server closes the stream, so nothing is rendered
        while the model is still generating.

        Returns:
            Every non-empty ``choices[0].delta.content`` fragment joined in
            arrival order. The final chunk's ``usage`` is discarded — iterate
            the stream itself for that.

        Examples:
            .. code-block:: python

                async def main() -> None:
                    stream = await pc.assistants.chat_completions(
                        assistant_name="acme-support-bot",
                        messages=[{"content": "Explain vector databases in one sentence."}],
                        stream=True,
                    )
                    print(await stream.collect())

        .. seealso::
           :meth:`text` — the fragments one at a time, for rendering the
           answer as it arrives.
        """
        parts: list[str] = []
        async for chunk in self._stream:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content is not None and content != "":
                    parts.append(content)
        return "".join(parts)


class ChatCompletionStreamDelta(StructDictMixin, Struct, kw_only=True):
    """The delta payload within a chat completion streaming chunk.

    Reached as ``chunk.choices[0].delta``. Both fields are optional and both
    are commonly absent: the first chunk of a response typically carries
    ``role`` and no ``content``, and the finish chunk carries neither.

    Attributes:
        role: The role of the message author, or ``None`` when the chunk does
            not restate it.
        content: The text fragment, or ``None`` when this chunk carries no
            text. Concatenate the non-empty fragments in arrival order to
            rebuild the answer.
    """

    role: str | None = None
    content: str | None = None

    @safe_display
    def __repr__(self) -> str:
        parts: list[str] = []
        if self.role is not None:
            parts.append(f"role={self.role!r}")
        if self.content is not None:
            parts.append(f"content={truncate_text(self.content, 80)!r}")
        inner = ", ".join(parts) if parts else "<empty>"
        return f"ChatCompletionStreamDelta({inner})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("ChatCompletionStreamDelta(...)")
            return
        parts: list[str] = []
        if self.role is not None:
            parts.append(f"role={self.role!r}")
        if self.content is not None:
            parts.append(f"content={truncate_text(self.content, 200)!r}")
        if not parts:
            p.text("ChatCompletionStreamDelta(<empty>)")
            return
        with p.group(2, "ChatCompletionStreamDelta(", ")"):
            for part in parts:
                p.breakable()
                p.text(f"{part},")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("ChatCompletionStreamDelta")
        if self.role is not None:
            builder.row("Role", self.role)
        if self.content is not None:
            builder.row("Content", truncate_text(self.content, 500))
        return builder.build()


class ChatCompletionStreamChoice(StructDictMixin, Struct, kw_only=True):
    """A single choice in a chat completion streaming chunk.

    Reached as ``chunk.choices[0]``, and the wrapper for the fragment of text
    this chunk carries.

    Attributes:
        index: Position of this choice in the chunk's ``choices`` list.
        delta: The :class:`ChatCompletionStreamDelta` for this choice; the
            text is at ``delta.content``.
        finish_reason: ``None`` while generation is ongoing. Once set, why
            generation stopped: ``"stop"`` (the model finished), ``"length"``
            (the token limit was reached), ``"content_filter"`` (content
            filtering rules blocked the output), or ``"tool_calls"`` (a tool
            call was triggered). The literal string ``"null"`` also reaches
            callers and is a different value from Python ``None``, so treat
            this as an open set of strings rather than switching exhaustively
            on the four above.
    """

    index: int
    delta: ChatCompletionStreamDelta
    finish_reason: str | None = None

    @safe_display
    def __repr__(self) -> str:
        parts: list[str] = [f"index={self.index!r}", f"delta={self.delta!r}"]
        if self.finish_reason is not None:
            parts.append(f"finish_reason={self.finish_reason!r}")
        return f"ChatCompletionStreamChoice({', '.join(parts)})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("ChatCompletionStreamChoice(...)")
            return
        with p.group(2, "ChatCompletionStreamChoice(", ")"):
            p.breakable()
            p.text(f"index={self.index!r},")
            p.breakable()
            p.text(f"delta={self.delta!r},")
            if self.finish_reason is not None:
                p.breakable()
                p.text(f"finish_reason={self.finish_reason!r},")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("ChatCompletionStreamChoice")
        builder.row("Index", self.index)
        if self.delta.role is not None:
            builder.row("Role", self.delta.role)
        if self.delta.content is not None:
            builder.row("Content", truncate_text(self.delta.content, 500))
        if self.finish_reason is not None:
            builder.row("Finish reason", self.finish_reason)
        return builder.build()


class ChatCompletionStreamChunk(StructDictMixin, Struct, kw_only=True):
    """One chunk of an OpenAI-compatible completion stream.

    Unlike the Pinecone-native stream there is a single chunk type, so there
    is nothing to branch on: the text is at ``chunk.choices[0].delta.content``
    and is ``None`` or ``""`` on the role-only first chunk and the finish
    chunk. ``choices`` can also arrive empty, so guard on it before indexing.

    Attributes:
        id: Identifier of the completion, the same on every chunk of the
            stream.
        choices: The streaming choices, normally one. Read the text from
            ``choices[0].delta.content``.
        model: Name of the model that generated the answer, or ``None`` if the
            server did not report it on this chunk.
        object: The object type (typically ``"chat.completion.chunk"``), or
            ``None``.
        created: Unix timestamp when the chunk was created, or ``None``.
        system_fingerprint: Opaque fingerprint of the serving configuration,
            or ``None``. Useful only for comparing two responses.
        usage: :class:`~pinecone.models.assistant.chat.ChatUsage` token counts,
            populated on the final chunk and ``None`` on every earlier one.

    .. seealso::
       :data:`ChatStreamChunk` — the Pinecone-native chunk types, which
       deliver citations as objects you can render.
    """

    id: str
    choices: list[ChatCompletionStreamChoice]
    model: str | None = None
    object: str | None = None
    created: int | None = None
    system_fingerprint: str | None = None
    usage: ChatUsage | None = None

    @safe_display
    def __repr__(self) -> str:
        parts: list[str] = [f"id={self.id!r}"]
        if self.model is not None:
            parts.append(f"model={self.model!r}")
        parts.append(f"choices={len(self.choices)}")
        if self.usage is not None:
            parts.append(f"usage={self.usage!r}")
        return f"ChatCompletionStreamChunk({', '.join(parts)})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("ChatCompletionStreamChunk(...)")
            return
        first_content: str | None = None
        if self.choices and self.choices[0].delta.content is not None:
            first_content = truncate_text(self.choices[0].delta.content, max_chars=200)
        with p.group(2, "ChatCompletionStreamChunk(", ")"):
            p.breakable()
            p.text(f"id={self.id!r},")
            if self.model is not None:
                p.breakable()
                p.text(f"model={self.model!r},")
            if self.object is not None:
                p.breakable()
                p.text(f"object={self.object!r},")
            if self.created is not None:
                p.breakable()
                p.text(f"created={self.created!r},")
            if self.system_fingerprint is not None:
                p.breakable()
                p.text(f"system_fingerprint={self.system_fingerprint!r},")
            p.breakable()
            p.text(f"choices={len(self.choices)},")
            if first_content is not None:
                p.breakable()
                p.text(f"first_choice_content={first_content!r},")
            if self.usage is not None:
                p.breakable()
                p.text(f"usage={self.usage!r},")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("ChatCompletionStreamChunk")
        builder.row("Id", self.id)
        if self.model is not None:
            builder.row("Model", self.model)
        if self.object is not None:
            builder.row("Object", self.object)
        if self.created is not None:
            builder.row("Created", self.created)
        if self.system_fingerprint is not None:
            builder.row("System fingerprint", self.system_fingerprint)
        builder.row("Choices", len(self.choices))
        if self.choices:
            first = self.choices[0]
            section_rows: list[tuple[str, Any]] = [("Index", first.index)]
            if first.delta.role is not None:
                section_rows.append(("Role", first.delta.role))
            if first.delta.content is not None:
                section_rows.append(("Content", truncate_text(first.delta.content, 500)))
            if first.finish_reason is not None:
                section_rows.append(("Finish reason", first.finish_reason))
            builder.section("First choice", section_rows)
        if self.usage is not None:
            builder.row("Usage", repr(self.usage))
        return builder.build()
