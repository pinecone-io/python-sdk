"""Chat response models for the Assistant API."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone.models._display import HtmlBuilder, abbreviate_list, safe_display, truncate_text
from pinecone.models.assistant._mixin import StructDictMixin
from pinecone.models.assistant.file_model import AssistantFileModel


class ChatUsage(StructDictMixin, Struct, kw_only=True):
    """Token counts the API reported for one assistant request.

    Reached as ``usage`` on :class:`ChatResponse`,
    :class:`ChatCompletionResponse`,
    :class:`~pinecone.models.assistant.context.ContextResponse`,
    :class:`~pinecone.models.assistant.evaluation.AlignmentResult`, and on the
    closing chunk of a stream.

    Attributes:
        prompt_tokens: Tokens counted in the prompt.
        completion_tokens: Tokens counted in the generated answer.
        total_tokens: Total the API reported for the request.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChatUsage:
        """Build a :class:`ChatUsage` from a plain dict.

        Args:
            d: Mapping with any of ``prompt_tokens``, ``completion_tokens``,
                and ``total_tokens``. A missing key becomes ``0`` rather than
                raising, so a partial payload yields a partial count.

        Returns:
            :class:`ChatUsage` with the three counts filled in.
        """
        return cls(
            prompt_tokens=d.get("prompt_tokens", 0),
            completion_tokens=d.get("completion_tokens", 0),
            total_tokens=d.get("total_tokens", 0),
        )

    @safe_display
    def __repr__(self) -> str:
        return (
            f"ChatUsage(prompt={self.prompt_tokens},"
            f" completion={self.completion_tokens},"
            f" total={self.total_tokens})"
        )

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("ChatUsage(...)")
            return
        with p.group(2, "ChatUsage(", ")"):
            p.breakable()
            p.text(f"prompt={self.prompt_tokens},")
            p.breakable()
            p.text(f"completion={self.completion_tokens},")
            p.breakable()
            p.text(f"total={self.total_tokens},")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("ChatUsage")
        builder.row("Prompt tokens:", self.prompt_tokens)
        builder.row("Completion tokens:", self.completion_tokens)
        builder.row("Total tokens:", self.total_tokens)
        return builder.build()


class ChatHighlight(StructDictMixin, Struct, kw_only=True):
    """The passage of a source document that a citation drew on.

    Reached as ``reference.highlight``, and present only when the chat request
    set ``include_highlights=True``. Render it to show the reader the source
    text behind a citation without fetching the file.

    Attributes:
        type: The kind of highlighted content (e.g. ``"text"``).
        content: The highlighted passage, taken from the source document.
    """

    type: str
    content: str

    @safe_display
    def __repr__(self) -> str:
        truncated = truncate_text(self.content, max_chars=80)
        return f"ChatHighlight(type={self.type!r}, content={truncated!r})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        truncated = truncate_text(self.content, max_chars=200)
        p.text(f"ChatHighlight(type={self.type!r}, content={truncated!r})")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("ChatHighlight")
        builder.row("Type", self.type)
        builder.row("Content", truncate_text(self.content, max_chars=500))
        return builder.build()


class ChatReference(StructDictMixin, Struct, kw_only=True):
    """One source document behind a citation.

    Reached as an entry of ``citation.references``. These three fields are
    what a RAG caller renders as a source link.

    Attributes:
        file: The source file, as an
            :class:`~pinecone.models.assistant.file_model.AssistantFileModel`
            — ``file.name`` for a label, ``file.id`` to fetch it again, and
            ``file.metadata`` for whatever you attached at upload.
        pages: Page numbers within the source file, for paginated documents
            such as PDFs. ``None`` for sources that have no pages.
        highlight: The :class:`ChatHighlight` passage this reference drew on,
            or ``None`` unless the chat request set
            ``include_highlights=True``.
    """

    file: AssistantFileModel
    pages: list[int] | None = None
    highlight: ChatHighlight | None = None

    @safe_display
    def __repr__(self) -> str:
        pages_str = abbreviate_list(self.pages) if self.pages is not None else "None"
        highlight_str = "yes" if self.highlight is not None else "no"
        return (
            f"ChatReference(file={self.file.name!r},"
            f" pages={pages_str}, highlight={highlight_str!r})"
        )

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("ChatReference(...)")
            return
        pages_str = abbreviate_list(self.pages) if self.pages is not None else "None"
        highlight_str = "yes" if self.highlight is not None else "no"
        with p.group(2, "ChatReference(", ")"):
            p.breakable()
            p.text(f"file={self.file.name!r},")
            p.breakable()
            p.text(f"pages={pages_str},")
            p.breakable()
            p.text(f"highlight={highlight_str!r},")

    @safe_display
    def _repr_html_(self) -> str:
        pages_val = abbreviate_list(self.pages) if self.pages is not None else "—"
        highlight_val = type(self.highlight).__name__ if self.highlight is not None else "—"
        builder = HtmlBuilder("ChatReference")
        builder.row("File", self.file.name)
        builder.row("Pages", pages_val)
        builder.row("Highlight", highlight_val)
        if self.highlight is not None:
            builder.section("Highlight", [("Content", truncate_text(self.highlight.content, 500))])
        return builder.build()


class ChatCitation(StructDictMixin, Struct, kw_only=True):
    """A point in the answer, tied to the documents that support it.

    Reached as an entry of ``response.citations`` on a
    :class:`ChatResponse`, or as ``chunk.citation`` on a
    :class:`~pinecone.models.assistant.streaming.StreamCitationChunk`.

    Attributes:
        position: Character position in ``response.message.content`` that this
            citation annotates. Insert a footnote marker there to render the
            answer with inline sources.
        references: The :class:`ChatReference` entries supporting the answer
            at that position. Can hold more than one document.
    """

    position: int
    references: list[ChatReference]

    @safe_display
    def __repr__(self) -> str:
        return f"ChatCitation(position={self.position}, references={len(self.references)})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("ChatCitation(...)")
            return
        names = [ref.file.name for ref in self.references[:3]]
        extra = len(self.references) - 3
        names_str = ", ".join(repr(n) for n in names)
        if extra > 0:
            names_str += f", ...{extra} more"
        with p.group(2, "ChatCitation(", ")"):
            p.breakable()
            p.text(f"position={self.position},")
            p.breakable()
            p.text(f"references=[{names_str}],")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("ChatCitation")
        builder.row("Position", self.position)
        builder.row("Reference count", len(self.references))
        if self.references:
            ref_rows: list[tuple[str, Any]] = []
            for ref in self.references[:5]:
                pages_val = abbreviate_list(ref.pages) if ref.pages is not None else "—"
                ref_rows.append((ref.file.name, pages_val))
            builder.section("References", ref_rows)
        else:
            builder.section("References", [("—", "")])
        return builder.build()


class ChatMessage(StructDictMixin, Struct, kw_only=True):
    """The assistant's reply inside a :class:`ChatResponse`.

    Reached as ``response.message``. To *send* a message, build a
    :class:`~pinecone.models.assistant.message.Message` instead — this class
    only comes back from the API.

    Attributes:
        role: The role of the message author (e.g. ``"user"``,
            ``"assistant"``).
        content: The answer text. Citation positions index into this string.
    """

    role: str
    content: str

    @safe_display
    def __repr__(self) -> str:
        truncated = truncate_text(self.content, max_chars=80)
        return f"ChatMessage(role={self.role!r}, content={truncated!r})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        truncated = truncate_text(self.content, max_chars=200)
        p.text(f"ChatMessage(role={self.role!r}, content={truncated!r})")

    @safe_display
    def _repr_html_(self) -> str:
        return (
            HtmlBuilder("ChatMessage")
            .row("Role", self.role)
            .row("Content", truncate_text(self.content, max_chars=500))
            .build()
        )


class ChatResponse(StructDictMixin, Struct, kw_only=True):
    """The generated answer to a chat request, with its citations.

    Returned by :meth:`~pinecone.client.assistants.Assistants.chat` when
    ``stream`` is left ``False``. The answer text is at
    ``response.message.content``, and the sources come back as structured
    objects rather than markers woven into that text — which is what a caller
    needs to render source links. The full path is
    ``response.citations[i].references[j].file.name``, with
    ``citations[i].position`` saying where in the answer each citation belongs.

    Attributes:
        id: Identifier of this chat response.
        model: Name of the model that generated the answer, which need not be
            the name you requested.
        usage: :class:`ChatUsage` token counts for the request.
        message: The assistant's reply as a :class:`ChatMessage`; the text is
            at ``message.content``.
        finish_reason: Why generation stopped: ``"stop"`` (the model
            finished), ``"length"`` (the token limit was reached),
            ``"content_filter"`` (content filtering rules blocked the output),
            or ``"tool_calls"`` (a tool call was triggered). The literal
            string ``"null"`` also reaches callers, so treat this as an open
            set of strings rather than switching exhaustively on the four
            above.
        citations: The :class:`ChatCitation` entries tying positions in
            ``message.content`` to source documents. Empty when the answer
            drew on no file.
        context_snippet_count: Number of retrieved context snippets that were
            provided to the model, or ``None`` if the server did not report it.
            ``0`` means no relevant context was found for the query, which
            explains an answer with no citations.
        content_filter_results: Safety classifications reported by the LLM
            provider, or ``None`` when the provider returned none. Read
            ``spec`` for the provider's name and ``results`` for a payload
            whose shape that provider defines.

    Examples:
        The answer is one string, and each citation names a position in it
        together with the documents backing the claim at that position:

        >>> response = pc.assistants.chat(
        ...     assistant_name="acme-support-bot",
        ...     messages=[{"content": "Which regions support BYOC?"}],
        ... )
        >>> response.message.content
        'BYOC is available in aws us-east-1.'
        >>> citation = response.citations[0]
        >>> citation.position
        34
        >>> citation.references[0].file.name
        'q3-revenue-review.pdf'
        >>> citation.references[0].pages
        [3]
        >>> citation.references[0].highlight is None
        True

        That last line is the default: pass ``include_highlights=True`` to
        :meth:`~pinecone.client.assistants.Assistants.chat` to get the source
        passage as well as the file name.

    .. seealso::
       - :class:`~pinecone.models.assistant.context.ContextResponse` — the
         retrieved snippets with no answer generated over them, from
         :meth:`~pinecone.client.assistants.Assistants.context`. Use that when
         you want to run your own model over Pinecone's retrieval.
       - :class:`ChatCompletionResponse` — the same answer in the
         OpenAI-compatible shape, without structured citations.
       - :class:`~pinecone.models.assistant.streaming.ChatStream` — the same
         answer delivered as chunks, from ``chat(..., stream=True)``.
    """

    id: str
    model: str
    usage: ChatUsage
    message: ChatMessage
    finish_reason: str
    citations: list[ChatCitation]
    context_snippet_count: int | None = None
    content_filter_results: dict[str, Any] | None = None

    @safe_display
    def __repr__(self) -> str:
        snippet_part = (
            f" context_snippet_count={self.context_snippet_count},"
            if self.context_snippet_count is not None
            else ""
        )
        return (
            f"ChatResponse(id={self.id!r}, model={self.model!r},"
            f" finish_reason={self.finish_reason!r},"
            f" citations={len(self.citations)},{snippet_part} usage={self.usage!r})"
        )

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("ChatResponse(...)")
            return
        with p.group(2, "ChatResponse(", ")"):
            p.breakable()
            p.text(f"id={self.id!r},")
            p.breakable()
            p.text(f"model={self.model!r},")
            p.breakable()
            p.text(f"finish_reason={self.finish_reason!r},")
            p.breakable()
            p.text(f"citations={len(self.citations)},")
            if self.context_snippet_count is not None:
                p.breakable()
                p.text(f"context_snippet_count={self.context_snippet_count},")
            if self.content_filter_results is not None:
                p.breakable()
                filter_text = truncate_text(str(self.content_filter_results), 200)
                p.text(f"content_filter_results={filter_text},")
            p.breakable()
            p.text(f"usage={self.usage!r},")
            p.breakable()
            p.text(f"message={self.message!r},")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("ChatResponse")
        builder.row("Id", self.id)
        builder.row("Model", self.model)
        builder.row("Finish reason", self.finish_reason)
        builder.row("Citations", len(self.citations))
        if self.context_snippet_count is not None:
            builder.row("Context snippets", self.context_snippet_count)
        if self.content_filter_results is not None:
            builder.row(
                "Content filter results", truncate_text(str(self.content_filter_results), 500)
            )
        builder.row("Usage", repr(self.usage))
        builder.section(
            "Message",
            [
                ("Role", self.message.role),
                ("Content", truncate_text(self.message.content, 500)),
            ],
        )
        return builder.build()


class ChatCompletionMessage(StructDictMixin, Struct, kw_only=True):
    """The answer message inside a chat completion choice.

    Reached as ``response.choices[0].message``. Both fields are optional, so
    guard on ``content`` before using it.

    Attributes:
        role: The role of the message author, or ``None`` when the API did not
            report one.
        content: The answer text, or ``None`` when the choice carries none.
    """

    role: str | None = None
    content: str | None = None

    @safe_display
    def __repr__(self) -> str:
        truncated = truncate_text(self.content or "", max_chars=80) if self.content else None
        return f"ChatCompletionMessage(role={self.role!r}, content={truncated!r})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        truncated = truncate_text(self.content or "", max_chars=200) if self.content else None
        p.text(f"ChatCompletionMessage(role={self.role!r}, content={truncated!r})")

    @safe_display
    def _repr_html_(self) -> str:
        return (
            HtmlBuilder("ChatCompletionMessage")
            .row("Role", self.role if self.role is not None else "—")
            .row("Content", truncate_text(self.content, max_chars=500) if self.content else "—")
            .build()
        )


class ChatCompletionChoice(StructDictMixin, Struct, kw_only=True):
    """A single answer in a chat completion response.

    Reached as ``response.choices[0]``.

    Attributes:
        index: Position of this choice in the response's ``choices`` list.
        message: The :class:`ChatCompletionMessage` for this choice; the text
            is at ``message.content``.
        finish_reason: Why generation stopped: ``"stop"`` (the model
            finished), ``"length"`` (the token limit was reached),
            ``"content_filter"`` (content filtering rules blocked the output),
            or ``"tool_calls"`` (a tool call was triggered). The literal
            string ``"null"`` also reaches callers, so treat this as an open
            set of strings rather than switching exhaustively on the four
            above.
    """

    index: int
    message: ChatCompletionMessage
    finish_reason: str

    @safe_display
    def __repr__(self) -> str:
        return (
            f"ChatCompletionChoice(index={self.index!r},"
            f" finish_reason={self.finish_reason!r},"
            f" message={self.message!r})"
        )

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("ChatCompletionChoice(...)")
            return
        with p.group(2, "ChatCompletionChoice(", ")"):
            p.breakable()
            p.text(f"index={self.index!r},")
            p.breakable()
            p.text(f"finish_reason={self.finish_reason!r},")
            p.breakable()
            p.text(f"message={self.message!r},")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("ChatCompletionChoice")
        builder.row("Index", self.index)
        builder.row("Finish reason", self.finish_reason)
        builder.row("Role", self.message.role if self.message.role is not None else "—")
        builder.row(
            "Content",
            truncate_text(self.message.content, max_chars=500)
            if self.message.content is not None
            else "—",
        )
        return builder.build()


class ChatCompletionResponse(StructDictMixin, Struct, kw_only=True):
    """The generated answer to a chat request, in OpenAI-compatible shape.

    Returned by
    :meth:`~pinecone.client.assistants.Assistants.chat_completions` when
    ``stream`` is left ``False``. The answer text is nested at
    ``response.choices[0].message.content``. There is no structured citation
    list here; citations arrive woven into the answer text, so prefer
    :class:`ChatResponse` unless you are pointing existing OpenAI client code
    at Pinecone.

    Attributes:
        id: Identifier of this completion.
        model: Name of the model that generated the answer, which need not be
            the name you requested.
        usage: :class:`ChatUsage` token counts for the request.
        choices: The :class:`ChatCompletionChoice` answers, normally one. Read
            the text from ``choices[0].message.content``.

    Examples:
        The text is two levels down, under ``choices``, and there is no
        ``citations`` attribute to read — that absence is the whole difference
        from :class:`ChatResponse`:

        >>> response = pc.assistants.chat_completions(
        ...     assistant_name="acme-support-bot",
        ...     messages=[{"content": "Which regions support BYOC?"}],
        ... )
        >>> response.choices[0].message.content
        'BYOC is available in aws us-east-1.'
        >>> response.choices[0].finish_reason
        'stop'
        >>> hasattr(response, "citations")
        False

    .. seealso::
       - :class:`ChatResponse` — the Pinecone-native shape, whose
         ``citations`` are objects you can render as source links.
       - :class:`~pinecone.models.assistant.streaming.ChatCompletionStream` —
         the same answer delivered as chunks, from
         ``chat_completions(..., stream=True)``.
    """

    id: str
    model: str
    usage: ChatUsage
    choices: list[ChatCompletionChoice]

    @safe_display
    def __repr__(self) -> str:
        return (
            f"ChatCompletionResponse(id={self.id!r}, model={self.model!r},"
            f" choices={len(self.choices)}, usage={self.usage!r})"
        )

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("ChatCompletionResponse(...)")
            return
        first_content: str | None = None
        if self.choices and self.choices[0].message.content is not None:
            first_content = truncate_text(self.choices[0].message.content, max_chars=200)
        with p.group(2, "ChatCompletionResponse(", ")"):
            p.breakable()
            p.text(f"id={self.id!r},")
            p.breakable()
            p.text(f"model={self.model!r},")
            p.breakable()
            p.text(f"usage={self.usage!r},")
            p.breakable()
            p.text(f"choices={len(self.choices)},")
            if first_content is not None:
                p.breakable()
                p.text(f"first_choice_content={first_content!r},")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("ChatCompletionResponse")
        builder.row("Id", self.id)
        builder.row("Model", self.model)
        builder.row("Choices", len(self.choices))
        builder.row("Usage", repr(self.usage))
        if self.choices:
            first = self.choices[0]
            builder.section(
                "First choice",
                [
                    ("Index", first.index),
                    ("Finish reason", first.finish_reason),
                    ("Role", first.message.role if first.message.role is not None else "—"),
                    (
                        "Content",
                        truncate_text(first.message.content, max_chars=500)
                        if first.message.content is not None
                        else "—",
                    ),
                ],
            )
        return builder.build()
