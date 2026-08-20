"""Chat response models for the Assistant API."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone.models._display import HtmlBuilder, abbreviate_list, safe_display, truncate_text
from pinecone.models.assistant._mixin import StructDictMixin
from pinecone.models.assistant.file_model import AssistantFileModel


class ChatUsage(StructDictMixin, Struct, kw_only=True):
    """Token usage information for a chat request.

    Attributes:
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total number of tokens used.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChatUsage:
        """Construct a ``ChatUsage`` from a plain dict representation.

        Missing token count fields default to 0.
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
    """A highlighted portion of a referenced document.

    Attributes:
        type: The type of highlight (e.g. ``"text"``).
        content: The highlighted text content.
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
    """A single reference within a citation.

    Attributes:
        file: The source file object with metadata.
        pages: Optional list of page numbers in the source file.
        highlight: Optional highlight from the referenced document,
            or ``None`` when highlights are not requested.
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
    """A citation linking a position in the response to source references.

    Attributes:
        position: The character position of the citation in the response content.
        references: The list of references supporting this citation.
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
    """A message in a chat conversation.

    Attributes:
        role: The role of the message author (e.g. ``"user"``, ``"assistant"``).
        content: The text content of the message.
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
    """Non-streaming response from the assistant chat endpoint.

    Attributes:
        id: Unique identifier for the chat response.
        model: The model used to generate the response.
        usage: Token usage statistics for the request.
        message: The assistant's response message.
        finish_reason: The reason the model stopped generating — one of
            ``"stop"`` (the model finished), ``"length"`` (the token limit was
            reached), ``"content_filter"`` (content filtering rules blocked the
            output), or ``"tool_calls"`` (a tool call was triggered).
        citations: List of citations linking response text to source documents.
        context_snippet_count: Number of retrieved context snippets that were
            provided to the model, or ``None`` if the server did not report it.
            ``0`` means no relevant context was found for the query.
        content_filter_results: Safety classifications reported by the LLM
            provider, or ``None`` when the provider returned none. The payload
            carries a ``spec`` key naming the provider (e.g. ``"openai"``,
            ``"gemini"``) and a ``results`` value whose structure is defined by
            that provider, so it is left as a plain dict.
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
    """A message in a chat completion response.

    Attributes:
        role: The role of the message author, or ``None`` if not provided.
        content: The text content of the message, or ``None`` if not provided.
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
    """A single choice in a chat completion response.

    Attributes:
        index: The index of this choice in the choices list.
        message: The message content for this choice.
        finish_reason: The reason the model stopped generating — one of
            ``"stop"`` (the model finished), ``"length"`` (the token limit was
            reached), ``"content_filter"`` (content filtering rules blocked the
            output), or ``"tool_calls"`` (a tool call was triggered).
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
    """Non-streaming response from the OpenAI-compatible chat completion endpoint.

    Attributes:
        id: Unique identifier for the chat completion.
        model: The model used to generate the response.
        usage: Token usage statistics for the request.
        choices: List of completion choices.
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
