"""Context response models for the Assistant API."""

from __future__ import annotations

from typing import Any, TypeAlias

from msgspec import Struct

from pinecone.models._display import HtmlBuilder, abbreviate_list, safe_display, truncate_text
from pinecone.models.assistant._mixin import StructDictMixin
from pinecone.models.assistant.chat import ChatUsage
from pinecone.models.assistant.file_model import AssistantFileModel


class ContextImageData(StructDictMixin, Struct, kw_only=True):
    """Base64-encoded image data within a context snippet.

    Attributes:
        type: The format of the image data (e.g. ``"base64"``).
        mime_type: The MIME type of the image (e.g. ``"image/jpeg"``).
        data: The base64-encoded image data string.
    """

    type: str
    mime_type: str
    data: str

    @safe_display
    def __repr__(self) -> str:
        return (
            f"ContextImageData(type={self.type!r}, mime_type={self.mime_type!r},"
            f" data=<{len(self.data):,} bytes>)"
        )

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("ContextImageData(...)")
            return
        preview = self.data[:32] + "..." if len(self.data) > 32 else self.data
        p.text(
            f"ContextImageData(\n"
            f"  type={self.type!r},\n"
            f"  mime_type={self.mime_type!r},\n"
            f"  data={preview!r}  # {len(self.data):,} bytes\n"
            f")"
        )

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("ContextImageData")
        builder.row("Type", self.type)
        builder.row("MIME type", self.mime_type)
        builder.row("Size", f"{len(self.data):,} chars")
        builder.row("Preview", truncate_text(self.data, 32))
        return builder.build()


class ContextImageBlock(
    Struct,
    kw_only=True,
    tag="image",
    tag_field="type",
    rename={"image_data": "image"},
):
    """An image block within a multimodal context snippet.

    Attributes:
        caption: A text caption describing the image.
        image_data: The image data, or ``None`` when binary content
            is excluded from the response.
    """

    caption: str
    image_data: ContextImageData | None = None

    @safe_display
    def __repr__(self) -> str:
        image_summary = "present" if self.image_data is not None else "absent"
        return (
            f"ContextImageBlock(caption={truncate_text(self.caption, 80)!r},"
            f" image_data=<{image_summary}>)"
        )

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("ContextImageBlock(...)")
            return
        image_summary = "present" if self.image_data is not None else "absent"
        p.text(
            f"ContextImageBlock(\n"
            f"  caption={truncate_text(self.caption, 80)!r},\n"
            f"  image_data=<{image_summary}>\n"
            f")"
        )

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("ContextImageBlock")
        builder.row("Caption", truncate_text(self.caption, 200))
        if self.image_data is not None:
            image_value = f"{self.image_data.mime_type} ({len(self.image_data.data):,} chars)"
        else:
            image_value = "—"
        builder.row("Image", image_value)
        return builder.build()


class ContextTextBlock(StructDictMixin, Struct, kw_only=True, tag="text", tag_field="type"):
    """A text block within a multimodal context snippet.

    Attributes:
        text: The text content of the block.
    """

    text: str

    @safe_display
    def __repr__(self) -> str:
        return f"ContextTextBlock(text={truncate_text(self.text, 80)!r})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("ContextTextBlock(...)")
            return
        p.text(f"ContextTextBlock(\n  text={truncate_text(self.text, 200)!r}\n)")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("ContextTextBlock")
        builder.row("Text", truncate_text(self.text, 500))
        return builder.build()


ContextContentBlock: TypeAlias = ContextTextBlock | ContextImageBlock
"""A content block within a multimodal snippet — either text or image."""


class FileReference(StructDictMixin, Struct, kw_only=True):
    """A reference to a source file.

    Attributes:
        file: The source file object returned by the API.
        pages: The list of page numbers relevant to the snippet, when
            the source is a paginated document (e.g. PDF). ``None`` for
            text, JSON, or Markdown sources.
        type: The kind of document referenced — ``"text"``, ``"json"``,
            ``"markdown"``, ``"pdf"``, or ``"doc_x"``. ``None`` only for
            payloads that omit it, which the API itself never does.
    """

    file: AssistantFileModel
    pages: list[int] | None = None
    type: str | None = None

    @safe_display
    def __repr__(self) -> str:
        pages_str = abbreviate_list(self.pages) if self.pages is not None else "None"
        type_part = f"type={self.type!r}, " if self.type is not None else ""
        return f"FileReference({type_part}file={self.file.name!r}, pages={pages_str})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("FileReference(...)")
            return
        pages_str = abbreviate_list(self.pages) if self.pages is not None else "None"
        with p.group(2, "FileReference(", ")"):
            if self.type is not None:
                p.breakable()
                p.text(f"type={self.type!r},")
            p.breakable()
            p.text(f"file={self.file.name!r},")
            p.breakable()
            p.text(f"pages={pages_str},")

    @safe_display
    def _repr_html_(self) -> str:
        pages_val = abbreviate_list(self.pages) if self.pages is not None else "—"
        builder = HtmlBuilder("FileReference")
        builder.row("Type", self.type if self.type is not None else "—")
        builder.row("File", self.file.name)
        builder.row("Pages", pages_val)
        return builder.build()


PageReference = FileReference
"""Alias kept for backwards compatibility. Use :class:`FileReference` instead."""

ContextReference: TypeAlias = FileReference
"""A reference to a source file."""


class TextSnippet(StructDictMixin, Struct, kw_only=True, tag="text", tag_field="type"):
    """A text context snippet from a source document.

    Attributes:
        content: The text content of the snippet.
        score: The relevance score of the snippet.
        reference: A reference to the source file.
    """

    content: str
    score: float
    reference: FileReference

    @safe_display
    def __repr__(self) -> str:
        return (
            f"TextSnippet(score={self.score!r},"
            f" reference={self.reference.file.name!r},"
            f" content={truncate_text(self.content, 80)!r})"
        )

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("TextSnippet(...)")
            return
        pages_str = (
            abbreviate_list(self.reference.pages) if self.reference.pages is not None else "None"
        )
        p.text(
            f"TextSnippet(\n"
            f"  score={self.score!r},\n"
            f"  reference={self.reference.file.name!r} pages={pages_str},\n"
            f"  content={truncate_text(self.content, 200)!r}\n"
            f")"
        )

    @safe_display
    def _repr_html_(self) -> str:
        pages_val = (
            abbreviate_list(self.reference.pages) if self.reference.pages is not None else "—"
        )
        builder = HtmlBuilder("TextSnippet")
        builder.row("Score", self.score)
        builder.row("Reference", self.reference.file.name)
        builder.row("Pages", pages_val)
        builder.row("Content", truncate_text(self.content, 500))
        return builder.build()


class MultimodalSnippet(StructDictMixin, Struct, kw_only=True, tag="multimodal", tag_field="type"):
    """A multimodal context snippet containing text and/or image blocks.

    Attributes:
        content: The list of content blocks (text and/or image).
        score: The relevance score of the snippet.
        reference: A reference to the source file.
    """

    content: list[ContextContentBlock]
    score: float
    reference: FileReference

    @safe_display
    def __repr__(self) -> str:
        n_text = sum(1 for b in self.content if isinstance(b, ContextTextBlock))
        n_image = len(self.content) - n_text
        return (
            f"MultimodalSnippet(score={self.score!r},"
            f" reference={self.reference.file.name!r},"
            f" blocks=<text:{n_text},image:{n_image}>)"
        )

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("MultimodalSnippet(...)")
            return
        n_text = sum(1 for b in self.content if isinstance(b, ContextTextBlock))
        n_image = len(self.content) - n_text
        p.text(
            f"MultimodalSnippet(\n"
            f"  score={self.score!r},\n"
            f"  reference={self.reference.file.name!r},\n"
            f"  blocks=<text:{n_text},image:{n_image}>\n"
            f")"
        )

    @safe_display
    def _repr_html_(self) -> str:
        n_text = sum(1 for b in self.content if isinstance(b, ContextTextBlock))
        n_image = len(self.content) - n_text
        pages_val = (
            abbreviate_list(self.reference.pages) if self.reference.pages is not None else "—"
        )
        builder = HtmlBuilder("MultimodalSnippet")
        builder.row("Score", self.score)
        builder.row("Reference", self.reference.file.name)
        builder.row("Pages", pages_val)
        builder.row("Blocks", f"{n_text} text, {n_image} image")
        section_rows: list[tuple[str, Any]] = []
        for block in self.content[:5]:
            if isinstance(block, ContextTextBlock):
                section_rows.append(("text", truncate_text(block.text, 60)))
            else:
                section_rows.append(("image", truncate_text(block.caption, 60)))
        if len(self.content) > 5:
            section_rows.append(("...", f"{len(self.content) - 5} more"))
        builder.section("Blocks", section_rows)
        return builder.build()


ContextSnippet: TypeAlias = TextSnippet | MultimodalSnippet
"""A context snippet — either text or multimodal."""


class ContextResponse(StructDictMixin, Struct, kw_only=True):
    """Response from the assistant context endpoint.

    Attributes:
        snippets: The list of context snippets.
        usage: Token usage statistics for the request.
        id: Unique identifier for the context response, or ``None`` if
            not included in the response.
    """

    snippets: list[ContextSnippet]
    usage: ChatUsage
    id: str | None = None

    @safe_display
    def __repr__(self) -> str:
        id_part = f"id={self.id!r}, " if self.id is not None else ""
        return f"ContextResponse({id_part}snippets={len(self.snippets)}, usage={self.usage!r})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("ContextResponse(...)")
            return
        with p.group(2, "ContextResponse(", ")"):
            if self.id is not None:
                p.breakable()
                p.text(f"id={self.id!r},")
            p.breakable()
            p.text(f"snippets={len(self.snippets)},")
            p.breakable()
            p.text(f"usage={self.usage!r},")
            for snippet in self.snippets[:3]:
                p.breakable()
                p.text(repr(snippet))

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("ContextResponse")
        if self.id is not None:
            builder.row("Id", self.id)
        builder.row("Snippets", len(self.snippets))
        builder.row("Usage", repr(self.usage))
        section_rows: list[tuple[str, Any]] = []
        for snippet in self.snippets[:5]:
            snippet_type = type(snippet).__name__
            score = snippet.score
            file_name = snippet.reference.file.name
            section_rows.append((snippet_type, f"score={score}, file={file_name}"))
        if len(self.snippets) > 5:
            section_rows.append(("...", f"{len(self.snippets) - 5} more"))
        builder.section("Snippets", section_rows)
        return builder.build()
