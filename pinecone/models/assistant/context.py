"""Context response models for the Assistant API."""

from __future__ import annotations

from typing import Any, TypeAlias

from msgspec import Struct

from pinecone.models._display import HtmlBuilder, abbreviate_list, safe_display, truncate_text
from pinecone.models.assistant._mixin import StructDictMixin
from pinecone.models.assistant.chat import ChatUsage
from pinecone.models.assistant.file_model import AssistantFileModel


class ContextImageData(StructDictMixin, Struct, kw_only=True):
    """The encoded bytes of an image in a multimodal context snippet.

    Reached as ``block.image_data``, and present only when the request set
    ``include_binary_content=True``. ``data`` is text, not bytes — decode it
    before writing a file.

    Attributes:
        type: The encoding of ``data`` (e.g. ``"base64"``).
        mime_type: The MIME type of the image (e.g. ``"image/jpeg"``).
        data: The encoded image as a string, ready for a data URI or for
            ``base64.b64decode``.
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
    """An image inside a :class:`MultimodalSnippet`, wire tag ``"image"``.

    The caption always arrives; the bytes do not. Ask for them with
    ``include_binary_content=True``, and expect a much larger response.

    Identify it with ``isinstance``; ``block.type`` gives you
    ``AttributeError: 'ContextImageBlock' object has no attribute 'type'``,
    because the tag selected this class during decoding and was then dropped.

    Attributes:
        caption: A text caption describing the image. Usable in a prompt on
            its own, without the image bytes.
        image_data: The :class:`ContextImageData` holding the encoded image,
            or ``None`` when the request did not set
            ``include_binary_content=True``.
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
    """Text inside a :class:`MultimodalSnippet`, wire tag ``"text"``.

    Identify it with ``isinstance``; ``block.type`` gives you
    ``AttributeError: 'ContextTextBlock' object has no attribute 'type'``,
    because the tag selected this class during decoding and was then dropped.

    Attributes:
        text: The text content of the block. Note the field is ``text`` here,
            not the ``content`` that :class:`TextSnippet` uses.
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
"""One block of a :class:`MultimodalSnippet`, text or image.

Branch with ``isinstance`` and read ``block.text`` on a
:class:`ContextTextBlock` or ``block.caption`` on a
:class:`ContextImageBlock`. These classes do not re-expose the wire tag, so
``block.type`` raises :exc:`AttributeError`.
"""


class FileReference(StructDictMixin, Struct, kw_only=True):
    """The source file a context snippet came from.

    Reached as ``snippet.reference``. Render ``reference.file.name`` as the
    label and ``reference.pages`` to point at the part of the document used.

    Attributes:
        file: The source file, as an
            :class:`~pinecone.models.assistant.file_model.AssistantFileModel`
            — ``file.name`` for a label, ``file.id`` to fetch it again, and
            ``file.metadata`` for whatever you attached at upload.
        pages: Page numbers relevant to the snippet, when the source is a
            paginated document such as a PDF. ``None`` for text, JSON, or
            Markdown sources.
        type: The kind of document referenced — ``"text"``, ``"json"``,
            ``"markdown"``, ``"pdf"``, or ``"doc_x"`` — or ``None`` when the
            payload omits it.
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
"""Alias for :class:`FileReference`, the type of ``snippet.reference``."""


class TextSnippet(StructDictMixin, Struct, kw_only=True, tag="text", tag_field="type"):
    """A retrieved passage of plain text, from the wire tag ``"text"``.

    The :data:`ContextSnippet` variant whose ``content`` is a single string.
    A request with ``multimodal=True`` can instead yield a
    :class:`MultimodalSnippet`, whose ``content`` is a list of blocks, so
    branch with ``isinstance`` before reading ``content``.

    Branching on the tag instead gives you
    ``AttributeError: 'TextSnippet' object has no attribute 'type'``. That
    does not mean the payload lacked a ``type``: the tag selected this class
    during decoding and was then dropped, so there is no attribute to read.
    The streaming chunk classes do keep theirs, which is why code moved over
    from a chat stream hits this.

    Attributes:
        content: The retrieved passage, ready to put in your own prompt.
        score: Relevance of the snippet to the query; higher is more relevant.
        reference: The :class:`FileReference` naming where the passage came
            from.
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
    """A retrieved passage of mixed text and images, wire tag ``"multimodal"``.

    The :data:`ContextSnippet` variant whose ``content`` is a **list** of
    blocks rather than a string, so iterate it and branch with ``isinstance``
    on :class:`ContextTextBlock` versus :class:`ContextImageBlock`.

    Branching on the tag instead gives you
    ``AttributeError: 'MultimodalSnippet' object has no attribute 'type'``,
    and the same for either block class. That does not mean the payload
    lacked a ``type``: the tag selected the class during decoding and was
    then dropped, so there is no attribute to read.

    Attributes:
        content: The blocks making up the snippet, in document order. Each is
            a :class:`ContextTextBlock` (read ``block.text``) or a
            :class:`ContextImageBlock` (read ``block.caption``, and
            ``block.image_data`` when the request set
            ``include_binary_content=True``).
        score: Relevance of the snippet to the query; higher is more relevant.
        reference: The :class:`FileReference` naming where the passage came
            from.
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
"""One retrieved snippet, dispatched from the wire on a ``type`` tag.

Both variants carry ``score`` and ``reference``; they differ in ``content``.
On a :class:`TextSnippet` it is a string; on a :class:`MultimodalSnippet` it
is a list of blocks, so string handling of one will fail on the other.

Branch with ``isinstance``: unlike the streaming chunks, these classes do not
re-expose the wire tag, so ``snippet.type`` raises :exc:`AttributeError`.
"""


class ContextResponse(StructDictMixin, Struct, kw_only=True):
    """The retrieved snippets for a query, with no answer generated over them.

    Returned by :meth:`~pinecone.client.assistants.Assistants.context`. This
    is Pinecone's retrieval step on its own: the snippets are source material
    for a prompt you assemble yourself, not prose to show a user. Reach for it
    when you want to run your own model over the assistant's retrieval, or to
    see what an assistant would have been given.

    ``snippets`` holds :data:`ContextSnippet`, which is two classes. A
    :class:`TextSnippet` has a string ``content``. A
    :class:`MultimodalSnippet` has a list of blocks instead — each a
    :class:`ContextTextBlock` (``block.text``) or a
    :class:`ContextImageBlock` (``block.caption``, plus ``block.image_data``
    when the request set ``include_binary_content=True``). Branch with
    ``isinstance``, not on a ``type`` attribute: the snippet and block classes
    do not re-expose their wire tag, so ``snippet.type`` raises
    :exc:`AttributeError`. Both snippet classes carry ``score`` and
    ``snippet.reference.file.name``.

    Attributes:
        snippets: The retrieved snippets.
        usage: :class:`~pinecone.models.assistant.chat.ChatUsage` token counts
            for the retrieval request.
        id: Identifier of this context response, or ``None`` when the server
            did not report one.

    Examples:
        What comes back is retrieved source text, scored and attributed — no
        model was asked to write anything, which is why the completion token
        count is zero:

        >>> from pinecone.models.assistant import TextSnippet
        >>> response = pc.assistants.context(
        ...     assistant_name="acme-support-bot",
        ...     query="Which regions support BYOC?",
        ... )
        >>> snippet = response.snippets[0]
        >>> isinstance(snippet, TextSnippet)
        True
        >>> snippet.score
        0.87
        >>> snippet.content
        'BYOC is available in aws us-east-1.'
        >>> snippet.reference.file.name
        'q3-revenue-review.pdf'
        >>> response.usage.completion_tokens
        0

        Reading ``snippet.type`` to decide which variant you have does not
        work, even though the wire payload carries that tag:

        >>> snippet.type
        Traceback (most recent call last):
            ...
        AttributeError: 'TextSnippet' object has no attribute 'type'

    .. seealso::
       - :class:`~pinecone.models.assistant.chat.ChatResponse` — the generated
         answer over the same retrieval, from
         :meth:`~pinecone.client.assistants.Assistants.chat`, with citations
         you can render.
       - :class:`~pinecone.models.assistant.options.ContextOptions` — the
         bundle that tunes retrieval for a chat request.
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
