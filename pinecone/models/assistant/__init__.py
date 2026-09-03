"""Assistant data models."""

from __future__ import annotations

from pinecone.models.assistant.chat import (
    ChatCitation,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionResponse,
    ChatHighlight,
    ChatMessage,
    ChatReference,
    ChatResponse,
    ChatUsage,
)
from pinecone.models.assistant.context import (
    ContextContentBlock,
    ContextImageBlock,
    ContextImageData,
    ContextReference,
    ContextResponse,
    ContextSnippet,
    ContextTextBlock,
    FileReference,
    MultimodalSnippet,
    PageReference,
    TextSnippet,
)
from pinecone.models.assistant.evaluation import (
    AlignmentResult,
    AlignmentScores,
    EntailmentResult,
)
from pinecone.models.assistant.file_model import AssistantFileModel
from pinecone.models.assistant.list import (
    ListAssistantsResponse,
    ListFilesResponse,
    ListOperationsResponse,
)
from pinecone.models.assistant.message import Message
from pinecone.models.assistant.model import AssistantModel
from pinecone.models.assistant.operation import OperationModel
from pinecone.models.assistant.options import ContextOptions
from pinecone.models.assistant.streaming import (
    AsyncChatCompletionStream,
    AsyncChatStream,
    ChatCompletionStream,
    ChatCompletionStreamChoice,
    ChatCompletionStreamChunk,
    ChatCompletionStreamDelta,
    ChatStream,
    ChatStreamChunk,
    StreamCitationChunk,
    StreamContentChunk,
    StreamContentDelta,
    StreamMessageEnd,
    StreamMessageStart,
)

# ---------------------------------------------------------------------------
# Backwards-compatibility aliases
#
# The following names are provided for callers migrating from the legacy
# pinecone_plugins.assistant package. Each alias points at the canonical
# class in the new SDK. Prefer the canonical name in new code.
# ---------------------------------------------------------------------------

FileModel = AssistantFileModel
"""Deprecated alias for :class:`AssistantFileModel`. Use the canonical name."""

Usage = ChatUsage
"""Deprecated alias for :class:`ChatUsage`. Use the canonical name."""

TokenCounts = ChatUsage
"""Deprecated alias for :class:`ChatUsage`. Use the canonical name."""

Citation = ChatCitation
"""Deprecated alias for :class:`ChatCitation`. Use the canonical name."""

Reference = ChatReference
"""Deprecated alias for :class:`ChatReference`. Use the canonical name."""

Highlight = ChatHighlight
"""Deprecated alias for :class:`ChatHighlight`. Use the canonical name."""

MessageDelta = StreamContentDelta
"""Deprecated alias for :class:`StreamContentDelta`. Use the canonical name."""

BaseStreamChatResponseChunk = ChatStreamChunk
"""Deprecated alias for :data:`ChatStreamChunk`. Use the canonical name.

This is a union of the four chunk classes, not a base class, so it works with
``isinstance`` but cannot be subclassed."""

StreamChatResponseMessageStart = StreamMessageStart
"""Deprecated alias for :class:`StreamMessageStart`."""

StreamChatResponseContentDelta = StreamContentChunk
"""Deprecated alias for :class:`StreamContentChunk`."""

StreamChatResponseCitation = StreamCitationChunk
"""Deprecated alias for :class:`StreamCitationChunk`."""

StreamChatResponseMessageEnd = StreamMessageEnd
"""Deprecated alias for :class:`StreamMessageEnd`."""

StreamingChatCompletionChunk = ChatCompletionStreamChunk
"""Deprecated alias for :class:`ChatCompletionStreamChunk`."""

StreamingChatCompletionChoice = ChatCompletionStreamChoice
"""Deprecated alias for :class:`ChatCompletionStreamChoice`."""

AlignmentResponse = AlignmentResult
"""Deprecated alias for :class:`AlignmentResult`."""

Metrics = AlignmentScores
"""Deprecated alias for :class:`AlignmentScores`."""

EvaluatedFact = EntailmentResult
"""Deprecated alias for :class:`EntailmentResult`."""

TextBlock = ContextTextBlock
"""Deprecated alias for :class:`ContextTextBlock`."""

ImageBlock = ContextImageBlock
"""Deprecated alias for :class:`ContextImageBlock`."""

Image = ContextImageData
"""Deprecated alias for :class:`ContextImageData`."""

PdfReference = FileReference
"""Deprecated alias for :class:`FileReference`.

All five per-format reference names here are the same class, so
``isinstance(ref, PdfReference)`` is true for a Markdown reference as well.
Read ``ref.type`` to tell the formats apart."""
TextReference = FileReference
"""Deprecated alias for :class:`FileReference`."""
JsonReference = FileReference
"""Deprecated alias for :class:`FileReference`."""
MarkdownReference = FileReference
"""Deprecated alias for :class:`FileReference`."""
DocxReference = FileReference
"""Deprecated alias for :class:`FileReference`."""

__all__ = [
    "AlignmentResponse",  # deprecated alias for AlignmentResult
    "AlignmentResult",
    "AlignmentScores",
    "AssistantFileModel",
    "AssistantModel",
    "AsyncChatCompletionStream",
    "AsyncChatStream",
    "BaseStreamChatResponseChunk",  # deprecated alias for ChatStreamChunk
    "ChatCitation",
    "ChatCompletionChoice",
    "ChatCompletionMessage",
    "ChatCompletionResponse",
    "ChatCompletionStream",
    "ChatCompletionStreamChoice",
    "ChatCompletionStreamChunk",
    "ChatCompletionStreamDelta",
    "ChatHighlight",
    "ChatMessage",
    "ChatReference",
    "ChatResponse",
    "ChatStream",
    "ChatStreamChunk",
    "ChatUsage",
    "Citation",  # deprecated alias for ChatCitation
    "ContextContentBlock",
    "ContextImageBlock",
    "ContextImageData",
    "ContextOptions",
    "ContextReference",
    "ContextResponse",
    "ContextSnippet",
    "ContextTextBlock",
    "DocxReference",  # deprecated alias for FileReference
    "EntailmentResult",
    "EvaluatedFact",  # deprecated alias for EntailmentResult
    "FileModel",  # deprecated alias for AssistantFileModel
    "FileReference",
    "Highlight",  # deprecated alias for ChatHighlight
    "Image",  # deprecated alias for ContextImageData
    "ImageBlock",  # deprecated alias for ContextImageBlock
    "JsonReference",  # deprecated alias for FileReference
    "ListAssistantsResponse",
    "ListFilesResponse",
    "ListOperationsResponse",
    "MarkdownReference",  # deprecated alias for FileReference
    "Message",
    "MessageDelta",  # deprecated alias for StreamContentDelta
    "Metrics",  # deprecated alias for AlignmentScores
    "MultimodalSnippet",
    "OperationModel",
    "PageReference",
    "PdfReference",  # deprecated alias for FileReference
    "Reference",  # deprecated alias for ChatReference
    "StreamChatResponseCitation",  # deprecated alias for StreamCitationChunk
    "StreamChatResponseContentDelta",  # deprecated alias for StreamContentChunk
    "StreamChatResponseMessageEnd",  # deprecated alias for StreamMessageEnd
    "StreamChatResponseMessageStart",  # deprecated alias for StreamMessageStart
    "StreamCitationChunk",
    "StreamContentChunk",
    "StreamContentDelta",
    "StreamMessageEnd",
    "StreamMessageStart",
    "StreamingChatCompletionChoice",  # deprecated alias for ChatCompletionStreamChoice
    "StreamingChatCompletionChunk",  # deprecated alias for ChatCompletionStreamChunk
    "TextBlock",  # deprecated alias for ContextTextBlock
    "TextReference",  # deprecated alias for FileReference
    "TextSnippet",
    "TokenCounts",  # deprecated alias for ChatUsage
    "Usage",  # deprecated alias for ChatUsage
]
