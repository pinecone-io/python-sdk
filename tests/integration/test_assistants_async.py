"""Integration tests for assistant operations (async / REST).

Tests cover using AsyncPinecone:
  - `assistant-lifecycle`: create, describe, list, update, delete
  - `assistant-files`: upload file, list, describe file, delete file
  - `assistant-chat`: non-streaming chat, verify ChatResponse structure
  - `assistant-context`: retrieve context snippets, verify ContextResponse
  - `assistant-chat-streaming`: streaming chat, verify chunk structure
  - `assistant-chat-completions`: OpenAI-compatible chat completions endpoint
  - `assistant-evaluate`: evaluate_alignment, verify AlignmentResult structure

These tests make real API calls and require PINECONE_API_KEY in the environment.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile

import pytest

from pinecone import ApiError, AsyncPinecone, PineconeError, PineconeValueError
from pinecone.models.assistant.chat import (
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
    ContextImageBlock,
    ContextResponse,
    MultimodalSnippet,
    TextSnippet,
)
from pinecone.models.assistant.evaluation import AlignmentResult, EntailmentResult
from pinecone.models.assistant.file_model import AssistantFileModel
from pinecone.models.assistant.list import ListFilesResponse
from pinecone.models.assistant.model import AssistantModel
from pinecone.models.assistant.options import ContextOptions
from pinecone.models.assistant.streaming import (
    ChatCompletionStreamChunk,
    ChatStreamChunk,
    StreamContentChunk,
    StreamMessageEnd,
    StreamMessageStart,
)
from tests.integration.conftest import async_cleanup_resource, async_poll_until, unique_name

# ---------------------------------------------------------------------------
# assistant-lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_assistant_lifecycle_create_describe_list_update_delete(
    async_client: AsyncPinecone,
) -> None:
    """Create assistant, verify fields, describe, list, update instructions, delete."""
    name = unique_name("asst")
    try:
        # --- Create ---
        assistant = await async_client.assistants.create(
            name=name,
            instructions="You are an async test assistant.",
        )
        assert isinstance(assistant, AssistantModel)
        assert assistant.name == name
        assert assistant.status in ("Initializing", "Ready")

        # Poll until Ready
        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # --- Describe ---
        described = await async_client.assistants.describe(name=name)
        assert isinstance(described, AssistantModel)
        assert described.name == name
        assert described.status == "Ready"
        assert described.instructions == "You are an async test assistant."
        assert described.created_at is not None
        assert isinstance(described.created_at, str)
        assert len(described.created_at) > 0
        assert described.host is not None

        # --- List ---
        assistants = await async_client.assistants.list().to_list()
        names = [a.name for a in assistants]
        assert name in names

        # --- Update ---
        updated = await async_client.assistants.update(
            name=name,
            instructions="Updated async instructions for the test assistant.",
        )
        assert isinstance(updated, AssistantModel)
        assert updated.name == name
        assert updated.instructions == "Updated async instructions for the test assistant."

        # Verify update persisted via describe
        re_described = await async_client.assistants.describe(name=name)
        assert re_described.instructions == "Updated async instructions for the test assistant."

    finally:
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-files
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_assistant_files_upload_list_describe_delete(
    async_client: AsyncPinecone,
) -> None:
    """Upload a file to an assistant, list files, describe file, delete file."""
    name = unique_name("asst")
    tmp_path: str | None = None
    file_id: str | None = None
    try:
        # Create assistant
        assistant = await async_client.assistants.create(
            name=name, instructions="Test file assistant."
        )
        assert isinstance(assistant, AssistantModel)

        # Wait until Ready
        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Create a small temp text file on disk
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-test-"
        ) as f:
            f.write("Pinecone is a vector database used for building AI applications.")
            tmp_path = f.name

        # Upload the file
        file_model = await async_client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )
        assert isinstance(file_model, AssistantFileModel)
        assert file_model.id is not None
        assert file_model.name is not None
        assert file_model.status in ("Available", "Processing", "Processed")
        file_id = file_model.id

        # Poll until file is processed (Available)
        await async_poll_until(
            lambda: async_client.assistants.describe_file(assistant_name=name, file_id=file_id),
            lambda f: f.status in ("Available", "Processed"),
            timeout=120,
            interval=5,
            description=f"file {file_id}",
        )

        # Describe file — verify fields
        described_file = await async_client.assistants.describe_file(
            assistant_name=name, file_id=file_id
        )
        assert isinstance(described_file, AssistantFileModel)
        assert described_file.id == file_id
        assert described_file.name is not None
        assert described_file.status in ("Available", "Processed")

        # List files — verify file appears
        files = await async_client.assistants.list_files(assistant_name=name).to_list()
        file_ids = [f.id for f in files]
        assert file_id in file_ids

        # Delete the file
        await async_client.assistants.delete_file(assistant_name=name, file_id=file_id, timeout=60)
        file_id = None  # mark as cleaned

    finally:
        # Clean up temp file
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        # Delete the assistant
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-chat
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_assistant_chat_non_streaming_response(
    async_client: AsyncPinecone,
) -> None:
    """Send a non-streaming chat message and verify ChatResponse structure."""
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        # Create assistant
        assistant = await async_client.assistants.create(
            name=name, instructions="You are a helpful assistant."
        )
        assert isinstance(assistant, AssistantModel)

        # Wait for Ready
        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Upload a small knowledge file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-chat-"
        ) as f:
            f.write("Pinecone is a managed vector database. It supports dense and sparse vectors.")
            tmp_path = f.name

        await async_client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # Send a non-streaming chat message
        response = await async_client.assistants.chat(
            assistant_name=name,
            messages=[{"role": "user", "content": "What is Pinecone?"}],
            stream=False,
        )

        # Verify ChatResponse structure
        assert isinstance(response, ChatResponse)
        assert isinstance(response.id, str)
        assert len(response.id) > 0
        assert isinstance(response.model, str)
        assert len(response.model) > 0
        assert hasattr(response, "message")
        assert response.message.role == "assistant"
        assert isinstance(response.message.content, str)
        assert len(response.message.content) > 0
        assert isinstance(response.finish_reason, str)

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-context
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_assistant_context_retrieval(
    async_client: AsyncPinecone,
) -> None:
    """Retrieve context from an assistant and verify ContextResponse structure."""
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        # Create assistant
        assistant = await async_client.assistants.create(
            name=name, instructions="You are a helpful assistant."
        )
        assert isinstance(assistant, AssistantModel)

        # Wait for Ready
        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Upload a knowledge file for context retrieval
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-ctx-"
        ) as f:
            f.write("Pinecone is a managed vector database. It provides fast similarity search.")
            tmp_path = f.name

        await async_client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # Retrieve context using a message
        response = await async_client.assistants.context(
            assistant_name=name,
            messages=[{"role": "user", "content": "What is Pinecone?"}],
        )

        # Verify ContextResponse structure
        assert isinstance(response, ContextResponse)
        assert hasattr(response, "snippets")
        assert isinstance(response.snippets, list)
        assert hasattr(response, "usage")
        assert response.usage is not None

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-chat-streaming
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_assistant_chat_streaming_returns_content_chunks(
    async_client: AsyncPinecone,
) -> None:
    """Stream chat(stream=True), verify at least one content chunk with non-empty text."""
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        # Create assistant
        assistant = await async_client.assistants.create(
            name=name, instructions="You are a helpful assistant."
        )
        assert isinstance(assistant, AssistantModel)

        # Wait for Ready
        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Upload a small knowledge file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-stream-"
        ) as f:
            f.write("Pinecone is a managed vector database for AI applications.")
            tmp_path = f.name

        await async_client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # Streaming chat — await chat() to get the AsyncIterator[ChatStreamChunk]
        stream = await async_client.assistants.chat(
            assistant_name=name,
            messages=[{"role": "user", "content": "What is Pinecone?"}],
            stream=True,
        )

        # Consume the entire async stream
        chunks: list[ChatStreamChunk] = [chunk async for chunk in stream]

        # Must have received at least one chunk
        assert len(chunks) > 0, "Expected at least one chunk from async streaming chat"

        # At least one chunk must be a StreamContentChunk with non-empty content
        content_chunks = [c for c in chunks if isinstance(c, StreamContentChunk)]
        assert len(content_chunks) > 0, "Expected at least one StreamContentChunk"

        # Concatenated content must be non-empty
        full_content = "".join(c.delta.content for c in content_chunks)
        assert len(full_content) > 0, "Concatenated streaming content must not be empty"

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-chat-completions
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_assistant_chat_completions_openai_compatible_response(
    async_client: AsyncPinecone,
) -> None:
    """chat_completions() returns ChatCompletionResponse with OpenAI-compatible structure."""
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        # Create assistant
        assistant = await async_client.assistants.create(
            name=name, instructions="You are a helpful assistant."
        )
        assert isinstance(assistant, AssistantModel)

        # Wait for Ready
        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Upload a small knowledge file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-cc-"
        ) as f:
            f.write("Pinecone is a managed vector database for similarity search.")
            tmp_path = f.name

        await async_client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # Non-streaming chat completions
        response = await async_client.assistants.chat_completions(
            assistant_name=name,
            messages=[{"role": "user", "content": "What is Pinecone?"}],
            stream=False,
        )

        # Verify ChatCompletionResponse structure (OpenAI-compatible)
        assert isinstance(response, ChatCompletionResponse)
        assert hasattr(response, "choices")
        assert isinstance(response.choices, list)
        assert len(response.choices) > 0, "Expected at least one choice"

        choice = response.choices[0]
        assert hasattr(choice, "message")
        assert isinstance(choice.message, ChatCompletionMessage)
        assert choice.message.content is not None
        assert len(choice.message.content) > 0, "Expected non-empty message content"
        assert hasattr(choice, "finish_reason")

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-chat-typed-messages
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_assistant_chat_typed_message_objects(
    async_client: AsyncPinecone,
) -> None:
    """Verify chat() accepts typed Message objects in async mode (unified-chat-0003).

    Sends a multi-turn conversation where both the initial user message and the
    follow-up (including the prior assistant turn) are typed Message objects.
    """
    from pinecone.models.assistant.message import Message

    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        assistant = await async_client.assistants.create(
            name=name, instructions="You are a helpful assistant."
        )
        assert isinstance(assistant, AssistantModel)

        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-typed-async-"
        ) as f:
            f.write("Pinecone is a managed vector database. It stores dense and sparse vectors.")
            tmp_path = f.name

        await async_client.assistants.upload_file(
            assistant_name=name, file_path=tmp_path, timeout=120
        )

        # --- Turn 1: typed Message object (not a plain dict) ---
        first_message = Message(role="user", content="What is Pinecone?")
        assert isinstance(first_message, Message)

        first_response = await async_client.assistants.chat(
            assistant_name=name,
            messages=[first_message],
            stream=False,
        )
        assert isinstance(first_response, ChatResponse)
        assert first_response.message.role == "assistant"
        assert isinstance(first_response.message.content, str)
        assert len(first_response.message.content) > 0

        # --- Turn 2: full typed-object history (user + assistant + user) ---
        history = [
            first_message,
            Message(role="assistant", content=first_response.message.content),
            Message(role="user", content="What vectors does it support?"),
        ]
        assert all(isinstance(m, Message) for m in history)

        second_response = await async_client.assistants.chat(
            assistant_name=name,
            messages=history,
            stream=False,
        )
        assert isinstance(second_response, ChatResponse)
        assert second_response.message.role == "assistant"
        assert isinstance(second_response.message.content, str)
        assert len(second_response.message.content) > 0
        assert second_response.id != first_response.id

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-evaluate
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_assistant_evaluate_alignment_scores(
    async_client: AsyncPinecone,
) -> None:
    """evaluate_alignment() returns AlignmentResult with correctness/completeness scores."""
    # evaluate_alignment is stateless — no assistant or file needed
    result = await async_client.assistants.evaluate_alignment(
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        ground_truth_answer="The capital of France is Paris.",
    )

    # Verify AlignmentResult structure
    assert isinstance(result, AlignmentResult)
    assert hasattr(result, "scores")
    assert hasattr(result.scores, "alignment")
    assert hasattr(result.scores, "correctness")
    assert hasattr(result.scores, "completeness")

    # Scores should be floats in [0, 1]
    assert isinstance(result.scores.alignment, float)
    assert isinstance(result.scores.correctness, float)
    assert isinstance(result.scores.completeness, float)
    assert 0.0 <= result.scores.alignment <= 1.0
    assert 0.0 <= result.scores.correctness <= 1.0
    assert 0.0 <= result.scores.completeness <= 1.0

    # A well-aligned answer should score high
    assert result.scores.alignment > 0.5, f"Expected high alignment, got {result.scores.alignment}"

    # facts list and usage should be present
    assert hasattr(result, "facts")
    assert isinstance(result.facts, list)
    assert hasattr(result, "usage")
    assert result.usage is not None


# ---------------------------------------------------------------------------
# assistant-validation
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_assistant_create_region_validation_and_chat_stream_json_conflict_async(
    async_client: AsyncPinecone,
) -> None:
    """Region validation and stream+json_response mutual exclusivity fire before any API call (async).

    Verifies (async path):
    - unified-assistant-0013: create() raises PineconeValueError for regions other than
      "us" or "eu"
    - unified-assistant-0015: Region validation is case-sensitive — "US" and "EU" are
      rejected
    - unified-chat-0012: chat() raises PineconeValueError when stream=True and
      json_response=True are specified together, before any API call

    No real assistant is created. All validations are client-side.
    """
    # unified-assistant-0013: invalid region raises PineconeValueError
    with pytest.raises(PineconeValueError):
        await async_client.assistants.create(name="validation-test", region="au")

    with pytest.raises(PineconeValueError):
        await async_client.assistants.create(name="validation-test", region="invalid-region")

    # unified-assistant-0015: case-sensitive — uppercase "US" and "EU" are rejected
    with pytest.raises(PineconeValueError):
        await async_client.assistants.create(name="validation-test", region="US")

    with pytest.raises(PineconeValueError):
        await async_client.assistants.create(name="validation-test", region="EU")

    # unified-chat-0012: stream=True + json_response=True raises PineconeValueError
    # before any network call; assistant_name does not need to exist
    with pytest.raises(PineconeValueError):
        await async_client.assistants.chat(
            assistant_name="does-not-matter",
            messages=[{"content": "test query"}],
            stream=True,
            json_response=True,
        )


# ---------------------------------------------------------------------------
# assistant-create: metadata default behavior (async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_create_assistant_metadata_default_async(async_client: AsyncPinecone) -> None:
    """create() without metadata produces an assistant with model.metadata is None (async).

    Verifies the fix for D3: the SDK previously sent "metadata": {} instead of
    omitting the key, causing the backend to store an empty object rather than None.
    """
    name = unique_name("asst")
    try:
        model = await async_client.assistants.create(name=name, timeout=-1)
        assert isinstance(model, AssistantModel)
        assert model.metadata is None
    finally:
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-files: byte-stream upload with file_name and metadata (async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_upload_file_from_byte_stream_with_metadata_async(
    async_client: AsyncPinecone,
) -> None:
    """Upload from a BytesIO stream with file_name and metadata; verify both are preserved (async).

    Verifies (async path):
    - unified-file-0002: Can upload a file from an in-memory byte stream
    - unified-file-0003: Can upload a file with optional user-provided metadata
    - unified-file-0033: Byte stream upload preserves the provided file name and
      metadata in the file record
    """
    name = unique_name("asst")
    file_id: str | None = None
    try:
        # Create assistant
        assistant = await async_client.assistants.create(
            name=name, instructions="Test async stream upload."
        )
        assert isinstance(assistant, AssistantModel)

        # Wait for assistant to be ready
        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Build an in-memory byte stream with a specific file name and metadata
        content = b"Pinecone is a managed vector database. It supports semantic search."
        stream = io.BytesIO(content)
        upload_name = "stream-knowledge-async.txt"
        upload_metadata: dict[str, object] = {"source": "async-stream-test", "category": "sdk-docs"}

        # Upload via file_stream — exercises the file_stream / file_name / metadata paths
        file_model = await async_client.assistants.upload_file(
            assistant_name=name,
            file_stream=stream,
            file_name=upload_name,
            metadata=upload_metadata,
            timeout=120,
        )
        assert isinstance(file_model, AssistantFileModel)
        assert file_model.id is not None
        file_id = file_model.id

        # Poll until file processing completes
        await async_poll_until(
            lambda: async_client.assistants.describe_file(assistant_name=name, file_id=file_id),
            lambda f: f.status in ("Available", "Processed"),
            timeout=120,
            interval=5,
            description=f"file {file_id}",
        )

        # Verify file_name and metadata are preserved in the described file record
        described = await async_client.assistants.describe_file(
            assistant_name=name, file_id=file_id
        )
        assert isinstance(described, AssistantFileModel)
        assert described.id == file_id
        assert described.name == upload_name, (
            f"Expected file name '{upload_name}', got '{described.name}'"
        )
        assert described.metadata is not None, "Expected metadata to be preserved, got None"
        assert described.metadata.get("source") == "async-stream-test"
        assert described.metadata.get("category") == "sdk-docs"

        # Clean up file
        await async_client.assistants.delete_file(assistant_name=name, file_id=file_id, timeout=60)
        file_id = None

    finally:
        if file_id is not None:
            with contextlib.suppress(Exception):
                await async_client.assistants.delete_file(
                    assistant_name=name, file_id=file_id, timeout=60
                )
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.timeout(300)
async def test_upload_file_input_validation_and_delete_returns_none_async(
    async_client: AsyncPinecone,
) -> None:
    """upload_file() client-side validation fires before any API call; delete_file() returns None.

    Verifies (async path):
    - unified-file-0035: Uploading a file from a nonexistent local path raises PineconeValueError.
    - unified-file-0030: Successful file deletion returns no value (None).
    Mutual-exclusivity check (both/neither file_path+file_stream) also verified.
    """
    # --- Part 1: Client-side validation (no API call needed) ---

    # Both file_path AND file_stream → PineconeValueError before any HTTP request
    with pytest.raises(PineconeValueError):
        await async_client.assistants.upload_file(
            assistant_name="doesnt-matter",
            file_path="/some/path.txt",
            file_stream=io.BytesIO(b"data"),
        )

    # Neither file_path nor file_stream → PineconeValueError
    with pytest.raises(PineconeValueError):
        await async_client.assistants.upload_file(assistant_name="doesnt-matter")

    # Nonexistent path → PineconeValueError (unified-file-0035)
    with pytest.raises(PineconeValueError, match="File not found"):
        await async_client.assistants.upload_file(
            assistant_name="doesnt-matter",
            file_path="/nonexistent/path/to/file.txt",
        )

    # --- Part 2: delete_file() returns None (unified-file-0030) ---
    name = unique_name("asst")
    tmp_path: str | None = None
    file_id: str | None = None
    try:
        await async_client.assistants.create(name=name, instructions="Validation test assistant.")

        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Create a small temp text file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-val-"
        ) as f:
            f.write("Validation test content.")
            tmp_path = f.name

        file_model = await async_client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )
        assert isinstance(file_model, AssistantFileModel)
        file_id = file_model.id

        # unified-file-0030: delete_file returns None
        result = await async_client.assistants.delete_file(
            assistant_name=name,
            file_id=file_id,
            timeout=60,
        )
        assert result is None, f"Expected delete_file to return None, got {result!r}"
        file_id = None  # mark as cleaned up

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        if file_id is not None:
            with contextlib.suppress(Exception):
                await async_client.assistants.delete_file(
                    assistant_name=name, file_id=file_id, timeout=60
                )
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.timeout(600)
async def test_describe_file_signed_url_async(async_client: AsyncPinecone) -> None:
    """describe_file(include_url=True) returns a non-None signed_url string;
    describe_file() without include_url returns signed_url=None.

    Verifies:
    - unified-file-0009: Can request a signed download URL when retrieving file metadata.
    - unified-file-0025: When requesting a signed URL, include-url='true' is sent to the API.
    - unified-file-0026: When not requesting a signed URL, the include-url param is omitted.
    - unified-file-0056: The signed URL defaults to None when absent from the API response.
    """
    name = unique_name("asst")
    file_id: str | None = None
    try:
        # Create assistant
        assistant = await async_client.assistants.create(name=name, instructions="Test signed URL.")
        assert isinstance(assistant, AssistantModel)

        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Upload a small text file
        content = b"Pinecone vector database enables fast semantic search."
        stream = io.BytesIO(content)
        file_model = await async_client.assistants.upload_file(
            assistant_name=name,
            file_stream=stream,
            file_name="signed-url-test-async.txt",
            timeout=120,
        )
        assert isinstance(file_model, AssistantFileModel)
        assert file_model.id is not None
        file_id = file_model.id

        # Wait until the file is ready
        await async_poll_until(
            lambda: async_client.assistants.describe_file(assistant_name=name, file_id=file_id),
            lambda f: f.status in ("Available", "Processed"),
            timeout=120,
            interval=5,
            description=f"file {file_id}",
        )

        # Without include_url — signed_url should be None (default behavior)
        without_url = await async_client.assistants.describe_file(
            assistant_name=name,
            file_id=file_id,
        )
        assert isinstance(without_url, AssistantFileModel)
        assert without_url.signed_url is None, (
            f"Expected signed_url=None without include_url=True, got {without_url.signed_url!r}"
        )

        # With include_url=True — signed_url should be a non-None string
        with_url = await async_client.assistants.describe_file(
            assistant_name=name,
            file_id=file_id,
            include_url=True,
        )
        assert isinstance(with_url, AssistantFileModel)
        assert with_url.signed_url is not None, (
            "Expected a signed_url string when include_url=True, got None"
        )
        assert isinstance(with_url.signed_url, str), (
            f"Expected signed_url to be a str, got {type(with_url.signed_url)}"
        )
        assert len(with_url.signed_url) > 0, "Expected non-empty signed_url"

        # The core file identity fields should be the same in both responses
        assert with_url.id == without_url.id
        assert with_url.name == without_url.name

        # Clean up file
        await async_client.assistants.delete_file(assistant_name=name, file_id=file_id, timeout=60)
        file_id = None

    finally:
        if file_id is not None:
            with contextlib.suppress(Exception):
                await async_client.assistants.delete_file(
                    assistant_name=name, file_id=file_id, timeout=60
                )
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-chat-dict-no-role
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_chat_message_dict_role_default_async(
    async_client: AsyncPinecone,
) -> None:
    """async chat() with a message dict that omits 'role' defaults role to 'user'.

    Async variant of test_chat_message_dict_role_default_rest.
    Passes {"content": "..."} (no role key) and verifies that:
    1. The SDK converts the dict via Message.from_dict(), defaulting role to "user"
    2. The API call succeeds — confirming the role was correctly populated
    3. A valid ChatResponse is returned

    Verifies unified-chat-0020 and unified-chat-0019.
    """
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        assistant = await async_client.assistants.create(
            name=name, instructions="You are a helpful assistant."
        )
        assert isinstance(assistant, AssistantModel)

        # Wait for Ready
        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-norol-"
        ) as f:
            f.write("Pinecone is a managed vector database supporting dense and sparse vectors.")
            tmp_path = f.name

        await async_client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # Pass a message dict WITHOUT a "role" key — SDK should default to "user"
        response = await async_client.assistants.chat(
            assistant_name=name,
            messages=[{"content": "What is Pinecone?"}],
            stream=False,
        )

        # A valid ChatResponse means the SDK defaulted role="user" and the API accepted it
        assert isinstance(response, ChatResponse)
        assert isinstance(response.id, str)
        assert len(response.id) > 0
        assert response.message.role == "assistant"
        assert isinstance(response.message.content, str)
        assert len(response.message.content) > 0

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-update: metadata replace semantics
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_assistant_update_metadata_replaces_not_merges_async(
    async_client: AsyncPinecone,
) -> None:
    """assistants.update(metadata=...) fully replaces existing metadata; returns updated AssistantModel.

    Verifies:
    - unified-assistant-0021: Updating metadata fully replaces the existing metadata rather
      than merging. Old keys absent from the new dict must disappear.
    - unified-assistant-0030: Updating an assistant returns an AssistantModel with the
      updated attributes visible immediately on the returned object.
    """
    name = unique_name("asst")
    try:
        # Create with two metadata keys that should disappear after the replace
        assistant = await async_client.assistants.create(
            name=name,
            instructions="Initial instructions.",
            metadata={"initial_key": "initial_val", "extra_key": "will_be_gone"},
        )
        assert isinstance(assistant, AssistantModel)

        # Wait for Ready before updating
        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Confirm initial metadata is present
        described = await async_client.assistants.describe(name=name)
        assert described.metadata is not None
        assert described.metadata.get("initial_key") == "initial_val"
        assert described.metadata.get("extra_key") == "will_be_gone"

        # Update with a completely different metadata dict (replace, not merge)
        updated = await async_client.assistants.update(name=name, metadata={"new_key": "new_val"})

        # unified-assistant-0030: update() returns the updated AssistantModel immediately
        assert isinstance(updated, AssistantModel)
        assert updated.name == name
        assert updated.metadata is not None

        # unified-assistant-0021: replace semantics — old keys must be gone
        assert updated.metadata.get("new_key") == "new_val", (
            f"Expected new_key='new_val', got: {updated.metadata}"
        )
        assert "initial_key" not in updated.metadata, (
            f"Expected initial_key to be absent after metadata replace, got: {updated.metadata}"
        )
        assert "extra_key" not in updated.metadata, (
            f"Expected extra_key to be absent after metadata replace, got: {updated.metadata}"
        )

        # Verify persistence via a fresh describe call
        re_described = await async_client.assistants.describe(name=name)
        assert re_described.metadata is not None
        assert re_described.metadata.get("new_key") == "new_val"
        assert "initial_key" not in re_described.metadata
        assert "extra_key" not in re_described.metadata

    finally:
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-context: context() input validation (async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_context_retrieval_validation_async(async_client: AsyncPinecone) -> None:
    """async context() raises PineconeValueError for invalid query/messages combos.

    Verifies:
    - unified-context-0008: Exactly one of query or messages must be truthy;
      empty string and empty list are treated as not provided
    - unified-context-0009: Providing both query and messages produces a
      PineconeValueError before any API call
    - unified-context-0010: Providing neither query nor messages produces a
      PineconeValueError before any API call

    No real assistant is created. All validations are client-side and fire
    before any HTTP request, so any assistant_name value is acceptable.
    """
    # unified-context-0009: both query and messages provided → rejected
    with pytest.raises(PineconeValueError):
        await async_client.assistants.context(
            assistant_name="validation-test",
            query="What is Pinecone?",
            messages=[{"role": "user", "content": "What is Pinecone?"}],
        )

    # unified-context-0010: neither query nor messages provided → rejected
    with pytest.raises(PineconeValueError):
        await async_client.assistants.context(
            assistant_name="validation-test",
        )

    # unified-context-0008: empty string query treated as not provided → rejected (neither truthy)
    with pytest.raises(PineconeValueError):
        await async_client.assistants.context(
            assistant_name="validation-test",
            query="",
        )

    # unified-context-0008: empty list messages treated as not provided → rejected (neither truthy)
    with pytest.raises(PineconeValueError):
        await async_client.assistants.context(
            assistant_name="validation-test",
            messages=[],
        )

    # unified-context-0008: empty string query + valid messages → only messages truthy → no validation error
    # (this call will reach the network; we only care it does NOT raise a PineconeValueError)
    try:
        await async_client.assistants.context(
            assistant_name="validation-test",
            query="",
            messages=[{"role": "user", "content": "test"}],
        )
    except PineconeValueError:
        raise AssertionError(
            "context() raised PineconeValueError when query='' and messages is non-empty — "
            "empty string should be treated as not provided, leaving messages as the sole input"
        ) from None
    except Exception:
        # Any other exception (ApiError, NotFoundError, etc.) is acceptable here —
        # the assistant does not exist and the API will reject the request.
        pass


# ---------------------------------------------------------------------------
# assistant-context-query-param (async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_context_retrieval_with_query_param_async(
    async_client: AsyncPinecone,
) -> None:
    """context() with query parameter (not messages) returns ContextResponse with typed snippet structure.

    Verifies:
    - unified-context-0001: Can retrieve context using a plain text query string (query param)
    - unified-context-0021: Response contains snippets list and usage statistics
    - unified-context-0023: Text snippets have content (str), score (float), and reference.file (str)
    The top_k=2 limit is also verified against the returned snippet count.
    """
    name = unique_name("asst")
    file_id: str | None = None
    tmp_path: str | None = None
    try:
        assistant = await async_client.assistants.create(
            name=name, instructions="You help answer questions about vector databases."
        )
        assert isinstance(assistant, AssistantModel)

        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Upload a text file with content the assistant can retrieve snippets from
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-qctx-"
        ) as f:
            f.write(
                "Pinecone is a managed vector database optimized for semantic similarity search. "
                "It stores vector embeddings and enables fast nearest-neighbor retrieval. "
                "Pinecone supports metadata filtering, sparse-dense hybrid search, and "
                "both serverless and pod-based deployment modes."
            )
            tmp_path = f.name

        file_model = await async_client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )
        assert isinstance(file_model, AssistantFileModel)
        file_id = file_model.id

        # Wait for file to be indexed before calling context
        await async_poll_until(
            lambda: async_client.assistants.describe_file(assistant_name=name, file_id=file_id),
            lambda f: f.status in ("Available", "Processed"),
            timeout=120,
            interval=5,
            description=f"file {file_id}",
        )

        # Use the query parameter (plain text string), not messages
        response = await async_client.assistants.context(
            assistant_name=name,
            query="What is Pinecone?",
            top_k=2,
        )

        # unified-context-0021: response has snippets list and usage
        assert isinstance(response, ContextResponse)
        assert isinstance(response.snippets, list)
        assert response.usage is not None

        # top_k=2 must limit the result to at most 2 snippets
        assert len(response.snippets) <= 2, (
            f"Expected at most 2 snippets with top_k=2, got {len(response.snippets)}"
        )

        # unified-context-0023: each text snippet has content, score (float), and reference.file
        for snippet in response.snippets:
            assert isinstance(snippet.score, float), (
                f"Expected float score, got {type(snippet.score)}"
            )
            assert hasattr(snippet, "reference"), "Snippet missing reference attribute"
            assert isinstance(snippet.reference.file, AssistantFileModel)
            assert isinstance(snippet.reference.file.name, str)
            assert snippet.reference.file.name != "", (
                f"Expected non-empty file reference name, got {snippet.reference.file.name!r}"
            )
            if isinstance(snippet, TextSnippet):
                assert isinstance(snippet.content, str) and len(snippet.content) > 0, (
                    "TextSnippet.content should be a non-empty string"
                )

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-chat-completions-streaming
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_chat_completions_streaming_async(
    async_client: AsyncPinecone,
) -> None:
    """chat_completions(stream=True) returns AsyncIterator[ChatCompletionStreamChunk] (async).

    Verifies:
    - unified-stream-0002: Chat completions support streaming mode that returns chunks
      lazily as they arrive (async variant).
    - ChatCompletionStreamChunk structure: each chunk has id (str) and choices list;
      each choice has index (int), delta with optional content (str), and optional
      finish_reason.
    - At least one chunk has non-empty delta.content; full concatenated response is non-empty.
    """
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        assistant = await async_client.assistants.create(
            name=name, instructions="You are a helpful assistant."
        )
        assert isinstance(assistant, AssistantModel)

        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-ccstream-"
        ) as f:
            f.write("Pinecone is a managed vector database for AI-powered search.")
            tmp_path = f.name

        await async_client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # chat_completions(stream=True) returns AsyncIterator[ChatCompletionStreamChunk]
        stream = await async_client.assistants.chat_completions(
            assistant_name=name,
            messages=[{"role": "user", "content": "What is Pinecone?"}],
            stream=True,
        )

        # Consume the full async stream
        chunks: list[ChatCompletionStreamChunk] = [chunk async for chunk in stream]

        # Must receive at least one chunk
        assert len(chunks) > 0, "Expected at least one ChatCompletionStreamChunk"

        # Every chunk must conform to the expected structure
        for chunk in chunks:
            assert isinstance(chunk, ChatCompletionStreamChunk), (
                f"Expected ChatCompletionStreamChunk, got {type(chunk)}"
            )
            assert isinstance(chunk.id, str) and chunk.id != "", (
                f"ChatCompletionStreamChunk.id must be a non-empty string, got {chunk.id!r}"
            )
            assert isinstance(chunk.choices, list), "choices must be a list"
            for choice in chunk.choices:
                assert isinstance(choice.index, int), "choice.index must be int"
                assert choice.delta.content is None or isinstance(choice.delta.content, str), (
                    f"delta.content must be str or None, got {type(choice.delta.content)}"
                )

        # At least one chunk must carry delta content
        content_choices = [
            c
            for chunk in chunks
            for c in chunk.choices
            if c.delta.content is not None and c.delta.content != ""
        ]
        assert len(content_choices) > 0, "Expected at least one chunk with non-empty delta.content"

        # Concatenated response must be meaningful
        full_content = "".join(
            c.delta.content
            for chunk in chunks
            for c in chunk.choices
            if c.delta.content is not None
        )
        assert len(full_content) > 0, "Concatenated streaming content must not be empty"

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# list_files_page — explicit single-page pagination (async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.timeout(600)
async def test_list_files_page_with_page_size_and_pagination_token_async(
    async_client: AsyncPinecone,
) -> None:
    """list_files_page() supports page_size and pagination_token.

    With page_size=1 and two uploaded files, the first call returns one file and a
    pagination token; the follow-up call with that token returns the second file.

    Verifies:
      - unified-file-0011: Can list one page of files with explicit pagination control
      - unified-file-0014: Can provide a pagination token to fetch the next page of files
    """
    name = unique_name("asst")
    file_id_a: str | None = None
    file_id_b: str | None = None
    try:
        assistant = await async_client.assistants.create(
            name=name, instructions="Async pagination test assistant."
        )
        assert isinstance(assistant, AssistantModel)

        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Upload file A
        file_a = await async_client.assistants.upload_file(
            assistant_name=name,
            file_stream=io.BytesIO(b"File A content for async pagination test."),
            file_name="file_a.txt",
            timeout=120,
        )
        assert isinstance(file_a, AssistantFileModel)
        file_id_a = file_a.id

        # Upload file B
        file_b = await async_client.assistants.upload_file(
            assistant_name=name,
            file_stream=io.BytesIO(b"File B content for async pagination test."),
            file_name="file_b.txt",
            timeout=120,
        )
        assert isinstance(file_b, AssistantFileModel)
        file_id_b = file_b.id

        # Wait until both files are available
        for fid in (file_id_a, file_id_b):
            await async_poll_until(
                lambda fid=fid: async_client.assistants.describe_file(
                    assistant_name=name, file_id=fid
                ),
                lambda f: f.status in ("Available", "Processed"),
                timeout=120,
                interval=5,
                description=f"file {fid}",
            )

        # First page: page_size=1 must return exactly one file plus a token
        page1 = await async_client.assistants.list_files_page(assistant_name=name, page_size=1)
        assert isinstance(page1, ListFilesResponse), (
            f"Expected ListFilesResponse, got {type(page1)}"
        )
        assert len(page1.files) == 1, f"page_size=1 should yield 1 file, got {len(page1.files)}"
        assert all(isinstance(f, AssistantFileModel) for f in page1.files)
        assert page1.next is not None and page1.next != ""

        # Second page: pass the token, expect the remaining file
        page2 = await async_client.assistants.list_files_page(
            assistant_name=name, page_size=1, pagination_token=page1.next
        )
        assert isinstance(page2, ListFilesResponse)
        assert len(page2.files) == 1

        # Together the two pages must cover both uploaded files with no duplication
        seen_ids = {f.id for f in page1.files} | {f.id for f in page2.files}
        assert seen_ids == {file_id_a, file_id_b}

    finally:
        for fid in filter(None, [file_id_a, file_id_b]):
            with contextlib.suppress(Exception):
                await async_client.assistants.delete_file(
                    assistant_name=name, file_id=fid, timeout=30
                )
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# chat-stream-structure
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_chat_stream_message_start_and_end_structure_async(
    async_client: AsyncPinecone,
) -> None:
    """Async streaming chat: first chunk is StreamMessageStart; last is StreamMessageEnd with usage.

    Verifies:
    - unified-stream-0016: First chunk in a chat stream is a message-start chunk
      containing the model and role.
    - unified-stream-0018: Final chunk in a chat stream is a message-end chunk
      containing token usage statistics (prompt_tokens, completion_tokens, total_tokens).
    """
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        # Create assistant
        assistant = await async_client.assistants.create(
            name=name, instructions="You are a helpful assistant."
        )
        assert isinstance(assistant, AssistantModel)

        # Wait for Ready
        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Upload a small file so the assistant can respond
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-struct-"
        ) as f:
            f.write("Pinecone is a vector database for machine learning applications.")
            tmp_path = f.name

        await async_client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # Streaming chat — await chat() to get AsyncIterator[ChatStreamChunk]
        stream = await async_client.assistants.chat(
            assistant_name=name,
            messages=[{"role": "user", "content": "What is Pinecone?"}],
            stream=True,
        )

        # Consume the entire async stream
        chunks: list[ChatStreamChunk] = [chunk async for chunk in stream]

        assert len(chunks) >= 2, (
            f"Expected at least 2 chunks (message_start + message_end), got {len(chunks)}"
        )

        # First chunk must be StreamMessageStart with non-empty model and role
        first = chunks[0]
        assert isinstance(first, StreamMessageStart), (
            f"Expected first chunk to be StreamMessageStart, got {type(first).__name__}"
        )
        assert isinstance(first.model, str) and len(first.model) > 0, (
            f"StreamMessageStart.model must be a non-empty string, got {first.model!r}"
        )
        assert isinstance(first.role, str) and len(first.role) > 0, (
            f"StreamMessageStart.role must be a non-empty string, got {first.role!r}"
        )

        # Last chunk must be StreamMessageEnd with token usage statistics
        last = chunks[-1]
        assert isinstance(last, StreamMessageEnd), (
            f"Expected last chunk to be StreamMessageEnd, got {type(last).__name__}"
        )
        assert isinstance(last.usage.prompt_tokens, int) and last.usage.prompt_tokens >= 0, (
            f"prompt_tokens must be a non-negative int, got {last.usage.prompt_tokens!r}"
        )
        assert (
            isinstance(last.usage.completion_tokens, int) and last.usage.completion_tokens >= 0
        ), f"completion_tokens must be a non-negative int, got {last.usage.completion_tokens!r}"
        assert isinstance(last.usage.total_tokens, int) and last.usage.total_tokens > 0, (
            f"total_tokens must be a positive int, got {last.usage.total_tokens!r}"
        )

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# chat — json_response=True
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.timeout(300)
async def test_chat_json_response_mode_returns_valid_json_async(
    async_client: AsyncPinecone,
) -> None:
    """chat(json_response=True) returns a ChatResponse whose message.content is valid JSON (async).

    Verifies:
    - unified-chat-0038: JSON response mode returns valid JSON in the message content.
    """
    import json

    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        assistant = await async_client.assistants.create(
            name=name, instructions="You are a helpful assistant."
        )
        assert isinstance(assistant, AssistantModel)

        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-jsonresp-async-"
        ) as f:
            f.write("Pinecone is a managed vector database service.")
            tmp_path = f.name

        await async_client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # Request a JSON-format response with a prompt that strongly guides JSON output
        response = await async_client.assistants.chat(
            assistant_name=name,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Reply ONLY with a JSON object. "
                        "The object must have exactly one key called 'answer' whose value "
                        "is a string describing what Pinecone is in one sentence."
                    ),
                }
            ],
            json_response=True,
        )

        # Verify ChatResponse structure
        assert isinstance(response, ChatResponse), f"Expected ChatResponse, got {type(response)}"
        assert isinstance(response.message.content, str) and len(response.message.content) > 0, (
            f"message.content must be a non-empty string, got {response.message.content!r}"
        )

        # The content must be valid JSON — this is the core claim
        try:
            parsed = json.loads(response.message.content)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"chat(json_response=True) returned content that is not valid JSON: "
                f"{response.message.content!r}"
            ) from exc

        # Must be a JSON object (dict), not a list or primitive
        assert isinstance(parsed, dict), (
            f"Expected a JSON object, got {type(parsed).__name__}: {parsed!r}"
        )

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# chat — include_highlights
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
async def test_chat_include_highlights_async(async_client: AsyncPinecone) -> None:
    """chat(include_highlights=True) succeeds and returns correctly-typed highlight objects (async).

    Verifies:
    - unified-chat-0009: Can request highlight snippets to be included in citations.
    - unified-chat-0046: Each reference may include a highlight with type and content
      text when highlights are requested.
    - unified-chat-0047: The highlight field within a reference is absent (None) when
      highlights are not requested.
    """
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        assistant = await async_client.assistants.create(
            name=name, instructions="You are a helpful assistant."
        )
        assert isinstance(assistant, AssistantModel)

        await async_poll_until(
            query_fn=lambda: async_client.assistants.describe(name=name),
            check_fn=lambda r: r.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name} Ready",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-hlightsasync-"
        ) as f:
            f.write(
                "Pinecone is a managed vector database that stores dense and sparse embeddings. "
                "It supports serverless and pod-based deployment options."
            )
            tmp_path = f.name

        await async_client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # --- Call 1: include_highlights=True ---
        response_with = await async_client.assistants.chat(
            assistant_name=name,
            messages=[{"role": "user", "content": "What is Pinecone?"}],
            stream=False,
            include_highlights=True,
        )

        assert isinstance(response_with, ChatResponse), (
            f"Expected ChatResponse, got {type(response_with)}"
        )
        assert isinstance(response_with.citations, list), "citations must be a list"

        # For every reference returned, highlight must be None or a valid ChatHighlight
        for citation in response_with.citations:
            for ref in citation.references:
                assert isinstance(ref, ChatReference), f"Expected ChatReference, got {type(ref)}"
                if ref.highlight is not None:
                    assert isinstance(ref.highlight, ChatHighlight), (
                        f"highlight must be ChatHighlight or None, got {type(ref.highlight)}"
                    )
                    assert isinstance(ref.highlight.type, str) and len(ref.highlight.type) > 0, (
                        f"ChatHighlight.type must be a non-empty string, got {ref.highlight.type!r}"
                    )
                    assert (
                        isinstance(ref.highlight.content, str) and len(ref.highlight.content) > 0
                    ), (
                        f"ChatHighlight.content must be a non-empty string, "
                        f"got {ref.highlight.content!r}"
                    )

        # --- Call 2: default include_highlights (False) ---
        response_without = await async_client.assistants.chat(
            assistant_name=name,
            messages=[{"role": "user", "content": "What deployment options does Pinecone have?"}],
            stream=False,
        )

        assert isinstance(response_without, ChatResponse), (
            f"Expected ChatResponse, got {type(response_without)}"
        )
        # When highlights are not requested, all reference highlights must be None
        for citation in response_without.citations:
            for ref in citation.references:
                assert ref.highlight is None, (
                    f"Expected highlight=None when include_highlights not set, "
                    f"got {ref.highlight!r}"
                )

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# chat-context-options
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_chat_context_options_typed_and_dict_async(async_client: AsyncPinecone) -> None:
    """chat() accepts context_options as both a typed ContextOptions object and a plain dict (async).

    Verifies:
    - unified-chat-0025: Context options can be provided as either a typed options object
      or a plain dictionary.
    - unified-chat-0010: Can control context retrieval parameters via context options.
    """
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        assistant = await async_client.assistants.create(
            name=name, instructions="You are a helpful assistant."
        )
        assert isinstance(assistant, AssistantModel)

        await async_poll_until(
            query_fn=lambda: async_client.assistants.describe(name=name),
            check_fn=lambda r: r.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name} Ready",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-ctxoptasync-"
        ) as f:
            f.write(
                "Pinecone is a managed vector database for high-performance similarity search. "
                "It supports both serverless and pod-based deployment options."
            )
            tmp_path = f.name

        await async_client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # --- Call 1: typed ContextOptions object ---
        response_typed = await async_client.assistants.chat(
            assistant_name=name,
            messages=[{"role": "user", "content": "What is Pinecone?"}],
            stream=False,
            context_options=ContextOptions(top_k=5),
        )

        assert isinstance(response_typed, ChatResponse), (
            f"Expected ChatResponse with typed ContextOptions, got {type(response_typed)}"
        )
        assert isinstance(response_typed.message, ChatMessage), (
            f"ChatResponse.message must be ChatMessage, got {type(response_typed.message)}"
        )
        assert isinstance(response_typed.message.content, str), (
            f"ChatMessage.content must be str, got {type(response_typed.message.content)}"
        )
        assert len(response_typed.message.content) > 0, "ChatMessage.content must be non-empty"

        # --- Call 2: plain dict context_options ---
        response_dict = await async_client.assistants.chat(
            assistant_name=name,
            messages=[{"role": "user", "content": "What deployment options does Pinecone have?"}],
            stream=False,
            context_options={"top_k": 5},
        )

        assert isinstance(response_dict, ChatResponse), (
            f"Expected ChatResponse with dict context_options, got {type(response_dict)}"
        )
        assert isinstance(response_dict.message, ChatMessage), (
            f"ChatResponse.message must be ChatMessage, got {type(response_dict.message)}"
        )
        assert isinstance(response_dict.message.content, str), (
            f"ChatMessage.content must be str, got {type(response_dict.message.content)}"
        )
        assert len(response_dict.message.content) > 0, "ChatMessage.content must be non-empty"

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# chat-completions-full-structure — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_chat_completions_full_response_structure_async(
    async_client: AsyncPinecone,
) -> None:
    """Verify the complete ChatCompletionResponse structure from async chat_completions().

    Verifies:
    - unified-chat-0042: non-streaming chat_completions response has NO citations field
    - unified-chat-0048: response contains id (str), choices list, model (str), usage
    - unified-chat-0049: each choice contains an index (int), message, and finish_reason
    - unified-chat-0050: usage includes prompt_tokens, completion_tokens, total_tokens

    Area tag: assistant-chat-completions
    Transport: rest-async
    """
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        assistant = await async_client.assistants.create(
            name=name, instructions="You are a helpful assistant."
        )
        assert isinstance(assistant, AssistantModel)

        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-cc-full-async-"
        ) as f:
            f.write("Pinecone is a managed vector database service for AI applications.")
            tmp_path = f.name

        await async_client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        response = await async_client.assistants.chat_completions(
            assistant_name=name,
            messages=[{"role": "user", "content": "What is Pinecone used for?"}],
            stream=False,
        )

        # --- unified-chat-0042: NO citations field on ChatCompletionResponse ---
        assert isinstance(response, ChatCompletionResponse), (
            f"Expected ChatCompletionResponse, got {type(response)}"
        )
        assert not hasattr(response, "citations"), (
            "ChatCompletionResponse must NOT have a citations field (OpenAI-compatible format)"
        )

        # --- unified-chat-0048: response has id, choices, model, usage ---
        assert isinstance(response.id, str) and len(response.id) > 0, (
            f"response.id must be a non-empty str, got {response.id!r}"
        )
        assert isinstance(response.model, str) and len(response.model) > 0, (
            f"response.model must be a non-empty str, got {response.model!r}"
        )
        assert isinstance(response.choices, list) and len(response.choices) > 0, (
            "response.choices must be a non-empty list"
        )
        assert isinstance(response.usage, ChatUsage), (
            f"response.usage must be ChatUsage, got {type(response.usage)}"
        )

        # --- unified-chat-0049: each choice has index, message, finish_reason ---
        for choice in response.choices:
            assert isinstance(choice, ChatCompletionChoice), (
                f"Each choice must be ChatCompletionChoice, got {type(choice)}"
            )
            assert isinstance(choice.index, int), (
                f"choice.index must be int, got {type(choice.index)}"
            )
            assert isinstance(choice.message, ChatCompletionMessage), (
                f"choice.message must be ChatCompletionMessage, got {type(choice.message)}"
            )
            assert choice.message.content is not None and len(choice.message.content) > 0, (
                "choice.message.content must be a non-empty str"
            )
            assert isinstance(choice.finish_reason, str) and len(choice.finish_reason) > 0, (
                "choice.finish_reason must be a non-empty str"
            )

        # --- unified-chat-0050: usage has prompt_tokens, completion_tokens, total_tokens ---
        usage = response.usage
        assert isinstance(usage.prompt_tokens, int) and usage.prompt_tokens >= 0, (
            f"usage.prompt_tokens must be non-negative int, got {usage.prompt_tokens!r}"
        )
        assert isinstance(usage.completion_tokens, int) and usage.completion_tokens >= 0, (
            f"usage.completion_tokens must be non-negative int, got {usage.completion_tokens!r}"
        )
        assert isinstance(usage.total_tokens, int) and usage.total_tokens > 0, (
            f"usage.total_tokens must be a positive int, got {usage.total_tokens!r}"
        )
        assert usage.prompt_tokens + usage.completion_tokens == usage.total_tokens, (
            f"prompt_tokens ({usage.prompt_tokens}) + completion_tokens "
            f"({usage.completion_tokens}) should equal total_tokens ({usage.total_tokens})"
        )

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-evaluate per-fact structure and usage token counts
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_evaluate_alignment_per_fact_structure_and_usage_tokens_async(
    async_client: AsyncPinecone,
) -> None:
    """evaluate_alignment() returns EntailmentResult facts with proper field types
    and ChatUsage with non-negative integer token counts (async variant).

    Verifies:
    - unified-eval-0003: Each fact in AlignmentResult.facts is an EntailmentResult
      with a non-empty 'fact' string, an 'entailment' value in
      {"entailed", "contradicted", "neutral"}, and a 'reasoning' string.
    - unified-eval-0004: AlignmentResult.usage is a ChatUsage with non-negative
      integer prompt_tokens, completion_tokens, and total_tokens, where
      prompt_tokens + completion_tokens == total_tokens.

    evaluate_alignment is stateless — no assistant or file is needed.
    """
    # Use a statement about a well-known fact so at least one "entailed" fact
    # is produced, giving the test something concrete to check.
    result = await async_client.assistants.evaluate_alignment(
        question="What is the boiling point of water at sea level?",
        answer="Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
        ground_truth_answer="The boiling point of water at standard atmospheric pressure is 100°C.",
    )

    assert isinstance(result, AlignmentResult)

    # --- unified-eval-0003: per-fact EntailmentResult structure ---
    assert isinstance(result.facts, list), "facts must be a list"
    assert len(result.facts) >= 1, (
        "At least one EntailmentResult fact expected for a well-defined question/answer pair"
    )
    valid_entailments = {"entailed", "contradicted", "neutral"}
    for fact_item in result.facts:
        assert isinstance(fact_item, EntailmentResult), (
            f"Each fact must be an EntailmentResult, got {type(fact_item)}"
        )
        assert isinstance(fact_item.fact, str) and len(fact_item.fact) > 0, (
            f"EntailmentResult.fact must be a non-empty string, got {fact_item.fact!r}"
        )
        assert fact_item.entailment in valid_entailments, (
            f"EntailmentResult.entailment must be one of {valid_entailments!r}, "
            f"got {fact_item.entailment!r}"
        )
        assert isinstance(fact_item.reasoning, str), (
            f"EntailmentResult.reasoning must be a string, got {type(fact_item.reasoning)}"
        )

    # --- unified-eval-0004: ChatUsage token counts ---
    assert result.usage is not None, "AlignmentResult.usage must not be None"
    usage = result.usage
    assert isinstance(usage, ChatUsage), (
        f"AlignmentResult.usage must be a ChatUsage, got {type(usage)}"
    )
    assert isinstance(usage.prompt_tokens, int) and usage.prompt_tokens >= 0, (
        f"usage.prompt_tokens must be a non-negative int, got {usage.prompt_tokens!r}"
    )
    assert isinstance(usage.completion_tokens, int) and usage.completion_tokens >= 0, (
        f"usage.completion_tokens must be a non-negative int, got {usage.completion_tokens!r}"
    )
    assert isinstance(usage.total_tokens, int) and usage.total_tokens > 0, (
        f"usage.total_tokens must be a positive int, got {usage.total_tokens!r}"
    )
    assert usage.prompt_tokens + usage.completion_tokens == usage.total_tokens, (
        f"prompt_tokens ({usage.prompt_tokens}) + completion_tokens "
        f"({usage.completion_tokens}) must equal total_tokens ({usage.total_tokens})"
    )


# ---------------------------------------------------------------------------
# chat with explicit model and temperature parameters (async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_chat_with_model_and_temperature_async(
    async_client: AsyncPinecone,
) -> None:
    """chat() accepts explicit model and temperature parameters and returns a valid response.

    Verifies:
    - unified-chat-0005: Can specify a language model for chat requests
    - unified-chat-0006: Can specify a temperature parameter to control randomness

    Async counterpart of test_chat_with_model_and_temperature_rest.
    """
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        assistant = await async_client.assistants.create(
            name=name, instructions="You are a helpful assistant."
        )
        assert isinstance(assistant, AssistantModel)

        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-mdl-"
        ) as f:
            f.write("Pinecone is a managed vector database supporting dense and sparse vectors.")
            tmp_path = f.name

        await async_client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        response = await async_client.assistants.chat(
            assistant_name=name,
            messages=[{"role": "user", "content": "What is Pinecone?"}],
            model="gpt-4o",
            temperature=0.7,
            stream=False,
        )

        assert isinstance(response, ChatResponse)
        assert isinstance(response.id, str) and len(response.id) > 0
        assert isinstance(response.model, str) and len(response.model) > 0
        assert "gpt-4o" in response.model, (
            f"Expected response.model to contain 'gpt-4o', got {response.model!r}"
        )
        assert response.message.role == "assistant"
        assert isinstance(response.message.content, str) and len(response.message.content) > 0

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistants.list_page() — explicit single-page pagination (async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.timeout(300)
async def test_assistants_list_page_response_structure_async(
    async_client: AsyncPinecone,
) -> None:
    """assistants.list_page() returns a ListAssistantsResponse with correct structure (async).

    Verifies:
    - unified-assistant-0007: Can list one page of assistants with explicit pagination control.

    Async counterpart of test_assistants_list_page_response_structure_rest.
    """
    from pinecone.models.assistant.list import ListAssistantsResponse

    name = unique_name("asst")
    try:
        # Create an assistant to ensure at least one exists
        assistant = await async_client.assistants.create(
            name=name, instructions="You are a helpful assistant."
        )
        assert isinstance(assistant, AssistantModel)

        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Call list_page() — single page, no explicit page_size
        page = await async_client.assistants.list_page()

        # Response type
        assert isinstance(page, ListAssistantsResponse), (
            f"Expected ListAssistantsResponse, got {type(page)}"
        )

        # assistants is a list
        assert isinstance(page.assistants, list), (
            f"Expected page.assistants to be a list, got {type(page.assistants)}"
        )

        # next is None or a string
        assert page.next is None or isinstance(page.next, str), (
            f"Expected page.next to be None or str, got {type(page.next)}"
        )

        # Each item in the list is an AssistantModel with expected fields
        for a in page.assistants:
            assert isinstance(a, AssistantModel), f"Expected AssistantModel, got {type(a)}"
            assert isinstance(a.name, str) and len(a.name) > 0, (
                f"AssistantModel.name must be a non-empty string, got {a.name!r}"
            )
            assert isinstance(a.status, str), (
                f"AssistantModel.status must be a string, got {type(a.status)}"
            )

        # The created assistant must appear in the result
        names_in_page = [a.name for a in page.assistants]
        assert name in names_in_page, (
            f"Expected newly created assistant {name!r} in list_page() result; got: {names_in_page}"
        )

    finally:
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# upload_file with caller-specified file_id — upsert behavior — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_upload_file_with_caller_specified_file_id_async(
    async_client: AsyncPinecone,
) -> None:
    """async upload_file(file_id=...) assigns the caller-specified ID; re-uploading with the same
    file_id replaces the file (upsert semantics).

    Async transport parity for test_upload_file_with_caller_specified_file_id_rest.

    Verifies:
    - unified-file-0005: Can upload a file with a caller-specified file identifier for upsert behavior
    - unified-file-0006: When a file identifier is provided, the upload creates the file if it
      does not exist or replaces it if it does
    """
    name = unique_name("asst")
    # file_id must be 1-128 chars, alphanumeric/hyphens/underscores
    custom_file_id = unique_name("fid")
    file_id: str | None = None

    try:
        assistant = await async_client.assistants.create(
            name=name, instructions="Async file upsert test assistant."
        )
        assert isinstance(assistant, AssistantModel)

        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # --- First upload: create with caller-specified file_id ---
        first_content = b"Initial content for async caller-specified file ID test."
        first_upload = await async_client.assistants.upload_file(
            assistant_name=name,
            file_stream=io.BytesIO(first_content),
            file_name="caller-id-test-async.txt",
            file_id=custom_file_id,
            timeout=120,
        )
        assert isinstance(first_upload, AssistantFileModel)
        # The server must honor the caller-specified file_id
        assert first_upload.id == custom_file_id, (
            f"Expected file_id {custom_file_id!r} to be preserved; got {first_upload.id!r}"
        )
        file_id = first_upload.id

        # Wait until Available
        await async_poll_until(
            lambda: async_client.assistants.describe_file(assistant_name=name, file_id=file_id),
            lambda f: f.status in ("Available", "Processed"),
            timeout=120,
            interval=5,
            description=f"first upload of file {file_id}",
        )

        described_first = await async_client.assistants.describe_file(
            assistant_name=name, file_id=file_id
        )
        first_size = described_first.size

        # --- Second upload: upsert with same file_id, different (larger) content ---
        second_content = (
            b"Replacement content for async caller-specified file ID upsert test. " + b"x" * 500
        )
        second_upload = await async_client.assistants.upload_file(
            assistant_name=name,
            file_stream=io.BytesIO(second_content),
            file_name="caller-id-test-v2-async.txt",
            file_id=custom_file_id,
            timeout=120,
        )
        assert isinstance(second_upload, AssistantFileModel)
        # After upsert the file_id must remain the same
        assert second_upload.id == custom_file_id, (
            f"After upsert, expected file_id {custom_file_id!r}; got {second_upload.id!r}"
        )

        # Wait until Available again
        await async_poll_until(
            lambda: async_client.assistants.describe_file(assistant_name=name, file_id=file_id),
            lambda f: f.status in ("Available", "Processed"),
            timeout=120,
            interval=5,
            description=f"second upload (upsert) of file {file_id}",
        )

        described_second = await async_client.assistants.describe_file(
            assistant_name=name, file_id=file_id
        )
        second_size = described_second.size

        # The upsert replaced the file — size must differ from the first upload
        assert second_size != first_size, (
            f"Expected file size to change after upsert replacement; "
            f"first={first_size}, second={second_size}"
        )

        # Clean up the file before assistant deletion
        await async_client.assistants.delete_file(assistant_name=name, file_id=file_id, timeout=60)
        file_id = None

    finally:
        if file_id is not None:
            with contextlib.suppress(Exception):
                await async_client.assistants.delete_file(
                    assistant_name=name, file_id=file_id, timeout=60
                )
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# AssistantModel dict-like mixin operations — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.timeout(300)
async def test_assistant_model_dict_mixin_operations_async(
    async_client: AsyncPinecone,
) -> None:
    """AssistantModel StructDictMixin methods work correctly on real API-deserialized objects (async).

    Async mirror of test_assistant_model_dict_mixin_operations_rest.

    Verifies:
    - unified-model-0003: len(model) returns the number of declared fields
    - unified-model-0004: model.keys(), model.values(), model.items() expose all fields
    - unified-model-0005: model.get(key, default) returns value or default
    - unified-model-0006: model.to_dict() recursively converts to a plain dict

    AssistantModel declares 7 fields: name, status, metadata, instructions,
    host, created_at, updated_at.
    """
    name = unique_name("asst")
    try:
        assistant = await async_client.assistants.create(
            name=name,
            instructions="Test mixin ops async.",
        )
        assert isinstance(assistant, AssistantModel)

        # Wait for Ready so describe() returns a stable model
        await async_poll_until(
            query_fn=lambda: async_client.assistants.describe(name=name),
            check_fn=lambda m: m.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name!r} ready",
        )

        model = await async_client.assistants.describe(name=name)
        assert isinstance(model, AssistantModel)

        # --- unified-model-0003: len(model) returns field count ---
        assert len(model) == 7, f"AssistantModel has 7 declared fields; got len()={len(model)}"

        # --- unified-model-0004: keys(), values(), items() ---
        keys = model.keys()
        assert isinstance(keys, tuple), f"keys() should return tuple, got {type(keys).__name__}"
        assert "name" in keys, "keys() must include 'name'"
        assert "status" in keys, "keys() must include 'status'"
        assert "metadata" in keys, "keys() must include 'metadata'"
        assert "instructions" in keys, "keys() must include 'instructions'"
        assert len(keys) == 7, f"keys() length should be 7, got {len(keys)}"

        values = model.values()
        assert isinstance(values, list), f"values() should return list, got {type(values).__name__}"
        assert len(values) == 7, f"values() length should be 7, got {len(values)}"
        assert values[0] == model.name, "values()[0] (name) should match model.name"
        assert values[1] == model.status, "values()[1] (status) should match model.status"

        items = model.items()
        assert isinstance(items, list), f"items() should return list, got {type(items).__name__}"
        assert len(items) == 7, f"items() length should be 7, got {len(items)}"
        items_dict = dict(items)
        assert items_dict["name"] == model.name, "items() 'name' should match model.name"
        assert items_dict["status"] == model.status, "items() 'status' should match model.status"

        # --- unified-model-0005: get() with and without default ---
        assert model.get("name") == model.name, "get('name') should return model.name"
        assert model.get("status") == model.status, "get('status') should return model.status"
        assert model.get("totally_nonexistent_field") is None, (
            "get() for unknown field should default to None"
        )
        assert model.get("totally_nonexistent_field", "sentinel") == "sentinel", (
            "get() for unknown field should return the specified default"
        )

        # --- unified-model-0006: to_dict() returns a plain dict ---
        d = model.to_dict()
        assert isinstance(d, dict), f"to_dict() should return dict, got {type(d).__name__}"
        assert "name" in d and d["name"] == model.name, (
            f"to_dict()['name'] should equal model.name={model.name!r}"
        )
        assert "status" in d and d["status"] == model.status, (
            f"to_dict()['status'] should equal model.status={model.status!r}"
        )
        assert d["metadata"] is None or isinstance(d["metadata"], dict), (
            f"to_dict()['metadata'] should be None or plain dict, "
            f"got {type(d['metadata']).__name__}"
        )

    finally:
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# chat-completions-streaming-finish-reason (async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.timeout(300)
async def test_chat_completions_streaming_finish_reason_async(
    async_client: AsyncPinecone,
) -> None:
    """chat_completions(stream=True) final chunk has finish_reason set; content chunks have None (async).

    Verifies:
    - unified-stream-0021: Each chat completion streaming chunk contains a list of choices
      with index, delta message, and finish reason. (async transport)
    - Content chunks (those carrying delta.content) have finish_reason == None.
    - At least one chunk has a non-None finish_reason (the terminal/final chunk).
    - The terminal finish_reason is a non-empty string (e.g. "stop").
    """
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        assistant = await async_client.assistants.create(
            name=name, instructions="You are a helpful assistant."
        )
        assert isinstance(assistant, AssistantModel)

        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-finish-async-"
        ) as f:
            f.write("Pinecone is a managed vector database.")
            tmp_path = f.name

        await async_client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        stream = await async_client.assistants.chat_completions(
            assistant_name=name,
            messages=[{"role": "user", "content": "What is Pinecone in one sentence?"}],
            stream=True,
        )

        chunks: list[ChatCompletionStreamChunk] = [chunk async for chunk in stream]
        assert len(chunks) > 0, "Expected at least one streaming chunk"

        # Flatten all choices across all chunks for analysis
        finish_reason_chunks: list[str] = []
        for chunk in chunks:
            for choice in chunk.choices:
                # finish_reason must be None or a string — never another type
                assert choice.finish_reason is None or isinstance(choice.finish_reason, str), (
                    f"finish_reason must be str or None, got {type(choice.finish_reason)}"
                )
                # Content-bearing choices must NOT carry a finish_reason
                if choice.delta.content is not None and choice.delta.content != "":
                    assert choice.finish_reason is None, (
                        f"Content chunk should have finish_reason=None, "
                        f"got {choice.finish_reason!r}"
                    )
                if choice.finish_reason is not None:
                    finish_reason_chunks.append(choice.finish_reason)

        # At least one chunk must have a non-None finish_reason (the terminal chunk)
        assert len(finish_reason_chunks) > 0, (
            "Expected at least one chunk with a non-None finish_reason (e.g. 'stop'), "
            f"but all {len(chunks)} chunks had finish_reason=None"
        )
        # The terminal finish_reason must be a non-empty string
        for fr in finish_reason_chunks:
            assert isinstance(fr, str) and len(fr) > 0, (
                f"finish_reason must be a non-empty string, got {fr!r}"
            )

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-name-length — boundary validation (async)
# ---------------------------------------------------------------------------


# Path to test fixture files (same directory as this test file).
_FIXTURES_DIR = os.path.dirname(os.path.realpath(__file__))

# All assistant chat models supported by the API as of 2025-10.
_SUPPORTED_CHAT_MODELS = (
    "gpt-4o",
    "gpt-4.1",
    "o4-mini",
    "claude-3-5-sonnet",
    "claude-3-7-sonnet",
    "gemini-2.5-pro",
)


def _padded_name(length: int) -> str:
    """Return a uniquely-prefixed name padded to exactly ``length`` chars using
    valid characters (lowercase alphanumerics and hyphens)."""
    base = unique_name("a")
    suffix = "0123456789abcdef" * ((length // 16) + 2)
    name = (base + suffix)[:length]
    assert len(name) == length, f"wanted {length}-char name, got {len(name)}"
    return name


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.timeout(180)
async def test_assistant_create_accepts_max_length_name(async_client: AsyncPinecone) -> None:
    """create() accepts a name at the documented max length of 63 characters (async)."""
    name = _padded_name(63)
    try:
        assistant = await async_client.assistants.create(name=name, timeout=-1)
        assert isinstance(assistant, AssistantModel)
        assert assistant.name == name
        assert assistant.status in ("Initializing", "Ready")
    finally:
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.timeout(60)
async def test_assistant_create_rejects_name_over_max_length(
    async_client: AsyncPinecone,
) -> None:
    """create() rejects names longer than the documented max of 63 characters (async)."""
    name = _padded_name(65)
    created = False
    try:
        with pytest.raises((ApiError, PineconeValueError)):
            await async_client.assistants.create(name=name, timeout=-1)
    except Exception:
        created = True
        raise
    finally:
        if created:
            await async_cleanup_resource(
                lambda: async_client.assistants.delete(name=name, timeout=60),
                name,
                "assistant",
            )


# ---------------------------------------------------------------------------
# environment parameter (async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.timeout(30)
async def test_create_assistant_environment_rejected(async_client: AsyncPinecone) -> None:
    """create() with environment= raises ApiError 400 INVALID_ARGUMENT or 403 non-internal-plan (async).

    Confirms the environment parameter is wired through to the backend:
    the backend returns 400 INVALID_ARGUMENT or 403 when the calling org is not an internal plan.
    """
    name = unique_name("env-test")
    with pytest.raises(ApiError) as exc_info:
        await async_client.assistants.create(name=name, environment="prod-us", timeout=-1)
    assert exc_info.value.status_code in (400, 403)
    if exc_info.value.status_code == 400:
        assert exc_info.value.error_code == "INVALID_ARGUMENT"


# ---------------------------------------------------------------------------
# chat-context-options boundary validation (async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.timeout(300)
async def test_chat_context_options_boundary_validation(
    async_client: AsyncPinecone,
) -> None:
    """chat() rejects context_options with out-of-range snippet_size and top_k (async)."""
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        await async_client.assistants.create(name=name, instructions="You are a helpful assistant.")
        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-ctxbound-a-"
        ) as f:
            f.write("Pinecone is a managed vector database for semantic search.")
            tmp_path = f.name
        await async_client.assistants.upload_file(
            assistant_name=name, file_path=tmp_path, timeout=120
        )

        msgs = [{"role": "user", "content": "What is Pinecone?"}]

        with pytest.raises((ApiError, PineconeError, PineconeValueError)):
            await async_client.assistants.chat(
                assistant_name=name,
                messages=msgs,
                context_options=ContextOptions(top_k=2, snippet_size=3),
                stream=False,
            )

        with pytest.raises((ApiError, PineconeError, PineconeValueError)):
            await async_client.assistants.chat(
                assistant_name=name,
                messages=msgs,
                context_options=ContextOptions(top_k=100, snippet_size=512),
                stream=False,
            )

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# upload_file — malformed PDF error (async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.timeout(300)
async def test_upload_file_rejects_malformed_pdf(async_client: AsyncPinecone) -> None:
    """upload_file() surfaces a processing-failed error when the PDF is malformed (async)."""
    name = unique_name("asst")
    pdf_path = os.path.join(_FIXTURES_DIR, "malformed.pdf")
    assert os.path.isfile(pdf_path), f"fixture missing: {pdf_path}"

    try:
        await async_client.assistants.create(name=name, instructions="Malformed PDF test.")
        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with pytest.raises((PineconeError, ApiError)):
            await async_client.assistants.upload_file(
                assistant_name=name,
                file_path=pdf_path,
                timeout=180,
            )

    finally:
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# chat — multi-model matrix and invalid-model rejection (async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.timeout(600)
async def test_chat_across_all_supported_models_and_rejects_invalid(
    async_client: AsyncPinecone,
) -> None:
    """chat() succeeds for every documented model and rejects an invalid model name (async)."""
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        await async_client.assistants.create(name=name, instructions="You are a helpful assistant.")
        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-models-a-"
        ) as f:
            f.write(
                "Pinecone is a managed vector database for AI apps. "
                "It supports dense, sparse, and hybrid similarity search."
            )
            tmp_path = f.name
        await async_client.assistants.upload_file(
            assistant_name=name, file_path=tmp_path, timeout=120
        )

        msgs = [{"role": "user", "content": "What is Pinecone?"}]

        for model_name in _SUPPORTED_CHAT_MODELS:
            response = await async_client.assistants.chat(
                assistant_name=name,
                messages=msgs,
                model=model_name,
                stream=False,
            )
            assert isinstance(response, ChatResponse), (
                f"model={model_name!r} did not return ChatResponse"
            )
            assert isinstance(response.message.content, str)
            assert len(response.message.content) > 0, f"model={model_name!r} returned empty content"

        with pytest.raises((ApiError, PineconeError, PineconeValueError)):
            await async_client.assistants.chat(
                assistant_name=name,
                messages=msgs,
                model="definitely-not-a-real-model",
                stream=False,
            )

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# chat — out-of-range temperature (async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.timeout(300)
async def test_chat_rejects_out_of_range_temperature(
    async_client: AsyncPinecone,
) -> None:
    """chat() rejects temperature values outside the supported range (async)."""
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        await async_client.assistants.create(name=name, instructions="You are a helpful assistant.")
        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-temp-a-"
        ) as f:
            f.write("Pinecone is a managed vector database.")
            tmp_path = f.name
        await async_client.assistants.upload_file(
            assistant_name=name, file_path=tmp_path, timeout=120
        )

        msgs = [{"role": "user", "content": "What is Pinecone?"}]

        with pytest.raises((ApiError, PineconeError, PineconeValueError)):
            await async_client.assistants.chat(
                assistant_name=name,
                messages=msgs,
                temperature=3.0,
                stream=False,
            )

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# context — filter parameter (async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.timeout(600)
async def test_context_filter_metadata_excludes_matching_files(
    async_client: AsyncPinecone,
) -> None:
    """context(filter=...) restricts snippets to files whose metadata matches the filter (async)."""
    name = unique_name("asst")
    tmp_keep: str | None = None
    tmp_skip: str | None = None
    try:
        await async_client.assistants.create(name=name, instructions="Filter test assistant.")
        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-flt-a-"
        ) as f:
            f.write("Pinecone is a managed vector database for AI applications.")
            tmp_keep = f.name
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-flt-b-"
        ) as f:
            f.write("Pinecone supports dense and sparse vectors for semantic search.")
            tmp_skip = f.name

        file_a = await async_client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_keep,
            metadata={"company": "anthropic"},
            timeout=120,
        )
        file_b = await async_client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_skip,
            metadata={"company": "openai"},
            timeout=120,
        )

        response = await async_client.assistants.context(
            assistant_name=name,
            query="What is Pinecone?",
            filter={"company": {"$ne": "anthropic"}},
        )
        assert isinstance(response, ContextResponse)
        for snippet in response.snippets:
            assert snippet.reference.file.id == file_b.id, (
                f"Expected only file_b ({file_b.id}); got {snippet.reference.file.id}"
            )

        response_flip = await async_client.assistants.context(
            assistant_name=name,
            query="What is Pinecone?",
            filter={"company": {"$ne": "openai"}},
        )
        for snippet in response_flip.snippets:
            assert snippet.reference.file.id == file_a.id

    finally:
        for p in (tmp_keep, tmp_skip):
            if p is not None:
                with contextlib.suppress(Exception):
                    os.unlink(p)
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-lifecycle — explicit status transitions (async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.timeout(300)
async def test_assistant_lifecycle_status_transitions_explicit(
    async_client: AsyncPinecone,
) -> None:
    """Verify async assistant explicitly moves through Initializing → Ready → Terminating."""
    name = unique_name("asst")
    deleted = False
    try:
        assistant = await async_client.assistants.create(name=name, timeout=-1)
        assert isinstance(assistant, AssistantModel)
        assert assistant.status == "Initializing", (
            f"After create (timeout=-1), expected 'Initializing'; got {assistant.status!r}"
        )

        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=180,
            interval=3,
            description=f"assistant {name} to become Ready",
        )
        ready = await async_client.assistants.describe(name=name)
        assert ready.status == "Ready"
        assert ready.host is not None

        await async_client.assistants.delete(name=name, timeout=-1)
        deleted = True

        try:
            terminating = await async_client.assistants.describe(name=name)
            assert terminating.status == "Terminating", (
                f"After non-blocking delete, expected 'Terminating'; got {terminating.status!r}"
            )
        except ApiError as exc:
            assert exc.status_code in (404, 410), (
                f"Expected 404/410 after delete; got {exc.status_code}"
            )

    finally:
        if not deleted:
            await async_cleanup_resource(
                lambda: async_client.assistants.delete(name=name, timeout=60),
                name,
                "assistant",
            )


# ---------------------------------------------------------------------------
# assistant-update — combined instructions + metadata change (async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.timeout(180)
async def test_assistant_update_instructions_and_metadata_together(
    async_client: AsyncPinecone,
) -> None:
    """update() applies instructions and metadata changes from a single call (async)."""
    name = unique_name("asst")
    try:
        await async_client.assistants.create(
            name=name,
            instructions="Initial instructions.",
            metadata={"original_key": "original_value"},
        )
        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        updated = await async_client.assistants.update(
            name=name,
            instructions="Updated instructions after combined update.",
            metadata={"new_key": "new_value"},
        )
        assert isinstance(updated, AssistantModel)
        assert updated.instructions == "Updated instructions after combined update."
        assert updated.metadata == {"new_key": "new_value"}

        re_described = await async_client.assistants.describe(name=name)
        assert re_described.instructions == "Updated instructions after combined update."
        assert re_described.metadata is not None
        assert re_described.metadata.get("new_key") == "new_value"
        assert "original_key" not in re_described.metadata

    finally:
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# describe_file — DOCX signed URL (async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.timeout(300)
async def test_describe_file_docx_with_signed_url(async_client: AsyncPinecone) -> None:
    """describe_file(include_url=True) returns a usable signed_url for a DOCX file (async)."""
    name = unique_name("asst")
    docx_path = os.path.join(_FIXTURES_DIR, "test_doc.docx")
    assert os.path.isfile(docx_path), f"fixture missing: {docx_path}"
    file_id: str | None = None

    try:
        await async_client.assistants.create(name=name, instructions="DOCX signed-URL test.")
        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        file_model = await async_client.assistants.upload_file(
            assistant_name=name,
            file_path=docx_path,
            timeout=180,
        )
        assert isinstance(file_model, AssistantFileModel)
        file_id = file_model.id

        with_url = await async_client.assistants.describe_file(
            assistant_name=name, file_id=file_id, include_url=True
        )
        assert isinstance(with_url, AssistantFileModel)
        assert with_url.name == "test_doc.docx"
        assert with_url.signed_url is not None, "expected signed_url for DOCX"
        assert isinstance(with_url.signed_url, str) and len(with_url.signed_url) > 0

        without_url = await async_client.assistants.describe_file(
            assistant_name=name, file_id=file_id
        )
        assert without_url.signed_url is None

    finally:
        if file_id is not None:
            with contextlib.suppress(Exception):
                await async_client.assistants.delete_file(
                    assistant_name=name, file_id=file_id, timeout=60
                )
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# describe_file — metadata persistence (async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.timeout(300)
async def test_describe_file_preserves_uploaded_metadata(
    async_client: AsyncPinecone,
) -> None:
    """describe_file returns the exact metadata dict that was supplied at upload (async)."""
    name = unique_name("asst")
    tmp_path: str | None = None
    file_id: str | None = None
    try:
        await async_client.assistants.create(name=name, instructions="Metadata persistence test.")
        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-meta-a-"
        ) as f:
            f.write("Pinecone content used to verify metadata persistence.")
            tmp_path = f.name

        metadata: dict[str, object] = {
            "source": "integration-test",
            "category": "sdk-docs",
            "priority": "high",
        }
        upload = await async_client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            metadata=metadata,
            timeout=180,
        )
        assert isinstance(upload, AssistantFileModel)
        file_id = upload.id

        described = await async_client.assistants.describe_file(
            assistant_name=name, file_id=file_id
        )
        assert isinstance(described, AssistantFileModel)
        assert described.metadata is not None, "expected metadata on describe_file"
        assert described.metadata.get("source") == "integration-test"
        assert described.metadata.get("category") == "sdk-docs"
        assert described.metadata.get("priority") == "high"

    finally:
        if file_id is not None:
            with contextlib.suppress(Exception):
                await async_client.assistants.delete_file(
                    assistant_name=name, file_id=file_id, timeout=60
                )
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# multimodal PDF — image blocks, binary toggle, text fallback, errors (async)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.timeout(600)
async def test_multimodal_pdf_context_image_text_and_errors(
    async_client: AsyncPinecone,
) -> None:
    """Full multimodal surface for the async client against multimodal_sample.pdf."""
    name = unique_name("asst")
    pdf_path = os.path.join(_FIXTURES_DIR, "multimodal_sample.pdf")
    docx_path = os.path.join(_FIXTURES_DIR, "test_doc.docx")
    assert os.path.isfile(pdf_path), f"fixture missing: {pdf_path}"
    assert os.path.isfile(docx_path), f"fixture missing: {docx_path}"

    try:
        await async_client.assistants.create(name=name, instructions="Multimodal test assistant.")
        await async_poll_until(
            lambda: async_client.assistants.describe(name=name),
            lambda a: a.status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # 1. Upload multimodal PDF
        file_model = await async_client.assistants.upload_file(
            assistant_name=name,
            file_path=pdf_path,
            multimodal=True,
            timeout=240,
        )
        assert isinstance(file_model, AssistantFileModel)
        assert file_model.multimodal is True

        query = "What does this document show in its diagrams?"

        # 2. Default multimodal context
        res = await async_client.assistants.context(
            assistant_name=name, query=query, top_k=1, snippet_size=4000
        )
        assert isinstance(res, ContextResponse)
        assert len(res.snippets) == 1
        snippet = res.snippets[0]
        assert isinstance(snippet, MultimodalSnippet)
        image_blocks_with_data = [
            b
            for b in snippet.content
            if isinstance(b, ContextImageBlock) and b.image_data is not None
        ]
        assert len(image_blocks_with_data) > 0, (
            "expected at least one ContextImageBlock with image_data populated"
        )

        # 3. include_binary_content=False → image_data stripped
        res_no_binary = await async_client.assistants.context(
            assistant_name=name,
            query=query,
            top_k=1,
            snippet_size=4000,
            include_binary_content=False,
        )
        snippet_no_binary = res_no_binary.snippets[0]
        assert isinstance(snippet_no_binary, MultimodalSnippet)
        image_blocks = [b for b in snippet_no_binary.content if isinstance(b, ContextImageBlock)]
        assert len(image_blocks) > 0
        for block in image_blocks:
            assert block.image_data is None

        # 4. multimodal=False → TextSnippet fallback
        res_text = await async_client.assistants.context(
            assistant_name=name,
            query=query,
            top_k=1,
            snippet_size=4000,
            multimodal=False,
        )
        text_snippet = res_text.snippets[0]
        assert isinstance(text_snippet, TextSnippet)
        assert len(text_snippet.content) > 0

        # 5. chat() with multimodal context_options
        response = await async_client.assistants.chat(
            assistant_name=name,
            messages=[{"role": "user", "content": query}],
            context_options={"top_k": 1, "snippet_size": 4000},
            stream=False,
        )
        assert isinstance(response, ChatResponse)
        assert len(response.message.content) > 0

        # 6. multimodal=True on non-PDF → rejected
        with pytest.raises((ApiError, PineconeError, PineconeValueError)):
            await async_client.assistants.upload_file(
                assistant_name=name,
                file_path=docx_path,
                multimodal=True,
                timeout=120,
            )

        # 7. context(multimodal=False, include_binary_content=True) → rejected
        with pytest.raises((ApiError, PineconeError, PineconeValueError)):
            await async_client.assistants.context(
                assistant_name=name,
                query=query,
                top_k=1,
                snippet_size=4000,
                multimodal=False,
                include_binary_content=True,
            )

    finally:
        await async_cleanup_resource(
            lambda: async_client.assistants.delete(name=name, timeout=120),
            name,
            "assistant",
        )
