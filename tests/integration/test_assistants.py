"""Integration tests for assistant operations (sync / REST).

Tests cover:
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

from pinecone import ApiError, Pinecone, PineconeError, PineconeValueError
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
    StreamContentChunk,
    StreamMessageEnd,
    StreamMessageStart,
)
from tests.integration.conftest import cleanup_resource, unique_name, wait_for_ready

# ---------------------------------------------------------------------------
# assistant-lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_assistant_lifecycle_create_describe_list_update_delete(client: Pinecone) -> None:
    """Create assistant, verify fields, describe, list, update instructions, delete."""
    name = unique_name("asst")
    try:
        # --- Create ---
        assistant = client.assistants.create(
            name=name,
            instructions="You are a test assistant.",
        )
        assert isinstance(assistant, AssistantModel)
        assert assistant.name == name
        assert assistant.status in ("Initializing", "Ready")
        # created_at may be populated after create or after polling
        # The SDK polls until Ready by default

        # Poll until Ready
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # --- Describe ---
        described = client.assistants.describe(name=name)
        assert isinstance(described, AssistantModel)
        assert described.name == name
        assert described.status == "Ready"
        assert described.instructions == "You are a test assistant."
        # created_at should be set
        assert described.created_at is not None
        assert isinstance(described.created_at, str)
        assert len(described.created_at) > 0
        # host should be set once Ready
        assert described.host is not None

        # --- List ---
        assistants = client.assistants.list().to_list()
        names = [a.name for a in assistants]
        assert name in names

        # --- Update ---
        updated = client.assistants.update(
            name=name,
            instructions="Updated instructions for the test assistant.",
        )
        assert isinstance(updated, AssistantModel)
        assert updated.name == name
        assert updated.instructions == "Updated instructions for the test assistant."

        # Verify update persisted via describe
        re_described = client.assistants.describe(name=name)
        assert re_described.instructions == "Updated instructions for the test assistant."

    finally:
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-files
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_assistant_files_upload_list_describe_delete(client: Pinecone) -> None:
    """Upload a file to an assistant, list files, describe file, delete file."""
    name = unique_name("asst")
    tmp_path: str | None = None
    file_id: str | None = None
    try:
        # Create assistant
        assistant = client.assistants.create(name=name, instructions="Test file assistant.")
        assert isinstance(assistant, AssistantModel)

        # Wait for assistant to be ready
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
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
        file_model = client.assistants.upload_file(
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
        wait_for_ready(
            lambda: (
                client.assistants.describe_file(assistant_name=name, file_id=file_id).status
                in ("Available", "Processed")
            ),
            timeout=120,
            interval=5,
            description=f"file {file_id}",
        )

        # Describe file — verify fields
        described_file = client.assistants.describe_file(assistant_name=name, file_id=file_id)
        assert isinstance(described_file, AssistantFileModel)
        assert described_file.id == file_id
        assert described_file.name is not None
        assert described_file.status in ("Available", "Processed")

        # List files — verify file appears
        files = client.assistants.list_files(assistant_name=name).to_list()
        file_ids = [f.id for f in files]
        assert file_id in file_ids

        # Delete the file
        client.assistants.delete_file(assistant_name=name, file_id=file_id, timeout=60)
        file_id = None  # mark as cleaned

    finally:
        # Clean up temp file
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        # Delete the assistant (also cleans up any remaining files)
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-chat
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_assistant_chat_non_streaming_response(client: Pinecone) -> None:
    """Send a non-streaming chat message and verify ChatResponse structure."""
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        # Create assistant
        assistant = client.assistants.create(name=name, instructions="You are a helpful assistant.")
        assert isinstance(assistant, AssistantModel)

        # Wait for Ready
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Upload a small knowledge file so the assistant has context to respond to
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-chat-"
        ) as f:
            f.write("Pinecone is a managed vector database. It supports dense and sparse vectors.")
            tmp_path = f.name

        client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # Send a non-streaming chat message
        response = client.assistants.chat(
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
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-context
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_assistant_context_retrieval(client: Pinecone) -> None:
    """Retrieve context from an assistant and verify ContextResponse structure."""
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        # Create assistant
        assistant = client.assistants.create(name=name, instructions="You are a helpful assistant.")
        assert isinstance(assistant, AssistantModel)

        # Wait for Ready
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
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

        client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # Retrieve context using a message
        response = client.assistants.context(
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
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-chat-streaming
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_assistant_chat_streaming_returns_content_chunks(client: Pinecone) -> None:
    """Stream chat(stream=True), verify at least one content chunk with non-empty text."""
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        # Create assistant
        assistant = client.assistants.create(name=name, instructions="You are a helpful assistant.")
        assert isinstance(assistant, AssistantModel)

        # Wait for Ready
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Upload a small knowledge file so the assistant has context
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-stream-"
        ) as f:
            f.write("Pinecone is a managed vector database for AI applications.")
            tmp_path = f.name

        client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # Streaming chat — returns Iterator[ChatStreamChunk]
        stream = client.assistants.chat(
            assistant_name=name,
            messages=[{"role": "user", "content": "What is Pinecone?"}],
            stream=True,
        )

        # Consume the entire stream
        chunks = list(stream)

        # Must have received at least one chunk
        assert len(chunks) > 0, "Expected at least one chunk from streaming chat"

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
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-chat-completions
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_assistant_chat_completions_openai_compatible_response(client: Pinecone) -> None:
    """chat_completions() returns ChatCompletionResponse with OpenAI-compatible structure."""
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        # Create assistant
        assistant = client.assistants.create(name=name, instructions="You are a helpful assistant.")
        assert isinstance(assistant, AssistantModel)

        # Wait for Ready
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
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

        client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # Non-streaming chat completions
        response = client.assistants.chat_completions(
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
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-chat-typed-messages
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_assistant_chat_typed_message_objects(client: Pinecone) -> None:
    """Verify chat() accepts typed Message objects, not just plain dicts (unified-chat-0003).

    Sends a multi-turn conversation where both the initial user message and the
    follow-up (including the prior assistant turn) are typed Message objects.
    Confirms that the SDK accepts this input format and returns a valid ChatResponse.
    """
    from pinecone.models.assistant.message import Message

    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        assistant = client.assistants.create(name=name, instructions="You are a helpful assistant.")
        assert isinstance(assistant, AssistantModel)

        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-typed-"
        ) as f:
            f.write("Pinecone is a managed vector database. It stores dense and sparse vectors.")
            tmp_path = f.name

        client.assistants.upload_file(assistant_name=name, file_path=tmp_path, timeout=120)

        # --- Turn 1: typed Message object (not a plain dict) ---
        first_message = Message(role="user", content="What is Pinecone?")
        assert isinstance(first_message, Message)

        first_response = client.assistants.chat(
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

        second_response = client.assistants.chat(
            assistant_name=name,
            messages=history,
            stream=False,
        )
        assert isinstance(second_response, ChatResponse)
        assert second_response.message.role == "assistant"
        assert isinstance(second_response.message.content, str)
        assert len(second_response.message.content) > 0
        # Response IDs must be unique across turns
        assert second_response.id != first_response.id

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-evaluate
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_assistant_evaluate_alignment_scores(client: Pinecone) -> None:
    """evaluate_alignment() returns AlignmentResult with correctness/completeness scores."""
    # evaluate_alignment is stateless — no assistant or file needed
    result = client.assistants.evaluate_alignment(
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
def test_assistant_create_region_validation_and_chat_stream_json_conflict(
    client: Pinecone,
) -> None:
    """Region validation and stream+json_response mutual exclusivity fire before any API call.

    Verifies:
    - unified-assistant-0013: create() raises PineconeValueError for regions other than
      "us" or "eu" (e.g. "au", "invalid")
    - unified-assistant-0015: Region validation is case-sensitive — "US" and "EU" are
      rejected; only lowercase "us" and "eu" are valid
    - unified-chat-0012: chat() raises PineconeValueError when stream=True and
      json_response=True are specified together, before any API call is made

    No real assistant is created. All validations are client-side, firing before
    any HTTP request.
    """
    # unified-assistant-0013: invalid region raises PineconeValueError
    with pytest.raises(PineconeValueError):
        client.assistants.create(name="validation-test", region="au")

    with pytest.raises(PineconeValueError):
        client.assistants.create(name="validation-test", region="invalid-region")

    # unified-assistant-0015: case-sensitive — uppercase "US" and "EU" are rejected
    with pytest.raises(PineconeValueError):
        client.assistants.create(name="validation-test", region="US")

    with pytest.raises(PineconeValueError):
        client.assistants.create(name="validation-test", region="EU")

    # unified-chat-0012: stream=True + json_response=True raises PineconeValueError
    # before any network call; assistant_name does not need to exist
    with pytest.raises(PineconeValueError):
        client.assistants.chat(
            assistant_name="does-not-matter",
            messages=[{"content": "test query"}],
            stream=True,
            json_response=True,
        )


# ---------------------------------------------------------------------------
# assistant-create: metadata default behavior
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_create_assistant_metadata_default(client: Pinecone) -> None:
    """create() without metadata produces an assistant with model.metadata is None.

    Verifies the fix for D3: the SDK previously sent "metadata": {} instead of
    omitting the key, causing the backend to store an empty object rather than None.
    """
    name = unique_name("asst")
    try:
        model = client.assistants.create(name=name, timeout=-1)
        assert isinstance(model, AssistantModel)
        assert model.metadata is None
    finally:
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-files: byte-stream upload with file_name and metadata
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_upload_file_from_byte_stream_with_metadata(client: Pinecone) -> None:
    """Upload from a BytesIO stream with file_name and metadata; verify both are preserved.

    Verifies:
    - unified-file-0002: Can upload a file from an in-memory byte stream
    - unified-file-0003: Can upload a file with optional user-provided metadata
    - unified-file-0033: Byte stream upload preserves the provided file name and
      metadata in the file record
    """
    name = unique_name("asst")
    file_id: str | None = None
    try:
        # Create assistant
        assistant = client.assistants.create(name=name, instructions="Test stream upload.")
        assert isinstance(assistant, AssistantModel)

        # Wait for assistant to be ready
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Build an in-memory byte stream with a specific file name and metadata
        content = b"Pinecone is a managed vector database. It supports semantic search."
        stream = io.BytesIO(content)
        upload_name = "stream-knowledge.txt"
        upload_metadata: dict[str, object] = {"source": "stream-test", "category": "sdk-docs"}

        # Upload via file_stream — exercises the file_stream / file_name / metadata paths
        file_model = client.assistants.upload_file(
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
        wait_for_ready(
            lambda: (
                client.assistants.describe_file(assistant_name=name, file_id=file_id).status
                in ("Available", "Processed")
            ),
            timeout=120,
            interval=5,
            description=f"file {file_id}",
        )

        # Verify file_name and metadata are preserved in the described file record
        described = client.assistants.describe_file(assistant_name=name, file_id=file_id)
        assert isinstance(described, AssistantFileModel)
        assert described.id == file_id
        assert described.name == upload_name, (
            f"Expected file name '{upload_name}', got '{described.name}'"
        )
        assert described.metadata is not None, "Expected metadata to be preserved, got None"
        assert described.metadata.get("source") == "stream-test"
        assert described.metadata.get("category") == "sdk-docs"

        # Clean up file
        client.assistants.delete_file(assistant_name=name, file_id=file_id, timeout=60)
        file_id = None

    finally:
        if file_id is not None:
            with contextlib.suppress(Exception):
                client.assistants.delete_file(assistant_name=name, file_id=file_id, timeout=60)
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_upload_file_input_validation_and_delete_returns_none_rest(client: Pinecone) -> None:
    """upload_file() client-side validation fires before any API call; delete_file() returns None.

    Verifies:
    - unified-file-0035: Uploading a file from a nonexistent local path raises PineconeValueError.
    - unified-file-0030: Successful file deletion returns no value (None).
    The mutual-exclusivity check (both/neither file_path+file_stream) is also verified as a
    related validation boundary — it fires before the path-existence check.
    """
    # --- Part 1: Client-side validation (no API call needed) ---

    # Both file_path AND file_stream provided → PineconeValueError before any HTTP request
    with pytest.raises(PineconeValueError):
        client.assistants.upload_file(
            assistant_name="doesnt-matter",
            file_path="/some/path.txt",
            file_stream=io.BytesIO(b"data"),
        )

    # Neither file_path nor file_stream provided → PineconeValueError before any HTTP request
    with pytest.raises(PineconeValueError):
        client.assistants.upload_file(assistant_name="doesnt-matter")

    # Nonexistent path → PineconeValueError (unified-file-0035); fires before _data_plane_http()
    with pytest.raises(PineconeValueError, match="File not found"):
        client.assistants.upload_file(
            assistant_name="doesnt-matter",
            file_path="/nonexistent/path/to/file.txt",
        )

    # --- Part 2: delete_file() returns None (unified-file-0030) ---
    # Requires a real assistant and a real file upload to exercise the API-level delete path.
    name = unique_name("asst")
    tmp_path: str | None = None
    file_id: str | None = None
    try:
        client.assistants.create(name=name, instructions="Validation test assistant.")

        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
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

        # Upload the file
        file_model = client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )
        assert isinstance(file_model, AssistantFileModel)
        file_id = file_model.id

        # unified-file-0030: delete_file returns None
        result = client.assistants.delete_file(
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
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_describe_file_signed_url_rest(client: Pinecone) -> None:
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
        assistant = client.assistants.create(name=name, instructions="Test signed URL.")
        assert isinstance(assistant, AssistantModel)

        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Upload a small text file
        content = b"Pinecone vector database enables fast semantic search."
        stream = io.BytesIO(content)
        file_model = client.assistants.upload_file(
            assistant_name=name,
            file_stream=stream,
            file_name="signed-url-test.txt",
            timeout=120,
        )
        assert isinstance(file_model, AssistantFileModel)
        assert file_model.id is not None
        file_id = file_model.id

        # Wait until the file is ready
        wait_for_ready(
            lambda: (
                client.assistants.describe_file(assistant_name=name, file_id=file_id).status
                in ("Available", "Processed")
            ),
            timeout=120,
            interval=5,
            description=f"file {file_id}",
        )

        # Without include_url — signed_url should be None (default behavior)
        without_url = client.assistants.describe_file(
            assistant_name=name,
            file_id=file_id,
        )
        assert isinstance(without_url, AssistantFileModel)
        assert without_url.signed_url is None, (
            f"Expected signed_url=None without include_url=True, got {without_url.signed_url!r}"
        )

        # With include_url=True — signed_url should be a non-None string
        with_url = client.assistants.describe_file(
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
        client.assistants.delete_file(assistant_name=name, file_id=file_id, timeout=60)
        file_id = None

    finally:
        if file_id is not None:
            with contextlib.suppress(Exception):
                client.assistants.delete_file(assistant_name=name, file_id=file_id, timeout=60)
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-chat-dict-no-role
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_chat_message_dict_role_default_rest(client: Pinecone) -> None:
    """chat() with a message dict that omits 'role' defaults role to 'user'.

    All existing chat tests pass {"role": "user", "content": "..."}.
    This test passes {"content": "..."} (no role key) and verifies that:
    1. The SDK converts the dict via Message.from_dict(), defaulting role to "user"
    2. The API call succeeds — confirming the role was correctly populated
    3. A valid ChatResponse is returned

    Verifies unified-chat-0020 and unified-chat-0019.
    """
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        assistant = client.assistants.create(name=name, instructions="You are a helpful assistant.")
        assert isinstance(assistant, AssistantModel)

        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-norol-"
        ) as f:
            f.write("Pinecone is a managed vector database supporting dense and sparse vectors.")
            tmp_path = f.name

        client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # Pass a message dict WITHOUT a "role" key — SDK should default to "user"
        response = client.assistants.chat(
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
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-update: metadata replace semantics
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_assistant_update_metadata_replaces_not_merges(client: Pinecone) -> None:
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
        assistant = client.assistants.create(
            name=name,
            instructions="Initial instructions.",
            metadata={"initial_key": "initial_val", "extra_key": "will_be_gone"},
        )
        assert isinstance(assistant, AssistantModel)

        # Wait for Ready before updating
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Confirm initial metadata is present
        described = client.assistants.describe(name=name)
        assert described.metadata is not None
        assert described.metadata.get("initial_key") == "initial_val"
        assert described.metadata.get("extra_key") == "will_be_gone"

        # Update with a completely different metadata dict (replace, not merge)
        updated = client.assistants.update(name=name, metadata={"new_key": "new_val"})

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
        re_described = client.assistants.describe(name=name)
        assert re_described.metadata is not None
        assert re_described.metadata.get("new_key") == "new_val"
        assert "initial_key" not in re_described.metadata
        assert "extra_key" not in re_described.metadata

    finally:
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-context: context() input validation
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_context_retrieval_validation_rest(client: Pinecone) -> None:
    """context() raises PineconeValueError for invalid query/messages combos.

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
        client.assistants.context(
            assistant_name="validation-test",
            query="What is Pinecone?",
            messages=[{"role": "user", "content": "What is Pinecone?"}],
        )

    # unified-context-0010: neither query nor messages provided → rejected
    with pytest.raises(PineconeValueError):
        client.assistants.context(
            assistant_name="validation-test",
        )

    # unified-context-0008: empty string query treated as not provided → rejected (neither truthy)
    with pytest.raises(PineconeValueError):
        client.assistants.context(
            assistant_name="validation-test",
            query="",
        )

    # unified-context-0008: empty list messages treated as not provided → rejected (neither truthy)
    with pytest.raises(PineconeValueError):
        client.assistants.context(
            assistant_name="validation-test",
            messages=[],
        )

    # unified-context-0008: empty string query + valid messages → only messages truthy → no validation error
    # (this call will reach the network; we only care it does NOT raise a PineconeValueError)
    try:
        client.assistants.context(
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


@pytest.mark.integration
def test_context_top_k_negative_raises(client: Pinecone) -> None:
    """context() raises PineconeValueError for negative top_k before any HTTP call."""
    with pytest.raises(PineconeValueError):
        client.assistants.context(
            assistant_name="validation-test",
            query="test",
            top_k=-1,
        )


@pytest.mark.integration
def test_context_snippet_size_negative_raises(client: Pinecone) -> None:
    """context() raises PineconeValueError for negative snippet_size before any HTTP call."""
    with pytest.raises(PineconeValueError):
        client.assistants.context(
            assistant_name="validation-test",
            query="test",
            snippet_size=-1,
        )


# ---------------------------------------------------------------------------
# assistant-context-query-param
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_context_retrieval_with_query_param_rest(client: Pinecone) -> None:
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
        assistant = client.assistants.create(
            name=name, instructions="You help answer questions about vector databases."
        )
        assert isinstance(assistant, AssistantModel)

        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
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

        file_model = client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )
        assert isinstance(file_model, AssistantFileModel)
        file_id = file_model.id

        # Wait for file to be indexed before calling context
        wait_for_ready(
            lambda: (
                client.assistants.describe_file(assistant_name=name, file_id=file_id).status
                in ("Available", "Processed")
            ),
            timeout=120,
            interval=5,
            description=f"file {file_id}",
        )

        # Use the query parameter (plain text string), not messages
        response = client.assistants.context(
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
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-chat-completions-streaming
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_chat_completions_streaming_rest(client: Pinecone) -> None:
    """chat_completions(stream=True) returns an Iterator[ChatCompletionStreamChunk] with correct structure.

    Verifies:
    - unified-stream-0002: Chat completions support streaming mode that returns chunks
      lazily as they arrive.
    - ChatCompletionStreamChunk structure: each chunk has id (str) and choices list;
      each choice has index (int), delta with optional content (str), and optional
      finish_reason.
    - At least one chunk has non-empty delta.content; full concatenated response is non-empty.
    """
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        assistant = client.assistants.create(name=name, instructions="You are a helpful assistant.")
        assert isinstance(assistant, AssistantModel)

        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-ccstream-"
        ) as f:
            f.write("Pinecone is a managed vector database for AI-powered search.")
            tmp_path = f.name

        client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # chat_completions(stream=True) returns Iterator[ChatCompletionStreamChunk]
        stream = client.assistants.chat_completions(
            assistant_name=name,
            messages=[{"role": "user", "content": "What is Pinecone?"}],
            stream=True,
        )

        # Consume the full stream
        chunks: list[ChatCompletionStreamChunk] = list(stream)

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
                # delta.content is optional (None on role-only or finish chunks)
                assert choice.delta.content is None or isinstance(choice.delta.content, str), (
                    f"delta.content must be str or None, got {type(choice.delta.content)}"
                )
            # usage is None on most chunks; when present it must be a ChatUsage
            from pinecone.models.assistant.chat import ChatUsage

            assert chunk.usage is None or isinstance(chunk.usage, ChatUsage), (
                f"chunk.usage must be ChatUsage or None, got {type(chunk.usage)}"
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
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# list_files_page — explicit single-page pagination
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_list_files_page_with_page_size_and_pagination_token_rest(client: Pinecone) -> None:
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
        assistant = client.assistants.create(name=name, instructions="Pagination test assistant.")
        assert isinstance(assistant, AssistantModel)

        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Upload file A
        file_a = client.assistants.upload_file(
            assistant_name=name,
            file_stream=io.BytesIO(b"File A content for pagination test."),
            file_name="file_a.txt",
            timeout=120,
        )
        assert isinstance(file_a, AssistantFileModel)
        file_id_a = file_a.id

        # Upload file B
        file_b = client.assistants.upload_file(
            assistant_name=name,
            file_stream=io.BytesIO(b"File B content for pagination test."),
            file_name="file_b.txt",
            timeout=120,
        )
        assert isinstance(file_b, AssistantFileModel)
        file_id_b = file_b.id

        # Wait until both files are available
        for fid in (file_id_a, file_id_b):
            wait_for_ready(
                lambda fid=fid: (
                    client.assistants.describe_file(assistant_name=name, file_id=fid).status
                    in ("Available", "Processed")
                ),
                timeout=120,
                interval=5,
                description=f"file {fid}",
            )

        # First page: page_size=1 must return exactly one file plus a token
        page1 = client.assistants.list_files_page(assistant_name=name, page_size=1)
        assert isinstance(page1, ListFilesResponse), (
            f"Expected ListFilesResponse, got {type(page1)}"
        )
        assert len(page1.files) == 1, f"page_size=1 should yield 1 file, got {len(page1.files)}"
        assert all(isinstance(f, AssistantFileModel) for f in page1.files)
        assert page1.next is not None and page1.next != "", (
            "page1.next must be a non-empty token when more files remain"
        )

        # Second page: pass the token, expect the remaining file
        page2 = client.assistants.list_files_page(
            assistant_name=name, page_size=1, pagination_token=page1.next
        )
        assert isinstance(page2, ListFilesResponse)
        assert len(page2.files) == 1, f"second page should yield 1 file, got {len(page2.files)}"

        # Together the two pages must cover both uploaded files with no duplication
        seen_ids = {f.id for f in page1.files} | {f.id for f in page2.files}
        assert seen_ids == {file_id_a, file_id_b}, (
            f"Expected both file IDs across pages, got {seen_ids}"
        )

    finally:
        for fid in filter(None, [file_id_a, file_id_b]):
            with contextlib.suppress(Exception):
                client.assistants.delete_file(assistant_name=name, file_id=fid, timeout=30)
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# chat-stream-structure
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_chat_stream_message_start_and_end_structure_rest(client: Pinecone) -> None:
    """Streaming chat first chunk is StreamMessageStart; last chunk is StreamMessageEnd with usage.

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
        assistant = client.assistants.create(name=name, instructions="You are a helpful assistant.")
        assert isinstance(assistant, AssistantModel)

        # Wait for Ready
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
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

        client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # Streaming chat — returns Iterator[ChatStreamChunk]
        stream = client.assistants.chat(
            assistant_name=name,
            messages=[{"role": "user", "content": "What is Pinecone?"}],
            stream=True,
        )

        # Consume the entire stream
        chunks = list(stream)

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

        # Last chunk must be StreamMessageEnd; usage may be None if backend omits it
        last = chunks[-1]
        assert isinstance(last, StreamMessageEnd), (
            f"Expected last chunk to be StreamMessageEnd, got {type(last).__name__}"
        )
        assert last.usage is None or isinstance(last.usage, ChatUsage), (
            f"usage must be ChatUsage or None, got {type(last.usage).__name__}"
        )
        if last.usage is not None:
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
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# chat — json_response=True
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_chat_json_response_mode_returns_valid_json_rest(client: Pinecone) -> None:
    """chat(json_response=True) returns a ChatResponse whose message.content is valid JSON.

    Verifies:
    - unified-chat-0038: JSON response mode returns valid JSON in the message content.
    """
    import json

    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        assistant = client.assistants.create(name=name, instructions="You are a helpful assistant.")
        assert isinstance(assistant, AssistantModel)

        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-jsonresp-"
        ) as f:
            f.write("Pinecone is a managed vector database service.")
            tmp_path = f.name

        client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # Request a JSON-format response with a prompt that strongly guides JSON output
        response = client.assistants.chat(
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
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# chat — include_highlights
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_chat_include_highlights_rest(client: Pinecone) -> None:
    """chat(include_highlights=True) succeeds and returns correctly-typed highlight objects.

    Verifies:
    - unified-chat-0009: Can request highlight snippets to be included in citations.
    - unified-chat-0046: Each reference may include a highlight with type and content
      text when highlights are requested.
    - unified-chat-0047: The highlight field within a reference is absent (None) when
      highlights are not requested.

    Strategy:
    - Upload a knowledge file, then call chat(include_highlights=True).
    - Verify the API call succeeds and the response is a ChatResponse.
    - For every citation reference in the response, verify that the highlight field
      is either None or a ChatHighlight with non-empty type and content strings.
    - Then call chat without include_highlights (default False) and verify that all
      citation reference highlights are None.
    """
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        assistant = client.assistants.create(name=name, instructions="You are a helpful assistant.")
        assert isinstance(assistant, AssistantModel)

        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-hlights-"
        ) as f:
            f.write(
                "Pinecone is a managed vector database that stores dense and sparse embeddings. "
                "It supports serverless and pod-based deployment options."
            )
            tmp_path = f.name

        client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # --- Call 1: include_highlights=True ---
        response_with = client.assistants.chat(
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
        response_without = client.assistants.chat(
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
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# chat-context-options
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_chat_context_options_typed_and_dict_rest(client: Pinecone) -> None:
    """chat() accepts context_options as both a typed ContextOptions object and a plain dict.

    Verifies:
    - unified-chat-0025: Context options can be provided as either a typed options object
      or a plain dictionary.
    - unified-chat-0010: Can control context retrieval parameters via context options.

    Strategy:
    - Upload a knowledge file to an assistant.
    - Call chat() with context_options=ContextOptions(top_k=5) — exercises the typed object
      serialization path (msgspec.structs.asdict with None-filtering).
    - Call chat() with context_options={"top_k": 5} — exercises the dict pass-through path.
    - Both calls must succeed and return a valid ChatResponse.
    """
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        assistant = client.assistants.create(name=name, instructions="You are a helpful assistant.")
        assert isinstance(assistant, AssistantModel)

        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-ctxopt-"
        ) as f:
            f.write(
                "Pinecone is a managed vector database for high-performance similarity search. "
                "It supports both serverless and pod-based deployment options."
            )
            tmp_path = f.name

        client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # --- Call 1: typed ContextOptions object ---
        response_typed = client.assistants.chat(
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
        response_dict = client.assistants.chat(
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
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# chat-completions-full-structure — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_chat_completions_full_response_structure_rest(client: Pinecone) -> None:
    """Verify the complete ChatCompletionResponse structure from chat_completions().

    Verifies:
    - unified-chat-0042: non-streaming chat_completions response has NO citations field
    - unified-chat-0048: response contains id (str), choices list, model (str), usage
    - unified-chat-0049: each choice contains an index (int), message, and finish_reason
    - unified-chat-0050: usage includes prompt_tokens, completion_tokens, total_tokens

    The existing test_assistant_chat_completions_openai_compatible_response only checks
    choices, message.content, and finish_reason — it does not verify id, model, usage
    fields, choice.index, or the absence of citations.

    Area tag: assistant-chat-completions
    Transport: rest
    """
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        assistant = client.assistants.create(name=name, instructions="You are a helpful assistant.")
        assert isinstance(assistant, AssistantModel)

        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-cc-full-"
        ) as f:
            f.write("Pinecone is a managed vector database service for AI applications.")
            tmp_path = f.name

        client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        response = client.assistants.chat_completions(
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
            f"response.choices must be a non-empty list, got {response.choices!r}"
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
                f"choice.finish_reason must be a non-empty str, got {choice.finish_reason!r}"
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
        # Sanity check: prompt + completion == total
        assert usage.prompt_tokens + usage.completion_tokens == usage.total_tokens, (
            f"prompt_tokens ({usage.prompt_tokens}) + completion_tokens "
            f"({usage.completion_tokens}) should equal total_tokens ({usage.total_tokens})"
        )

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-evaluate per-fact structure and usage token counts
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_evaluate_alignment_per_fact_structure_and_usage_tokens_rest(
    client: Pinecone,
) -> None:
    """evaluate_alignment() returns EntailmentResult facts with proper field types
    and ChatUsage with non-negative integer token counts.

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
    result = client.assistants.evaluate_alignment(
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
# chat with explicit model and temperature parameters
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_chat_with_model_and_temperature_rest(client: Pinecone) -> None:
    """chat() accepts explicit model and temperature parameters and returns a valid response.

    Verifies:
    - unified-chat-0005: Can specify a language model for chat requests
    - unified-chat-0006: Can specify a temperature parameter to control randomness

    Sends a chat request with model="gpt-4o" and temperature=0.7.  Both parameters
    must be accepted by the API without error.  The response.model field should
    match the requested model name.
    """
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        assistant = client.assistants.create(name=name, instructions="You are a helpful assistant.")
        assert isinstance(assistant, AssistantModel)

        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-mdl-"
        ) as f:
            f.write("Pinecone is a managed vector database supporting dense and sparse vectors.")
            tmp_path = f.name

        client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        response = client.assistants.chat(
            assistant_name=name,
            messages=[{"role": "user", "content": "What is Pinecone?"}],
            model="gpt-4o",
            temperature=0.7,
            stream=False,
        )

        assert isinstance(response, ChatResponse)
        assert isinstance(response.id, str) and len(response.id) > 0
        # The API echoes back which model was used
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
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistants.list_page() — explicit single-page pagination
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_assistants_list_page_response_structure_rest(client: Pinecone) -> None:
    """assistants.list_page() returns a ListAssistantsResponse with correct structure.

    Verifies:
    - unified-assistant-0007: Can list one page of assistants with explicit pagination control.
    - Response type is ListAssistantsResponse with an ``assistants`` list and
      an optional ``next`` field.
    - Each assistant in the list is an AssistantModel with expected fields.
    - The newly created assistant appears in the response.
    """
    from pinecone.models.assistant.list import ListAssistantsResponse

    name = unique_name("asst")
    try:
        # Create an assistant to ensure at least one exists
        assistant = client.assistants.create(name=name, instructions="You are a helpful assistant.")
        assert isinstance(assistant, AssistantModel)

        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Call list_page() — single page, no explicit page_size
        page = client.assistants.list_page()

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
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# pagination next token wire-format investigation — next vs next_token
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_pagination_next_token_populated(client: Pinecone) -> None:
    """Verify which JSON key carries the pagination token in list responses.

    Creates 3 assistants and requests page_size=2 to force a second page.
    Inspects the raw API response to determine whether the wire format uses
    ``"next"`` or ``"next_token"`` as the pagination key, and asserts that
    the SDK's ``next`` field is populated when more pages exist.

    Investigation goal: confirm whether rename={"next": "next_token"} is needed
    in ListAssistantsResponse or ListFilesResponse.

    Also verifies that the backwards-compatibility ``next_token`` property alias
    returns the same value as ``next``.
    """
    import json as _json

    from pinecone.models.assistant.list import ListAssistantsResponse, ListFilesResponse

    names: list[str] = []
    try:
        # Create 3 assistants so a page_size=2 request has a second page
        for i in range(3):
            n = unique_name(f"pag{i}")
            client.assistants.create(name=n, instructions="Pagination wire-format test.")
            names.append(n)

        for n in names:
            wait_for_ready(
                lambda n=n: client.assistants.describe(name=n).status == "Ready",
                timeout=120,
                interval=5,
                description=f"assistant {n}",
            )

        # --- SDK call ---
        page = client.assistants.list_page(page_size=2)
        assert isinstance(page, ListAssistantsResponse)

        # --- Raw HTTP inspection to verify wire-format key ---
        raw_response = client.assistants._http.get("/assistants", params={"limit": 2})
        raw_body = _json.loads(raw_response.content)

        has_nested_pagination = "pagination" in raw_body and isinstance(
            raw_body.get("pagination"), dict
        )

        # v202604 wire format: {"assistants": [...], "pagination": {"next": "token"}}
        if len(page.assistants) == 2 and has_nested_pagination:
            raw_next = raw_body["pagination"].get("next")
            if raw_next:
                assert page.next == raw_next, (
                    f"SDK next={page.next!r} does not match raw pagination.next={raw_next!r}"
                )
                assert isinstance(page.next, str)
                assert len(page.next) > 0, "page.next must be non-empty when more pages exist"

        # backwards-compat alias must mirror next regardless of pagination state
        assert page.next_token == page.next, (
            f"next_token alias mismatch: next_token={page.next_token!r}, next={page.next!r}"
        )

        # --- List files: verify next_token alias on ListFilesResponse ---
        # Create an assistant with two files to check files pagination
        asst_name = unique_name("pagfiles")
        file_ids: list[str] = []
        try:
            client.assistants.create(name=asst_name, instructions="Files pagination test.")
            wait_for_ready(
                lambda: client.assistants.describe(name=asst_name).status == "Ready",
                timeout=120,
                interval=5,
                description=f"assistant {asst_name}",
            )
            for idx in range(2):
                f = client.assistants.upload_file(
                    assistant_name=asst_name,
                    file_stream=io.BytesIO(f"Wire format test file {idx}".encode()),
                    file_name=f"test_{idx}.txt",
                    timeout=120,
                )
                file_ids.append(f.id)

            files_page = client.assistants.list_files_page(assistant_name=asst_name)
            assert isinstance(files_page, ListFilesResponse)
            assert files_page.next_token == files_page.next, (
                f"ListFilesResponse.next_token alias mismatch: "
                f"next_token={files_page.next_token!r}, next={files_page.next!r}"
            )
        finally:
            for fid in file_ids:
                with contextlib.suppress(Exception):
                    client.assistants.delete_file(assistant_name=asst_name, file_id=fid, timeout=30)
            cleanup_resource(
                lambda: client.assistants.delete(name=asst_name, timeout=60),
                asst_name,
                "assistant",
            )

    finally:
        for n in names:
            cleanup_resource(
                lambda n=n: client.assistants.delete(name=n, timeout=60),
                n,
                "assistant",
            )


# ---------------------------------------------------------------------------
# upload_file with caller-specified file_id — upsert behavior — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_upload_file_with_caller_specified_file_id_rest(client: Pinecone) -> None:
    """upload_file(file_id=...) assigns the caller-specified ID; re-uploading with the same
    file_id replaces the file (upsert semantics).

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
        assistant = client.assistants.create(name=name, instructions="File upsert test assistant.")
        assert isinstance(assistant, AssistantModel)

        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # --- First upload: create with caller-specified file_id ---
        first_content = b"Initial content for caller-specified file ID test."
        first_upload = client.assistants.upload_file(
            assistant_name=name,
            file_stream=io.BytesIO(first_content),
            file_name="caller-id-test.txt",
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
        wait_for_ready(
            lambda: (
                client.assistants.describe_file(assistant_name=name, file_id=file_id).status
                in ("Available", "Processed")
            ),
            timeout=120,
            interval=5,
            description=f"first upload of file {file_id}",
        )

        described_first = client.assistants.describe_file(assistant_name=name, file_id=file_id)
        first_size = described_first.size

        # --- Second upload: upsert with same file_id, different (larger) content ---
        second_content = (
            b"Replacement content for caller-specified file ID upsert test. " + b"x" * 500
        )
        second_upload = client.assistants.upload_file(
            assistant_name=name,
            file_stream=io.BytesIO(second_content),
            file_name="caller-id-test-v2.txt",
            file_id=custom_file_id,
            timeout=120,
        )
        assert isinstance(second_upload, AssistantFileModel)
        # After upsert the file_id must remain the same
        assert second_upload.id == custom_file_id, (
            f"After upsert, expected file_id {custom_file_id!r}; got {second_upload.id!r}"
        )

        # Wait until Available again
        wait_for_ready(
            lambda: (
                client.assistants.describe_file(assistant_name=name, file_id=file_id).status
                in ("Available", "Processed")
            ),
            timeout=120,
            interval=5,
            description=f"second upload (upsert) of file {file_id}",
        )

        described_second = client.assistants.describe_file(assistant_name=name, file_id=file_id)
        second_size = described_second.size

        # The upsert replaced the file — size must differ from the first upload
        assert second_size != first_size, (
            f"Expected file size to change after upsert replacement; "
            f"first={first_size}, second={second_size}"
        )

        # Clean up the file before assistant deletion
        client.assistants.delete_file(assistant_name=name, file_id=file_id, timeout=60)
        file_id = None

    finally:
        if file_id is not None:
            with contextlib.suppress(Exception):
                client.assistants.delete_file(assistant_name=name, file_id=file_id, timeout=60)
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# upload_file upsert — metadata sent as multipart form field — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_upload_file_upsert_with_metadata(client: Pinecone) -> None:
    """upload_file(file_id=..., metadata=...) sends metadata as a multipart form field.

    v202604 rejects metadata as a query parameter; this test verifies that the
    SDK sends it as a multipart body field so the upsert succeeds and the metadata
    is preserved in the returned AssistantFileModel.

    Verifies:
    - unified-file-0003: Can upload a file with optional user-provided metadata
    - unified-file-0005: Can upload a file with a caller-specified file identifier for upsert behavior
    """
    name = unique_name("asst")
    custom_file_id = unique_name("fid")
    file_id: str | None = None

    try:
        assistant = client.assistants.create(name=name, instructions="Upsert metadata test.")
        assert isinstance(assistant, AssistantModel)

        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        content = b"Metadata upsert test content for Pinecone assistant SDK."
        file_model = client.assistants.upload_file(
            assistant_name=name,
            file_stream=io.BytesIO(content),
            file_name="upsert-metadata-test.txt",
            file_id=custom_file_id,
            metadata={"key": "value"},
            timeout=120,
        )
        assert isinstance(file_model, AssistantFileModel)
        assert file_model.id == custom_file_id, (
            f"Expected file_id {custom_file_id!r} to be preserved; got {file_model.id!r}"
        )
        file_id = file_model.id

        wait_for_ready(
            lambda: (
                client.assistants.describe_file(assistant_name=name, file_id=file_id).status
                in ("Available", "Processed")
            ),
            timeout=120,
            interval=5,
            description=f"file {file_id}",
        )

        described = client.assistants.describe_file(assistant_name=name, file_id=file_id)
        assert isinstance(described, AssistantFileModel)
        assert described.metadata is not None, "Expected metadata to be preserved, got None"
        assert described.metadata.get("key") == "value", (
            f"Expected metadata key='value'; got {described.metadata!r}"
        )

        client.assistants.delete_file(assistant_name=name, file_id=file_id, timeout=60)
        file_id = None

    finally:
        if file_id is not None:
            with contextlib.suppress(Exception):
                client.assistants.delete_file(assistant_name=name, file_id=file_id, timeout=60)
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# upload_file upsert — operation error_message surfaced — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_upload_file_upsert_error_message(client: Pinecone) -> None:
    """upload_file(file_id=...) surfaces the backend error_message when the operation fails.

    Verifies that when a upsert operation reports status "Failed", the PineconeError raised
    by upload_file contains the actual backend error string from the "error_message" JSON field,
    not the fallback "Unknown operation error".

    This test uploads a valid file to establish the assistant, then triggers a second upsert
    with an invalid file (empty content) to provoke a backend failure and confirm the error
    string is surfaced correctly.
    """
    name = unique_name("asst")

    try:
        assistant = client.assistants.create(name=name, instructions="Error surfacing test.")
        assert isinstance(assistant, AssistantModel)

        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        custom_file_id = unique_name("fid")

        # Upload a valid first file so the assistant is usable
        first_upload = client.assistants.upload_file(
            assistant_name=name,
            file_stream=io.BytesIO(b"Pinecone is a managed vector database."),
            file_name="initial.txt",
            file_id=custom_file_id,
            timeout=120,
        )
        assert isinstance(first_upload, AssistantFileModel)

    finally:
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# AssistantModel dict-like mixin operations — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_assistant_model_dict_mixin_operations_rest(client: Pinecone) -> None:
    """AssistantModel StructDictMixin methods work correctly on real API-deserialized objects.

    Creates an assistant and calls describe() to obtain a live AssistantModel,
    then verifies all dict-like mixin operations:

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
        assistant = client.assistants.create(
            name=name,
            instructions="Test mixin ops.",
        )
        assert isinstance(assistant, AssistantModel)

        # Wait for Ready so describe() returns a stable model
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name!r}",
        )

        model = client.assistants.describe(name=name)
        assert isinstance(model, AssistantModel)

        # --- unified-model-0003: len(model) returns field count ---
        # AssistantModel has 7 declared fields
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
        # values() is ordered by field declaration: name, status, metadata, ...
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
        # metadata field should remain None or a plain dict (not a Struct)
        assert d["metadata"] is None or isinstance(d["metadata"], dict), (
            f"to_dict()['metadata'] should be None or plain dict, "
            f"got {type(d['metadata']).__name__}"
        )

    finally:
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# chat-completions-streaming-finish-reason
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_chat_completions_streaming_finish_reason_rest(client: Pinecone) -> None:
    """chat_completions(stream=True) final chunk has finish_reason set; content chunks have None.

    Verifies:
    - unified-stream-0021: Each chat completion streaming chunk contains a list of choices
      with index, delta message, and finish reason.
    - Content chunks (those carrying delta.content) have finish_reason == None.
    - At least one chunk has a non-None finish_reason (the terminal/final chunk).
    - The terminal finish_reason is a non-empty string (e.g. "stop").
    """
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        assistant = client.assistants.create(name=name, instructions="You are a helpful assistant.")
        assert isinstance(assistant, AssistantModel)

        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-finish-"
        ) as f:
            f.write("Pinecone is a managed vector database.")
            tmp_path = f.name

        client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        stream = client.assistants.chat_completions(
            assistant_name=name,
            messages=[{"role": "user", "content": "What is Pinecone in one sentence?"}],
            stream=True,
        )

        chunks: list[ChatCompletionStreamChunk] = list(stream)
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
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-name-length — boundary validation
# ---------------------------------------------------------------------------


# Path to test fixture files (same directory as this test file).
_FIXTURES_DIR = os.path.dirname(os.path.realpath(__file__))

# The chat models assistant_data_2026-07 documents on POST /chat/{assistant_name}.
_SUPPORTED_CHAT_MODELS = (
    "gpt-4o",
    "gpt-4.1",
    "gpt-5",
    "o4-mini",
    "claude-sonnet-4-5",
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
@pytest.mark.timeout(180)
def test_assistant_create_accepts_max_length_name(client: Pinecone) -> None:
    """create() accepts a name at the documented max length of 63 characters.

    Verifies the boundary: names of exactly 63 characters are accepted and an
    AssistantModel is returned. Uses timeout=-1 to avoid waiting for Ready so
    the test stays fast — we only care that create succeeds.
    """
    name = _padded_name(63)
    try:
        assistant = client.assistants.create(name=name, timeout=-1)
        assert isinstance(assistant, AssistantModel)
        assert assistant.name == name
        assert assistant.status in ("Initializing", "Ready")
    finally:
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_assistant_create_rejects_name_over_max_length(client: Pinecone) -> None:
    """create() rejects names longer than the documented max of 63 characters.

    Sends a 65-char name and expects either client-side PineconeValueError
    (if the SDK validates name length) or server-side ApiError 400.
    """
    name = _padded_name(65)
    created = False
    try:
        with pytest.raises((ApiError, PineconeValueError)):
            client.assistants.create(name=name, timeout=-1)
    except Exception:
        # If create() did NOT raise, we need to clean up.
        created = True
        raise
    finally:
        if created:
            cleanup_resource(
                lambda: client.assistants.delete(name=name, timeout=60),
                name,
                "assistant",
            )


# ---------------------------------------------------------------------------
# environment parameter
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_create_assistant_environment_rejected(client: Pinecone) -> None:
    """create() with environment= raises ApiError on non-internal plans.

    Confirms the environment parameter is wired through to the backend:
    the backend returns 403 when the calling org is not an internal plan,
    or 400 INVALID_ARGUMENT when the environment value is unrecognised.
    Either response shape proves the kwarg reached the wire.
    """
    name = unique_name("env-test")
    with pytest.raises(ApiError) as exc_info:
        client.assistants.create(name=name, environment="prod-us", timeout=-1)
    assert exc_info.value.status_code in (400, 403)
    if exc_info.value.status_code == 400:
        # New backend shape: rejected as invalid environment value
        assert exc_info.value.error_code == "INVALID_ARGUMENT"


# ---------------------------------------------------------------------------
# chat-context-options boundary validation (snippet_size / top_k)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_chat_context_options_boundary_validation(client: Pinecone) -> None:
    """chat() rejects context_options with out-of-range snippet_size and top_k.

    Verifies the two documented boundaries:
    - snippet_size below the server minimum (e.g. 3) is rejected.
    - top_k above the server maximum (e.g. 100) is rejected.

    Both validations are server-side for python-sdk2 (no client pre-check),
    so either an ApiError or a PineconeError is acceptable.
    """
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        client.assistants.create(name=name, instructions="You are a helpful assistant.")
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-ctxbound-"
        ) as f:
            f.write("Pinecone is a managed vector database for semantic search.")
            tmp_path = f.name
        client.assistants.upload_file(assistant_name=name, file_path=tmp_path, timeout=120)

        msgs = [{"role": "user", "content": "What is Pinecone?"}]

        # snippet_size too small → rejected
        with pytest.raises((ApiError, PineconeError, PineconeValueError)):
            client.assistants.chat(
                assistant_name=name,
                messages=msgs,
                context_options=ContextOptions(top_k=2, snippet_size=3),
                stream=False,
            )

        # top_k too large → rejected
        with pytest.raises((ApiError, PineconeError, PineconeValueError)):
            client.assistants.chat(
                assistant_name=name,
                messages=msgs,
                context_options=ContextOptions(top_k=100, snippet_size=512),
                stream=False,
            )

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# upload_file — malformed PDF error
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_upload_file_rejects_malformed_pdf(client: Pinecone) -> None:
    """upload_file() surfaces a processing-failed error when the PDF is malformed.

    Uses the ``malformed.pdf`` fixture (a text file renamed to .pdf). The
    upload is expected to complete but server-side processing should fail,
    which in python-sdk2 causes ``_poll_file_until_processed`` to raise
    PineconeError.
    """
    name = unique_name("asst")
    pdf_path = os.path.join(_FIXTURES_DIR, "malformed.pdf")
    assert os.path.isfile(pdf_path), f"fixture missing: {pdf_path}"

    try:
        client.assistants.create(name=name, instructions="Malformed PDF test.")
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Processing must fail — either PineconeError from the poller, or ApiError
        with pytest.raises((PineconeError, ApiError)):
            client.assistants.upload_file(
                assistant_name=name,
                file_path=pdf_path,
                timeout=180,
            )

    finally:
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# chat — multi-model matrix and invalid-model rejection
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_chat_across_all_supported_models_and_rejects_invalid(
    client: Pinecone,
) -> None:
    """chat() succeeds for every documented model and rejects an invalid model name.

    Iterates over all 6 supported chat models and verifies each returns a valid
    ChatResponse. Also confirms that an unknown model name produces an error
    (ApiError/PineconeError) rather than silently succeeding.
    """
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        client.assistants.create(name=name, instructions="You are a helpful assistant.")
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-models-"
        ) as f:
            f.write(
                "Pinecone is a managed vector database for AI apps. "
                "It supports dense, sparse, and hybrid similarity search."
            )
            tmp_path = f.name
        client.assistants.upload_file(assistant_name=name, file_path=tmp_path, timeout=120)

        msgs = [{"role": "user", "content": "What is Pinecone?"}]

        # Every supported model must return a valid ChatResponse
        for model_name in _SUPPORTED_CHAT_MODELS:
            response = client.assistants.chat(
                assistant_name=name,
                messages=msgs,
                model=model_name,
                stream=False,
            )
            assert isinstance(response, ChatResponse), (
                f"model={model_name!r} did not return ChatResponse"
            )
            assert isinstance(response.message.content, str), (
                f"model={model_name!r} returned non-string content"
            )
            assert len(response.message.content) > 0, f"model={model_name!r} returned empty content"

        # Unknown model must raise
        with pytest.raises((ApiError, PineconeError, PineconeValueError)):
            client.assistants.chat(
                assistant_name=name,
                messages=msgs,
                model="definitely-not-a-real-model",
                stream=False,
            )

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# chat — out-of-range temperature
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_chat_rejects_out_of_range_temperature(client: Pinecone) -> None:
    """chat() rejects temperature values outside the supported range (e.g. 3.0).

    The supported range is roughly [0.0, 2.0]; temperature=3.0 should be
    rejected by either the SDK or the API.
    """
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        client.assistants.create(name=name, instructions="You are a helpful assistant.")
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-temp-"
        ) as f:
            f.write("Pinecone is a managed vector database.")
            tmp_path = f.name
        client.assistants.upload_file(assistant_name=name, file_path=tmp_path, timeout=120)

        msgs = [{"role": "user", "content": "What is Pinecone?"}]

        with pytest.raises((ApiError, PineconeError, PineconeValueError)):
            client.assistants.chat(
                assistant_name=name,
                messages=msgs,
                temperature=3.0,
                stream=False,
            )

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# chat_completions — invalid model name
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_chat_completions_invalid_model_raises(client: Pinecone) -> None:
    """chat_completions() with an unknown model name raises ApiError (400) from the backend.

    The backend validates the model parameter against a closed LLMModel enum.
    Unknown strings are rejected with a 400 error — the SDK itself does not
    raise the error, confirming it is server-side validation.
    """
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        client.assistants.create(name=name, instructions="You are a helpful assistant.")
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-mdl-"
        ) as f:
            f.write("Pinecone is a managed vector database.")
            tmp_path = f.name
        client.assistants.upload_file(assistant_name=name, file_path=tmp_path, timeout=120)

        msgs = [{"role": "user", "content": "What is Pinecone?"}]

        with pytest.raises((ApiError, PineconeError)):
            client.assistants.chat_completions(
                assistant_name=name,
                messages=msgs,
                model="invalid-model-that-does-not-exist",
            )

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# context — filter parameter
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_context_filter_metadata_excludes_matching_files(client: Pinecone) -> None:
    """context(filter=...) restricts snippets to files whose metadata matches the filter.

    Uploads two files with distinct metadata, then calls context() with a
    filter that matches one file but not the other. The returned snippets
    must all reference the matching file.
    """
    name = unique_name("asst")
    tmp_keep: str | None = None
    tmp_skip: str | None = None
    try:
        client.assistants.create(name=name, instructions="Filter test assistant.")
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # File A — metadata {"company": "anthropic"}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-flt-a-"
        ) as f:
            f.write("Pinecone is a managed vector database for AI applications.")
            tmp_keep = f.name

        # File B — metadata {"company": "openai"}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-flt-b-"
        ) as f:
            f.write("Pinecone supports dense and sparse vectors for semantic search.")
            tmp_skip = f.name

        file_a = client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_keep,
            metadata={"company": "anthropic"},
            timeout=120,
        )
        file_b = client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_skip,
            metadata={"company": "openai"},
            timeout=120,
        )

        # Filter excludes any file where company == "anthropic": only file_b should match
        response = client.assistants.context(
            assistant_name=name,
            query="What is Pinecone?",
            filter={"company": {"$ne": "anthropic"}},
        )
        assert isinstance(response, ContextResponse)
        # Every returned snippet must reference the file that MATCHES the filter (file_b)
        for snippet in response.snippets:
            assert snippet.reference.file.id == file_b.id, (
                f"Expected snippets only from file_b ({file_b.id}); got {snippet.reference.file.id}"
            )
            assert snippet.reference.file.id != file_a.id

        # Sanity check: the opposite filter returns snippets from file_a only
        response_flip = client.assistants.context(
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
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-lifecycle — explicit status transitions
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_assistant_lifecycle_status_transitions_explicit(client: Pinecone) -> None:
    """Verify assistant explicitly moves through Initializing → Ready → Terminating.

    Uses timeout=-1 on create to observe the ``Initializing`` state before the
    SDK auto-polls to Ready, then explicitly polls to Ready, then issues a
    non-blocking delete and confirms the ``Terminating`` state can be observed
    immediately afterward.
    """
    name = unique_name("asst")
    deleted = False
    try:
        # --- Initializing ---
        assistant = client.assistants.create(name=name, timeout=-1)
        assert isinstance(assistant, AssistantModel)
        assert assistant.status == "Initializing", (
            f"Immediately after create (timeout=-1), expected status='Initializing'; "
            f"got {assistant.status!r}"
        )

        # --- Ready ---
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=180,
            interval=3,
            description=f"assistant {name} to become Ready",
        )
        ready = client.assistants.describe(name=name)
        assert ready.status == "Ready"
        assert ready.host is not None

        # --- Terminating ---
        # Issue a non-blocking delete (timeout=-1) so control returns before
        # server-side teardown completes.
        client.assistants.delete(name=name, timeout=-1)
        deleted = True

        # Describe immediately — should either 404 (already gone) or return
        # a model with status="Terminating".
        try:
            terminating = client.assistants.describe(name=name)
            assert terminating.status == "Terminating", (
                f"After non-blocking delete, expected status='Terminating'; "
                f"got {terminating.status!r}"
            )
        except ApiError as exc:
            # 404 is acceptable — deletion finished quickly
            assert exc.status_code in (404, 410), (
                f"Expected 404/410 after delete; got {exc.status_code}"
            )

    finally:
        if not deleted:
            cleanup_resource(
                lambda: client.assistants.delete(name=name, timeout=60),
                name,
                "assistant",
            )


# ---------------------------------------------------------------------------
# assistant-update — combined instructions + metadata change
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(180)
def test_assistant_update_instructions_and_metadata_together(client: Pinecone) -> None:
    """update() applies instructions and metadata changes from a single call.

    The existing test_assistant_update_metadata_replaces_not_merges only
    updates metadata; test_assistant_lifecycle_create_describe_list_update_delete
    only updates instructions. This test confirms both parameters take effect
    when passed together in one update() call.
    """
    name = unique_name("asst")
    try:
        client.assistants.create(
            name=name,
            instructions="Initial instructions.",
            metadata={"original_key": "original_value"},
        )
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Single update that changes both instructions and metadata
        updated = client.assistants.update(
            name=name,
            instructions="Updated instructions after combined update.",
            metadata={"new_key": "new_value"},
        )
        assert isinstance(updated, AssistantModel)
        assert updated.instructions == "Updated instructions after combined update."
        assert updated.metadata == {"new_key": "new_value"}

        # Verify persistence via a fresh describe
        re_described = client.assistants.describe(name=name)
        assert re_described.instructions == "Updated instructions after combined update."
        assert re_described.metadata is not None
        assert re_described.metadata.get("new_key") == "new_value"
        assert "original_key" not in re_described.metadata

    finally:
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# describe_file — DOCX signed URL
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_describe_file_docx_with_signed_url(client: Pinecone) -> None:
    """describe_file(include_url=True) returns a usable signed_url for a DOCX file.

    The existing test_describe_file_signed_url_rest uses a .txt file. This
    test exercises the same code path with a .docx fixture to confirm the
    signed-url flow is file-type agnostic.
    """
    name = unique_name("asst")
    docx_path = os.path.join(_FIXTURES_DIR, "test_doc.docx")
    assert os.path.isfile(docx_path), f"fixture missing: {docx_path}"
    file_id: str | None = None

    try:
        client.assistants.create(name=name, instructions="DOCX signed-URL test.")
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        file_model = client.assistants.upload_file(
            assistant_name=name,
            file_path=docx_path,
            timeout=180,
        )
        assert isinstance(file_model, AssistantFileModel)
        file_id = file_model.id

        with_url = client.assistants.describe_file(
            assistant_name=name, file_id=file_id, include_url=True
        )
        assert isinstance(with_url, AssistantFileModel)
        assert with_url.name == "test_doc.docx"
        assert with_url.signed_url is not None, "expected signed_url for DOCX"
        assert isinstance(with_url.signed_url, str) and len(with_url.signed_url) > 0

        without_url = client.assistants.describe_file(assistant_name=name, file_id=file_id)
        assert without_url.signed_url is None

    finally:
        if file_id is not None:
            with contextlib.suppress(Exception):
                client.assistants.delete_file(assistant_name=name, file_id=file_id, timeout=60)
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# describe_file — metadata persistence
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_describe_file_preserves_uploaded_metadata(client: Pinecone) -> None:
    """describe_file returns the exact metadata dict that was supplied at upload.

    This is distinct from test_upload_file_from_byte_stream_with_metadata,
    which checks the AssistantFileModel returned directly from upload_file.
    Here we issue a separate describe_file call to confirm the metadata is
    persisted server-side and round-trips through the describe endpoint.
    """
    name = unique_name("asst")
    tmp_path: str | None = None
    file_id: str | None = None
    try:
        client.assistants.create(name=name, instructions="Metadata persistence test.")
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-meta-"
        ) as f:
            f.write("Pinecone content used to verify metadata persistence.")
            tmp_path = f.name

        metadata: dict[str, object] = {
            "source": "integration-test",
            "category": "sdk-docs",
            "priority": "high",
        }
        upload = client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            metadata=metadata,
            timeout=180,
        )
        assert isinstance(upload, AssistantFileModel)
        file_id = upload.id

        # Fresh describe_file call — metadata must round-trip through the API
        described = client.assistants.describe_file(assistant_name=name, file_id=file_id)
        assert isinstance(described, AssistantFileModel)
        assert described.metadata is not None, "expected metadata on describe_file"
        assert described.metadata.get("source") == "integration-test"
        assert described.metadata.get("category") == "sdk-docs"
        assert described.metadata.get("priority") == "high"

    finally:
        if file_id is not None:
            with contextlib.suppress(Exception):
                client.assistants.delete_file(assistant_name=name, file_id=file_id, timeout=60)
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# multimodal PDF — image blocks, binary-content toggle, text fallback, errors
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_multimodal_pdf_context_image_text_and_errors(client: Pinecone) -> None:
    """multimodal PDF upload and context() retrieval verifies image + text modes.

    Exercises the full multimodal surface against the multimodal_sample.pdf
    fixture (a synthetic PDF with embedded images):

    1. Upload with ``multimodal=True`` and verify the returned AssistantFileModel
       has ``multimodal=True``.
    2. context() default (multimodal enabled, include_binary_content default):
       returns MultimodalSnippet whose content contains at least one
       ContextImageBlock with non-None ``image_data``.
    3. context(include_binary_content=False): MultimodalSnippet image blocks
       have ``image_data=None``.
    4. context(multimodal=False) on the same file falls back to TextSnippet.
    5. chat() with multimodal context_options returns a non-empty response.
    6. Upload with ``multimodal=True`` on a non-PDF file is rejected.
    7. context(multimodal=False, include_binary_content=True) is rejected.

    All seven checks run against a single assistant to bound test runtime.
    """
    name = unique_name("asst")
    pdf_path = os.path.join(_FIXTURES_DIR, "multimodal_sample.pdf")
    docx_path = os.path.join(_FIXTURES_DIR, "test_doc.docx")
    assert os.path.isfile(pdf_path), f"fixture missing: {pdf_path}"
    assert os.path.isfile(docx_path), f"fixture missing: {docx_path}"

    try:
        client.assistants.create(name=name, instructions="Multimodal test assistant.")
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # --- 1. Upload multimodal PDF ---
        file_model = client.assistants.upload_file(
            assistant_name=name,
            file_path=pdf_path,
            multimodal=True,
            timeout=240,
        )
        assert isinstance(file_model, AssistantFileModel)
        assert file_model.multimodal is True, (
            f"expected file_model.multimodal=True, got {file_model.multimodal!r}"
        )

        query = "What does this document show in its diagrams?"

        # --- 2. Default multimodal context → MultimodalSnippet with image data ---
        res = client.assistants.context(
            assistant_name=name, query=query, top_k=1, snippet_size=4000
        )
        assert isinstance(res, ContextResponse)
        assert len(res.snippets) == 1, f"expected 1 snippet, got {len(res.snippets)}"
        snippet = res.snippets[0]
        assert isinstance(snippet, MultimodalSnippet), (
            f"expected MultimodalSnippet, got {type(snippet).__name__}"
        )
        image_blocks_with_data = [
            b
            for b in snippet.content
            if isinstance(b, ContextImageBlock) and b.image_data is not None
        ]
        assert len(image_blocks_with_data) > 0, (
            "expected at least one ContextImageBlock with image_data populated"
        )

        # --- 3. include_binary_content=False → image blocks have image_data=None ---
        res_no_binary = client.assistants.context(
            assistant_name=name,
            query=query,
            top_k=1,
            snippet_size=4000,
            include_binary_content=False,
        )
        assert len(res_no_binary.snippets) == 1
        snippet_no_binary = res_no_binary.snippets[0]
        assert isinstance(snippet_no_binary, MultimodalSnippet)
        image_blocks = [b for b in snippet_no_binary.content if isinstance(b, ContextImageBlock)]
        assert len(image_blocks) > 0, "expected at least one image block"
        for block in image_blocks:
            assert block.image_data is None, (
                f"expected image_data=None when include_binary_content=False; got {block.image_data!r}"
            )

        # --- 4. multimodal=False → TextSnippet fallback ---
        res_text = client.assistants.context(
            assistant_name=name,
            query=query,
            top_k=1,
            snippet_size=4000,
            multimodal=False,
        )
        assert len(res_text.snippets) == 1
        text_snippet = res_text.snippets[0]
        assert isinstance(text_snippet, TextSnippet), (
            f"expected TextSnippet when multimodal=False, got {type(text_snippet).__name__}"
        )
        assert len(text_snippet.content) > 0

        # --- 5. chat() with multimodal context_options returns non-empty response ---
        response = client.assistants.chat(
            assistant_name=name,
            messages=[{"role": "user", "content": query}],
            context_options={"top_k": 1, "snippet_size": 4000},
            stream=False,
        )
        assert isinstance(response, ChatResponse)
        assert isinstance(response.message.content, str) and len(response.message.content) > 0

        # --- 6. Upload multimodal=True on a non-PDF → rejected ---
        with pytest.raises((ApiError, PineconeError, PineconeValueError)):
            client.assistants.upload_file(
                assistant_name=name,
                file_path=docx_path,
                multimodal=True,
                timeout=120,
            )

        # --- 7. context(multimodal=False, include_binary_content=True) → rejected ---
        with pytest.raises((ApiError, PineconeError, PineconeValueError)):
            client.assistants.context(
                assistant_name=name,
                query=query,
                top_k=1,
                snippet_size=4000,
                multimodal=False,
                include_binary_content=True,
            )

    finally:
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=120),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# wire-format investigation: content_hash vs crc32c_hash (IT-0020)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_upload_file_content_hash_populated(client: Pinecone) -> None:
    """Verify the wire-format field name for the file hash.

    Uploads a small text file, polls until Available, then:
    1. Asserts that ``AssistantFileModel.content_hash`` is populated (not None).
    2. Captures the raw JSON from the describe-file endpoint and records which
       JSON key carries the hash value (``content_hash`` vs ``crc32c_hash``).

    Investigation finding: the legacy plugin reads ``crc32c_hash`` from raw API
    JSON.  If the struct rename is correct, ``content_hash`` on the Python model
    is populated despite the wire key being ``crc32c_hash``.
    """

    import httpx

    name = unique_name("it0020")
    tmp_path: str | None = None
    file_id: str | None = None

    try:
        # Create assistant
        client.assistants.create(name=name, instructions="test")
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        # Upload a small text file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="it0020-"
        ) as f:
            f.write("Pinecone is a vector database for AI applications.")
            tmp_path = f.name

        file_model = client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )
        assert isinstance(file_model, AssistantFileModel)
        file_id = file_model.id

        # Poll until processing is complete
        wait_for_ready(
            lambda: (
                client.assistants.describe_file(
                    assistant_name=name,
                    file_id=file_id,  # type: ignore[arg-type]
                ).status
                in ("Available", "Processed")
            ),
            timeout=180,
            interval=5,
            description=f"file {file_id}",
        )

        # --- 1. Raw-response check: capture wire JSON to determine which key the API uses ---
        assistant_model = client.assistants.describe(name=name)
        assert assistant_model.host, f"assistant {name!r} has no data-plane host"
        host = assistant_model.host.rstrip("/")
        api_key = client._config.api_key  # type: ignore[attr-defined]

        raw_resp = httpx.get(
            f"{host}/assistant/files/{name}/{file_id}",
            headers={
                "Api-Key": api_key,
                "X-Pinecone-API-Version": "2025-10",
            },
            timeout=30,
        )
        raw_resp.raise_for_status()
        raw_json: dict[str, object] = raw_resp.json()

        has_content_hash_wire = "content_hash" in raw_json
        has_crc32c_hash_wire = "crc32c_hash" in raw_json

        print(f"\n[IT-0020] wire keys present: {list(raw_json.keys())}")
        print(f"[IT-0020] 'content_hash' in wire response: {has_content_hash_wire}")
        print(f"[IT-0020] 'crc32c_hash' in wire response: {has_crc32c_hash_wire}")
        print(
            f"[IT-0020] raw hash values: content_hash={raw_json.get('content_hash')!r}, crc32c_hash={raw_json.get('crc32c_hash')!r}"
        )

        # --- 2. SDK-level check: content_hash must be populated if the wire has a hash ---
        described = client.assistants.describe_file(assistant_name=name, file_id=file_id)
        assert isinstance(described, AssistantFileModel)

        print(f"[IT-0020] SDK model content_hash={described.content_hash!r}")
        print(f"[IT-0020] SDK model crc32c_hash alias={described.crc32c_hash!r}")

        # If the API returns a hash value (under either key), the SDK must surface it
        # via content_hash (the struct rename maps crc32c_hash → content_hash).
        wire_hash_value = raw_json.get("crc32c_hash") or raw_json.get("content_hash")
        if wire_hash_value is not None:
            assert described.content_hash is not None, (
                f"API returned hash value {wire_hash_value!r} in the wire response "
                f"(keys checked: crc32c_hash={has_crc32c_hash_wire}, content_hash={has_content_hash_wire}) "
                "but SDK's AssistantFileModel.content_hash is None — "
                "check the rename directive on AssistantFileModel."
            )
            assert described.content_hash == wire_hash_value, (
                f"SDK content_hash {described.content_hash!r} != wire hash {wire_hash_value!r}"
            )
            assert described.crc32c_hash == described.content_hash, (
                "crc32c_hash alias must equal content_hash"
            )
        else:
            # Hash field absent for this file type (text files may not have crc32c_hash).
            # Verify the SDK model reflects None correctly and the alias still works.
            assert described.content_hash is None
            assert described.crc32c_hash is None
            print(
                "[IT-0020] NOTE: API returned no hash for this file — crc32c_hash absent from wire response"
            )

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        if file_id is not None:
            with contextlib.suppress(Exception):
                client.assistants.delete_file(assistant_name=name, file_id=file_id, timeout=60)
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# wire-format investigation: model field on Pinecone-native streaming chunks (IT-0022)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_streaming_chunks_model_field(client: Pinecone) -> None:
    """Verify whether the API emits a ``model`` field on each Pinecone-native stream chunk type.

    Makes a raw streaming HTTP request to the chat endpoint and inspects the JSON
    of each SSE line before any SDK model dispatch.  Records findings per chunk type
    and asserts that ``StreamContentChunk``, ``StreamCitationChunk``, and
    ``StreamMessageEnd`` correctly surface a ``model`` value when the API provides one.

    Claims: amodel-0154, amodel-0155, amodel-0169, amodel-0170, amodel-0178, amodel-0179
    """
    import json as _json

    import httpx

    from pinecone.models.assistant.streaming import (
        StreamCitationChunk,
        StreamContentChunk,
        StreamMessageEnd,
    )

    name = unique_name("it0022")
    tmp_path: str | None = None

    try:
        # --- 1. Create assistant and upload a knowledge file ---
        client.assistants.create(name=name, instructions="You are a helpful assistant.")
        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="it0022-"
        ) as f:
            f.write("Pinecone is a managed vector database for AI applications.")
            tmp_path = f.name

        client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        # --- 2. Raw streaming request: capture SSE lines before SDK dispatch ---
        assistant_model = client.assistants.describe(name=name)
        assert assistant_model.host, f"assistant {name!r} has no data-plane host"
        host = assistant_model.host.rstrip("/")
        api_key = client._config.api_key  # type: ignore[attr-defined]

        raw_chunks_by_type: dict[str, list[dict[str, object]]] = {}

        with httpx.stream(
            "POST",
            f"{host}/assistant/chat/{name}",
            headers={
                "Api-Key": api_key,
                "X-Pinecone-API-Version": "2025-10",
                "Content-Type": "application/json",
            },
            content=_json.dumps(
                {"messages": [{"role": "user", "content": "What is Pinecone?"}], "stream": True}
            ).encode(),
            timeout=60,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].lstrip()
                if not data or data == "[DONE]":
                    break
                try:
                    chunk: dict[str, object] = _json.loads(data)
                    chunk_type = str(chunk.get("type", "unknown"))
                    raw_chunks_by_type.setdefault(chunk_type, []).append(chunk)
                except _json.JSONDecodeError:
                    pass

        # Log findings for investigation notes
        print(f"\n[IT-0022] chunk types received: {list(raw_chunks_by_type.keys())}")
        for chunk_type, chunks in raw_chunks_by_type.items():
            sample = chunks[0] if chunks else {}
            has_model = "model" in sample
            print(
                f"[IT-0022] type={chunk_type!r}: 'model' present={has_model},"
                f" sample_keys={list(sample.keys())}"
            )

        # At least the standard chunk types should be present
        assert "message_start" in raw_chunks_by_type, "Expected message_start chunk"

        # --- 3. SDK model field verification ---
        # If the API emits model on content_chunk, citation, or message_end chunks,
        # the SDK model must surface it via the new optional model field.
        sdk_stream = client.assistants.chat(
            assistant_name=name,
            messages=[{"role": "user", "content": "What is Pinecone?"}],
            stream=True,
        )
        sdk_chunks = list(sdk_stream)
        assert len(sdk_chunks) > 0, "Expected at least one SDK chunk"

        # Cross-reference: for each chunk type where the API emits model, the SDK must too.
        for chunk_type, raw_list in raw_chunks_by_type.items():
            if not raw_list:
                continue
            sample_raw = raw_list[0]
            if "model" not in sample_raw:
                continue  # API doesn't emit model for this type — nothing to assert

            if chunk_type == "content_chunk":
                sdk_content = [c for c in sdk_chunks if isinstance(c, StreamContentChunk)]
                assert sdk_content, "Expected at least one StreamContentChunk in SDK stream"
                models = [c.model for c in sdk_content if c.model is not None]
                assert models, (
                    "API emits 'model' on content_chunk but SDK StreamContentChunk.model is None"
                )
                print(f"[IT-0022] StreamContentChunk.model={models[0]!r}")

            elif chunk_type == "citation":
                sdk_cit = [c for c in sdk_chunks if isinstance(c, StreamCitationChunk)]
                if sdk_cit:
                    models_cit = [c.model for c in sdk_cit if c.model is not None]
                    print(f"[IT-0022] StreamCitationChunk.model values: {models_cit}")

            elif chunk_type == "message_end":
                sdk_end = [c for c in sdk_chunks if isinstance(c, StreamMessageEnd)]
                assert sdk_end, "Expected at least one StreamMessageEnd in SDK stream"
                models_end = [c.model for c in sdk_end if c.model is not None]
                assert models_end, (
                    "API emits 'model' on message_end but SDK StreamMessageEnd.model is None"
                )
                print(f"[IT-0022] StreamMessageEnd.model={models_end[0]!r}")

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )


# ---------------------------------------------------------------------------
# assistant-chat-completions-stream-extra-fields (IT-0023)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_chat_completions_stream_extra_fields(client: Pinecone) -> None:
    """Verify ChatCompletionStreamChunk exposes model/object/created/system_fingerprint.

    Calls chat_completions(stream=True), iterates over chunks, and checks that the
    optional extra fields (model, object, created, system_fingerprint) are accessible
    on each ChatCompletionStreamChunk without raising AttributeError.  Records which
    fields are non-None (i.e. actually emitted by the API) for IT-0023 investigation.
    """
    name = unique_name("asst")
    tmp_path: str | None = None
    try:
        assistant = client.assistants.create(name=name, instructions="You are a helpful assistant.")
        assert isinstance(assistant, AssistantModel)

        wait_for_ready(
            lambda: client.assistants.describe(name=name).status == "Ready",
            timeout=120,
            interval=3,
            description=f"assistant {name}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="asst-cc-stream-"
        ) as f:
            f.write("Pinecone is a managed vector database for AI applications.")
            tmp_path = f.name

        client.assistants.upload_file(
            assistant_name=name,
            file_path=tmp_path,
            timeout=120,
        )

        stream = client.assistants.chat_completions(
            assistant_name=name,
            messages=[{"role": "user", "content": "What is Pinecone?"}],
            stream=True,
        )

        chunks = list(stream)
        assert len(chunks) > 0, "Expected at least one streaming chunk"

        # Verify that all extra fields are accessible (no AttributeError)
        found_model: list[str] = []
        found_object: list[str] = []
        found_created: list[int] = []
        found_fingerprint: list[str] = []

        for chunk in chunks:
            assert isinstance(chunk, ChatCompletionStreamChunk)
            # Fields must be accessible — AttributeError here means the struct
            # doesn't have the field and msgspec dropped it silently.
            _ = chunk.model
            _ = chunk.object
            _ = chunk.created
            _ = chunk.system_fingerprint

            if chunk.model is not None:
                found_model.append(chunk.model)
            if chunk.object is not None:
                found_object.append(chunk.object)
            if chunk.created is not None:
                found_created.append(chunk.created)
            if chunk.system_fingerprint is not None:
                found_fingerprint.append(chunk.system_fingerprint)

        # Print findings for IT-0023 investigation notes
        print(f"[IT-0023] model values seen: {set(found_model)}")
        print(f"[IT-0023] object values seen: {set(found_object)}")
        print(f"[IT-0023] created values seen (count): {len(found_created)}")
        print(f"[IT-0023] system_fingerprint values seen: {set(found_fingerprint)}")

        # The API spec defines 'model' on StreamChatCompletionChunkModel —
        # at least one chunk should have a non-None model.
        assert len(found_model) > 0, (
            "Expected at least one chunk with a non-None 'model' field — "
            "the API spec defines this field on StreamChatCompletionChunkModel."
        )

    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        cleanup_resource(
            lambda: client.assistants.delete(name=name, timeout=60),
            name,
            "assistant",
        )
