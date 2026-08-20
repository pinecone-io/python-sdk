"""2026-07 conformance for the five assistant_data file-lifecycle operations.

``assistant_data_2026-07.oas.yaml`` reworks the write half of the file surface:

* ``upload_file`` (line 130) loses the ``metadata`` query parameter — it is now
  a ``metadata`` multipart form field on ``FileData`` (line 1542) — and answers
  ``202`` with an ``OperationModel`` plus a ``Location`` header instead of
  ``200`` with the file (line 165).
* ``upsert_file`` (line 334) is the same envelope on ``PUT``.
* ``delete_file`` (lines 483, 496) answers ``202`` with an ``OperationModel``
  when the deletion is asynchronous, or ``204`` with no body when the backend
  removed the file at once.
* ``describe_file`` (line 240) and ``list_files`` (line 29) are unchanged
  request-side; ``AssistantFileModel`` lost ``error_message``/``percent_done``
  and gained ``size``.

Backend behavior is authoritative and matches: ``operation_accepted_response``
builds the 202 + ``Location: /assistant/operations/{kb}/{op}`` body
(``svc-knowledge-engine/src/ingest/service/routes/v202604/mod.rs:45-74``), the
delete handler maps ``DeleteOutcome::Immediate``/``AlreadyDeleted`` to ``204``
(same file, 241-275), ``FileUploadParams::try_from`` turns a ``metadata``
*query* parameter into a ``400 INVALID_ARGUMENT`` rather than ignoring it (same
file, 306-330), and ``classify_upload_field`` accepts exactly the ``file`` and
``metadata`` multipart parts and rejects anything else
(``svc-knowledge-engine/src/ingest/service/handler.rs:1113-1142``). All at
pinecone-db@f6fd0a4019.

Paths carry the ``/assistant`` prefix the SDK really sends, which the spec's
``servers`` URL omits (#173, registered as a ``base_path_overrides`` entry).
"""

from __future__ import annotations

import io
import re
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import httpx
import orjson
import pytest
import respx
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pinecone._internal.config import PineconeConfig
from pinecone.client.assistants import Assistants
from pinecone.errors.exceptions import ApiError, PineconeError
from pinecone.models.assistant.file_model import AssistantFileModel
from pinecone.models.assistant.list import ListFilesResponse
from pinecone.models.assistant.operation import OperationModel
from tests.unit.conformance import api_op

BASE_URL = "https://api.test.pinecone.io"
CONTROL_URL = f"{BASE_URL}/assistant"
DATA_HOST = "https://prod-1-data.ke.pinecone.io"
DATA_URL = f"{DATA_HOST}/assistant"

ASSISTANT_NAME = "files-conformance"
FILE_ID = "ae79e447-b89e-4994-994b-3232ca52a654"
UPSERT_FILE_ID = "my-file-id-123"
OPERATION_ID = "op-1234-abcd-5678"

ASSISTANT: dict[str, Any] = {
    "name": ASSISTANT_NAME,
    "status": "Ready",
    "host": DATA_HOST,
    "region": "us",
}

FILE: dict[str, Any] = {
    "id": FILE_ID,
    "name": "pinecone-guide.pdf",
    "size": 25000,
    "status": "Available",
    "multimodal": False,
    "metadata": {"tags": ["report", "Q4"]},
    "signed_url": None,
    "created_on": "2026-07-01T00:00:00Z",
    "updated_on": "2026-07-01T00:01:00Z",
}

FILE_LIST: dict[str, Any] = {
    "files": [FILE],
    "pagination": {"next": "eyJza2lwX3Bhc3QiOiI5OTNlNzRhIn0="},
}


def operation(**overrides: Any) -> dict[str, Any]:
    """A 2026-07 ``OperationModel`` body, with every field the schema requires."""
    body: dict[str, Any] = {
        "id": OPERATION_ID,
        "operation_type": "upload_file",
        "file_id": FILE_ID,
        "status": "Processing",
        "created_on": "2026-07-01T00:00:00Z",
        "completed_on": None,
        "percent_complete": 0,
        "error_message": None,
        "ingestion_units": None,
    }
    body.update(overrides)
    return body


ACCEPTED_UPLOAD = operation()
ACCEPTED_UPSERT = operation(operation_type="upsert_file", file_id=UPSERT_FILE_ID)
ACCEPTED_DELETE = operation(operation_type="delete_file")
COMPLETED_UPLOAD = operation(
    status="Completed",
    percent_complete=100,
    completed_on="2026-07-01T00:00:30Z",
    ingestion_units=12.5,
)

METADATA_QUERY_PARAM_ERROR = (
    "metadata query parameter is not supported in this API version; include metadata "
    "as a multipart form field instead"
)


def multipart_fields(request: httpx.Request) -> dict[str, bytes]:
    """The multipart parts of *request*, by form-field name.

    The backend seals the set of field names it accepts, so the test has to see
    the real names rather than trust that the right bytes appear somewhere in
    the payload.
    """
    content_type = request.headers["content-type"]
    match = re.search(r"boundary=([^;]+)", content_type)
    assert match is not None, content_type
    boundary = match.group(1).strip('"').encode()
    fields: dict[str, bytes] = {}
    for part in request.content.split(b"--" + boundary):
        head, separator, body = part.partition(b"\r\n\r\n")
        if not separator:
            continue
        name = re.search(rb'\bname="([^"]*)"', head)
        if name is None:
            continue
        fields[name.group(1).decode()] = body[:-2] if body.endswith(b"\r\n") else body
    return fields


@pytest.fixture
def assistants(respx_mock: respx.MockRouter) -> Iterator[Assistants]:
    respx_mock.get(f"{CONTROL_URL}/assistants/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=ASSISTANT)
    )
    client = Assistants(config=PineconeConfig(api_key="conformance-key", host=BASE_URL))
    yield client
    client.close()


def mock_upload_completion(respx_mock: respx.MockRouter, file_id: str = FILE_ID) -> None:
    respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}/{OPERATION_ID}").mock(
        return_value=httpx.Response(200, json=COMPLETED_UPLOAD)
    )
    respx_mock.get(f"{DATA_URL}/files/{ASSISTANT_NAME}/{file_id}").mock(
        return_value=httpx.Response(200, json={**FILE, "id": file_id})
    )


@api_op("assistant_data:upload_file")
def test_upload_file(claim: Any, assistants: Assistants, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{DATA_URL}/files/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(
            202,
            json=ACCEPTED_UPLOAD,
            headers={"Location": f"/assistant/operations/{ASSISTANT_NAME}/{OPERATION_ID}"},
        )
    )
    mock_upload_completion(respx_mock)

    result = assistants.upload_file(
        assistant_name=ASSISTANT_NAME,
        file_stream=io.BytesIO(b"%PDF-1.7 pinecone"),
        file_name="pinecone-guide.pdf",
        metadata={"tags": ["report", "Q4"]},
        multimodal=True,
        timeout=-1,
    )
    assert isinstance(result, AssistantFileModel)
    assert result.id == FILE_ID
    assert result.size == 25000

    request = route.calls.last.request
    fields = multipart_fields(request)
    assert set(fields) == {"file", "metadata"}
    assert fields["file"] == b"%PDF-1.7 pinecone"
    assert orjson.loads(fields["metadata"]) == {"tags": ["report", "Q4"]}
    assert "metadata" not in request.url.params
    assert request.url.params["multimodal"] == "true"

    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(
        OperationModel,
        ACCEPTED_UPLOAD,
        optional_absent=["file_id", "percent_complete", "operation_type"],
    )


@api_op("assistant_data:upsert_file")
def test_upsert_file(claim: Any, assistants: Assistants, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.put(f"{DATA_URL}/files/{ASSISTANT_NAME}/{UPSERT_FILE_ID}").mock(
        return_value=httpx.Response(
            202,
            json=ACCEPTED_UPSERT,
            headers={"Location": f"/assistant/operations/{ASSISTANT_NAME}/{OPERATION_ID}"},
        )
    )
    mock_upload_completion(respx_mock, file_id=UPSERT_FILE_ID)

    with patch("pinecone.client.assistants.time.sleep"):
        result = assistants.upload_file(
            assistant_name=ASSISTANT_NAME,
            file_stream=io.BytesIO(b"plain text"),
            file_name="notes.txt",
            file_id=UPSERT_FILE_ID,
            metadata={"published": "2025-10-01"},
        )
    assert isinstance(result, AssistantFileModel)
    assert result.id == UPSERT_FILE_ID

    request = route.calls.last.request
    fields = multipart_fields(request)
    assert set(fields) == {"file", "metadata"}
    assert orjson.loads(fields["metadata"]) == {"published": "2025-10-01"}
    assert "metadata" not in request.url.params

    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(
        OperationModel,
        ACCEPTED_UPSERT,
        optional_absent=["file_id", "created_on", "percent_complete"],
    )


@api_op("assistant_data:delete_file")
def test_delete_file(claim: Any, assistants: Assistants, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.delete(f"{DATA_URL}/files/{ASSISTANT_NAME}/{FILE_ID}").mock(
        return_value=httpx.Response(
            202,
            json=ACCEPTED_DELETE,
            headers={"Location": f"/assistant/operations/{ASSISTANT_NAME}/{OPERATION_ID}"},
        )
    )
    respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}/{OPERATION_ID}").mock(
        return_value=httpx.Response(
            200, json=operation(operation_type="delete_file", status="Completed")
        )
    )

    assert assistants.delete_file(assistant_name=ASSISTANT_NAME, file_id=FILE_ID) is None

    request = route.calls.last.request
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(
        OperationModel,
        ACCEPTED_DELETE,
        optional_absent=["file_id", "operation_type", "percent_complete"],
    )


@api_op("assistant_data:describe_file")
def test_describe_file(claim: Any, assistants: Assistants, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{DATA_URL}/files/{ASSISTANT_NAME}/{FILE_ID}").mock(
        return_value=httpx.Response(200, json=FILE)
    )

    result = assistants.describe_file(
        assistant_name=ASSISTANT_NAME, file_id=FILE_ID, include_url=True
    )
    assert isinstance(result, AssistantFileModel)
    assert result.size == 25000
    assert result.metadata == {"tags": ["report", "Q4"]}

    request = route.calls.last.request
    assert request.url.params["include_url"] == "true"
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(AssistantFileModel, FILE, optional_absent=["size", "status", "metadata"])


@api_op("assistant_data:list_files")
def test_list_files(claim: Any, assistants: Assistants, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{DATA_URL}/files/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=FILE_LIST)
    )

    page = assistants.list_files_page(
        assistant_name=ASSISTANT_NAME,
        page_size=10,
        filter={"tags": {"$in": ["report"]}},
    )
    assert isinstance(page, ListFilesResponse)
    assert [f.id for f in page.files] == [FILE_ID]
    assert page.next == FILE_LIST["pagination"]["next"]

    request = route.calls.last.request
    assert request.url.params["limit"] == "10"
    assert orjson.loads(request.url.params["filter"]) == {"tags": {"$in": ["report"]}}
    claim.assert_request(request)
    claim.assert_api_version(request)
    claim.assert_roundtrip(ListFilesResponse, FILE_LIST, optional_absent=["pagination"])


def test_delete_file_204_is_success_with_no_body(
    assistants: Assistants, respx_mock: respx.MockRouter
) -> None:
    """A file the backend could remove at once answers 204 and never polls."""
    route = respx_mock.delete(f"{DATA_URL}/files/{ASSISTANT_NAME}/{FILE_ID}").mock(
        return_value=httpx.Response(204)
    )
    operations = respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}/{OPERATION_ID}").mock(
        return_value=httpx.Response(200, json=COMPLETED_UPLOAD)
    )

    assert assistants.delete_file(assistant_name=ASSISTANT_NAME, file_id=FILE_ID) is None

    assert route.call_count == 1
    assert operations.call_count == 0
    assert route.calls.last.request.headers["x-pinecone-api-version"] == "2026-07"


def test_metadata_is_never_sent_as_a_query_parameter(
    assistants: Assistants, respx_mock: respx.MockRouter
) -> None:
    """The 2025-10 call shape is a 400 on 2026-07, so the SDK must not produce it."""
    route = respx_mock.post(f"{DATA_URL}/files/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(202, json=ACCEPTED_UPLOAD)
    )
    mock_upload_completion(respx_mock)

    assistants.upload_file(
        assistant_name=ASSISTANT_NAME,
        file_stream=io.BytesIO(b"data"),
        metadata={"created_by": "Jane Doe"},
        timeout=-1,
    )

    assert "metadata=" not in str(route.calls.last.request.url)


def test_metadata_query_param_rejection_reaches_the_caller_verbatim(
    assistants: Assistants, respx_mock: respx.MockRouter
) -> None:
    """A caller pinned to the old shape gets the backend's remediation text."""
    respx_mock.post(f"{DATA_URL}/files/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(
            400,
            json={
                "status": 400,
                "error": {"code": "INVALID_ARGUMENT", "message": METADATA_QUERY_PARAM_ERROR},
            },
        )
    )

    with pytest.raises(ApiError) as excinfo:
        assistants.upload_file(
            assistant_name=ASSISTANT_NAME, file_stream=io.BytesIO(b"data"), timeout=-1
        )

    assert excinfo.value.message == METADATA_QUERY_PARAM_ERROR
    assert "multipart form field" in str(excinfo.value)


def test_a_2025_10_file_body_on_upload_fails_pointing_at_the_operations_api(
    assistants: Assistants, respx_mock: respx.MockRouter
) -> None:
    """The old 200 ``AssistantFileModel`` upload body no longer yields the file.

    Read as the operation envelope it now is, such a body names no ``file_id``,
    so there is nothing to describe — and the SDK says so rather than returning
    a file it guessed at.
    """
    respx_mock.post(f"{DATA_URL}/files/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json={**FILE, "status": "Processing"})
    )

    with pytest.raises(PineconeError) as excinfo:
        assistants.upload_file(
            assistant_name=ASSISTANT_NAME,
            file_stream=io.BytesIO(b"data"),
            file_name="pinecone-guide.pdf",
        )

    message = str(excinfo.value)
    assert "did not name the file it created" in message
    assert "describe_operation()" in message
    assert FILE["id"] in message


def test_a_failed_upload_operation_quotes_the_server_message(
    assistants: Assistants, respx_mock: respx.MockRouter
) -> None:
    """The 400-example text from the spec must survive to the raised error."""
    server_message = "Uploaded file can only currently be either a pdf or txt file"
    respx_mock.post(f"{DATA_URL}/files/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(202, json=ACCEPTED_UPLOAD)
    )
    respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}/{OPERATION_ID}").mock(
        return_value=httpx.Response(
            200,
            json=operation(
                status="Failed",
                completed_on="2026-07-01T00:00:05Z",
                error_message=server_message,
            ),
        )
    )

    with pytest.raises(PineconeError) as excinfo:
        assistants.upload_file(
            assistant_name=ASSISTANT_NAME, file_stream=io.BytesIO(b"gif89a"), file_name="a.gif"
        )

    message = str(excinfo.value)
    assert server_message in message
    assert OPERATION_ID in message
    assert FILE_ID in message


def test_the_failure_path_does_not_read_the_removed_file_fields(
    assistants: Assistants, respx_mock: respx.MockRouter
) -> None:
    """``AssistantFileModel`` no longer carries the failure detail at all."""
    respx_mock.get(f"{DATA_URL}/files/{ASSISTANT_NAME}/{FILE_ID}").mock(
        return_value=httpx.Response(200, json={**FILE, "status": "ProcessingFailed"})
    )

    file = assistants.describe_file(assistant_name=ASSISTANT_NAME, file_id=FILE_ID)
    for removed in ("error_message", "percent_done"):
        with pytest.raises(AttributeError, match="describe_operation"):
            getattr(file, removed)


def test_a_16kb_metadata_document_still_travels_in_the_multipart_body(
    assistants: Assistants, respx_mock: respx.MockRouter
) -> None:
    """16KB is the documented ceiling; a URL could not carry it even if allowed."""
    metadata = {"notes": "x" * 16_000}
    route = respx_mock.post(f"{DATA_URL}/files/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(202, json=ACCEPTED_UPLOAD)
    )
    mock_upload_completion(respx_mock)

    assistants.upload_file(
        assistant_name=ASSISTANT_NAME,
        file_stream=io.BytesIO(b"data"),
        metadata=metadata,
        timeout=-1,
    )

    request = route.calls.last.request
    assert orjson.loads(multipart_fields(request)["metadata"]) == metadata
    assert "metadata" not in request.url.params


metadata_values = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-(2**53), max_value=2**53),
        st.text(max_size=40),
        st.dates().map(str),
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.text(min_size=1, max_size=20), children, max_size=4),
    ),
    max_leaves=12,
)


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    metadata=st.dictionaries(
        st.text(min_size=1, max_size=30), metadata_values, min_size=1, max_size=6
    )
)
def test_metadata_round_trips_through_the_multipart_field(metadata: dict[str, Any]) -> None:
    """Any metadata document arrives as valid JSON in the field, never in the URL.

    Nested containers, unicode keys and dates-as-strings all have to survive
    intact: the field is the only place 2026-07 reads metadata from, and a leak
    into the query string is a 400 rather than a silent loss.
    """
    with respx.mock:
        respx.get(f"{CONTROL_URL}/assistants/{ASSISTANT_NAME}").mock(
            return_value=httpx.Response(200, json=ASSISTANT)
        )
        route = respx.post(f"{DATA_URL}/files/{ASSISTANT_NAME}").mock(
            return_value=httpx.Response(202, json=ACCEPTED_UPLOAD)
        )
        respx.get(f"{DATA_URL}/files/{ASSISTANT_NAME}/{FILE_ID}").mock(
            return_value=httpx.Response(200, json=FILE)
        )
        client = Assistants(config=PineconeConfig(api_key="conformance-key", host=BASE_URL))
        try:
            client.upload_file(
                assistant_name=ASSISTANT_NAME,
                file_stream=io.BytesIO(b"data"),
                metadata=metadata,
                timeout=-1,
            )
        finally:
            client.close()

        request = route.calls.last.request

    fields = multipart_fields(request)
    assert set(fields) == {"file", "metadata"}
    assert orjson.loads(fields["metadata"]) == metadata
    assert "metadata" not in request.url.params


def test_no_metadata_means_no_metadata_part(
    assistants: Assistants, respx_mock: respx.MockRouter
) -> None:
    """Omitting metadata must not send an empty part — the backend seals the set."""
    route = respx_mock.post(f"{DATA_URL}/files/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(202, json=ACCEPTED_UPLOAD)
    )
    mock_upload_completion(respx_mock)

    assistants.upload_file(
        assistant_name=ASSISTANT_NAME, file_stream=io.BytesIO(b"data"), timeout=-1
    )

    request = route.calls.last.request
    assert set(multipart_fields(request)) == {"file"}
    assert "multimodal" not in request.url.params


def test_upload_polls_the_operation_not_the_file(
    assistants: Assistants, respx_mock: respx.MockRouter
) -> None:
    """Progress comes from the operation record, which reports percent_complete."""
    respx_mock.post(f"{DATA_URL}/files/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(202, json=ACCEPTED_UPLOAD)
    )
    operations = respx_mock.get(f"{DATA_URL}/operations/{ASSISTANT_NAME}/{OPERATION_ID}").mock(
        side_effect=[
            httpx.Response(200, json=operation(percent_complete=25)),
            httpx.Response(200, json=operation(percent_complete=80)),
            httpx.Response(200, json=COMPLETED_UPLOAD),
        ]
    )
    describe = respx_mock.get(f"{DATA_URL}/files/{ASSISTANT_NAME}/{FILE_ID}").mock(
        return_value=httpx.Response(200, json=FILE)
    )

    with patch("pinecone.client.assistants.time.sleep"):
        assistants.upload_file(assistant_name=ASSISTANT_NAME, file_stream=io.BytesIO(b"data"))

    assert operations.call_count == 3
    assert describe.call_count == 1
    for call in operations.calls:
        assert call.request.headers["x-pinecone-api-version"] == "2026-07"


def test_one_cached_data_plane_client_serves_every_file_operation(
    respx_mock: respx.MockRouter,
) -> None:
    """List, describe and delete share the cached client — no per-call rebuild.

    ``_list_files_http``/``_upsert_http`` used to construct an uncached client
    per call, which meant a fresh control-plane describe for every list and
    every upsert.
    """
    describe_assistant = respx_mock.get(f"{CONTROL_URL}/assistants/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=ASSISTANT)
    )
    respx_mock.get(f"{DATA_URL}/files/{ASSISTANT_NAME}").mock(
        return_value=httpx.Response(200, json=FILE_LIST)
    )
    respx_mock.get(f"{DATA_URL}/files/{ASSISTANT_NAME}/{FILE_ID}").mock(
        return_value=httpx.Response(200, json=FILE)
    )
    respx_mock.delete(f"{DATA_URL}/files/{ASSISTANT_NAME}/{FILE_ID}").mock(
        return_value=httpx.Response(204)
    )

    client = Assistants(config=PineconeConfig(api_key="conformance-key", host=BASE_URL))
    try:
        client.list_files_page(assistant_name=ASSISTANT_NAME)
        client.list_files_page(assistant_name=ASSISTANT_NAME)
        client.describe_file(assistant_name=ASSISTANT_NAME, file_id=FILE_ID)
        client.delete_file(assistant_name=ASSISTANT_NAME, file_id=FILE_ID)
    finally:
        client.close()

    assert describe_assistant.call_count == 1
