"""Assistants namespace: manage assistants and their files, and chat with them."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from typing import IO, TYPE_CHECKING, Any
from urllib.parse import quote

import msgspec
import msgspec.structs
import orjson

from pinecone._internal.adapters.assistants_adapter import AssistantsAdapter
from pinecone._internal.constants import (
    ASSISTANT_API_VERSION,
    ASSISTANT_EVALUATION_BASE_URL,
    DEFAULT_BASE_URL,
)
from pinecone._internal.validation import require_str
from pinecone.client._assistants_legacy import AssistantsLegacyNamespaceMixin
from pinecone.errors.exceptions import (
    NotFoundError,
    PineconeError,
    PineconeTimeoutError,
    PineconeValueError,
)
from pinecone.models.assistant.chat import ChatCompletionResponse, ChatResponse
from pinecone.models.assistant.context import ContextResponse
from pinecone.models.assistant.evaluation import AlignmentResult
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
    ChatCompletionStream,
    ChatCompletionStreamChunk,
    ChatStream,
    ChatStreamChunk,
)
from pinecone.models.pagination import Page, Paginator

if TYPE_CHECKING:
    from pinecone._internal.config import PineconeConfig
    from pinecone._internal.http_client import HTTPClient

logger = logging.getLogger(__name__)

_VALID_REGIONS = ("us", "eu")
_CREATE_POLL_INTERVAL_SECONDS = 0.5
_DELETE_POLL_INTERVAL_SECONDS = 5
_UPLOAD_POLL_INTERVAL_SECONDS = 5

# A read timeout on an SSE stream measures the gap between tokens, and
# POST /chat/{name}/chat/completions sends nothing during that gap: no keepalive,
# unlike POST /chat/{name}, which heartbeats every 15s. A model that thinks for
# longer than PineconeConfig.timeout (30s) is therefore indistinguishable from a
# dead connection. Both endpoints get the raised floor so they fail alike.
_STREAM_TIMEOUT_FLOOR_SECONDS = 300.0

# Checked client-side because the backend answers an unparseable filter with a
# 400 that does not enumerate what it would have accepted.
_VALID_OPERATION_TYPES = ("upload_file", "upsert_file", "update_file_metadata", "delete_file")
_VALID_OPERATION_STATUSES = ("Processing", "Completed", "Failed")

# Statuses that mean delete() will never see the 404 it polls for: the
# server-side reconciler only rescues Create operations, so an assistant that
# fails while being deleted stays put and an untimed poll spins forever.
# "Terminating" is deliberately absent — that one is a delete in flight.
_DELETE_TERMINAL_STATUSES = ("Failed", "InitializationFailed")


def _operation_target(file_id: str | None) -> str:
    return f"file {file_id!r}" if file_id is not None else "the file"


def _stream_upload_name(file_name: str | None) -> str:
    """The multipart filename to send for a ``file_stream`` upload.

    The server types an uploaded file by its filename extension alone — it
    never sniffs the bytes — so a stream sent without a usable filename is a
    guaranteed 400. Refusing here saves the round trip and names the fix.
    """
    if file_name is None or not os.path.splitext(file_name)[1].lstrip("."):
        raise PineconeValueError(
            "upload_file(file_stream=...) needs a file_name with an extension: the "
            "server types an uploaded file by its extension alone, so a stream with "
            "no filename is rejected whatever the bytes are. Pass "
            "file_name='report.pdf' (supported extensions: .txt, .pdf, .json, .md, "
            ".docx)."
        )
    return file_name


def _validate_choice(name: str, value: str, valid: tuple[str, ...]) -> str:
    if value not in valid:
        raise PineconeValueError(f"{name} must be one of {valid!r}, got {value!r}")
    return value


def _stream_timeout(config_timeout: float, override: float | None) -> float:
    """Resolve the HTTP timeout for a streaming chat request.

    An explicit per-call *override* is used verbatim, including values below the
    floor. Otherwise *config_timeout* is raised to
    :data:`_STREAM_TIMEOUT_FLOOR_SECONDS` — raised only, never lowered, so a
    client configured with a longer timeout keeps it.
    """
    if override is not None:
        return override
    return max(config_timeout, _STREAM_TIMEOUT_FLOOR_SECONDS)


def _operation_failure_message(action: str, file_id: str | None, operation: OperationModel) -> str:
    """The failure text a caller sees when a file operation reports ``"Failed"``.

    The server's ``error_message`` is quoted verbatim: it is the only part that
    says *why* ("Uploaded file can only currently be either a pdf or txt file"),
    and paraphrasing it is the difference between a caller fixing the input and
    a caller guessing.
    """
    detail = operation.error or "the server reported no error message"
    return (
        f"{action} of {_operation_target(file_id)} failed "
        f"(operation_id={operation.operation_id!r}): {detail}"
    )


class Assistants(AssistantsLegacyNamespaceMixin):
    """Control-plane operations for Pinecone assistants.

    A Pinecone assistant is a managed question-answering service grounded in
    documents you upload to it: create the assistant, upload files, then chat
    against them and get answers with citations back to the files that
    supported each claim.

    Reached as ``pc.assistants``; not constructed directly. Unlike an index,
    which you query for records you then feed to your own model, an assistant
    does the retrieval and the generation for you.

    Examples:

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone(api_key="your-api-key")
            for assistant in pc.assistants.list():
                print(assistant.name, assistant.status)

    .. seealso::
       `Error handling <https://sdk.pinecone.io/python/guides/error-handling.html>`__:
       the exceptions any of these methods can raise, and which ones are
       worth retrying.
    """

    def __init__(self, config: PineconeConfig) -> None:
        from pinecone._internal.config import PineconeConfig as _PineconeConfig
        from pinecone._internal.http_client import HTTPClient as _HTTPClient

        self._config = config
        # Internal env-var escape hatches (undocumented, used by Pinecone CI
        # to redirect to non-prod KE clusters). Precedence: explicit
        # config.host (user) > env var > hardcoded default.
        env_control_host = os.getenv("PINECONE_PLUGIN_ASSISTANT_CONTROL_HOST")
        cp_host = (config.host or env_control_host or DEFAULT_BASE_URL).rstrip("/")
        cp_config = _PineconeConfig(
            api_key=config.api_key,
            host=f"{cp_host}/assistant",
            timeout=config.timeout,
            additional_headers=config.additional_headers,
            source_tag=config.source_tag or "",
            proxy_url=config.proxy_url or "",
            proxy_headers=config.proxy_headers,
            ssl_ca_certs=config.ssl_ca_certs,
            ssl_verify=config.ssl_verify,
            connection_pool_maxsize=config.connection_pool_maxsize,
            retry_config=config.retry_config,
        )
        self._http = _HTTPClient(cp_config, ASSISTANT_API_VERSION)
        self._adapter = AssistantsAdapter()
        self._data_plane_clients: dict[str, HTTPClient] = {}

        env_data_host = os.getenv("PINECONE_PLUGIN_ASSISTANT_DATA_HOST")
        eval_host = (
            f"{env_data_host.rstrip('/')}/assistant"
            if env_data_host
            else ASSISTANT_EVALUATION_BASE_URL
        )
        eval_config = _PineconeConfig(
            api_key=config.api_key,
            host=eval_host,
            timeout=config.timeout,
            additional_headers=config.additional_headers,
            source_tag=config.source_tag or "",
            proxy_url=config.proxy_url or "",
            proxy_headers=config.proxy_headers,
            ssl_ca_certs=config.ssl_ca_certs,
            ssl_verify=config.ssl_verify,
            connection_pool_maxsize=config.connection_pool_maxsize,
            retry_config=config.retry_config,
        )
        self._eval_http = _HTTPClient(eval_config, ASSISTANT_API_VERSION)

    def _attach_ref(self, model: AssistantModel) -> AssistantModel:
        """Attach a back-reference to *self* on *model* for legacy method delegation.

        Called after every API response that constructs an :class:`AssistantModel`
        so that legacy instance methods (``assistant.chat(...)``, etc.) can
        delegate to this :class:`Assistants` namespace.

        ``AssistantModel`` is declared with ``dict=True`` so instances have a
        ``__dict__``. We write directly into ``model.__dict__`` to bypass
        msgspec's field-restricted ``__setattr__``, which only allows setting
        declared struct fields.
        """
        model.__dict__["_assistants"] = self
        return model

    def close(self) -> None:
        """Release the HTTP connections held by this namespace.

        Call this when you're done using ``pc.assistants`` to free pooled
        connections, including any opened for individual assistants.
        """
        self._http.close()
        self._eval_http.close()
        for client in self._data_plane_clients.values():
            client.close()
        self._data_plane_clients.clear()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "Assistants()"

    def _data_plane_http(self, assistant_name: str) -> HTTPClient:
        """Return an HTTPClient targeting the assistant's data-plane host.

        Caches clients by assistant name to avoid repeated describe calls.
        """
        require_str("assistant_name", assistant_name)
        if assistant_name not in self._data_plane_clients:
            from pinecone._internal.config import PineconeConfig as _PineconeConfig
            from pinecone._internal.http_client import HTTPClient as _HTTPClient

            assistant = self.describe(name=assistant_name)
            if not assistant.host:
                raise PineconeValueError(f"Assistant '{assistant_name}' has no data-plane host")
            data_config = _PineconeConfig(
                api_key=self._config.api_key,
                host=f"{assistant.host.rstrip('/')}/assistant",
                timeout=self._config.timeout,
                additional_headers=self._config.additional_headers,
                source_tag=self._config.source_tag or "",
                proxy_url=self._config.proxy_url or "",
                proxy_headers=self._config.proxy_headers,
                ssl_ca_certs=self._config.ssl_ca_certs,
                ssl_verify=self._config.ssl_verify,
                connection_pool_maxsize=self._config.connection_pool_maxsize,
                retry_config=self._config.retry_config,
            )
            self._data_plane_clients[assistant_name] = _HTTPClient(
                data_config, ASSISTANT_API_VERSION
            )
        return self._data_plane_clients[assistant_name]

    def _poll_operation_until_done(
        self,
        assistant_name: str,
        operation_id: str,
        timeout: float | None,
        *,
        action: str,
        file_id: str | None = None,
        poll_interval: float = _UPLOAD_POLL_INTERVAL_SECONDS,
    ) -> OperationModel:
        """Poll :meth:`describe_operation` until the operation is done.

        Returns the terminal :class:`OperationModel`. Raises
        :exc:`PineconeError` when the operation reports ``"Failed"``, quoting
        the server's ``error_message`` verbatim, and
        :exc:`PineconeTimeoutError` when *timeout* elapses first.
        """
        start = time.monotonic()
        while True:
            operation = self.describe_operation(
                assistant_name=assistant_name, operation_id=operation_id
            )

            if operation.status != "Processing":
                if operation.status == "Failed":
                    raise PineconeError(_operation_failure_message(action, file_id, operation))
                return operation

            if timeout is not None:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    raise PineconeTimeoutError(
                        f"{action} of {_operation_target(file_id)} did not finish within "
                        f"{timeout}s (operation_id={operation_id!r}, "
                        f"percent_complete={operation.percent_complete}). The operation is "
                        "still running server-side; call describe_operation() to follow it."
                    )
            time.sleep(poll_interval)

    def upload_file(
        self,
        *,
        assistant_name: str,
        file_path: str | None = None,
        file_stream: IO[bytes] | None = None,
        file_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        multimodal: bool | None = None,
        file_id: str | None = None,
        timeout: float | None = None,
    ) -> AssistantFileModel:
        """Upload a file to a Pinecone assistant.

        Uploads a file from a local path or an in-memory byte stream, then
        waits until processing finishes before returning. The server decides
        how to process the file from its extension, which must be ``.txt``,
        ``.pdf``, ``.json``, ``.md`` or ``.docx``.

        Args:
            assistant_name: Name of the target assistant.
            file_path: Path to a local file to upload. Mutually exclusive
                with *file_stream*.
            file_stream: An open byte stream to upload. Mutually exclusive
                with *file_path*. Requires *file_name*.
            file_name: Filename for *file_stream*, including its extension,
                e.g. ``"q3-revenue-review.pdf"``. Required with
                *file_stream*. Ignored with *file_path*, whose basename is
                used instead.
            metadata: Metadata to attach to the file, e.g.
                ``{"department": "finance"}``. The *filter* argument of
                :meth:`list_files`, :meth:`chat` and :meth:`context` matches
                against it.
            multimodal: Whether to enable multimodal processing for PDFs.
            file_id: Optional identifier for the uploaded file. When given,
                any existing file with that id is replaced. Otherwise the
                server assigns one.
            timeout: Seconds to wait for processing to complete. ``None``
                (default) polls indefinitely. Use ``-1`` to return as soon as
                the server accepts the upload, while the file may still be
                ``"Processing"``. Raises :exc:`PineconeTimeoutError` if
                processing is not done before the deadline.

        Returns:
            :class:`AssistantFileModel` describing the uploaded file, once
            processing completes.

        Raises:
            :exc:`PineconeTypeError`: If *assistant_name* is not a ``str``, or
                *file_id* is given and is not a ``str``.
            :exc:`PineconeValueError`: If both or neither of *file_path*
                and *file_stream* are provided, if *file_path* does not
                exist, or if *file_stream* is used without a *file_name*
                carrying a file extension.
            :exc:`PineconeTimeoutError`: If processing does not complete
                before *timeout*.
            :exc:`PineconeError`: If processing fails.

        Examples:
            Upload from a local path. The basename supplies the extension the
            server types the file by:

            .. code-block:: python

                file = pc.assistants.upload_file(
                    assistant_name="research-assistant",
                    file_path="/data/q3-revenue-review.pdf",
                )
                print(file.status)

            Or upload from an open byte stream instead. Pass exactly one of
            ``file_path`` and ``file_stream``; a stream needs ``file_name`` to
            carry the extension a path would have supplied:

            >>> import io
            >>> file = pc.assistants.upload_file(
            ...     assistant_name="research-assistant",
            ...     file_stream=io.BytesIO(b"%PDF-1.4 Q3 revenue review"),
            ...     file_name="q3-revenue-review.pdf",
            ...     metadata={"department": "finance", "quarter": "2024-Q3"},
            ... )
            >>> file.status
            'Available'
        """
        import json as _json

        if (file_path is None) == (file_stream is None):
            raise PineconeValueError("Exactly one of file_path or file_stream must be provided")
        if file_id is not None:
            require_str("file_id", file_id)

        opened_file: IO[bytes] | None = None
        if file_path is not None:
            if not os.path.isfile(file_path):
                raise PineconeValueError(f"File not found: {file_path}")
            opened_file = open(file_path, "rb")  # noqa: SIM115
            handle: IO[bytes] = opened_file
            upload_name = os.path.basename(file_path)
        else:
            if file_stream is None:
                raise PineconeValueError("Exactly one of file_path or file_stream must be provided")
            handle = file_stream
            upload_name = _stream_upload_name(file_name)

        try:
            data_http = self._data_plane_http(assistant_name)

            form: dict[str, Any] = {"file": (upload_name, handle)}
            if metadata is not None:
                form["metadata"] = (None, _json.dumps(metadata))
            params: dict[str, str] = {}
            if multimodal is not None:
                params["multimodal"] = str(multimodal).lower()

            if file_id is not None:
                action = "Upsert"
                logger.info(
                    "Upserting file %r (id=%s) to assistant %r",
                    upload_name,
                    file_id,
                    assistant_name,
                )
                response = data_http.put(
                    f"/files/{quote(assistant_name, safe='')}/{quote(file_id, safe='')}",
                    files=form,
                    params=params,
                )
            else:
                action = "Upload"
                logger.info("Uploading file %r to assistant %r", upload_name, assistant_name)
                response = data_http.post(
                    f"/files/{quote(assistant_name, safe='')}", files=form, params=params
                )
            operation = self._adapter.to_operation(response.content)
        finally:
            if opened_file is not None:
                opened_file.close()

        uploaded_id = file_id if file_id is not None else operation.file_id
        if uploaded_id is None:
            raise PineconeError(
                f"{action} of {upload_name!r} was accepted (operation_id="
                f"{operation.operation_id!r}) but the response did not name the file it "
                "created, so there is nothing to describe. Call describe_operation() with "
                "that operation id to find the file."
            )
        logger.debug(
            "%s of %r accepted (file_id=%s, operation_id=%s)",
            action,
            upload_name,
            uploaded_id,
            operation.operation_id,
        )

        if timeout == -1:
            return self.describe_file(assistant_name=assistant_name, file_id=uploaded_id)

        self._poll_operation_until_done(
            assistant_name,
            operation.operation_id,
            timeout,
            action=action,
            file_id=uploaded_id,
        )
        return self.describe_file(assistant_name=assistant_name, file_id=uploaded_id)

    def describe_file(
        self,
        *,
        assistant_name: str,
        file_id: str,
        include_url: bool = False,
    ) -> AssistantFileModel:
        """Get the status and metadata of a file uploaded to an assistant.

        Args:
            assistant_name: Name of the assistant that owns the file.
            file_id: Unique identifier of the file to retrieve.
            include_url: If ``True``, include a signed download URL in the
                response. Defaults to ``False``.

        Returns:
            :class:`AssistantFileModel` with ``id``, ``name``, ``status``,
            ``metadata``, ``size``, ``multimodal``, ``created_on``,
            ``updated_on`` and ``content_hash``, plus ``signed_url`` when
            *include_url* is ``True``.

        Raises:
            :exc:`PineconeTypeError`: If *assistant_name* or *file_id* is not a
                ``str``.
            :exc:`NotFoundError`: If the file does not exist.

        Examples:
            >>> file = pc.assistants.describe_file(
            ...     assistant_name="research-assistant",
            ...     file_id="file-abc123",
            ... )
            >>> file.status
            'Available'

        .. seealso::
           :meth:`list_files`: every file on the assistant. That listing
           drops a ``"ProcessingFailed"`` file once it is old enough; this
           method still returns it by id.
        """
        require_str("file_id", file_id)
        data_http = self._data_plane_http(assistant_name)
        params: dict[str, str] = {}
        if include_url:
            params["include_url"] = "true"
        logger.info("Describing file %r in assistant %r", file_id, assistant_name)
        response = data_http.get(
            f"/files/{quote(assistant_name, safe='')}/{quote(file_id, safe='')}",
            params=params,
        )
        return self._adapter.to_file(response.content)

    def list_files(
        self,
        *,
        assistant_name: str,
        filter: dict[str, Any] | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> Paginator[AssistantFileModel]:
        """List files for an assistant with lazy pagination.

        A ``"ProcessingFailed"`` file drops out of this listing once its
        ``created_on`` passes the listing's age cutoff. It is not deleted:
        :meth:`describe_file` still returns it by id.

        Args:
            assistant_name: Name of the assistant whose files to list.
            filter: Metadata filter, e.g. ``{"department": "finance"}``. Only
                files whose metadata matches are listed.
            limit: Maximum number of files to yield across all pages. ``None``
                (default) yields all files.
            pagination_token: Token to resume pagination from a previous call.

        Returns:
            :class:`Paginator` over :class:`AssistantFileModel` objects.
            Supports ``for`` loops, ``.to_list()``, ``.pages()``, and
            ``limit``.

        Raises:
            :exc:`PineconeTypeError`: If *assistant_name* is not a ``str``.
            :exc:`NotFoundError`: If the assistant does not exist.

        Examples:
            .. code-block:: python

                for f in pc.assistants.list_files(assistant_name="research-assistant"):
                    print(f.name, f.status)

            The paginator fetches pages lazily as you iterate. Call
            ``to_list()`` instead when you want every file materialized up
            front:

            .. code-block:: python

                files = pc.assistants.list_files(assistant_name="research-assistant").to_list()

        .. seealso::
           - :meth:`describe_file`: one file by id, with no age filter.
           - :meth:`list_files_page`: one page at a time, when you want to
             hold the continuation token yourself.
           - `Pagination <https://sdk.pinecone.io/python/guides/pagination.html>`__:
             how the paginator and the continuation tokens work.
        """
        logger.info("Listing files for assistant %r", assistant_name)

        def fetch_page(token: str | None) -> Page[AssistantFileModel]:
            result = self.list_files_page(
                assistant_name=assistant_name,
                pagination_token=token,
                filter=filter,
            )
            return Page(items=result.files, pagination_token=result.next)

        return Paginator(fetch_page=fetch_page, initial_token=pagination_token, limit=limit)

    def list_files_page(
        self,
        *,
        assistant_name: str,
        page_size: int | None = None,
        pagination_token: str | None = None,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ListFilesResponse:
        """List one page of files for an assistant with explicit pagination control.

        Args:
            assistant_name: Name of the assistant whose files to list.
            page_size: Maximum number of files in this page. Omit it to let
                the server choose the page size.
            pagination_token: Token from a previous response to fetch the
                next page.
            filter: Metadata filter, e.g. ``{"department": "finance"}``. Only
                files whose metadata matches are listed.
            **kwargs (Any): Accepts the legacy alias ``limit`` for
                *page_size*. Passing both, or any other keyword, raises
                :exc:`PineconeValueError`.

        Returns:
            :class:`ListFilesResponse` with a ``files`` list and an optional
            ``next`` continuation token.

        Raises:
            :exc:`PineconeTypeError`: If *assistant_name* is not a ``str``.
            :exc:`NotFoundError`: If the assistant does not exist.

        Examples:
            .. code-block:: python

                page = pc.assistants.list_files_page(
                    assistant_name="research-assistant",
                    page_size=10,
                )
                for f in page.files:
                    print(f.name)
                if page.next:
                    next_page = pc.assistants.list_files_page(
                        assistant_name="research-assistant",
                        page_size=10,
                        pagination_token=page.next,
                    )

        .. seealso::
           `Pagination <https://sdk.pinecone.io/python/guides/pagination.html>`__:
           the continuation-token loop this method expects you to drive, and
           the paginator that drives it for you.
        """
        from pinecone._internal.kwargs_aliases import (
            reject_unknown_kwargs,
            remap_legacy_kwargs,
        )

        remapped = remap_legacy_kwargs(
            kwargs,
            aliases={"limit": "page_size"},
            method_name="list_files_page",
        )
        reject_unknown_kwargs(remapped, allowed={"page_size"}, method_name="list_files_page")
        if "page_size" in remapped:
            if page_size is not None:
                raise PineconeValueError(
                    "list_files_page() received both 'limit' (legacy) and 'page_size'. "
                    "Pass only one — prefer 'page_size'."
                )
            page_size = remapped["page_size"]

        import json as _json

        list_http = self._data_plane_http(assistant_name)
        params: dict[str, str | int] = {}
        if page_size is not None:
            params["limit"] = page_size
        if pagination_token is not None:
            params["pagination_token"] = pagination_token
        if filter is not None:
            params["filter"] = _json.dumps(filter)

        logger.info("Listing files page for assistant %r", assistant_name)
        response = list_http.get(f"/files/{quote(assistant_name, safe='')}", params=params)
        result = self._adapter.to_file_list(response.content)
        logger.debug(
            "Listed %d files for assistant %r (has_next=%s)",
            len(result.files),
            assistant_name,
            result.next is not None,
        )
        return result

    def delete_file(
        self,
        *,
        assistant_name: str,
        file_id: str,
        timeout: float | None = None,
    ) -> None:
        """Delete a file from a Pinecone assistant.

        Deletion can finish immediately or run as a pending operation,
        depending on the file's state. When it is pending, this method polls
        until it finishes, unless you pass ``timeout=-1``.

        Args:
            assistant_name: Name of the assistant that owns the file.
            file_id: Unique identifier of the file to delete.
            timeout: Seconds to wait for the deletion to finish. Use ``None``
                (default) to poll indefinitely. Use ``-1`` to return as soon
                as the request is accepted; the file may still exist when
                this returns. Use a positive value to poll with a deadline.
                Raises :exc:`PineconeTimeoutError` if the deletion is not
                done before the deadline.

        Returns:
            ``None``

        Raises:
            :exc:`PineconeTypeError`: If *assistant_name* or *file_id* is not a
                ``str``.
            :exc:`NotFoundError`: If *file_id* does not name a file on this
                assistant, including a file that has already been deleted.
            :exc:`PineconeError`: If the deletion operation reports failure.
            :exc:`PineconeTimeoutError`: If the deletion has not finished
                after *timeout* seconds.

        Examples:
            >>> pc.assistants.delete_file(
            ...     assistant_name="research-assistant",
            ...     file_id="file-abc123",
            ... )
        """
        require_str("file_id", file_id)
        data_http = self._data_plane_http(assistant_name)
        logger.info("Deleting file %r from assistant %r", file_id, assistant_name)
        response = data_http.delete(
            f"/files/{quote(assistant_name, safe='')}/{quote(file_id, safe='')}"
        )

        if response.status_code == 204 or not response.content:
            logger.debug("File %r was deleted immediately (no operation)", file_id)
            return

        operation = self._adapter.to_operation(response.content)
        logger.debug(
            "Deletion of file %r accepted (operation_id=%s)", file_id, operation.operation_id
        )

        if timeout == -1:
            return

        self._poll_operation_until_done(
            assistant_name,
            operation.operation_id,
            timeout,
            action="Deletion",
            file_id=file_id,
            poll_interval=_DELETE_POLL_INTERVAL_SECONDS,
        )

    def describe_operation(
        self,
        *,
        assistant_name: str,
        operation_id: str,
    ) -> OperationModel:
        """Get the current status of a long-running assistant operation.

        :meth:`upload_file` and :meth:`delete_file` poll their own operation
        for you by default. Reach for this method when you called one of
        them with ``timeout=-1`` and want to check on it later, for
        example to find the file a fire-and-forget upload created, via
        :attr:`OperationModel.file_id`.

        Args:
            assistant_name: Name of the assistant that owns the operation.
            operation_id: Identifier of the operation to describe, as
                returned by :meth:`upload_file`, :meth:`delete_file`, or
                :meth:`list_operations`.

        Returns:
            :class:`OperationModel` with ``status``, ``operation_type``,
            ``file_id``, ``percent_complete``, ``created_at``,
            ``completed_on``, ``ingestion_units`` and ``error``. ``status`` is
            ``"Processing"``, ``"Completed"`` or ``"Failed"``. Check
            ``status``, not ``error``, to tell whether the operation failed.

        Raises:
            :exc:`PineconeTypeError`: If *assistant_name* or *operation_id* is
                not a ``str``.
            :exc:`NotFoundError`: If the assistant or the operation does not
                exist. The server keeps finished operations for a limited
                time, so an old operation id also raises this.

        Examples:
            >>> operation = pc.assistants.describe_operation(
            ...     assistant_name="research-assistant",
            ...     operation_id="op-1234-abcd-5678",
            ... )
            >>> operation.status, operation.percent_complete
            ('Completed', 100)

        .. seealso::
           :meth:`list_operations`: every operation on the assistant, when
           you did not keep the operation id.
        """
        require_str("operation_id", operation_id)
        data_http = self._data_plane_http(assistant_name)
        logger.info("Describing operation %r in assistant %r", operation_id, assistant_name)
        response = data_http.get(
            f"/operations/{quote(assistant_name, safe='')}/{quote(operation_id, safe='')}"
        )
        return self._adapter.to_operation(response.content)

    def list_operations(
        self,
        *,
        assistant_name: str,
        operation_type: str | None = None,
        status: str | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> Paginator[OperationModel]:
        """List an assistant's operations with lazy pagination.

        Includes operations still in progress and finished ones, successful
        or failed. The server keeps finished operations for a limited time.

        Args:
            assistant_name: Name of the assistant whose operations to list.
            operation_type: Restrict the listing to one kind of operation. One
                of ``"upload_file"``, ``"upsert_file"``,
                ``"update_file_metadata"`` or ``"delete_file"``.
            status: Restrict the listing to one status. One of
                ``"Processing"``, ``"Completed"`` or ``"Failed"``
                (case-sensitive).
            limit: Maximum number of operations to yield across all pages.
                ``None`` (default) yields all of them.
            pagination_token: Token to resume pagination from a previous call.

        Returns:
            :class:`Paginator` over :class:`OperationModel` objects. Supports
            ``for`` loops, ``.to_list()``, ``.pages()``, and ``limit``.

        Raises:
            :exc:`PineconeTypeError`: If *assistant_name* is not a ``str``.
            :exc:`PineconeValueError`: If *operation_type* or *status* is not
                one of the values above.
            :exc:`NotFoundError`: If the assistant does not exist.

        Examples:
            .. code-block:: python

                for op in pc.assistants.list_operations(assistant_name="research-assistant"):
                    print(op.operation_id, op.status, op.percent_complete)

            Filter server-side to narrow the listing, here to uploads that
            have not finished yet:

            .. code-block:: python

                pending = pc.assistants.list_operations(
                    assistant_name="research-assistant",
                    operation_type="upload_file",
                    status="Processing",
                ).to_list()

        .. seealso::
           - :meth:`describe_operation`: one operation by id, when you kept
             the id a ``timeout=-1`` call handed back.
           - :meth:`list_operations_page`: one page at a time, when you want
             to hold the continuation token yourself.
           - `Pagination <https://sdk.pinecone.io/python/guides/pagination.html>`__:
             how the paginator and the continuation tokens work.
        """
        logger.info("Listing operations for assistant %r", assistant_name)

        def fetch_page(token: str | None) -> Page[OperationModel]:
            result = self.list_operations_page(
                assistant_name=assistant_name,
                operation_type=operation_type,
                status=status,
                pagination_token=token,
            )
            return Page(items=result.operations, pagination_token=result.next)

        return Paginator(fetch_page=fetch_page, initial_token=pagination_token, limit=limit)

    def list_operations_page(
        self,
        *,
        assistant_name: str,
        operation_type: str | None = None,
        status: str | None = None,
        page_size: int | None = None,
        pagination_token: str | None = None,
    ) -> ListOperationsResponse:
        """List one page of an assistant's operations with explicit pagination control.

        Args:
            assistant_name: Name of the assistant whose operations to list.
            operation_type: Restrict the listing to one kind of operation. One
                of ``"upload_file"``, ``"upsert_file"``,
                ``"update_file_metadata"`` or ``"delete_file"``.
            status: Restrict the listing to one status. One of
                ``"Processing"``, ``"Completed"`` or ``"Failed"``
                (case-sensitive).
            page_size: Maximum number of operations in this page. Omit it to
                let the server choose the page size.
            pagination_token: Token from a previous response to fetch the next
                page.

        Returns:
            :class:`ListOperationsResponse` with an ``operations`` list and an
            optional ``next`` continuation token.

        Raises:
            :exc:`PineconeTypeError`: If *assistant_name* is not a ``str``.
            :exc:`PineconeValueError`: If *operation_type* or *status* is not
                one of the values above.
            :exc:`NotFoundError`: If the assistant does not exist.

        Examples:
            .. code-block:: python

                page = pc.assistants.list_operations_page(
                    assistant_name="research-assistant",
                    status="Failed",
                    page_size=10,
                )
                for op in page.operations:
                    print(op.operation_id, op.error)
                if page.next:
                    next_page = pc.assistants.list_operations_page(
                        assistant_name="research-assistant",
                        status="Failed",
                        page_size=10,
                        pagination_token=page.next,
                    )

        .. seealso::
           `Pagination <https://sdk.pinecone.io/python/guides/pagination.html>`__:
           the continuation-token loop this method expects you to drive, and
           the paginator that drives it for you.
        """
        params: dict[str, str | int] = {}
        if operation_type is not None:
            params["operation_type"] = _validate_choice(
                "operation_type", operation_type, _VALID_OPERATION_TYPES
            )
        if status is not None:
            params["status"] = _validate_choice("status", status, _VALID_OPERATION_STATUSES)
        if page_size is not None:
            params["limit"] = page_size
        if pagination_token is not None:
            params["pagination_token"] = pagination_token

        data_http = self._data_plane_http(assistant_name)
        logger.info("Listing operations page for assistant %r", assistant_name)
        response = data_http.get(f"/operations/{quote(assistant_name, safe='')}", params=params)
        result = self._adapter.to_operation_list(response.content)
        logger.debug(
            "Listed %d operations for assistant %r (has_next=%s)",
            len(result.operations),
            assistant_name,
            result.next is not None,
        )
        return result

    def create(
        self,
        *,
        name: str | None = None,
        instructions: str | None = None,
        metadata: dict[str, Any] | None = None,
        region: str = "us",
        environment: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> AssistantModel:
        """Create a new Pinecone assistant.

        By default, blocks until the assistant reaches ``"Ready"`` status, so
        you can upload files to it as soon as the call returns.

        Args:
            name (str): Name for the new assistant, e.g. ``"docs-assistant"``.
                Must be unique within the project. The server validates the
                name and rejects an invalid one.
            instructions (str | None): Guidance the assistant applies to every
                response, e.g. ``"Always cite the source document."``.
            metadata (dict[str, Any] | None): Optional metadata to attach to
                the assistant, e.g. ``{"team": "docs"}``.
            region (str): Region to deploy the assistant in, ``"us"`` or
                ``"eu"``. Defaults to ``"us"``. Cannot be changed afterwards:
                an assistant in the wrong region has to be recreated.
            environment (str | None): Reserved for internal Pinecone use. Leave
                it unset; the server rejects it for other projects.
            timeout (float | None): Seconds to wait for the assistant to become
                ready. Use ``None`` (default) to poll indefinitely, ``-1`` to
                return immediately without polling, or a non-negative value to
                poll with a deadline.
            **kwargs (Any): Accepts the legacy alias ``assistant_name`` for
                *name*. Passing both, or any other keyword, raises
                :exc:`PineconeValueError`.

        Returns:
            :class:`AssistantModel` describing the created assistant.

        Raises:
            :exc:`PineconeTypeError`: If *name* is not a ``str``.
            :exc:`PineconeValueError`: If *region* is not ``"us"`` or ``"eu"``.
            :exc:`PineconeError`: If the assistant fails to initialize or is
                deleted while the call waits for it.
            :exc:`PineconeTimeoutError`: If the assistant does not become ready
                before the deadline.
            :exc:`ConflictError`: If an assistant of this name already exists
                in the project.
            :exc:`RateLimitError`: If the project has reached its assistant
                limit. Retrying does not help; delete an assistant you no
                longer need first.

        Examples:
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> assistant = pc.assistants.create(name="research-assistant")
            >>> assistant.status
            'Ready'

            Instructions, metadata and region are all optional. ``create``
            returns once the assistant reaches ``"Ready"``, so the assistant
            below is usable as soon as the call returns:

            >>> assistant = pc.assistants.create(
            ...     name="support-docs-assistant",
            ...     instructions="Always cite the source document.",
            ...     metadata={"team": "support", "cost_center": "R-4120"},
            ...     region="eu",
            ... )
            >>> assistant.status
            'Ready'
        """
        from pinecone._internal.kwargs_aliases import (
            reject_unknown_kwargs,
            remap_legacy_kwargs,
        )

        remapped = remap_legacy_kwargs(
            kwargs,
            aliases={"assistant_name": "name"},
            method_name="create",
        )
        reject_unknown_kwargs(remapped, allowed={"name"}, method_name="create")
        if "name" in remapped:
            if name is not None:
                raise PineconeValueError(
                    "create() received both 'assistant_name' (legacy) and 'name'. "
                    "Pass only one — prefer 'name'."
                )
            name = remapped["name"]
        if name is None:
            raise PineconeValueError(
                "create() missing required argument: 'name' (or legacy alias 'assistant_name')."
            )
        require_str("name", name)

        if region not in _VALID_REGIONS:
            raise PineconeValueError(f"region must be one of {_VALID_REGIONS!r}, got {region!r}")

        body: dict[str, Any] = {
            "name": name,
            "instructions": instructions,
            "region": region,
        }
        if metadata is not None:
            body["metadata"] = metadata
        if environment is not None:
            body["environment"] = environment

        logger.info("Creating assistant %r", name)
        response = self._http.post("/assistants", json=body)
        model = self._attach_ref(self._adapter.to_assistant(response.content))
        logger.debug("Created assistant %r (status=%s)", name, model.status)

        if timeout == -1:
            return model

        return self._poll_until_ready(name, timeout)

    def describe(self, *, name: str | None = None, **kwargs: Any) -> AssistantModel:
        """Get detailed information about a named assistant.

        Args:
            name (str): The name of the assistant to describe.
            **kwargs (Any): Accepts the legacy alias ``assistant_name`` for
                *name*. Passing both, or any other keyword, raises
                :exc:`PineconeValueError`.

        Returns:
            :class:`AssistantModel` with ``name``, ``status``, ``region``,
            ``host``, ``instructions``, ``metadata``, ``created_at`` and
            ``updated_at``.

        Raises:
            :exc:`PineconeTypeError`: If *name* is not a ``str``.
            :exc:`NotFoundError`: If the assistant does not exist.

        Examples:
            >>> assistant = pc.assistants.describe(name="research-assistant")
            >>> assistant.status
            'Ready'
        """
        from pinecone._internal.kwargs_aliases import (
            reject_unknown_kwargs,
            remap_legacy_kwargs,
        )

        remapped = remap_legacy_kwargs(
            kwargs,
            aliases={"assistant_name": "name"},
            method_name="describe",
        )
        reject_unknown_kwargs(remapped, allowed={"name"}, method_name="describe")
        if "name" in remapped:
            if name is not None:
                raise PineconeValueError(
                    "describe() received both 'assistant_name' (legacy) and 'name'. "
                    "Pass only one — prefer 'name'."
                )
            name = remapped["name"]
        if name is None:
            raise PineconeValueError(
                "describe() missing required argument: 'name' (or legacy alias 'assistant_name')."
            )
        require_str("name", name)

        logger.info("Describing assistant %r", name)
        response = self._http.get(f"/assistants/{quote(name, safe='')}")
        model = self._attach_ref(self._adapter.to_assistant(response.content))
        logger.debug("Described assistant %r (status=%s)", name, model.status)
        return model

    def list(
        self,
        *,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> Paginator[AssistantModel]:
        """List assistants in the project with lazy pagination.

        Args:
            limit (int | None): Maximum number of assistants to yield across
                all pages. ``None`` (default) yields all assistants.
            pagination_token (str | None): Token to resume pagination from a
                previous call.

        Returns:
            :class:`Paginator` over :class:`AssistantModel` objects. Supports
            ``for`` loops, ``.to_list()``, ``.pages()``, and ``limit``.

        Examples:
            .. code-block:: python

                for assistant in pc.assistants.list():
                    print(assistant.name, assistant.status)

            The paginator fetches pages lazily as you iterate. Call
            ``to_list()`` instead when you want every assistant materialized
            up front:

            .. code-block:: python

                all_assistants = pc.assistants.list().to_list()

        .. seealso::
           - :meth:`list_page`: one page at a time, when you want to hold
             the continuation token yourself.
           - `Pagination <https://sdk.pinecone.io/python/guides/pagination.html>`__:
             how the paginator and the continuation tokens work.
        """
        logger.info("Listing assistants")

        def fetch_page(token: str | None) -> Page[AssistantModel]:
            result = self.list_page(pagination_token=token)
            return Page(items=result.assistants, pagination_token=result.next)

        return Paginator(fetch_page=fetch_page, initial_token=pagination_token, limit=limit)

    def list_page(
        self,
        *,
        page_size: int | None = None,
        pagination_token: str | None = None,
        **kwargs: Any,
    ) -> ListAssistantsResponse:
        """List one page of assistants with explicit pagination control.

        Args:
            page_size (int | None): Maximum number of assistants per page.
                Omit it to let the server choose the page size.
            pagination_token (str | None): Token from a previous response
                to fetch the next page.
            **kwargs (Any): Accepts the legacy alias ``limit`` for
                *page_size*. Passing both, or any other keyword, raises
                :exc:`PineconeValueError`.

        Returns:
            :class:`ListAssistantsResponse` with an ``assistants`` list and
            an optional ``next`` continuation token.

        Examples:
            .. code-block:: python

                page = pc.assistants.list_page(page_size=10)
                for assistant in page.assistants:
                    print(assistant.name)
                if page.next:
                    next_page = pc.assistants.list_page(
                        page_size=10,
                        pagination_token=page.next,
                    )

        .. seealso::
           `Pagination <https://sdk.pinecone.io/python/guides/pagination.html>`__:
           the continuation-token loop this method expects you to drive, and
           the paginator that drives it for you.
        """
        from pinecone._internal.kwargs_aliases import (
            reject_unknown_kwargs,
            remap_legacy_kwargs,
        )

        remapped = remap_legacy_kwargs(
            kwargs,
            aliases={"limit": "page_size"},
            method_name="list_page",
        )
        reject_unknown_kwargs(remapped, allowed={"page_size"}, method_name="list_page")
        if "page_size" in remapped:
            if page_size is not None:
                raise PineconeValueError(
                    "list_page() received both 'limit' (legacy) and 'page_size'. "
                    "Pass only one — prefer 'page_size'."
                )
            page_size = remapped["page_size"]

        params: dict[str, str | int] = {}
        if page_size is not None:
            params["limit"] = page_size
        if pagination_token is not None:
            params["pagination_token"] = pagination_token

        logger.info("Listing assistants page")
        response = self._http.get("/assistants", params=params)
        result = self._adapter.to_assistant_list(response.content)
        for item in result.assistants:
            self._attach_ref(item)
        logger.debug(
            "Listed %d assistants (has_next=%s)",
            len(result.assistants),
            result.next is not None,
        )
        return result

    def update(
        self,
        *,
        name: str | None = None,
        instructions: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AssistantModel:
        """Update an existing Pinecone assistant.

        Pass at least one of *instructions* and *metadata*. A field left as
        ``None`` keeps its current value; to clear a field, pass its empty
        value (``instructions=""`` or ``metadata={}``).

        Args:
            name (str): The name of the assistant to update.
            instructions (str | None): New instructions for the assistant.
                Pass an empty string to clear existing instructions.
            metadata (dict[str, Any] | None): New metadata dictionary. Fully
                replaces any existing metadata rather than merging. Pass an
                empty dict to clear existing metadata.
            **kwargs (Any): Accepts the legacy alias ``assistant_name`` for
                *name*. Passing both, or any other keyword, raises
                :exc:`PineconeValueError`.

        Returns:
            :class:`AssistantModel` describing the updated assistant.

        Raises:
            :exc:`PineconeTypeError`: If *name* is not a ``str``.
            :exc:`PineconeValueError`: If neither *instructions* nor
                *metadata* is provided.
            :exc:`NotFoundError`: If the assistant does not exist.

        Examples:
            Patch only the instructions. ``metadata`` is left out of the
            request body entirely, so whatever metadata the assistant already
            carries survives untouched:

            >>> assistant = pc.assistants.update(
            ...     name="research-assistant",
            ...     instructions="Always cite the source document.",
            ... )

            Passing ``metadata`` replaces the whole dictionary instead of
            merging into it. An assistant carrying
            ``{"team": "research", "cost_center": "R-4120"}`` is left with
            only ``team`` after the call below. Its instructions stay
            unchanged, since they were not named:

            >>> assistant = pc.assistants.update(
            ...     name="research-assistant",
            ...     metadata={"team": "docs-platform"},
            ... )
        """
        from pinecone._internal.kwargs_aliases import (
            reject_unknown_kwargs,
            remap_legacy_kwargs,
        )

        remapped = remap_legacy_kwargs(
            kwargs,
            aliases={"assistant_name": "name"},
            method_name="update",
        )
        reject_unknown_kwargs(remapped, allowed={"name"}, method_name="update")
        if "name" in remapped:
            if name is not None:
                raise PineconeValueError(
                    "update() received both 'assistant_name' (legacy) and 'name'. "
                    "Pass only one — prefer 'name'."
                )
            name = remapped["name"]
        if name is None:
            raise PineconeValueError(
                "update() missing required argument: 'name' (or legacy alias 'assistant_name')."
            )
        require_str("name", name)
        if instructions is None and metadata is None:
            raise PineconeValueError(
                "update() needs at least one of 'instructions' or 'metadata'. With both "
                "omitted the patch body is empty and the server answers 400 'No updates "
                "provided'. To clear a field, send its empty value: instructions='' or "
                "metadata={}."
            )

        body: dict[str, Any] = {}
        if instructions is not None:
            body["instructions"] = instructions
        if metadata is not None:
            body["metadata"] = metadata

        logger.info("Updating assistant %r", name)
        response = self._http.patch(f"/assistants/{quote(name, safe='')}", json=body)
        model = self._attach_ref(self._adapter.to_assistant(response.content))
        logger.debug("Updated assistant %r", name)
        return model

    def delete(
        self,
        *,
        name: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Delete a Pinecone assistant by name.

        Deleting an assistant also deletes its files. By default, blocks
        until the assistant is confirmed gone.

        Args:
            name (str): The name of the assistant to delete.
            timeout (float | None): Seconds to wait for the assistant to
                disappear. Use ``None`` (default) to poll indefinitely.
                Use ``-1`` to return immediately without polling.
                Use a positive value to poll with a deadline. Raises
                :exc:`PineconeTimeoutError` if the assistant is not gone
                before the deadline.
            **kwargs (Any): Accepts the legacy alias ``assistant_name`` for
                *name*. Passing both, or any other keyword, raises
                :exc:`PineconeValueError`.

        Returns:
            None

        Raises:
            :exc:`PineconeTypeError`: If *name* is not a ``str``.
            :exc:`PineconeError`: If the assistant is in a failed state and
                will never finish deleting.
            :exc:`PineconeTimeoutError`: If the assistant still exists after
                *timeout* seconds.

        Examples:
            .. code-block:: python

                pc.assistants.delete(name="research-assistant")

            The call above blocks until the assistant is confirmed gone. Pass
            ``timeout=-1`` to return as soon as the request is accepted; the
            assistant may still be terminating when this returns:

            .. code-block:: python

                pc.assistants.delete(name="stale-prototype", timeout=-1)
        """
        from pinecone._internal.kwargs_aliases import (
            reject_unknown_kwargs,
            remap_legacy_kwargs,
        )

        remapped = remap_legacy_kwargs(
            kwargs,
            aliases={"assistant_name": "name"},
            method_name="delete",
        )
        reject_unknown_kwargs(remapped, allowed={"name"}, method_name="delete")
        if "name" in remapped:
            if name is not None:
                raise PineconeValueError(
                    "delete() received both 'assistant_name' (legacy) and 'name'. "
                    "Pass only one — prefer 'name'."
                )
            name = remapped["name"]
        if name is None:
            raise PineconeValueError(
                "delete() missing required argument: 'name' (or legacy alias 'assistant_name')."
            )
        require_str("name", name)

        logger.info("Deleting assistant %r", name)
        self._http.delete(f"/assistants/{quote(name, safe='')}")
        logger.debug("Deleted assistant %r", name)

        if timeout == -1:
            return

        start = time.monotonic()
        while True:
            try:
                model = self.describe(name=name)
            except NotFoundError:
                return
            if model.status == "Terminated":
                logger.debug("Assistant %r reported 'Terminated'; deletion is complete", name)
                return
            if model.status in _DELETE_TERMINAL_STATUSES:
                raise PineconeError(
                    f"Assistant '{name}' entered terminal state '{model.status}' while being "
                    f"deleted, so it will never disappear on its own. Check status with "
                    f"pc.assistants.describe(name='{name}') and retry the delete."
                )
            if timeout is not None:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    raise PineconeTimeoutError(f"Assistant '{name}' still exists after {timeout}s")
            time.sleep(_DELETE_POLL_INTERVAL_SECONDS)

    def context(
        self,
        *,
        assistant_name: str,
        query: str | None = None,
        messages: Sequence[Message | Mapping[str, str]] | None = None,
        filter: dict[str, Any] | None = None,
        top_k: int | None = None,
        snippet_size: int | None = None,
        multimodal: bool | None = None,
        include_binary_content: bool | None = None,
    ) -> ContextResponse:
        """Retrieve relevant context snippets from a Pinecone assistant.

        Returns the snippets from the assistant's files that match a text
        query or a conversation, without generating an answer, so you can
        prompt your own model with them. Provide exactly one of *query* or
        *messages*.

        Args:
            assistant_name: Name of the assistant to retrieve context from.
            query: Text query to use for context retrieval. Mutually exclusive
                with *messages*. An empty string is treated as not provided.
            messages: Conversation messages to use for context retrieval.
                Mutually exclusive with *query*. An empty list is treated as
                not provided. Dicts are converted to :class:`Message` objects.
                Each role is ``"user"`` or ``"assistant"``.
            filter: Filter on file metadata, such as
                ``{"department": "finance"}``, restricting which files
                contribute snippets.
            top_k: Maximum number of snippets to return. Omit it to let the
                server choose.
            snippet_size: Maximum size of each snippet, in tokens. Omit it to
                let the server choose.
            multimodal: Whether to include snippets drawn from images in files
                uploaded with ``multimodal=True``.
            include_binary_content: Whether image snippets include base64
                image data. Only meaningful when *multimodal* is ``True``.

        Returns:
            :class:`ContextResponse` with ``snippets`` (each carrying
            ``content``, a relevance ``score``, and a ``reference`` naming the
            source file and, for paginated documents, the pages) and
            ``usage``.

        Raises:
            :exc:`PineconeTypeError`: If *assistant_name* is not a ``str``.
            :exc:`PineconeValueError`: If both or neither of *query* and
                *messages* are provided, or if *top_k* or *snippet_size* is
                negative.
            :exc:`ApiError`: If the assistant has no file in ``"Available"``
                status yet, for example while uploads are still processing.

        Examples:
            .. code-block:: python

                response = pc.assistants.context(
                    assistant_name="research-assistant",
                    query="What is Pinecone?",
                )
                for snippet in response.snippets:
                    print(snippet.content)

        .. seealso::
           - :meth:`chat`: a generated answer with structured citations,
             when you want Pinecone to do the generation as well.
           - :meth:`chat_completions`: a generated answer in OpenAI's
             response shape.
        """
        query_truthy = query is not None and query != ""
        messages_truthy = messages is not None and len(messages) > 0

        if query_truthy and messages_truthy:
            raise PineconeValueError("Exactly one of query or messages must be provided, not both.")
        if not query_truthy and not messages_truthy:
            raise PineconeValueError("Exactly one of query or messages must be provided.")

        body: dict[str, Any] = {}

        if query_truthy:
            body["query"] = query
        else:
            if messages is None:
                raise PineconeValueError("Exactly one of query or messages must be provided.")
            parsed: list[Message] = [
                m if isinstance(m, Message) else Message.from_dict(m) for m in messages
            ]
            body["messages"] = [{"role": m.role, "content": m.content} for m in parsed]

        if top_k is not None and top_k < 0:
            raise PineconeValueError("top_k must be a non-negative integer.")
        if snippet_size is not None and snippet_size < 0:
            raise PineconeValueError("snippet_size must be a non-negative integer.")

        if filter is not None:
            body["filter"] = filter
        if top_k is not None:
            body["top_k"] = top_k
        if snippet_size is not None:
            body["snippet_size"] = snippet_size
        if multimodal is not None:
            body["multimodal"] = multimodal
        if include_binary_content is not None:
            body["include_binary_content"] = include_binary_content

        http = self._data_plane_http(assistant_name)
        response = http.post(f"/chat/{quote(assistant_name, safe='')}/context", json=body)
        return self._adapter.to_context_response(response.content)

    def chat(
        self,
        *,
        assistant_name: str,
        messages: Sequence[Message | Mapping[str, str]],
        model: str = "gpt-4o",
        stream: bool = False,
        temperature: float | None = None,
        filter: dict[str, Any] | None = None,
        json_response: bool = False,
        include_highlights: bool = False,
        context_options: ContextOptions | dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> ChatResponse | ChatStream:
        """Chat with an assistant and receive citations in Pinecone-native format.

        Citations come back as a structured list keyed to character positions
        in the answer, which is what separates this from
        :meth:`chat_completions`. The assistant answers only from the files
        you uploaded to it, so an assistant with nothing ingested yet errors
        rather than replying from general knowledge.

        Args:
            assistant_name (str): Name of the assistant to chat with.
            messages (list[Message | dict[str, str]]): Conversation messages.
                Dicts are converted to :class:`Message` objects; role defaults
                to ``"user"`` when not present. Each role is ``"user"`` or
                ``"assistant"``.
            model (str): Name of the large language model that generates the
                answer. Defaults to ``"gpt-4o"``. The models the API
                documents for this endpoint are ``"gpt-4o"``, ``"gpt-4.1"``,
                ``"gpt-5"``, ``"o4-mini"``, ``"claude-sonnet-4-5"``, and
                ``"gemini-2.5-pro"``. A name outside that list may be served
                by a successor model rather than rejected, so the response's
                ``model`` field, not this argument, says which model
                answered.
            stream (bool): If ``True``, return a :class:`ChatStream`. Defaults
                to ``False``.
            temperature (float | None): Controls randomness. Lower values produce
                more deterministic responses.
            filter (dict[str, Any] | None): Filter on file metadata, such as
                ``{"department": "finance"}``, restricting which files the
                answer draws on.
            json_response (bool): If ``True``, instruct the assistant to return
                a JSON response. Cannot be combined with ``stream=True``.
            include_highlights (bool): If ``True``, include highlight snippets
                from referenced documents in citations.
            context_options (ContextOptions | dict[str, Any] | None):
                Retrieval settings, such as ``top_k`` and ``snippet_size``, as
                a :class:`ContextOptions` or an equivalent dict.
            timeout (float | None): Per-call HTTP timeout in seconds, overriding
                the client-level default. On a streaming request this bounds the
                gap between chunks rather than the whole response (see below).

        Returns:
            :class:`ChatResponse` for non-streaming requests, carrying
            ``message`` (the answer), ``citations`` (each with the ``position``
            in the answer it supports and the ``references`` behind it),
            ``model`` (the model that answered), ``finish_reason`` and
            ``usage``. For streaming requests, a :class:`ChatStream`.

        Raises:
            :exc:`PineconeTypeError`: If *assistant_name* is not a ``str``.
            :exc:`PineconeValueError`: If both ``stream=True`` and
                ``json_response=True`` are specified.
            :exc:`ApiError`: If the assistant has no file in ``"Available"``
                status yet, for example while uploads are still processing.

        Examples:
            .. code-block:: python

                from pinecone import Pinecone

                pc = Pinecone(api_key="your-api-key")
                response = pc.assistants.chat(
                    assistant_name="research-assistant",
                    messages=[{"content": "What is Pinecone?"}],
                )
                print(response.message.content)
                for citation in response.citations:
                    for reference in citation.references:
                        print(citation.position, reference.file.name)

            Set ``stream=True`` for a :class:`ChatStream` instead of a
            single response. Its ``text()`` method yields content fragments as
            they arrive, skipping the start, citation and end chunks:

            .. code-block:: python

                stream = pc.assistants.chat(
                    assistant_name="research-assistant",
                    messages=[{"content": "What is Pinecone?"}],
                    stream=True,
                )
                for text in stream.text():
                    print(text, end="", flush=True)

        .. seealso::
           - :meth:`chat_completions`: the same conversation in OpenAI's
             response shape, with citations woven into the message text
             instead of returned as a structured list.
           - :meth:`context`: the retrieved snippets on their own, with no
             generated answer, when you want to prompt your own model.

        .. note::
           On a streaming request the timeout applies to the gap between
           chunks rather than the whole response, and the default is raised so
           a model that thinks for a while isn't mistaken for a dead
           connection. Pass *timeout* to change it. A stream that exceeds its
           timeout raises :exc:`PineconeTimeoutError` partway through
           iteration, after earlier chunks have already been yielded.
        """
        if stream and json_response:
            raise PineconeValueError("json_response cannot be used with stream=True")

        parsed: list[Message] = [
            m if isinstance(m, Message) else Message.from_dict(m) for m in messages
        ]

        body: dict[str, Any] = {
            "messages": [{"role": m.role, "content": m.content} for m in parsed],
            "model": model,
            "stream": stream,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if filter is not None:
            body["filter"] = filter
        if json_response:
            body["json_response"] = json_response
        # Streaming requests always include include_highlights (defaults to False)
        if stream or include_highlights:
            body["include_highlights"] = include_highlights
        if context_options is not None:
            if isinstance(context_options, dict):
                body["context_options"] = context_options
            else:
                body["context_options"] = {
                    k: v
                    for k, v in msgspec.structs.asdict(context_options).items()
                    if v is not None
                }

        http = self._data_plane_http(assistant_name)
        if stream:
            return ChatStream(
                self._chat_streaming(
                    http=http,
                    url=f"/chat/{quote(assistant_name, safe='')}",
                    body=body,
                    timeout=timeout,
                )
            )

        response = http.post(f"/chat/{quote(assistant_name, safe='')}", timeout=timeout, json=body)
        return self._adapter.to_chat_response(response.content)

    def chat_completions(
        self,
        *,
        assistant_name: str,
        messages: Sequence[Message | Mapping[str, str]],
        model: str = "gpt-4o",
        stream: bool = False,
        temperature: float | None = None,
        filter: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> ChatCompletionResponse | ChatCompletionStream:
        """Chat with an assistant using an OpenAI-compatible interface.

        Use it when your code already consumes OpenAI's chat completion
        shape. Citations arrive inline in the message text rather than as a
        structured list, and the *include_highlights*, *context_options* and
        *json_response* options of :meth:`chat` are not available.

        Args:
            assistant_name (str): Name of the assistant to chat with.
            messages (list[Message | dict[str, str]]): Conversation messages.
                Dicts are converted to :class:`Message` objects; role defaults
                to ``"user"`` when not present. Each role is ``"user"`` or
                ``"assistant"``.
            model (str): Name of the large language model that generates the
                answer. Defaults to ``"gpt-4o"``. The models the API
                documents for this endpoint are ``"gpt-4o"``, ``"gpt-4.1"``,
                ``"o4-mini"``, ``"claude-sonnet-4-5"``, and
                ``"gemini-2.5-pro"``. A name outside that list may be served
                by a successor model rather than rejected, so the response's
                ``model`` field, not this argument, says which model
                answered.
            stream (bool): If ``True``, return a :class:`ChatCompletionStream`.
                Defaults to ``False``.
            temperature (float | None): Controls randomness. Lower values produce
                more deterministic responses.
            filter (dict[str, Any] | None): Filter on file metadata, such as
                ``{"department": "finance"}``, restricting which files the
                answer draws on.
            timeout (float | None): Per-call HTTP timeout in seconds, overriding
                the client-level default. On a streaming request this bounds the
                gap between chunks rather than the whole response (see below).

        Returns:
            :class:`ChatCompletionResponse` for non-streaming requests,
            carrying ``choices`` (read ``choices[0].message.content`` for the
            answer, with citations woven into that text), ``model`` (the model
            that answered), and ``usage``. For streaming requests, a
            :class:`ChatCompletionStream`.

        Raises:
            :exc:`PineconeTypeError`: If *assistant_name* is not a ``str``.
            :exc:`ApiError`: If the assistant has no file in ``"Available"``
                status yet, for example while uploads are still processing.

        Examples:
            .. code-block:: python

                from pinecone import Pinecone

                pc = Pinecone(api_key="your-api-key")
                response = pc.assistants.chat_completions(
                    assistant_name="research-assistant",
                    messages=[{"content": "Explain quantum entanglement briefly."}],
                )
                print(response.choices[0].message.content)

            Set ``stream=True`` for a :class:`ChatCompletionStream`. Its
            ``text()`` method yields content fragments as they arrive:

            .. code-block:: python

                stream = pc.assistants.chat_completions(
                    assistant_name="research-assistant",
                    messages=[{"content": "Explain quantum entanglement briefly."}],
                    stream=True,
                )
                for text in stream.text():
                    print(text, end="", flush=True)

        .. seealso::
           - :meth:`chat`: the Pinecone-native shape, with a structured
             ``citations`` list.
           - :meth:`context`: the retrieved snippets on their own, with no
             generated answer.

        .. note::
           On a streaming request the timeout applies to the gap between
           chunks rather than the whole response, and the default is raised so
           a model that pauses for longer while reasoning isn't mistaken for a
           dead connection. Pass *timeout* to widen it further. A stream that
           exceeds its timeout raises :exc:`PineconeTimeoutError` partway
           through iteration, after earlier chunks have already been yielded.
        """
        parsed: list[Message] = [
            m if isinstance(m, Message) else Message.from_dict(m) for m in messages
        ]

        body: dict[str, Any] = {
            "messages": [{"role": m.role, "content": m.content} for m in parsed],
            "model": model,
            "stream": stream,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if filter is not None:
            body["filter"] = filter

        http = self._data_plane_http(assistant_name)
        if stream:
            url = f"/chat/{quote(assistant_name, safe='')}/chat/completions"
            return ChatCompletionStream(
                self._chat_completions_streaming(http=http, url=url, body=body, timeout=timeout)
            )

        response = http.post(
            f"/chat/{quote(assistant_name, safe='')}/chat/completions",
            timeout=timeout,
            json=body,
        )
        return self._adapter.to_chat_completion_response(response.content)

    def _chat_streaming(
        self,
        *,
        http: HTTPClient,
        url: str,
        body: dict[str, Any],
        timeout: float | None = None,
    ) -> Iterator[ChatStreamChunk]:
        """Stream Pinecone-native chat chunks via SSE.

        POSTs to the given URL with ``stream=True`` in the body,
        parses each SSE line, and yields typed chunk objects dispatched by the
        ``type`` field.

        Args:
            http: Pre-resolved data-plane HTTP client for the assistant.
            url: Request URL path (e.g. ``/chat/{assistant_name}``).
            body: Pre-built request body (must include ``stream=True``).
            timeout: Per-call HTTP timeout override. When ``None`` the client
                timeout is raised to :data:`_STREAM_TIMEOUT_FLOOR_SECONDS`.

        Yields:
            :class:`StreamMessageStart`, :class:`StreamContentChunk`,
            :class:`StreamCitationChunk`, or :class:`StreamMessageEnd`
            depending on the ``type`` field of each SSE chunk.

        Raises:
            :exc:`ApiError`: If the server returns an HTTP error.
            :exc:`PineconeTimeoutError`: If the gap between chunks exceeds the
                resolved timeout, possibly after some chunks have been yielded.
        """
        from pinecone._internal.http_client import _encode_json

        with http.stream(
            "POST",
            url,
            content=_encode_json(body),
            headers={"Content-Type": "application/json"},
            timeout=_stream_timeout(self._config.timeout, timeout),
        ) as response:
            response.encoding = "utf-8-sig"
            for line in response.iter_lines():
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                line = line[5:].lstrip()
                if not line:
                    continue
                # Unreachable against the 2026-07 backend, which ends the stream
                # by closing it. Kept because an intermediary that terminates the
                # SSE conventionally would otherwise reach orjson.loads and raise.
                if line == "[DONE]":
                    break
                chunk_data: dict[str, Any] = orjson.loads(line)
                try:
                    yield msgspec.convert(chunk_data, ChatStreamChunk)
                except msgspec.ValidationError:
                    logger.debug("Skipping unknown chunk type: %s", chunk_data.get("type"))

    def _chat_completions_streaming(
        self,
        *,
        http: HTTPClient,
        url: str,
        body: dict[str, Any],
        timeout: float | None = None,
    ) -> Iterator[ChatCompletionStreamChunk]:
        """Stream OpenAI-compatible chat completion chunks via SSE.

        POSTs to the given URL with ``stream=True`` in the body and yields each
        SSE line parsed as a :class:`ChatCompletionStreamChunk`.

        Args:
            http: Pre-resolved data-plane HTTP client for the assistant.
            url: Request URL path (e.g. ``/chat/{assistant_name}/chat/completions``).
            body: Pre-built request body (must include ``stream=True``).
            timeout: Per-call HTTP timeout override. When ``None`` the client
                timeout is raised to :data:`_STREAM_TIMEOUT_FLOOR_SECONDS`.

        Yields:
            :class:`ChatCompletionStreamChunk` for each non-empty SSE line.
            Lines that do not fit the struct are logged and skipped rather than
            aborting the stream, matching :meth:`_chat_streaming`.

        Raises:
            :exc:`ApiError`: If the server returns an HTTP error.
            :exc:`PineconeTimeoutError`: If the gap between chunks exceeds the
                resolved timeout, possibly after some chunks have been yielded.
        """
        from pinecone._internal.http_client import _encode_json

        with http.stream(
            "POST",
            url,
            content=_encode_json(body),
            headers={"Content-Type": "application/json"},
            timeout=_stream_timeout(self._config.timeout, timeout),
        ) as response:
            response.encoding = "utf-8-sig"
            for line in response.iter_lines():
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                line = line[5:].lstrip()
                if not line:
                    continue
                # Unreachable against the 2026-07 backend, which ends the stream
                # by closing it. Kept because an intermediary that terminates the
                # SSE conventionally would otherwise reach orjson.loads and raise.
                if line == "[DONE]":
                    break
                chunk_data: dict[str, Any] = orjson.loads(line)
                try:
                    yield msgspec.convert(chunk_data, ChatCompletionStreamChunk)
                except msgspec.ValidationError:
                    logger.debug("Skipping unparseable completion chunk: %s", chunk_data)

    def evaluate_alignment(
        self,
        *,
        question: str,
        answer: str,
        ground_truth_answer: str,
    ) -> AlignmentResult:
        """Evaluate answer alignment against a ground truth answer.

        Scores a generated answer, such as one from :meth:`chat`, against an
        answer you know is right. Alignment is the harmonic mean of
        correctness (precision) and completeness (recall).

        Args:
            question: The question for which the answer was generated.
            answer: The generated answer to evaluate.
            ground_truth_answer: The ground truth answer to compare against.

        Returns:
            :class:`AlignmentResult` with ``scores`` (``correctness``,
            ``completeness`` and ``alignment``), ``facts`` (the per-fact
            entailment verdicts behind the scores) and token ``usage``.

        Examples:
            The answer below contradicts the ground truth on purpose, so the
            scores come back low and ``result.facts`` records where the
            contradiction is:

            >>> result = pc.assistants.evaluate_alignment(
            ...     question="What is the capital of Spain?",
            ...     answer="Barcelona.",
            ...     ground_truth_answer="Madrid.",
            ... )
            >>> result.scores.alignment
            0.0
            >>> [fact.entailment for fact in result.facts]
            ['contradicted']
        """
        body = {
            "question": question,
            "answer": answer,
            "ground_truth_answer": ground_truth_answer,
        }
        logger.info("Evaluating alignment for question %r", question)
        response = self._eval_http.post("/evaluation/metrics/alignment", json=body)
        result = self._adapter.to_alignment_result(response.content)
        logger.debug("Alignment evaluation complete (alignment=%.3f)", result.scores.alignment)
        return result

    def _poll_until_ready(self, name: str, timeout: float | None) -> AssistantModel:
        """Poll ``GET /assistants/{name}`` until status is ``"Ready"`` or timeout."""
        start = time.monotonic()
        while True:
            response = self._http.get(f"/assistants/{quote(name, safe='')}")
            model = self._attach_ref(self._adapter.to_assistant(response.content))
            if model.status == "Ready":
                return model
            if model.status in ("Failed", "InitializationFailed", "Terminated", "Terminating"):
                raise PineconeError(
                    f"Assistant '{name}' entered terminal state '{model.status}'. "
                    f"Check status with pc.assistants.describe(name='{name}')."
                )
            if timeout is not None:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    raise PineconeTimeoutError(
                        f"Assistant '{name}' not ready after {timeout}s. "
                        f"Check status with pc.assistants.describe(name='{name}')."
                    )
            time.sleep(_CREATE_POLL_INTERVAL_SECONDS)
