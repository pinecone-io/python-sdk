"""Assistants namespace — control-plane operations for Pinecone assistants."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from typing import IO, TYPE_CHECKING, Any

import msgspec
import msgspec.structs
import orjson

from pinecone._internal.adapters.assistants_adapter import AssistantsAdapter
from pinecone._internal.constants import (
    ASSISTANT_API_VERSION,
    ASSISTANT_EVALUATION_BASE_URL,
    DEFAULT_BASE_URL,
)
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
from pinecone.models.assistant.list import ListAssistantsResponse, ListFilesResponse
from pinecone.models.assistant.message import Message
from pinecone.models.assistant.model import AssistantModel
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


class Assistants(AssistantsLegacyNamespaceMixin):
    """Control-plane operations for Pinecone assistants.

    Args:
        config (PineconeConfig): SDK configuration used to construct an
            HTTP client targeting the assistant API version.

    Examples:

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone(api_key="your-api-key")
            assistants = pc.assistants
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
        """Close the underlying HTTP client and any cached data-plane clients."""
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

    def _list_files_http(self, assistant_name: str) -> HTTPClient:
        """Return an uncached HTTPClient for the assistant's data-plane host."""
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
        return _HTTPClient(data_config, ASSISTANT_API_VERSION)

    def _upsert_http(self, assistant_name: str) -> HTTPClient:
        """Return an HTTPClient targeting the assistant's data-plane host for upserts."""
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
        return _HTTPClient(data_config, ASSISTANT_API_VERSION)

    def _poll_operation_until_done(
        self,
        upsert_http: HTTPClient,
        assistant_name: str,
        operation_id: str,
        timeout: float | None,
    ) -> None:
        """Poll ``GET /operations/{assistant_name}/{operation_id}`` until done."""
        start = time.monotonic()
        while True:
            response = upsert_http.get(f"/operations/{assistant_name}/{operation_id}")
            op_model = self._adapter.to_operation(response.content)

            if op_model.status != "Processing":
                if op_model.status == "Failed":
                    error_msg = op_model.error or "Unknown operation error"
                    raise PineconeError(
                        f"Upsert operation failed for operation '{operation_id}': {error_msg}"
                    )
                return

            if timeout is not None:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    raise PineconeTimeoutError(
                        f"Upsert operation timed out after {timeout}s (operation_id={operation_id})"
                    )
            time.sleep(_UPLOAD_POLL_INTERVAL_SECONDS)

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
        polls until server-side processing completes.

        Args:
            assistant_name: Name of the target assistant.
            file_path: Path to a local file to upload. Mutually exclusive
                with *file_stream*.
            file_stream: An open byte stream to upload. Mutually exclusive
                with *file_path*. Use *file_name* to set the filename.
            file_name: Filename to associate with *file_stream*. Ignored
                when *file_path* is provided.
            metadata: Optional metadata dictionary. Sent as a JSON string.
            multimodal: Whether to enable multimodal processing for PDFs.
            file_id: Optional caller-specified file identifier for upsert
                behavior.
            timeout: Seconds to wait for processing to complete. ``None``
                (default) polls indefinitely. Use ``-1`` to return
                immediately after upload with one describe call. Raises
                :exc:`PineconeTimeoutError` if processing is not done
                before the deadline.

        Returns:
            :class:`AssistantFileModel` fetched fresh from the API after
            processing completes.

        Raises:
            :exc:`PineconeValueError`: If both or neither of *file_path*
                and *file_stream* are provided, or if *file_path* does not
                exist.
            :exc:`PineconeTimeoutError`: If processing does not complete
                before *timeout*.
            :exc:`PineconeError`: If server-side processing fails.

        Examples:
            >>> file = pc.assistants.upload_file(
            ...     assistant_name="research-assistant",
            ...     file_path="/data/report.pdf",
            ... )
            >>> file.status  # doctest: +SKIP
            'Available'
        """
        import json as _json

        if (file_path is None) == (file_stream is None):
            raise PineconeValueError("Exactly one of file_path or file_stream must be provided")

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
            upload_name = file_name or "upload"

        try:
            data_http = self._data_plane_http(assistant_name)

            params: dict[str, str] = {}
            if metadata is not None:
                params["metadata"] = _json.dumps(metadata)
            if multimodal is not None:
                params["multimodal"] = str(multimodal).lower()

            if file_id is not None:
                # Use the 2026-04 upsert endpoint: PUT /files/{assistant_name}/{file_id}
                upsert_http = self._upsert_http(assistant_name)
                logger.info(
                    "Upserting file %r (id=%s) to assistant %r",
                    upload_name,
                    file_id,
                    assistant_name,
                )
                # v202604 rejects metadata as a query param; send it as a multipart field instead.
                upsert_files: dict[str, Any] = {"file": (upload_name, handle)}
                if metadata is not None:
                    upsert_files["metadata"] = (None, _json.dumps(metadata))
                upsert_query: dict[str, str] = {}
                if multimodal is not None:
                    upsert_query["multimodal"] = str(multimodal).lower()
                upsert_response = upsert_http.put(
                    f"/files/{assistant_name}/{file_id}",
                    files=upsert_files,
                    params=upsert_query,
                )
                op_model = self._adapter.to_operation(upsert_response.content)
                operation_id = op_model.operation_id
                if timeout == -1:
                    return self.describe_file(assistant_name=assistant_name, file_id=file_id)
                self._poll_operation_until_done(upsert_http, assistant_name, operation_id, timeout)
                return self.describe_file(assistant_name=assistant_name, file_id=file_id)

            logger.info("Uploading file %r to assistant %r", upload_name, assistant_name)
            response = data_http.post(
                f"/files/{assistant_name}",
                files={"file": (upload_name, handle)},
                params=params,
            )
            file_model = self._adapter.to_file(response.content)
            logger.debug(
                "Uploaded file %r (id=%s, status=%s)",
                upload_name,
                file_model.id,
                file_model.status,
            )
        finally:
            if opened_file is not None:
                opened_file.close()

        if timeout == -1:
            return self.describe_file(assistant_name=assistant_name, file_id=file_model.id)

        return self._poll_file_until_processed(data_http, assistant_name, file_model.id, timeout)

    def _poll_file_until_processed(
        self,
        data_http: HTTPClient,
        assistant_name: str,
        file_id: str,
        timeout: float | None,
    ) -> AssistantFileModel:
        """Poll ``GET /files/{assistant_name}/{file_id}`` until processing completes."""
        start = time.monotonic()
        while True:
            response = data_http.get(f"/files/{assistant_name}/{file_id}")
            file_model = self._adapter.to_file(response.content)

            if file_model.status != "Processing":
                if file_model.status == "ProcessingFailed":
                    raise PineconeError(
                        f"File processing failed for '{file_id}'. Call describe_operation() "
                        "for the failure reason."
                    )
                return file_model

            if timeout is not None:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    raise PineconeTimeoutError(
                        f"File processing timed out after {timeout}s (operation_id={file_id})"
                    )
            time.sleep(_UPLOAD_POLL_INTERVAL_SECONDS)

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
            :class:`AssistantFileModel` with file metadata and status.

        Raises:
            :exc:`NotFoundError`: If the file does not exist.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> file = pc.assistants.describe_file(
            ...     assistant_name="my-assistant",
            ...     file_id="file-abc123",
            ... )
            >>> file.status  # doctest: +SKIP
            'Available'
        """
        data_http = self._data_plane_http(assistant_name)
        params: dict[str, str] = {}
        if include_url:
            params["include_url"] = "true"
        logger.info("Describing file %r in assistant %r", file_id, assistant_name)
        response = data_http.get(f"/files/{assistant_name}/{file_id}", params=params)
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

        Args:
            assistant_name: Name of the assistant whose files to list.
            filter: Optional metadata filter expression. Serialized to a JSON
                string before being sent to the API.
            limit: Maximum number of files to yield across all pages. ``None``
                (default) yields all files.
            pagination_token: Token to resume pagination from a previous call.

        Returns:
            :class:`Paginator` over :class:`AssistantFileModel` objects.
            Supports ``for`` loops, ``.to_list()``, ``.pages()``, and
            ``limit``.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            .. code-block:: python

                for f in pc.assistants.list_files(assistant_name="my-assistant"):
                    print(f.name, f.status)

                files = pc.assistants.list_files(assistant_name="my-assistant").to_list()
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

        Only the parameters that are explicitly provided are sent in the
        request. Omitted parameters are not included as query params.

        Args:
            assistant_name: Name of the assistant whose files to list.
            page_size: Maximum number of files per page.
            pagination_token: Token from a previous response to fetch the
                next page.
            filter: Optional metadata filter expression. Serialized to a JSON
                string before being sent to the API.

        Returns:
            :class:`ListFilesResponse` with a ``files`` list and an optional
            ``next`` continuation token.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            .. code-block:: python

                page = pc.assistants.list_files_page(assistant_name="my-assistant")
                names = [f.name for f in page.files]
                token = page.next  # use as pagination_token for the next call
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

        list_http = self._list_files_http(assistant_name)
        params: dict[str, str | int] = {}
        if page_size is not None:
            params["limit"] = page_size
        if pagination_token is not None:
            params["pagination_token"] = pagination_token
        if filter is not None:
            params["filter"] = _json.dumps(filter)

        logger.info("Listing files page for assistant %r", assistant_name)
        response = list_http.get(f"/files/{assistant_name}", params=params)
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

        Sends a DELETE request, then polls every 5 seconds until the file is
        confirmed gone (404 from describe_file). Other errors during polling
        propagate immediately.

        Args:
            assistant_name: Name of the assistant that owns the file.
            file_id: Unique identifier of the file to delete.
            timeout: Seconds to wait for the file to be deleted. Use ``None``
                (default) to poll indefinitely. Use ``-1`` to return
                immediately without polling. Use a positive value to poll with
                a deadline. Raises :exc:`PineconeTimeoutError` if the file
                is not gone before the deadline.

        Returns:
            ``None``

        Raises:
            :exc:`PineconeError`: If server-side file deletion fails.
            :exc:`PineconeTimeoutError`: If the file still exists after
                *timeout* seconds.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> pc.assistants.delete_file(
            ...     assistant_name="my-assistant",
            ...     file_id="file-abc123",
            ... )
        """
        data_http = self._data_plane_http(assistant_name)
        logger.info("Deleting file %r from assistant %r", file_id, assistant_name)
        data_http.delete(f"/files/{assistant_name}/{file_id}")
        logger.debug("Deleted file %r from assistant %r", file_id, assistant_name)

        if timeout == -1:
            return

        start = time.monotonic()
        while True:
            try:
                file_model = self.describe_file(assistant_name=assistant_name, file_id=file_id)
            except NotFoundError:
                return
            if file_model.status not in ("Deleting", None):
                raise PineconeError(
                    f"File deletion failed for '{file_id}': the file is in state "
                    f"{file_model.status!r}. Call describe_operation() for the failure reason."
                )
            if timeout is not None:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    raise PineconeTimeoutError(f"File '{file_id}' still exists after {timeout}s")
            time.sleep(_DELETE_POLL_INTERVAL_SECONDS)

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

        Creates an assistant and optionally polls until it reaches ``"Ready"``
        status. The assistant starts in ``"Initializing"`` status.

        Args:
            name (str): Name for the new assistant. Must be 1-63 characters,
                start and end with an alphanumeric character, and consist only
                of lowercase alphanumeric characters or hyphens.
            instructions (str | None): Optional directive for the assistant to
                apply to all responses. Maximum 16 KB.
            metadata (dict[str, Any] | None): Optional metadata dictionary.
                When omitted or ``None``, no metadata is sent and the assistant
                is created without metadata (``None``).
            region (str): Region to deploy the assistant in. Must be ``"us"``
                or ``"eu"`` (case-sensitive). Defaults to ``"us"``.
            environment (str | None): Optional environment override. Restricted
                to Pinecone-internal org plans; passing this on a non-internal
                plan raises a 403 error from the backend.
            timeout (float | None): Seconds to wait for the assistant to become
                ready. Use ``None`` (default) to poll indefinitely. Use ``-1``
                to return immediately without polling. Use ``0`` or a positive
                value to poll with a deadline. Raises
                :exc:`PineconeTimeoutError` if the assistant is not ready
                before the deadline.

        Returns:
            :class:`AssistantModel` describing the created assistant.

        Raises:
            :exc:`PineconeValueError`: If *region* is not ``"us"`` or ``"eu"``.
            :exc:`PineconeTimeoutError`: If the assistant does not become ready
                before the deadline.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> assistant = pc.assistants.create(name="my-assistant")  # doctest: +SKIP

            >>> assistant = pc.assistants.create(  # doctest: +SKIP
            ...     name="research-assistant",
            ...     instructions="You are a helpful research assistant.",
            ...     metadata={"team": "engineering", "version": "1"},
            ...     region="eu",
            ... )
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

        Returns:
            :class:`AssistantModel` with name, status, created_at, updated_at,
            metadata, instructions, and host.

        Raises:
            :exc:`ApiError`: If the API returns an error response (e.g. 404
                when the assistant does not exist).

        Examples:
            >>> assistant = pc.assistants.describe(name="my-assistant")
            >>> assistant.status  # doctest: +SKIP
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

        logger.info("Describing assistant %r", name)
        response = self._http.get(f"/assistants/{name}")
        model = self._attach_ref(self._adapter.to_assistant(response.content))
        logger.debug("Described assistant %r (status=%s)", name, model.status)
        return model

    def list(
        self,
        *,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> Paginator[AssistantModel]:
        """List assistants in the project with transparent lazy pagination.

        Args:
            limit (int | None): Maximum number of assistants to yield across
                all pages. ``None`` (default) yields all assistants.
            pagination_token (str | None): Token to resume pagination from a
                previous call.

        Returns:
            :class:`Paginator` over :class:`AssistantModel` objects. Supports
            ``for`` loops, ``.to_list()``, ``.pages()``, and ``limit``.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            .. code-block:: python

                for a in pc.assistants.list():
                    print(a.name, a.status)

                all_assistants = pc.assistants.list().to_list()
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

        Only the parameters that are explicitly provided are sent in the
        request. Omitted parameters are not included as query params.

        Args:
            page_size (int | None): Maximum number of assistants per page.
                Only sent when explicitly provided.
            pagination_token (str | None): Token from a previous response
                to fetch the next page.

        Returns:
            :class:`ListAssistantsResponse` with an ``assistants`` list and
            an optional ``next`` continuation token.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            .. code-block:: python

                page = pc.assistants.list_page(page_size=10)
                names = [a.name for a in page.assistants]
                token = page.next  # use as pagination_token for the next call
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

        Updates the specified assistant's instructions and/or metadata.
        Metadata is fully replaced (not merged) when provided.

        Args:
            name (str): The name of the assistant to update.
            instructions (str | None): New instructions for the assistant.
                Pass an empty string to clear existing instructions.
            metadata (dict[str, Any] | None): New metadata dictionary. Fully
                replaces any existing metadata rather than merging.

        Returns:
            :class:`AssistantModel` describing the updated assistant.

        Raises:
            :exc:`ApiError`: If the API returns an error response (e.g. 404
                when the assistant does not exist).

        Examples:
            >>> assistant = pc.assistants.update(  # doctest: +SKIP
            ...     name="my-assistant",
            ...     instructions="You are a helpful research assistant.",
            ... )

            >>> assistant = pc.assistants.update(  # doctest: +SKIP
            ...     name="my-assistant",
            ...     metadata={"team": "ml", "version": "2"},
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

        body: dict[str, Any] = {}
        if instructions is not None:
            body["instructions"] = instructions
        if metadata is not None:
            body["metadata"] = metadata

        logger.info("Updating assistant %r", name)
        response = self._http.patch(f"/assistants/{name}", json=body)
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

        Sends a DELETE request, then polls every 5 seconds until the
        assistant is confirmed gone (404 from describe). Other errors
        during polling propagate immediately.

        Args:
            name (str): The name of the assistant to delete.
            timeout (float | None): Seconds to wait for the assistant to
                disappear. Use ``None`` (default) to poll indefinitely.
                Use ``-1`` to return immediately without polling.
                Use a positive value to poll with a deadline. Raises
                :exc:`PineconeTimeoutError` if the assistant is not gone
                before the deadline.

        Returns:
            None

        Raises:
            :exc:`PineconeTimeoutError`: If the assistant still exists after
                *timeout* seconds.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            .. code-block:: python

                pc.assistants.delete(name="my-assistant")

                # Return immediately without waiting for deletion
                pc.assistants.delete(name="my-assistant", timeout=-1)
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

        logger.info("Deleting assistant %r", name)
        self._http.delete(f"/assistants/{name}")
        logger.debug("Deleted assistant %r", name)

        if timeout == -1:
            return

        start = time.monotonic()
        while True:
            try:
                self.describe(name=name)
            except NotFoundError:
                return
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

        Retrieves context snippets matching a text query or conversation
        history. Exactly one of *query* or *messages* must be provided
        and non-empty.

        Args:
            assistant_name: Name of the assistant to retrieve context from.
            query: Text query to use for context retrieval. Mutually exclusive
                with *messages*. Empty string is treated as not provided.
            messages: Conversation messages to use for context retrieval.
                Mutually exclusive with *query*. Empty list is treated as not
                provided. Dicts are converted to :class:`Message` objects.
            filter: Metadata filter restricting which documents contribute
                context. Omitted from request when ``None``.
            top_k: Maximum number of context snippets to return. Omitted
                from request when ``None``.
            snippet_size: Maximum snippet size in tokens. Omitted from
                request when ``None``.
            multimodal: Whether to include image-related context snippets.
                Omitted from request when ``None``.
            include_binary_content: Whether image snippets include base64
                image data. Only meaningful when *multimodal* is ``True``.
                Omitted from request when ``None``.

        Returns:
            :class:`ContextResponse` containing the matching context snippets.

        Raises:
            :exc:`PineconeValueError`: If both or neither of *query* and
                *messages* are provided (or if they are empty).
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            .. code-block:: python

                response = pc.assistants.context(
                    assistant_name="my-assistant",
                    query="What is Pinecone?",
                )
                for snippet in response.snippets:
                    print(snippet.content)
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
        response = http.post(f"/chat/{assistant_name}/context", json=body)
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
    ) -> ChatResponse | ChatStream:
        """Chat with an assistant and receive citations in Pinecone-native format.

        Args:
            assistant_name (str): Name of the assistant to chat with.
            messages (list[Message | dict[str, str]]): Conversation messages.
                Dicts are converted to :class:`Message` objects; role defaults
                to ``"user"`` when not present.
            model (str): Large language model to use. Defaults to ``"gpt-4o"``.
                The models the ``2026-07`` API documents for this endpoint are
                ``"gpt-4o"``, ``"gpt-4.1"``, ``"gpt-5"``, ``"o4-mini"``,
                ``"claude-sonnet-4-5"``, and ``"gemini-2.5-pro"``. The removed
                aliases ``"claude-3-5-sonnet"`` and ``"claude-3-7-sonnet"`` are
                still accepted but deprecated — the backend silently remaps
                them to ``"claude-sonnet-4-5"``, so migrate to that name.
                The value is not validated client-side: the backend is
                authoritative and gains models between SDK releases, and it
                rejects an unknown name with a 400 whose message lists the
                values it accepts.
            stream (bool): If ``True``, return a :class:`ChatStream`. Defaults
                to ``False``.
            temperature (float | None): Controls randomness. Lower values produce
                more deterministic responses. Omitted from request when ``None``.
            filter (dict[str, Any] | None): Metadata filter restricting which
                documents are used as context. Omitted from request when ``None``.
            json_response (bool): If ``True``, instruct the assistant to return
                a JSON response. Cannot be used with streaming.
            include_highlights (bool): If ``True``, include highlight snippets
                from referenced documents in citations.
            context_options (ContextOptions | dict[str, Any] | None): Options
                controlling context retrieval. Omitted from request when ``None``.

        Returns:
            :class:`ChatResponse` for non-streaming requests, or a
            :class:`ChatStream` for streaming requests.

        Raises:
            :exc:`PineconeValueError`: If both ``stream=True`` and
                ``json_response=True`` are specified.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            .. code-block:: python

                from pinecone import Pinecone
                pc = Pinecone(api_key="your-api-key")
                response = pc.assistants.chat(
                    assistant_name="my-assistant",
                    messages=[{"content": "What is Pinecone?"}],
                )

            .. code-block:: python

                stream = pc.assistants.chat(
                    assistant_name="my-assistant",
                    messages=[{"content": "What is Pinecone?"}],
                    stream=True,
                )
                for text in stream.text():
                    print(text, end="", flush=True)
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
                self._chat_streaming(http=http, url=f"/chat/{assistant_name}", body=body)
            )

        response = http.post(f"/chat/{assistant_name}", json=body)
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
    ) -> ChatCompletionResponse | ChatCompletionStream:
        """Chat with an assistant using an OpenAI-compatible interface.

        Returns responses in OpenAI chat completion format. Useful when you
        need inline citations or OpenAI-compatible responses. Has limited
        functionality compared to the standard :meth:`chat` interface — does
        not support ``include_highlights``, ``context_options``, or
        ``json_response`` parameters.

        Args:
            assistant_name (str): Name of the assistant to chat with.
            messages (list[Message | dict[str, str]]): Conversation messages.
                Dicts are converted to :class:`Message` objects; role defaults
                to ``"user"`` when not present.
            model (str): Large language model to use. Defaults to ``"gpt-4o"``.
                The models the ``2026-07`` API documents for this endpoint are
                ``"gpt-4o"``, ``"gpt-4.1"``, ``"o4-mini"``,
                ``"claude-sonnet-4-5"``, and ``"gemini-2.5-pro"`` — the same
                list :meth:`chat` accepts, minus ``"gpt-5"``, which the spec
                documents only on the Pinecone-native chat endpoint. The
                removed aliases ``"claude-3-5-sonnet"`` and
                ``"claude-3-7-sonnet"`` are still accepted but deprecated — the
                backend silently remaps them to ``"claude-sonnet-4-5"``, so
                migrate to that name. The value is not validated client-side:
                the backend is authoritative and gains models between SDK
                releases, and it rejects an unknown name with a 400 whose
                message lists the values it accepts.
            stream (bool): If ``True``, return a :class:`ChatCompletionStream`.
                Defaults to ``False``.
            temperature (float | None): Controls randomness. Lower values produce
                more deterministic responses. Omitted from request when ``None``.
            filter (dict[str, Any] | None): Metadata filter restricting which
                documents are used as context. Omitted from request when ``None``.

        Returns:
            :class:`ChatCompletionResponse` for non-streaming requests, or a
            :class:`ChatCompletionStream` for streaming requests.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            .. code-block:: python

                from pinecone import Pinecone
                pc = Pinecone(api_key="your-api-key")
                response = pc.assistants.chat_completions(
                    assistant_name="research-assistant",
                    messages=[{"content": "Explain quantum entanglement briefly."}],
                )
                response.choices[0].message.content

            .. code-block:: python

                stream = pc.assistants.chat_completions(
                    assistant_name="research-assistant",
                    messages=[{"content": "Explain quantum entanglement briefly."}],
                    stream=True,
                )
                for chunk in stream:
                    print(chunk)
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
            url = f"/chat/{assistant_name}/chat/completions"
            return ChatCompletionStream(
                self._chat_completions_streaming(http=http, url=url, body=body)
            )

        response = http.post(f"/chat/{assistant_name}/chat/completions", json=body)
        return self._adapter.to_chat_completion_response(response.content)

    def _chat_streaming(
        self,
        *,
        http: HTTPClient,
        url: str,
        body: dict[str, Any],
    ) -> Iterator[ChatStreamChunk]:
        """Stream Pinecone-native chat chunks via SSE.

        POSTs to the given URL with ``stream=True`` in the body,
        parses each SSE line, and yields typed chunk objects dispatched by the
        ``type`` field.

        Args:
            http: Pre-resolved data-plane HTTP client for the assistant.
            url: Request URL path (e.g. ``/chat/{assistant_name}``).
            body: Pre-built request body (must include ``stream=True``).

        Yields:
            :class:`StreamMessageStart`, :class:`StreamContentChunk`,
            :class:`StreamCitationChunk`, or :class:`StreamMessageEnd`
            depending on the ``type`` field of each SSE chunk.

        Raises:
            :exc:`ApiError`: If the server returns an HTTP error.
        """
        with http.stream(
            "POST",
            url,
            content=orjson.dumps(body),
            headers={"Content-Type": "application/json"},
        ) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                line = line[5:].lstrip()
                if not line:
                    continue
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
    ) -> Iterator[ChatCompletionStreamChunk]:
        """Stream OpenAI-compatible chat completion chunks via SSE.

        POSTs to the given URL with ``stream=True`` in the body and yields each
        SSE line parsed as a :class:`ChatCompletionStreamChunk`.

        Args:
            http: Pre-resolved data-plane HTTP client for the assistant.
            url: Request URL path (e.g. ``/chat/{assistant_name}/chat/completions``).
            body: Pre-built request body (must include ``stream=True``).

        Yields:
            :class:`ChatCompletionStreamChunk` for each non-empty SSE line.

        Raises:
            :exc:`ApiError`: If the server returns an HTTP error.
        """
        with http.stream(
            "POST",
            url,
            content=orjson.dumps(body),
            headers={"Content-Type": "application/json"},
        ) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                line = line[5:].lstrip()
                if not line:
                    continue
                if line == "[DONE]":
                    break
                yield msgspec.convert(orjson.loads(line), ChatCompletionStreamChunk)

    def evaluate_alignment(
        self,
        *,
        question: str,
        answer: str,
        ground_truth_answer: str,
    ) -> AlignmentResult:
        """Evaluate answer alignment against a ground truth answer.

        Measures the correctness and completeness of a generated answer with
        respect to a ground truth answer. Alignment is the harmonic mean of
        correctness (precision) and completeness (recall).

        Args:
            question: The question for which the answer was generated.
            answer: The generated answer to evaluate.
            ground_truth_answer: The ground truth answer to compare against.

        Returns:
            :class:`AlignmentResult` with aggregate scores, per-fact entailment
            results, and token usage statistics.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> result = pc.assistants.evaluate_alignment(
            ...     question="What is the capital of Spain?",
            ...     answer="Barcelona.",
            ...     ground_truth_answer="Madrid.",
            ... )
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
            response = self._http.get(f"/assistants/{name}")
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
