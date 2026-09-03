"""Every exception the Pinecone SDK raises.

Two families live here. :class:`ApiError` and its subclasses mean the server
answered with an HTTP error status. The rest are raised by the client with no
round trip: argument validation, transport failures, and responses the SDK
could not decode. All of them derive from :class:`PineconeError`.
"""

from __future__ import annotations

TYPE_CHECKING = False

if TYPE_CHECKING:
    from typing import Any


class PineconeError(Exception):
    """Base class for every exception the SDK raises.

    Catch this when one handler should cover any SDK failure; catch a subclass
    when a particular failure needs its own recovery. ``message`` holds the
    text the SDK or the server produced.

    Three subclasses also derive from a builtin — :class:`PineconeValueError`
    from :class:`ValueError`, :class:`PineconeTypeError` from
    :class:`TypeError`, :class:`PineconeTimeoutError` from
    :class:`TimeoutError` — so an ``except ValueError`` already in your code
    catches those without importing anything from Pinecone.

    Examples:
        >>> from pinecone import NotFoundError, PineconeError
        >>> try:
        ...     raise NotFoundError(message="No index named 'movie-recommendations'")
        ... except PineconeError as exc:
        ...     print(type(exc).__name__, exc.message)
        NotFoundError No index named 'movie-recommendations'

    .. seealso::
       :doc:`/guides/error-handling` — which errors a given call produces, what
       the SDK retries before raising, and how to order handlers.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class ApiError(PineconeError):
    """The server answered with an HTTP error status.

    Every status-specific class below derives from this one, so
    ``except ApiError`` catches all of them. A status with no dedicated
    subclass reaches the caller as a bare ``ApiError`` — ``400`` and ``422``,
    which are what a request the server considers malformed produces.

    Two attributes are worth reading. ``status_code`` says which class of
    failure it was; ``body`` is the parsed JSON response, and it is where a
    field-level explanation lives when the server sent one. ``str(exc)``
    already renders the status, the server's error code, and the request id,
    so logging the exception loses nothing. Quote ``request_id`` when you open
    a support ticket.

    Examples:
        >>> from pinecone import ApiError
        >>> exc = ApiError(
        ...     "No index named 'movie-recommendations'",
        ...     404,
        ...     body={"error": {"code": "NOT_FOUND"}},
        ...     request_id="req-9f2c",
        ... )
        >>> print(exc)
        [404] No index named 'movie-recommendations' (request_id: req-9f2c)
        >>> exc.status_code, exc.body["error"]["code"]
        (404, 'NOT_FOUND')

    .. seealso::
       :doc:`/guides/error-handling` — the full attribute table, and the
       handler shape for each status.
    """

    def __init__(
        self,
        message: str,
        status_code: int,
        body: dict[str, Any] | None = None,
        *,
        reason: str | None = None,
        headers: dict[str, str] | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.body = body
        self.reason = reason
        self.headers = headers
        self.error_code = error_code
        self.request_id = request_id
        super().__init__(message)

    def __str__(self) -> str:
        try:
            prefix = f"{self.status_code}"
            if self.error_code:
                prefix = f"{prefix} {self.error_code}"
            base = f"[{prefix}] {self.message}"
            if self.request_id:
                base = f"{base} (request_id: {self.request_id})"
            return base
        except Exception:
            # Never let __str__ raise — that would mask the original error.
            try:
                return f"[{self.status_code}] {self.message}"
            except Exception:
                return "<ApiError: unrenderable>"

    def __repr__(self) -> str:
        try:
            msg = self.message
            if len(msg) > 100:
                msg = msg[:97] + "..."
        except Exception:
            msg = "<unrenderable>"
        parts = [
            f"status_code={self.status_code}",
            f"message={msg!r}",
        ]
        if self.error_code is not None:
            parts.append(f"error_code={self.error_code!r}")
        if self.request_id is not None:
            parts.append(f"request_id={self.request_id!r}")
        if self.body is not None:
            try:
                parts.append(f"body={self.body!r}")
            except Exception:
                parts.append("body=<unrenderable>")
        return f"{type(self).__name__}({', '.join(parts)})"


class NotFoundError(ApiError):
    """404 — the resource the request named does not exist.

    A misspelled name, a resource someone else already deleted, or an API key
    scoped to a different project than the one that owns the resource all
    produce this. When you only want to know whether an index is there,
    :meth:`~pinecone.client.indexes.Indexes.exists` is clearer than catching
    this.

    .. note::
       A ``404`` is not proof of absence everywhere.
       :meth:`~pinecone.client.restore_jobs.RestoreJobs.describe` answers
       ``404`` for any failure to read the restore-job store, so there it
       means "could not produce this job", not "no such job" — do not key
       control flow on it.
    """

    def __init__(
        self,
        message: str = "Resource not found",
        status_code: int = 404,
        body: dict[str, Any] | None = None,
        *,
        reason: str | None = None,
        headers: dict[str, str] | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            body=body,
            reason=reason,
            headers=headers,
            error_code=error_code,
            request_id=request_id,
        )


class ConflictError(ApiError):
    """409 — the request conflicts with the resource's current state.

    Creating an index, collection, or backup under a name that is already
    taken is the common case. Guard the create with
    :meth:`~pinecone.client.indexes.Indexes.exists`, or catch this and treat
    it as a no-op when concurrent callers make the check pointless.
    """

    def __init__(
        self,
        message: str = "Resource conflict",
        status_code: int = 409,
        body: dict[str, Any] | None = None,
        *,
        reason: str | None = None,
        headers: dict[str, str] | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            body=body,
            reason=reason,
            headers=headers,
            error_code=error_code,
            request_id=request_id,
        )


class UnauthorizedError(ApiError):
    """401 — the request carried no usable credential.

    The API key was missing, malformed, or has been deleted, or an
    :class:`~pinecone.admin.Admin` client's OAuth2 credentials were rejected.
    Nothing about the request itself will fix it and retrying will not help:
    supply a valid credential. Check ``PINECONE_API_KEY`` first when you did
    not pass ``api_key`` explicitly.
    """

    def __init__(
        self,
        message: str = "Invalid or missing API key",
        status_code: int = 401,
        body: dict[str, Any] | None = None,
        *,
        reason: str | None = None,
        headers: dict[str, str] | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            body=body,
            reason=reason,
            headers=headers,
            error_code=error_code,
            request_id=request_id,
        )


class ForbiddenError(ApiError):
    """403 — the credential is valid but the operation is not permitted.

    Three causes account for most of these: the key's roles do not cover the
    operation, a quota on the project or organization has been reached, or a
    protection setting on the resource forbids it — deletion protection makes
    :meth:`~pinecone.client.indexes.Indexes.delete` answer ``403`` until you
    turn it off with :meth:`~pinecone.client.indexes.Indexes.configure`.

    Retrying never helps. ``message`` carries the server's explanation, which
    is what distinguishes the three.
    """

    def __init__(
        self,
        message: str = "Forbidden",
        status_code: int = 403,
        body: dict[str, Any] | None = None,
        *,
        reason: str | None = None,
        headers: dict[str, str] | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            body=body,
            reason=reason,
            headers=headers,
            error_code=error_code,
            request_id=request_id,
        )


class PaymentRequiredError(ApiError):
    """402 — the organization's billing state blocks the operation.

    Raised where the control plane gates resource creation on payment, notably
    :meth:`~pinecone.admin.projects.Projects.create` and
    :meth:`~pinecone.admin.api_keys.ApiKeys.create`, which need the
    organization to have an active payment method or a plan that permits the
    request.

    Retrying will not help: you or an organization owner has to resolve the
    billing state first. ``message`` carries the server's explanation
    verbatim, so it is the authoritative description of what needs fixing.
    """

    def __init__(
        self,
        message: str = "Payment required",
        status_code: int = 402,
        body: dict[str, Any] | None = None,
        *,
        reason: str | None = None,
        headers: dict[str, str] | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            body=body,
            reason=reason,
            headers=headers,
            error_code=error_code,
            request_id=request_id,
        )


class FailedPreconditionError(ApiError):
    """412 — the target resource is not in a state that permits the request.

    The request was well-formed. This is the dominant admin failure class:
    deleting a project or organization that still owns resources, or deleting
    a backup with a restore job still in flight, all answer 412.

    The precondition is usually satisfiable — delete the blocking resources, or
    wait for the in-flight operation to finish, then retry. ``message`` carries
    the server's explanation verbatim and typically names the specific resources
    or job ids that are in the way.
    """

    def __init__(
        self,
        message: str = "Precondition failed",
        status_code: int = 412,
        body: dict[str, Any] | None = None,
        *,
        reason: str | None = None,
        headers: dict[str, str] | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            body=body,
            reason=reason,
            headers=headers,
            error_code=error_code,
            request_id=request_id,
        )


class RateLimitError(ApiError):
    """429 — the request was throttled.

    The SDK retries ``429`` on its own, so one reaching your code means the
    retry budget was already spent. Retrying immediately in a loop will not
    get through; reduce the request rate, or raise the retry allowance.

    ``retry_after`` is how long the server asked you to wait, in seconds. It
    is parsed from the ``Retry-After`` response header when that header is
    present and expressible as a non-negative number of seconds; an HTTP-date
    value is not parsed, and leaves ``retry_after`` as ``None``.

    .. seealso::
       :doc:`/guides/retries` — what the SDK retries by default and how to
       change it.
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        status_code: int = 429,
        body: dict[str, Any] | None = None,
        *,
        reason: str | None = None,
        headers: dict[str, str] | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(
            message=message,
            status_code=status_code,
            body=body,
            reason=reason,
            headers=headers,
            error_code=error_code,
            request_id=request_id,
        )


class ServiceError(ApiError):
    """5xx — the server failed to handle a well-formed request.

    Nothing about the request needs changing. The SDK already retries the
    common 5xx statuses, so one reaching your code means the retry budget was
    spent — back off further before trying again. A 5xx outside the retryable
    set arrives on the first attempt instead.

    Read ``status_code`` to tell the two apart, and quote ``request_id`` if
    the failure persists.

    .. seealso::
       :doc:`/guides/retries` — which statuses are retried by default.
    """

    def __init__(
        self,
        message: str = "Internal server error",
        status_code: int = 500,
        body: dict[str, Any] | None = None,
        *,
        reason: str | None = None,
        headers: dict[str, str] | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            body=body,
            reason=reason,
            headers=headers,
            error_code=error_code,
            request_id=request_id,
        )


class IndexInitFailedError(PineconeError):
    """An index entered ``InitializationFailed`` while the SDK waited for it.

    Only a call that polls for readiness raises this —
    :meth:`~pinecone.client.indexes.Indexes.create` and
    :meth:`~pinecone.client.indexes.Indexes.create_for_model` do so unless you
    pass ``timeout=-1`` to return immediately. The index exists but will never
    become ready: delete it and create again, changing the deployment if that
    is what failed. ``index_name`` is the index that failed.
    """

    def __init__(self, index_name: str) -> None:
        super().__init__(f"Index '{index_name}' entered InitializationFailed state")
        self.index_name = index_name


class IndexTerminatedError(PineconeError):
    """An index reached a terminal state while the SDK waited for it.

    The terminal states are ``Terminating`` and ``Disabled``. Something
    outside this call deleted or disabled the index mid-wait, so waiting
    longer cannot succeed. ``name`` and ``state`` say which index and which
    state; :meth:`~pinecone.client.indexes.Indexes.describe` confirms whether
    it still exists.
    """

    def __init__(self, name: str, state: str) -> None:
        super().__init__(
            f"Index '{name}' entered terminal state '{state}'. "
            f"Check status with pc.describe_index(name='{name}')."
        )
        self.name = name
        self.state = state


class PineconeTimeoutError(PineconeError, TimeoutError):
    """An operation exceeded its timeout.

    Two deadlines produce this: a single HTTP request that outran the
    client's request timeout, and a readiness wait — creating or deleting an
    index, say — that outran the ``timeout`` you passed. The first is worth
    retrying; the second usually means the resource is still working, so
    describe it rather than re-issuing the call.

    Multiply inherits from Python's built-in :class:`TimeoutError` so that
    ``except TimeoutError`` blocks in caller code catch SDK timeouts without
    having to import a Pinecone-specific class. This is the same pattern used
    by :class:`PineconeValueError` (extends :class:`ValueError`).

    Args:
        message: Description of what timed out.
        response: Partial result, when the timeout interrupted a bulk operation
            that had already applied some of its work. Carrying it means the
            caller can tell what landed instead of having to re-send everything;
            ``response.failed_items`` is what remains. ``None`` for timeouts with
            nothing partial to report.
    """

    def __init__(self, message: str, *, response: Any | None = None) -> None:
        self.response = response
        super().__init__(message)


class PineconeConnectionError(PineconeError):
    """The connection failed before any response arrived.

    Covers DNS resolution failures, connection refused, read/write errors,
    and other transport-level problems. The SDK retries transport failures,
    so one reaching your code means the retry budget was spent — look at DNS,
    egress rules, and any proxy between you and Pinecone rather than at the
    request.
    """

    pass


class PineconeValueError(PineconeError, ValueError):
    """An argument had the right type but an unusable value.

    The SDK's own validation raises this before the request goes out, so
    nothing was created or changed. Fix the argument the message names.

    ``path`` locates the offending field when the raiser supplied one, and
    ``str(exc)`` then prefixes the message with ``at <path>:``. Also derives
    from :class:`ValueError`, so an ``except ValueError`` already in your code
    catches it.

    Examples:
        >>> from pinecone import PineconeValueError
        >>> print(PineconeValueError("dimension must be positive", "fields.embedding"))
        at fields.embedding: dimension must be positive
    """

    def __init__(self, message: str, path: str | None = None) -> None:
        self.path = path
        super().__init__(message)

    def __str__(self) -> str:
        try:
            if isinstance(self.path, str) and self.path:
                return f"at {self.path}: {self.message}"
            return self.message
        except Exception:
            return self.message if isinstance(self.message, str) else super().__str__()


class PineconeTypeError(PineconeError, TypeError):
    """An argument was of a type the SDK cannot use.

    Raised before the request goes out, so nothing was created or changed.
    Passing a keyword this SDK version no longer accepts produces this too,
    with a message naming the current argument to use instead.

    When the failure is a value inside a request body that will not
    JSON-encode, ``path`` locates it — ``records[2].embedding`` — which is
    the part worth reading on a bulk call. ``str(exc)`` prefixes the message
    with ``at <path>:``. Also derives from :class:`TypeError`.
    """

    def __init__(self, message: str, path: str | None = None) -> None:
        self.path = path
        super().__init__(message)

    def __str__(self) -> str:
        try:
            if isinstance(self.path, str) and self.path:
                return f"at {self.path}: {self.message}"
            return self.message
        except Exception:
            return self.message if isinstance(self.message, str) else super().__str__()


class ResponseParsingError(PineconeError):
    """The response arrived but the SDK could not decode it.

    The request succeeded, so this is not a failure you can fix by changing
    it. The usual cause is a response carrying a shape this SDK version does
    not model — a new deployment type or read-capacity mode, for instance —
    which an SDK upgrade resolves.

    ``cause`` holds the underlying deserialization error, and ``str(exc)``
    appends it, so the message already names the field that would not decode.
    Wrapping it this way is what lets an ``except PineconeError`` block catch
    a decode failure at all.
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        self.cause = cause
        super().__init__(message)

    def __str__(self) -> str:
        try:
            if self.cause is None:
                return self.message
            cause_name = type(self.cause).__name__
            try:
                cause_str = str(self.cause)
            except Exception:
                cause_str = "<unrenderable>"
            return f"{self.message} (caused by {cause_name}: {cause_str})"
        except Exception:
            return self.message if isinstance(self.message, str) else super().__str__()


# Backwards-compatible alias — most validation is value validation
ValidationError = PineconeValueError  # Deprecated: use PineconeValueError instead

# ---------------------------------------------------------------------------
# Legacy name aliases — :meta private:
# New code should use the canonical names above.
# ---------------------------------------------------------------------------

# Backcompat alias, :meta private:
PineconeException = PineconeError
# Backcompat alias, :meta private:
PineconeApiException = ApiError
# Backcompat alias, :meta private:
NotFoundException = NotFoundError
# Backcompat alias, :meta private:
UnauthorizedException = UnauthorizedError
# Backcompat alias, :meta private:
ForbiddenException = ForbiddenError
# Backcompat alias, :meta private:
ServiceException = ServiceError
# Backcompat alias, :meta private:
RateLimitException = RateLimitError
# Backcompat alias, :meta private:
PineconeConfigurationError = PineconeValueError
# Backcompat alias, :meta private:
PineconeProtocolError = PineconeError
# Backcompat alias, :meta private:
PineconeApiTypeError = PineconeTypeError
# Backcompat alias, :meta private:
PineconeApiValueError = PineconeValueError
# Backcompat alias, :meta private:
PineconeApiAttributeError = PineconeError
# Backcompat alias, :meta private:
PineconeApiKeyError = PineconeError
# Backcompat alias, :meta private:
ListConversionException = PineconeError
