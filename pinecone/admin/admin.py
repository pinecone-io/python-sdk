"""Admin client for Pinecone organization and project management."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import httpx
import orjson

from pinecone import __version__
from pinecone._internal.adapters._decode import decode_response
from pinecone._internal.config import PineconeConfig, normalize_host
from pinecone._internal.constants import ADMIN_API_VERSION, API_VERSION_HEADER, DEFAULT_BASE_URL
from pinecone._internal.http_client import HTTPClient, _build_socket_options, _RetryTransport
from pinecone._internal.user_agent import build_user_agent
from pinecone.errors.exceptions import (
    ApiError,
    PineconeConnectionError,
    PineconeTimeoutError,
    ResponseParsingError,
    ValidationError,
)
from pinecone.models.admin.token import TokenResponse

if TYPE_CHECKING:
    from pinecone.admin.api_keys import ApiKeys
    from pinecone.admin.invites import Invites
    from pinecone.admin.organizations import Organizations
    from pinecone.admin.projects import Projects
    from pinecone.admin.service_accounts import ServiceAccounts
    from pinecone.admin.users import Users

_OAUTH_URL: str = "https://login.pinecone.io/oauth/token"
_OAUTH_AUDIENCE: str = "https://api.pinecone.io/"


class Admin:
    """Admin client for Pinecone organization and project management.

    Authenticates via OAuth2 client credentials flow to obtain a Bearer
    token used for all admin API calls.

    **Auth model:** :class:`Admin` uses OAuth2 client credentials (service account), while
    :class:`~pinecone.Pinecone` uses API keys.  These serve different purposes:

    - :class:`Admin` — organization/project/key management (create projects, rotate keys, etc.)
    - :class:`~pinecone.Pinecone` — index and vector operations (upsert, query, etc.)

    A common workflow bridges both: use :class:`Admin` to create a project and API key, then
    pass that key to :class:`~pinecone.Pinecone` for data-plane operations::

        from pinecone import Admin, Pinecone, ServerlessSpec

        admin = Admin(client_id="...", client_secret="...")
        project = admin.projects.create(name="my-project")
        key = admin.api_keys.create(project_id=project.id, name="my-key")
        pc = Pinecone(api_key=key.value)
        pc.indexes.create(name="my-index", dimension=1536, metric="cosine",
                          spec=ServerlessSpec(cloud="aws", region="us-east-1"))

    Projects are created within the organization associated with your OAuth credentials.

    .. note::
        **Obtaining OAuth credentials** — Service account credentials (``client_id`` and
        ``client_secret``) are created in the Pinecone console:

        1. Go to `console.pinecone.io <https://console.pinecone.io>`_.
        2. Navigate to **Organization Settings** → **Service Accounts**.
        3. Click **Create Service Account**, assign the desired role, and save the generated
           ``client_id`` and ``client_secret``.

        These differ from the API keys used by :class:`~pinecone.Pinecone`; they are scoped to
        your organization and used exclusively for admin operations.

    Args:
        client_id (str | None): OAuth2 client ID. Falls back to ``PINECONE_CLIENT_ID`` env var.
        client_secret (str | None): OAuth2 client secret. Falls back to ``PINECONE_CLIENT_SECRET``
            env var.
        additional_headers (dict[str, str] | None): Extra headers included in every admin API
            request. Merged last, so an entry keyed exactly ``"Authorization"`` or
            ``"X-Pinecone-Api-Version"`` replaces the Bearer token or the API version the SDK
            would otherwise send. Matching is case-sensitive: any other spelling is sent
            alongside the SDK's own header rather than replacing it.
        proxy_url (str | None): HTTP proxy URL for outgoing requests.
        ssl_verify (bool): Whether to verify SSL certificates. Defaults to ``True``.
        source_tag (str | None): Tag appended to the User-Agent string for request attribution.
        host (str | None): Admin API host. Falls back to ``PINECONE_CONTROLLER_HOST`` env var,
            then defaults to ``https://api.pinecone.io``. A value with no scheme is prefixed
            with ``https://``, matching :class:`~pinecone.Pinecone`. Intended for pointing the
            client at a local simulator in tests or at a private Pinecone deployment; leave it
            unset against production.
        oauth_url (str | None): Full URL of the OAuth2 token endpoint, including its path.
            Defaults to ``https://login.pinecone.io/oauth/token``. A value with no scheme is
            prefixed with ``https://``. Intended for pointing the token exchange at a local
            simulator in tests or at a private Pinecone deployment; leave it unset against
            production. There is no environment-variable fallback for this parameter.

    Raises:
        :exc:`~pinecone.errors.exceptions.PineconeValueError`:
            If client_id or client_secret cannot be resolved.
        :exc:`ApiError`: If the OAuth token request fails.

    Examples:

        >>> from pinecone import Admin
        >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
        >>> for org in admin.organizations.list():
        ...     print(org.name)
    """

    def __init__(
        self,
        *,
        client_id: str | None = None,
        client_secret: str | None = None,
        additional_headers: dict[str, str] | None = None,
        proxy_url: str | None = None,
        ssl_verify: bool = True,
        source_tag: str | None = None,
        host: str | None = None,
        oauth_url: str | None = None,
    ) -> None:
        resolved_id = client_id or os.environ.get("PINECONE_CLIENT_ID", "")
        resolved_secret = client_secret or os.environ.get("PINECONE_CLIENT_SECRET", "")

        if not resolved_id or not resolved_id.strip():
            raise ValidationError(
                "No client_id provided. Pass client_id='...' or set the "
                "PINECONE_CLIENT_ID environment variable."
            )
        if not resolved_secret or not resolved_secret.strip():
            raise ValidationError(
                "No client_secret provided. Pass client_secret='...' or set the "
                "PINECONE_CLIENT_SECRET environment variable."
            )

        resolved_source_tag = source_tag or ""

        token = self._fetch_token(
            resolved_id,
            resolved_secret,
            proxy_url=proxy_url,
            ssl_verify=ssl_verify,
            source_tag=resolved_source_tag,
            oauth_url=normalize_host(oauth_url) or _OAUTH_URL,
        )

        headers: dict[str, str] = {
            "Authorization": f"Bearer {token}",
            API_VERSION_HEADER: ADMIN_API_VERSION,
        }
        if additional_headers:
            headers.update(additional_headers)

        config = PineconeConfig(
            api_key="",
            host=host or "",
            additional_headers=headers,
            proxy_url=proxy_url or "",
            ssl_verify=ssl_verify,
            source_tag=resolved_source_tag,
        )
        # __post_init__ already applied the PINECONE_CONTROLLER_HOST fallback and
        # scheme normalization; only the api.pinecone.io default is left to fill in.
        if not config.host:
            config = replace(config, host=DEFAULT_BASE_URL)

        # Prevent __post_init__ from falling back to PINECONE_API_KEY env var.
        # The Admin client authenticates via OAuth Bearer token, not Api-Key.
        # Must follow the replace() above, which re-runs __post_init__.
        object.__setattr__(config, "api_key", "")

        self._http = HTTPClient(config, ADMIN_API_VERSION)

        self._organizations: Organizations | None = None
        self._projects: Projects | None = None
        self._api_keys: ApiKeys | None = None
        self._users: Users | None = None
        self._invites: Invites | None = None
        self._service_accounts: ServiceAccounts | None = None

    def _fetch_token(
        self,
        client_id: str,
        client_secret: str,
        *,
        proxy_url: str | None = None,
        ssl_verify: bool = True,
        source_tag: str | None = None,
        oauth_url: str | None = None,
    ) -> str:
        """Exchange client credentials for a Bearer token.

        Args:
            client_id: OAuth2 client ID.
            client_secret: OAuth2 client secret.
            proxy_url: Optional HTTP proxy URL.
            ssl_verify: Whether to verify SSL certificates.
            source_tag: Optional source tag to append to the User-Agent string.
            oauth_url: Token endpoint URL. Defaults to the production endpoint; overridden
                when testing against a simulator or a private deployment.

        Returns:
            The access token string.

        Raises:
            ApiError: If the token request fails.
        """
        body = orjson.dumps(
            {
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "client_credentials",
                "audience": _OAUTH_AUDIENCE,
            }
        )

        transport = _RetryTransport(
            transport=httpx.HTTPTransport(http2=False, socket_options=_build_socket_options()),
        )
        with httpx.Client(
            transport=transport,
            proxy=proxy_url or None,
            verify=ssl_verify,
        ) as client:
            try:
                response = client.post(
                    oauth_url or _OAUTH_URL,
                    content=body,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": build_user_agent(__version__, source_tag),
                        API_VERSION_HEADER: ADMIN_API_VERSION,
                    },
                )
            except httpx.TimeoutException as exc:
                raise PineconeTimeoutError(str(exc)) from exc
            except httpx.TransportError as exc:
                raise PineconeConnectionError(str(exc)) from exc

        if not response.is_success:
            err_body: dict[str, Any] | None = None
            try:
                err_body = response.json()
            except Exception:
                err_body = None

            message = "OAuth token request failed"
            if err_body and isinstance(err_body.get("error_description"), str):
                message = err_body["error_description"]
            elif err_body and isinstance(err_body.get("error"), str):
                message = err_body["error"]

            raise ApiError(
                message=message,
                status_code=response.status_code,
                body=err_body,
            )

        try:
            token = decode_response(response.content, TokenResponse)
        except ResponseParsingError:
            token = None

        if token is None or not token.access_token:
            missing_body: dict[str, Any] | None
            try:
                missing_body = response.json()
            except Exception:
                missing_body = None
            raise ApiError(
                message="OAuth response missing access_token",
                status_code=response.status_code,
                body=missing_body,
            )

        return token.access_token

    @property
    def organizations(self) -> Organizations:
        """Access the Organizations namespace for organization operations.

        Lazily imported and instantiated on first access.

        Returns:
            :class:`Organizations` namespace instance.

        Examples:

            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> for org in admin.organizations.list():
            ...     print(org.name)
        """
        if self._organizations is None:
            from pinecone.admin.organizations import Organizations as _Organizations

            self._organizations = _Organizations(http=self._http)
        return self._organizations

    @property
    def projects(self) -> Projects:
        """Access the Projects namespace for project operations.

        Lazily imported and instantiated on first access.

        Returns:
            :class:`Projects` namespace instance.

        Examples:

            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> for project in admin.projects.list():
            ...     print(project.name)
        """
        if self._projects is None:
            from pinecone.admin.projects import Projects as _Projects

            self._projects = _Projects(http=self._http, admin=self)
        return self._projects

    @property
    def api_keys(self) -> ApiKeys:
        """Access the ApiKeys namespace for API key operations.

        Lazily imported and instantiated on first access.

        Returns:
            :class:`ApiKeys` namespace instance.

        Examples:

            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> keys = admin.api_keys.list(project_id="proj-abc123")
            >>> for key in keys:
            ...     print(key.key.id)
        """
        if self._api_keys is None:
            from pinecone.admin.api_keys import ApiKeys as _ApiKeys

            self._api_keys = _ApiKeys(http=self._http)
        return self._api_keys

    @property
    def users(self) -> Users:
        """Access the Users namespace for organization-member operations.

        Lazily imported and instantiated on first access.

        Returns:
            :class:`Users` namespace instance.

        Examples:

            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> for user in admin.users.list():
            ...     print(user.email)
        """
        if self._users is None:
            from pinecone.admin.users import Users as _Users

            self._users = _Users(http=self._http)
        return self._users

    @property
    def invites(self) -> Invites:
        """Access the Invites namespace for organization-invite operations.

        Lazily imported and instantiated on first access.

        Returns:
            :class:`Invites` namespace instance.

        Examples:

            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> for invite in admin.invites.list():
            ...     print(invite.email, invite.status)
        """
        if self._invites is None:
            from pinecone.admin.invites import Invites as _Invites

            self._invites = _Invites(http=self._http)
        return self._invites

    @property
    def service_accounts(self) -> ServiceAccounts:
        """Access the ServiceAccounts namespace for service-account operations.

        These are the same credentials this client authenticates with, so
        rotating or deleting the account behind ``client_id``/``client_secret``
        breaks this client. Lazily imported and instantiated on first access.

        Returns:
            :class:`ServiceAccounts` namespace instance.

        Examples:

            >>> from pinecone import Admin
            >>> admin = Admin(client_id="your-client-id", client_secret="your-client-secret")
            >>> for account in admin.service_accounts.list():
            ...     print(account.id, account.name)
        """
        if self._service_accounts is None:
            from pinecone.admin.service_accounts import ServiceAccounts as _ServiceAccounts

            self._service_accounts = _ServiceAccounts(http=self._http)
        return self._service_accounts

    def __repr__(self) -> str:
        return (
            "Admin(organizations=<Organizations>, projects=<Projects>, "
            "api_keys=<ApiKeys>, users=<Users>, invites=<Invites>, "
            "service_accounts=<ServiceAccounts>)"
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()

    def __enter__(self) -> Admin:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
