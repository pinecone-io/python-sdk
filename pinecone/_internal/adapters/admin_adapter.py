"""Adapter for Admin API responses."""

from __future__ import annotations

from msgspec import Struct

from pinecone._internal.adapters._decode import decode_response
from pinecone.models.admin.api_key import APIKeyList, APIKeyModel, APIKeyWithSecret
from pinecone.models.admin.invite import InviteList, InviteModel
from pinecone.models.admin.organization import OrganizationList, OrganizationModel
from pinecone.models.admin.project import ProjectList, ProjectModel
from pinecone.models.admin.user import UserList, UserModel


class _OrganizationListEnvelope(Struct, kw_only=True):
    """Internal envelope for the list-organizations response."""

    data: list[OrganizationModel] = []


class _APIKeyListEnvelope(Struct, kw_only=True):
    """Internal envelope for the list-api-keys response."""

    data: list[APIKeyModel] = []


class _ProjectListEnvelope(Struct, kw_only=True):
    """Internal envelope for the list-projects response."""

    data: list[ProjectModel] = []


class AdminAdapter:
    """Transforms raw Admin API JSON into domain models."""

    @staticmethod
    def to_organization(data: bytes) -> OrganizationModel:
        """Decode raw JSON bytes into an :class:`OrganizationModel`.

        Args:
            data (bytes): Raw JSON response bytes from the Admin API.

        Returns:
            :class:`OrganizationModel`: Decoded organization.

        Raises:
            :exc:`ResponseParsingError`: If ``data`` cannot be decoded into
                :class:`OrganizationModel`.
        """
        return decode_response(data, OrganizationModel)

    @staticmethod
    def to_organization_list(data: bytes) -> OrganizationList:
        """Decode raw JSON bytes from a list-organizations response into an
        :class:`OrganizationList`.

        Args:
            data (bytes): Raw JSON response bytes from the Admin API.

        Returns:
            :class:`OrganizationList`: Decoded list of organizations.

        Raises:
            :exc:`ResponseParsingError`: If ``data`` cannot be decoded.
        """
        envelope = decode_response(data, _OrganizationListEnvelope)
        return OrganizationList(envelope.data)

    @staticmethod
    def to_project(data: bytes) -> ProjectModel:
        """Decode raw JSON bytes into a :class:`ProjectModel`.

        Args:
            data (bytes): Raw JSON response bytes from the Admin API.

        Returns:
            :class:`ProjectModel`: Decoded project.

        Raises:
            :exc:`ResponseParsingError`: If ``data`` cannot be decoded into :class:`ProjectModel`.
        """
        return decode_response(data, ProjectModel)

    @staticmethod
    def to_project_list(data: bytes) -> ProjectList:
        """Decode raw JSON bytes from a list-projects response into a :class:`ProjectList`.

        Args:
            data (bytes): Raw JSON response bytes from the Admin API.

        Returns:
            :class:`ProjectList`: Decoded list of projects.

        Raises:
            :exc:`ResponseParsingError`: If ``data`` cannot be decoded.
        """
        envelope = decode_response(data, _ProjectListEnvelope)
        return ProjectList(envelope.data)

    @staticmethod
    def to_api_key(data: bytes) -> APIKeyModel:
        """Decode raw JSON bytes into an :class:`APIKeyModel`.

        Args:
            data (bytes): Raw JSON response bytes from the Admin API.

        Returns:
            :class:`APIKeyModel`: Decoded API key.

        Raises:
            :exc:`ResponseParsingError`: If ``data`` cannot be decoded into :class:`APIKeyModel`.
        """
        return decode_response(data, APIKeyModel)

    @staticmethod
    def to_api_key_with_secret(data: bytes) -> APIKeyWithSecret:
        """Decode raw JSON bytes into an :class:`APIKeyWithSecret`.

        Args:
            data (bytes): Raw JSON response bytes from the Admin API.

        Returns:
            :class:`APIKeyWithSecret`: Decoded API key including the secret value.

        Raises:
            :exc:`ResponseParsingError`: If ``data`` cannot be decoded into
                :class:`APIKeyWithSecret`.
        """
        return decode_response(data, APIKeyWithSecret)

    @staticmethod
    def to_api_key_list(data: bytes) -> APIKeyList:
        """Decode raw JSON bytes from a list-api-keys response into an :class:`APIKeyList`.

        Args:
            data (bytes): Raw JSON response bytes from the Admin API.

        Returns:
            :class:`APIKeyList`: Decoded list of API keys.

        Raises:
            :exc:`ResponseParsingError`: If ``data`` cannot be decoded.
        """
        envelope = decode_response(data, _APIKeyListEnvelope)
        return APIKeyList(envelope.data)

    @staticmethod
    def to_invite(data: bytes) -> InviteModel:
        """Decode raw JSON bytes into an :class:`InviteModel`.

        Args:
            data (bytes): Raw JSON response bytes from the Admin API.

        Returns:
            :class:`InviteModel`: Decoded invite.

        Raises:
            :exc:`ResponseParsingError`: If ``data`` cannot be decoded into :class:`InviteModel`.
        """
        return decode_response(data, InviteModel)

    @staticmethod
    def to_invite_list(data: bytes) -> InviteList:
        """Decode raw JSON bytes from a list-invites response into an :class:`InviteList`.

        Like ``UserList`` and unlike the unpaginated admin list responses,
        ``InviteList`` is itself the wire schema — it carries the ``pagination``
        cursor envelope alongside ``data`` — so no internal envelope struct is
        needed.

        Args:
            data (bytes): Raw JSON response bytes from the Admin API.

        Returns:
            :class:`InviteList`: Decoded page of invites plus the next-page cursor.

        Raises:
            :exc:`ResponseParsingError`: If ``data`` cannot be decoded.
        """
        return decode_response(data, InviteList)

    @staticmethod
    def to_user(data: bytes) -> UserModel:
        """Decode raw JSON bytes into a :class:`UserModel`.

        Args:
            data (bytes): Raw JSON response bytes from the Admin API.

        Returns:
            :class:`UserModel`: Decoded user.

        Raises:
            :exc:`ResponseParsingError`: If ``data`` cannot be decoded into :class:`UserModel`.
        """
        return decode_response(data, UserModel)

    @staticmethod
    def to_user_list(data: bytes) -> UserList:
        """Decode raw JSON bytes from a list-users response into a :class:`UserList`.

        Unlike the unpaginated admin list responses, ``UserList`` is itself the
        wire schema — it carries the ``pagination`` cursor envelope alongside
        ``data`` — so no internal envelope struct is needed.

        Args:
            data (bytes): Raw JSON response bytes from the Admin API.

        Returns:
            :class:`UserList`: Decoded page of users plus the next-page cursor.

        Raises:
            :exc:`ResponseParsingError`: If ``data`` cannot be decoded.
        """
        return decode_response(data, UserList)
