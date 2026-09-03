"""Proxy that makes :attr:`Pinecone.assistant` both callable and attribute-accessible.

Legacy callers used ``pc.assistant`` as a namespace alias for ``pc.assistants``
*and* called ``pc.assistant("my-name")`` as a shortcut for
``pc.assistants.describe(name="my-name")``. This proxy preserves both forms
while the canonical namespace is the plural ``pc.assistants``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pinecone.async_client.assistants import AsyncAssistants
    from pinecone.client.assistants import Assistants
    from pinecone.models.assistant.model import AssistantModel


class _AssistantNamespaceProxy:
    """Callable + attribute-access proxy for the singular ``assistant`` alias."""

    _assistants: Assistants

    def __init__(self, assistants: Assistants) -> None:
        # Store via object.__setattr__ so __getattr__ is not triggered for
        # the private slot while still keeping _assistants out of the normal
        # attribute lookup path (preventing infinite recursion in __getattr__).
        object.__setattr__(self, "_assistants", assistants)

    def __call__(self, name: str) -> AssistantModel:
        """Look up an assistant by name (``pc.assistant(name)``).

        Equivalent to :meth:`Assistants.describe`.

        Args:
            name: The name of the assistant to look up.

        Returns:
            :class:`AssistantModel` for the named assistant.

        Examples:
            >>> assistant = pc.assistant("my-assistant")  # doctest: +SKIP
            >>> assistant.status  # doctest: +SKIP
            'Ready'
        """
        assistants: Assistants = object.__getattribute__(self, "_assistants")
        return assistants.describe(name=name)

    def __getattr__(self, attr: str) -> Any:
        """Forward attribute access to the underlying :class:`Assistants` namespace.

        Lets ``pc.assistant`` act as a stand-in for ``pc.assistants``, so any
        method available on :class:`Assistants` can also be called through
        the singular alias.

        Args:
            attr: Name of the attribute or method to look up on
                :class:`Assistants`.

        Returns:
            The attribute from the underlying :class:`Assistants` instance.

        Examples:
            >>> pc.assistant.create_assistant(assistant_name="my-assistant")  # doctest: +SKIP
        """
        # Called only for attributes not found on the proxy itself, so
        # forward to the underlying Assistants namespace.
        assistants: Assistants = object.__getattribute__(self, "_assistants")
        return getattr(assistants, attr)

    def __repr__(self) -> str:
        assistants: Assistants = object.__getattribute__(self, "_assistants")
        return f"<AssistantNamespaceProxy for {assistants!r}>"


class _AsyncAssistantNamespaceProxy:
    """Callable + attribute-access proxy for the singular ``assistant`` alias (async)."""

    _assistants: AsyncAssistants

    def __init__(self, assistants: AsyncAssistants) -> None:
        object.__setattr__(self, "_assistants", assistants)

    async def __call__(self, name: str) -> AssistantModel:
        """Look up an assistant by name (``await pc.assistant(name)``).

        Equivalent to :meth:`AsyncAssistants.describe`.

        Args:
            name: The name of the assistant to look up.

        Returns:
            :class:`AssistantModel` for the named assistant.

        Examples:
            >>> assistant = await pc.assistant("my-assistant")  # doctest: +SKIP
            >>> assistant.status  # doctest: +SKIP
            'Ready'
        """
        assistants: AsyncAssistants = object.__getattribute__(self, "_assistants")
        return await assistants.describe(name=name)

    def __getattr__(self, attr: str) -> Any:
        """Forward attribute access to the underlying :class:`AsyncAssistants` namespace.

        Lets ``pc.assistant`` act as a stand-in for ``pc.assistants``, so any
        method available on :class:`AsyncAssistants` can also be called
        through the singular alias.

        Args:
            attr: Name of the attribute or method to look up on
                :class:`AsyncAssistants`.

        Returns:
            The attribute from the underlying :class:`AsyncAssistants` instance.

        Examples:
            >>> await pc.assistant.create_assistant(assistant_name="my-assistant")  # doctest: +SKIP
        """
        assistants: AsyncAssistants = object.__getattribute__(self, "_assistants")
        return getattr(assistants, attr)

    def __repr__(self) -> str:
        assistants: AsyncAssistants = object.__getattribute__(self, "_assistants")
        return f"<AsyncAssistantNamespaceProxy for {assistants!r}>"
