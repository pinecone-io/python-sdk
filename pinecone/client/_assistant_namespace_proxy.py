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
        """Look up an assistant by name; equivalent to ``Assistants.describe``.

        :meta private:
        """
        assistants: Assistants = object.__getattribute__(self, "_assistants")
        return assistants.describe(name=name)

    def __getattr__(self, attr: str) -> Any:
        """Forward attribute access to the underlying ``Assistants`` namespace.

        :meta private:
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
        """Look up an assistant by name; equivalent to ``AsyncAssistants.describe``.

        :meta private:
        """
        assistants: AsyncAssistants = object.__getattribute__(self, "_assistants")
        return await assistants.describe(name=name)

    def __getattr__(self, attr: str) -> Any:
        """Forward attribute access to the underlying ``AsyncAssistants`` namespace.

        :meta private:
        """
        assistants: AsyncAssistants = object.__getattribute__(self, "_assistants")
        return getattr(assistants, attr)

    def __repr__(self) -> str:
        assistants: AsyncAssistants = object.__getattribute__(self, "_assistants")
        return f"<AsyncAssistantNamespaceProxy for {assistants!r}>"
