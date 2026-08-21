"""Backwards-compatibility method shims for the async Assistants namespace.

Each method here mirrors a legacy method name from the removed
pinecone_plugins.assistant package. They delegate to the canonical
new-SDK method and remap legacy parameter names.

Keep this mixin narrowly scoped — it should contain only aliases, never
new behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pinecone.errors.exceptions import PineconeValueError

if TYPE_CHECKING:
    from pinecone.models.assistant.evaluation import AlignmentResult
    from pinecone.models.assistant.list import ListAssistantsResponse
    from pinecone.models.assistant.model import AssistantModel


class _AsyncAlignmentMetricsProxy:
    """Legacy nested proxy: ``assistants.evaluation.metrics`` (async)."""

    def __init__(self, assistants: AsyncAssistantsLegacyNamespaceMixin) -> None:
        self._assistants = assistants

    async def alignment(
        self,
        question: str,
        answer: str,
        ground_truth_answer: str,
        **kwargs: Any,
    ) -> AlignmentResult:
        """Deprecated alias for :meth:`AsyncAssistants.evaluate_alignment`."""
        return cast(
            "AlignmentResult",
            await self._assistants.evaluate_alignment(  # type: ignore[attr-defined]
                question=question,
                answer=answer,
                ground_truth_answer=ground_truth_answer,
                **kwargs,
            ),
        )


class _AsyncAlignmentEvaluationProxy:
    """Legacy nested proxy: ``assistants.evaluation`` (async)."""

    def __init__(self, assistants: AsyncAssistantsLegacyNamespaceMixin) -> None:
        self._assistants = assistants
        self.metrics = _AsyncAlignmentMetricsProxy(assistants)


class AsyncAssistantsLegacyNamespaceMixin:
    """Legacy-name method shims for the :class:`AsyncAssistants` namespace.

    Mixed into :class:`AsyncAssistants` so that callers upgrading from
    ``pinecone_plugins.assistant`` can keep using names like
    ``create_assistant`` and parameter names like ``assistant_name``.
    """

    async def list_assistants(self) -> list[AssistantModel]:
        """Deprecated alias for iterating :meth:`AsyncAssistants.list`.

        Returns a materialized list (auto-paginated) for compatibility with
        legacy callers that expected ``list_assistants() -> List[AssistantModel]``.
        Prefer :meth:`AsyncAssistants.list` which returns a lazy async paginator.
        """
        return [assistant async for assistant in self.list()]  # type: ignore[attr-defined]

    async def list_assistants_paginated(
        self,
        limit: int | None = None,
        pagination_token: str | None = None,
        *,
        page_size: int | None = None,
        **kwargs: Any,
    ) -> ListAssistantsResponse:
        """Deprecated alias for :meth:`AsyncAssistants.list_page`.

        Returns a :class:`ListAssistantsResponse`-shaped object built from
        the new SDK's ``list_page`` result. Accepts ``limit`` (legacy) or
        ``page_size`` (current).
        """

        resolved = limit if limit is not None else page_size
        return cast(
            "ListAssistantsResponse",
            await self.list_page(  # type: ignore[attr-defined]
                page_size=resolved,
                pagination_token=pagination_token,
                **kwargs,
            ),
        )

    async def describe_assistant(
        self,
        assistant_name: str | None = None,
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> AssistantModel:
        """Deprecated alias for :meth:`AsyncAssistants.describe`.

        Accepts ``assistant_name`` (legacy) or ``name`` (current), but not both.
        """
        if assistant_name is not None and name is not None:
            raise PineconeValueError(
                "describe_assistant() received both 'assistant_name' (legacy) and 'name'. "
                "Pass only one — prefer 'name'."
            )
        resolved_name = assistant_name if assistant_name is not None else name
        return cast(
            "AssistantModel",
            await self.describe(  # type: ignore[attr-defined]
                name=resolved_name,
                **kwargs,
            ),
        )

    async def update_assistant(
        self,
        assistant_name: str | None = None,
        instructions: str | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> AssistantModel:
        """Deprecated alias for :meth:`AsyncAssistants.update`."""
        resolved_name = assistant_name if assistant_name is not None else name
        return cast(
            "AssistantModel",
            await self.update(  # type: ignore[attr-defined]
                name=resolved_name,
                instructions=instructions,
                metadata=metadata,
                **kwargs,
            ),
        )

    async def create_assistant(
        self,
        assistant_name: str | None = None,
        instructions: str | None = None,
        metadata: dict[str, Any] | None = None,
        region: str = "us",
        timeout: int | None = None,
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> AssistantModel:
        """Deprecated alias for :meth:`AsyncAssistants.create`.

        Accepts either ``assistant_name`` (legacy) or ``name`` (current),
        but not both. All other parameters are forwarded unchanged.
        """
        resolved_name = assistant_name if assistant_name is not None else name
        return cast(
            "AssistantModel",
            await self.create(  # type: ignore[attr-defined]
                name=resolved_name,
                instructions=instructions,
                metadata=metadata,
                region=region,
                timeout=timeout,
                **kwargs,
            ),
        )

    async def delete_assistant(
        self,
        assistant_name: str | None = None,
        timeout: int | None = None,
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Deprecated alias for :meth:`AsyncAssistants.delete`."""
        resolved_name = assistant_name if assistant_name is not None else name
        await self.delete(name=resolved_name, timeout=timeout, **kwargs)  # type: ignore[attr-defined]

    @property
    def evaluation(self) -> _AsyncAlignmentEvaluationProxy:
        """Deprecated nested proxy for alignment evaluation.

        Equivalent to ``await pc.assistants.evaluate_alignment(...)``. Prefer
        the flat method in new code.
        """
        cached = getattr(self, "_legacy_evaluation", None)
        if cached is None:
            cached = _AsyncAlignmentEvaluationProxy(self)
            # Cache on the instance.
            object.__setattr__(self, "_legacy_evaluation", cached)
        return cached
