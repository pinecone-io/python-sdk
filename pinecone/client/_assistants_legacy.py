"""Backwards-compatibility method shims for the Assistants namespace.

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


class _AlignmentMetricsProxy:
    """Legacy nested proxy: ``assistants.evaluation.metrics``."""

    def __init__(self, assistants: AssistantsLegacyNamespaceMixin) -> None:
        self._assistants = assistants

    def alignment(
        self,
        question: str,
        answer: str,
        ground_truth_answer: str,
        **kwargs: Any,
    ) -> AlignmentResult:
        """Deprecated alias for :meth:`Assistants.evaluate_alignment`."""
        return cast(
            "AlignmentResult",
            self._assistants.evaluate_alignment(  # type: ignore[attr-defined]
                question=question,
                answer=answer,
                ground_truth_answer=ground_truth_answer,
                **kwargs,
            ),
        )


class _AlignmentEvaluationProxy:
    """Legacy nested proxy: ``assistants.evaluation``."""

    def __init__(self, assistants: AssistantsLegacyNamespaceMixin) -> None:
        self._assistants = assistants
        self.metrics = _AlignmentMetricsProxy(assistants)


class AssistantsLegacyNamespaceMixin:
    """Legacy-name method shims for the :class:`Assistants` namespace.

    Mixed into :class:`Assistants` so that callers upgrading from
    ``pinecone_plugins.assistant`` can keep using names like
    ``create_assistant`` and parameter names like ``assistant_name``.
    """

    def list_assistants(self) -> list[AssistantModel]:
        """Deprecated alias for iterating :meth:`Assistants.list`.

        Returns a materialized list (auto-paginated) for compatibility with
        legacy callers that expected ``list_assistants() -> List[AssistantModel]``.
        Prefer :meth:`Assistants.list` which returns a lazy paginator.
        """
        return list(self.list())  # type: ignore[attr-defined]

    def list_assistants_paginated(
        self,
        limit: int | None = None,
        pagination_token: str | None = None,
        *,
        page_size: int | None = None,
        **kwargs: Any,
    ) -> ListAssistantsResponse:
        """Deprecated alias for :meth:`Assistants.list_page`.

        Returns a :class:`ListAssistantsResponse`-shaped object built from
        the new SDK's ``list_page`` result. Accepts ``limit`` (legacy) or
        ``page_size`` (current).
        """

        resolved = limit if limit is not None else page_size
        return cast(
            "ListAssistantsResponse",
            self.list_page(  # type: ignore[attr-defined]
                page_size=resolved,
                pagination_token=pagination_token,
                **kwargs,
            ),
        )

    def describe_assistant(
        self,
        assistant_name: str | None = None,
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> AssistantModel:
        """Deprecated alias for :meth:`Assistants.describe`.

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
            self.describe(  # type: ignore[attr-defined]
                name=resolved_name,
                **kwargs,
            ),
        )

    def update_assistant(
        self,
        assistant_name: str | None = None,
        instructions: str | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> AssistantModel:
        """Deprecated alias for :meth:`Assistants.update`."""
        resolved_name = assistant_name if assistant_name is not None else name
        return cast(
            "AssistantModel",
            self.update(  # type: ignore[attr-defined]
                name=resolved_name,
                instructions=instructions,
                metadata=metadata,
                **kwargs,
            ),
        )

    def create_assistant(
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
        """Deprecated alias for :meth:`Assistants.create`.

        Accepts either ``assistant_name`` (legacy) or ``name`` (current),
        but not both. All other parameters are forwarded unchanged.
        """
        resolved_name = assistant_name if assistant_name is not None else name
        return cast(
            "AssistantModel",
            self.create(  # type: ignore[attr-defined]
                name=resolved_name,
                instructions=instructions,
                metadata=metadata,
                region=region,
                timeout=timeout,
                **kwargs,
            ),
        )

    def delete_assistant(
        self,
        assistant_name: str | None = None,
        timeout: int | None = None,
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Deprecated alias for :meth:`Assistants.delete`."""
        resolved_name = assistant_name if assistant_name is not None else name
        self.delete(name=resolved_name, timeout=timeout, **kwargs)  # type: ignore[attr-defined]

    @property
    def evaluation(self) -> _AlignmentEvaluationProxy:
        """Deprecated nested proxy for alignment evaluation.

        Equivalent to ``pc.assistants.evaluate_alignment(...)``. Prefer the
        flat method in new code.
        """
        cached = getattr(self, "_legacy_evaluation", None)
        if cached is None:
            cached = _AlignmentEvaluationProxy(self)
            # Cache on the instance.
            object.__setattr__(self, "_legacy_evaluation", cached)
        return cached
