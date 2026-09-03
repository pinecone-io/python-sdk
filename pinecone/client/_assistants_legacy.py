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
        """Legacy alias for :meth:`Assistants.evaluate_alignment`.

        Args:
            question: The question for which the answer was generated.
            answer: The generated answer to evaluate.
            ground_truth_answer: The ground truth answer to compare
                against.

        Returns:
            :class:`AlignmentResult` with aggregate scores, per-fact
            entailment results, and token usage statistics.

        Raises:
            :exc:`ApiError`: If the API returns an error response. See
                :meth:`Assistants.evaluate_alignment` for this endpoint's
                error responses.
        """
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
        """Legacy alias for :meth:`Assistants.list`.

        Eagerly materializes every page into a list, matching the legacy
        signature ``list_assistants() -> List[AssistantModel]``. Prefer
        :meth:`Assistants.list`, which returns a lazy paginator and only
        fetches the pages you actually iterate.

        Returns:
            List of :class:`AssistantModel` objects for every assistant in
            the project.
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
        """Legacy alias for :meth:`Assistants.list_page`.

        Accepts ``limit`` (legacy) or ``page_size`` (current) for the page
        size. If both are given, ``limit`` wins silently.

        Args:
            limit: Legacy name for the page size. Takes priority over
                *page_size* when both are given.
            pagination_token: Token from a previous response to fetch the
                next page.
            page_size: Current name for the page size. Ignored if *limit*
                is also given.

        Returns:
            :class:`ListAssistantsResponse` with an ``assistants`` list
            and an optional ``next`` continuation token.

        Examples:
            .. code-block:: python

                page = pc.assistants.list_assistants_paginated(limit=10)
                names = [a.name for a in page.assistants]
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
        """Legacy alias for :meth:`Assistants.describe`.

        Accepts ``assistant_name`` (legacy) or ``name`` (current), but not
        both.

        Args:
            assistant_name: Legacy name of the assistant to describe.
            name: Current name of the assistant to describe.

        Returns:
            :class:`AssistantModel` with name, status, created_at,
            updated_at, metadata, instructions, and host.

        Raises:
            :exc:`PineconeValueError`: If both *assistant_name* and *name*
                are given.
            :exc:`NotFoundError`: If the assistant does not exist.
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
        """Legacy alias for :meth:`Assistants.update`.

        Accepts ``assistant_name`` (legacy) or ``name`` (current) for the
        assistant to update. If both are given, ``assistant_name`` wins
        silently.

        Args:
            assistant_name: Legacy name of the assistant to update.
            instructions: New instructions for the assistant. Pass an
                empty string to clear existing instructions.
            metadata: New metadata dictionary. Fully replaces any
                existing metadata rather than merging. Pass an empty dict
                to clear existing metadata.
            name: Current name of the assistant to update.

        Returns:
            :class:`AssistantModel` describing the updated assistant.

        Raises:
            :exc:`PineconeValueError`: If neither *instructions* nor
                *metadata* is given.
            :exc:`NotFoundError`: If the assistant does not exist.
        """
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
        """Legacy alias for :meth:`Assistants.create`.

        Accepts ``assistant_name`` (legacy) or ``name`` (current) for the
        new assistant's name. If both are given, ``assistant_name`` wins
        silently. All other parameters are forwarded unchanged.

        Args:
            assistant_name: Legacy name for the new assistant.
            instructions: Optional directive for the assistant to apply
                to all responses.
            metadata: Optional metadata dictionary. When omitted, the
                assistant is created without metadata.
            region: Region to deploy the assistant in. Must be ``"us"``
                or ``"eu"``. Defaults to ``"us"``.
            timeout: Seconds to wait for the assistant to become ready.
                Use ``None`` (default) to poll indefinitely, ``-1`` to
                return immediately without polling, or a positive value
                to poll with a deadline.
            name: Current name for the new assistant.

        Returns:
            :class:`AssistantModel` describing the created assistant.

        Raises:
            :exc:`PineconeValueError`: If *region* is not ``"us"`` or
                ``"eu"``.
            :exc:`PineconeTimeoutError`: If the assistant does not become
                ready before *timeout*.

        Examples:
            .. code-block:: python

                assistant = pc.assistants.create_assistant(
                    assistant_name="research-assistant",
                    instructions="You are a helpful research assistant.",
                )
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
        """Legacy alias for :meth:`Assistants.delete`.

        Accepts ``assistant_name`` (legacy) or ``name`` (current) for the
        assistant to delete. If both are given, ``assistant_name`` wins
        silently.

        Args:
            assistant_name: Legacy name of the assistant to delete.
            timeout: Seconds to wait for the assistant to disappear. Use
                ``None`` (default) to poll indefinitely, ``-1`` to return
                immediately without polling, or a positive value to poll
                with a deadline.
            name: Current name of the assistant to delete.

        Returns:
            None

        Raises:
            :exc:`PineconeError`: If the assistant enters a terminal
                failure state while being deleted.
            :exc:`PineconeTimeoutError`: If the assistant still exists
                after *timeout* seconds.
        """
        resolved_name = assistant_name if assistant_name is not None else name
        self.delete(name=resolved_name, timeout=timeout, **kwargs)  # type: ignore[attr-defined]

    @property
    def evaluation(self) -> _AlignmentEvaluationProxy:
        """Legacy nested proxy for alignment evaluation.

        Mirrors the ``pinecone_plugins.assistant`` access pattern
        ``pc.assistants.evaluation.metrics.alignment(question=..., answer=...,
        ground_truth_answer=...)``. That method is a legacy alias for
        :meth:`Assistants.evaluate_alignment`; prefer calling it directly
        in new code.

        Returns:
            A proxy object exposing ``.metrics.alignment()``.
        """
        cached = getattr(self, "_legacy_evaluation", None)
        if cached is None:
            cached = _AlignmentEvaluationProxy(self)
            # Cache on the instance.
            object.__setattr__(self, "_legacy_evaluation", cached)
        return cached
