"""Evaluation response models for the Assistant API."""

from __future__ import annotations

from typing import Any, Literal

from msgspec import Struct

from pinecone.models._display import HtmlBuilder, safe_display, truncate_text
from pinecone.models.assistant._mixin import StructDictMixin
from pinecone.models.assistant.chat import ChatUsage

EntailmentType = Literal["entailed", "contradicted", "neutral"] | str


class EntailmentResult(StructDictMixin, Struct, kw_only=True):
    """One evaluated fact, and how the answer stood against it.

    Reached as an entry of ``result.facts``. Filtering for
    ``"contradicted"`` gives you the specific places the answer and the ground
    truth disagree, which the aggregate scores cannot tell you.

    Attributes:
        fact: The fact under evaluation, as a sentence.
        entailment: How the answer stood against the fact — ``"entailed"``,
            ``"contradicted"``, or ``"neutral"``. Typed as :class:`str` rather
            than a closed set, so an unrecognized value decodes instead of
            raising.
        reasoning: Why the judgment was made. ``""`` when the API returned
            none, so test for truthiness rather than for ``None``.
    """

    fact: str
    entailment: EntailmentType
    reasoning: str = ""

    @safe_display
    def __repr__(self) -> str:
        return (
            f"EntailmentResult(entailment={self.entailment!r},"
            f" fact={truncate_text(self.fact, max_chars=80)!r})"
        )

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("EntailmentResult(...)")
            return
        with p.group(2, "EntailmentResult(", ")"):
            p.breakable()
            p.text(f"entailment={self.entailment!r},")
            p.breakable()
            p.text(f"fact={truncate_text(self.fact, max_chars=200)!r},")
            if self.reasoning:
                p.breakable()
                p.text(f"reasoning={truncate_text(self.reasoning, max_chars=200)!r},")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("EntailmentResult")
        builder.row("Entailment", self.entailment)
        builder.row("Fact", truncate_text(self.fact, max_chars=500))
        if self.reasoning:
            builder.row("Reasoning", truncate_text(self.reasoning, max_chars=500))
        if self.entailment == "contradicted":
            rows: list[tuple[str, str]] = [
                ("Fact", truncate_text(self.fact, max_chars=500)),
            ]
            if self.reasoning:
                rows.append(("Reasoning", truncate_text(self.reasoning, max_chars=500)))
            builder.section("Contradiction", rows, theme="error")
        return builder.build()


class AlignmentScores(StructDictMixin, Struct, kw_only=True):
    """The three aggregate scores of an alignment evaluation.

    Reached as ``result.scores``. Because ``alignment`` is a harmonic mean, a
    low score on either input drags it down, so read all three rather than
    tracking ``alignment`` alone.

    Attributes:
        correctness: Precision of the generated answer — how much of what it
            said holds up.
        completeness: Recall of the generated answer — how much of the ground
            truth it covered.
        alignment: Harmonic mean of ``correctness`` and ``completeness``.
    """

    correctness: float
    completeness: float
    alignment: float

    @safe_display
    def __repr__(self) -> str:
        return (
            f"AlignmentScores(correctness={self.correctness:.3f},"
            f" completeness={self.completeness:.3f},"
            f" alignment={self.alignment:.3f})"
        )

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("AlignmentScores(...)")
            return
        with p.group(2, "AlignmentScores(", ")"):
            p.breakable()
            p.text(f"correctness={self.correctness:.3f},")
            p.breakable()
            p.text(f"completeness={self.completeness:.3f},")
            p.breakable()
            p.text(f"alignment={self.alignment:.3f},")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("AlignmentScores")
        builder.row("Correctness", f"{self.correctness:.3f}")
        builder.row("Completeness", f"{self.completeness:.3f}")
        builder.row("Alignment", f"{self.alignment:.3f}")
        return builder.build()


class AlignmentResult(StructDictMixin, Struct, kw_only=True):
    """How well a generated answer matched a ground-truth answer.

    Returned by
    :meth:`~pinecone.client.assistants.Assistants.evaluate_alignment`. Read
    ``result.scores`` for the aggregate numbers and ``result.facts`` for the
    per-fact judgments that explain them — the scores tell you an answer is
    wrong, and the facts tell you where.

    Attributes:
        scores: The :class:`AlignmentScores` for the answer as a whole —
            ``scores.correctness`` (precision of what the answer said),
            ``scores.completeness`` (recall against the ground truth), and
            ``scores.alignment``, their harmonic mean. Read all three: a low
            score on either input drags the mean down.
        facts: An :class:`EntailmentResult` per fact, each with a judgment and
            the reasoning behind it.
        usage: :class:`~pinecone.models.assistant.chat.ChatUsage` token counts
            for the evaluation request itself, not for the answer being
            evaluated.

    Examples:
        The answer below contradicts the ground truth, so the scores come back
        low and ``facts`` records exactly where the disagreement is:

        >>> result = pc.assistants.evaluate_alignment(
        ...     question="What is the capital of Spain?",
        ...     answer="Barcelona.",
        ...     ground_truth_answer="Madrid.",
        ... )
        >>> result.scores
        AlignmentScores(correctness=0.000, completeness=0.000, alignment=0.000)
        >>> result.facts[0].entailment
        'contradicted'
        >>> result.facts[0].reasoning
        'The answer names Barcelona instead of Madrid.'
        >>> [f.fact for f in result.facts if f.entailment == "contradicted"]
        ['The capital of Spain is Madrid.']
        >>> result.usage.total_tokens
        38
    """

    scores: AlignmentScores
    facts: list[EntailmentResult]
    usage: ChatUsage

    @safe_display
    def __repr__(self) -> str:
        return (
            f"AlignmentResult(alignment={self.scores.alignment:.3f},"
            f" correctness={self.scores.correctness:.3f},"
            f" completeness={self.scores.completeness:.3f},"
            f" facts={len(self.facts)}, usage={self.usage!r})"
        )

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("AlignmentResult(...)")
            return
        with p.group(2, "AlignmentResult(", ")"):
            p.breakable()
            p.text(f"alignment={self.scores.alignment:.3f},")
            p.breakable()
            p.text(f"correctness={self.scores.correctness:.3f},")
            p.breakable()
            p.text(f"completeness={self.scores.completeness:.3f},")
            p.breakable()
            p.text(f"facts={len(self.facts)},")
            p.breakable()
            p.text(f"usage={self.usage!r},")
            for fact in self.facts[:3]:
                p.breakable()
                p.text(repr(fact))

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("AlignmentResult")
        builder.row("Correctness", f"{self.scores.correctness:.3f}")
        builder.row("Completeness", f"{self.scores.completeness:.3f}")
        builder.row("Alignment", f"{self.scores.alignment:.3f}")
        builder.row("Facts", len(self.facts))
        builder.row("Usage", repr(self.usage))
        if self.facts:
            fact_rows: list[tuple[str, str]] = [
                (f.entailment, truncate_text(f.fact, 80)) for f in self.facts[:5]
            ]
            builder.section("Facts", fact_rows)
        contradictions = [f for f in self.facts if f.entailment == "contradicted"]
        if contradictions:
            contradiction_rows: list[tuple[str, str]] = [
                (truncate_text(f.fact, 80), f.reasoning or "") for f in contradictions[:5]
            ]
            builder.section("Contradictions", contradiction_rows, theme="error")
        return builder.build()
