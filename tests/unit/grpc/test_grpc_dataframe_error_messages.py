"""Every error this method can raise has to be actionable from the text alone.

These messages are increasingly read by a coding agent working on someone's
pipeline, not by a person with the SDK source open. An agent has to patch the
call site from the message and nothing else, which raises the bar: name the
parameter or column exactly as it appears in the signature, say what was
received, say where in the data, and state the remedy as something executable.

Walking every error in one parametrized case is what keeps that bar from
eroding one message at a time.
"""

from __future__ import annotations

import re
from typing import Any, NamedTuple
from unittest.mock import MagicMock, patch

import pytest

from pinecone.errors.exceptions import PineconeTimeoutError
from pinecone.grpc import GrpcIndex

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"

_REMEDY_VERBS = (
    "pass",
    "add",
    "give",
    "build",
    "rename",
    "convert",
    "drop",
    "strip",
    "raise",
    "lower",
    "set",
    "re-encode",
    "retry",
)

#: A stated legal range is a remedy in its own right: "must be between 1 and 64,
#: got 99" tells a caller exactly what to write without naming a verb. The knobs
#: validated by ``pinecone._internal.validation.require_in_range`` raise in that
#: shape, and the message text is the 2026-07 contract — ``docs/migration/
#: v10-migration.md`` prints it verbatim and 29 assertions across the vector-op
#: and gRPC parity suites pin it character for character, so it cannot carry a
#: trailing "Pass a limit within that range." clause just to satisfy a verb
#: scan. Every case with no stated range still has to name a verb.
_STATED_RANGE = re.compile(r"must be between \S+ and \S+")


def _states_a_remedy(message: str) -> bool:
    lowered = message.lower()
    return any(verb in lowered for verb in _REMEDY_VERBS) or bool(_STATED_RANGE.search(lowered))


class Case(NamedTuple):
    name: str
    kwargs: dict[str, Any]
    frame: Any
    names: str
    shows: str


def _index(channel: MagicMock | None = None) -> GrpcIndex:
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = channel if channel is not None else MagicMock()
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        return GrpcIndex(
            host="test-index-abc123.svc.pinecone.io", api_key="test-api-key", timeout=20.0
        )


def _frame(rows: list[dict[str, Any]]) -> Any:
    return pd.DataFrame(rows)


_CASES = [
    Case(
        name="not-a-dataframe",
        kwargs={},
        frame=[{"id": "v1", "values": [0.1]}],
        names="df",
        shows="list",
    ),
    Case(
        name="missing-id-column",
        kwargs={},
        frame=_frame([{"values": [0.1]}]),
        names="id",
        shows="values",
    ),
    Case(
        name="missing-values-column",
        kwargs={},
        frame=_frame([{"id": "v1"}]),
        names="values",
        shows="id",
    ),
    Case(
        name="bad-batch-size",
        kwargs={"batch_size": 0},
        frame=_frame([{"id": "v1", "values": [0.1]}]),
        names="batch_size",
        shows="0",
    ),
    Case(
        name="non-integer-batch-size",
        kwargs={"batch_size": 3.5},
        frame=_frame([{"id": "v1", "values": [0.1]}]),
        names="batch_size",
        shows="3.5",
    ),
    Case(
        name="bad-max-concurrency",
        kwargs={"max_concurrency": 99},
        frame=_frame([{"id": "v1", "values": [0.1]}]),
        names="max_concurrency",
        shows="99",
    ),
    Case(
        name="bad-on-error",
        kwargs={"on_error": "explode"},
        frame=_frame([{"id": "v1", "values": [0.1]}]),
        names="on_error",
        shows="explode",
    ),
    Case(
        name="non-string-id",
        kwargs={},
        frame=_frame([{"id": "v0", "values": [0.1]}, {"id": 7, "values": [0.2]}]),
        names="id",
        shows="7",
    ),
    Case(
        name="non-ascii-id",
        kwargs={},
        frame=_frame([{"id": "café", "values": [0.1]}]),
        names="id",
        shows="café",
    ),
    Case(
        name="empty-values",
        kwargs={},
        frame=_frame([{"id": "v0", "values": [0.1]}, {"id": "v1", "values": []}]),
        names="values",
        shows="v1",
    ),
    Case(
        name="metadata-not-a-dict",
        kwargs={},
        frame=_frame(
            [
                {"id": "v0", "values": [0.1], "metadata": {"a": 1}},
                {"id": "v1", "values": [0.2], "metadata": 3.5},
            ]
        ),
        names="metadata",
        shows="3.5",
    ),
]


class TestEveryErrorIsActionable:
    @pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
    def test_message_names_shows_and_remedies(self, case: Case) -> None:
        index = _index()

        with pytest.raises(Exception) as excinfo:
            index.upsert_from_dataframe(case.frame, show_progress=False, **case.kwargs)

        message = str(excinfo.value)
        assert case.names in message, f"message does not name the knob: {message}"
        assert case.shows in message, f"message does not show the input: {message}"
        assert _states_a_remedy(message), f"message states no remedy: {message}"

    @pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
    def test_no_bare_keyerror_escapes(self, case: Case) -> None:
        index = _index()

        with pytest.raises(Exception) as excinfo:
            index.upsert_from_dataframe(case.frame, show_progress=False, **case.kwargs)

        assert not isinstance(excinfo.value, KeyError), (
            "a bare KeyError names neither the method nor the schema it wanted"
        )


class TestPerRowErrorsAreLocated:
    @pytest.mark.parametrize(
        ("frame", "row", "vector_id"),
        [
            (_frame([{"id": "v0", "values": [0.1]}, {"id": 7, "values": [0.2]}]), 1, None),
            (
                _frame(
                    [
                        {"id": "v0", "values": [0.1]},
                        {"id": "v1", "values": [0.2]},
                        {"id": "v2", "values": []},
                    ]
                ),
                2,
                "v2",
            ),
        ],
        ids=["bad-id-type", "empty-values"],
    )
    def test_row_index_and_id_are_both_present(
        self, frame: Any, row: int, vector_id: str | None
    ) -> None:
        """An agent filtering a frame has one or the other, not reliably both."""
        index = _index()

        with pytest.raises(Exception) as excinfo:
            index.upsert_from_dataframe(frame, show_progress=False)

        message = str(excinfo.value)
        assert f"row {row}" in message
        if vector_id is not None:
            assert vector_id in message


class TestSparseCellsAreLocatedToo:
    """sparse_values is a supported optional column; its failures need a row too."""

    @pytest.mark.parametrize(
        ("sparse", "fragment"),
        [
            ({"indices": [0]}, "missing required keys"),
            ({"indices": [0, 1], "values": [0.5]}, "same length"),
            ("not-a-dict", "must be a dict"),
        ],
        ids=["missing-key", "length-mismatch", "wrong-type"],
    )
    def test_row_and_id_are_appended_to_sparse_errors(self, sparse: Any, fragment: str) -> None:
        index = _index()
        frame = _frame(
            [
                {"id": "sp-0", "values": [0.1], "sparse_values": {"indices": [0], "values": [0.5]}},
                {"id": "sp-1", "values": [0.2], "sparse_values": sparse},
            ]
        )

        with pytest.raises(Exception) as excinfo:
            index.upsert_from_dataframe(frame, show_progress=False)

        message = str(excinfo.value)
        assert fragment in message
        assert "row 1" in message
        assert "sp-1" in message


class TestTimeoutSaysWhichLayerFired:
    @staticmethod
    def _timing_out_index() -> GrpcIndex:
        channel = MagicMock()
        channel.upsert.side_effect = PineconeTimeoutError("deadline exceeded")
        return _index(channel)

    def test_names_the_layer_the_value_and_the_knobs(self) -> None:
        index = self._timing_out_index()

        with pytest.raises(PineconeTimeoutError) as excinfo:
            index.upsert_from_dataframe(
                _frame([{"id": "v1", "values": [0.1]}]),
                show_progress=False,
                timeout=42.0,
                on_error="raise",
            )

        message = str(excinfo.value)
        assert "per-attempt" in message
        assert "42.0" in message
        assert "total_timeout" in message

    def test_names_both_deadlines_when_a_per_call_one_is_passed(self) -> None:
        """The channel keeps its Endpoint deadline, so the shorter of the two fired."""
        index = self._timing_out_index()

        with pytest.raises(PineconeTimeoutError) as excinfo:
            index.upsert_from_dataframe(
                _frame([{"id": "v1", "values": [0.1]}]),
                show_progress=False,
                timeout=42.0,
                on_error="raise",
            )

        message = str(excinfo.value)
        assert "timeout=42.0" in message
        assert "index-level timeout of 20.0s" in message
        assert "The per-attempt deadline fired: 20.0s" in message

    def test_rules_max_retries_out_rather_than_pointing_at_it(self) -> None:
        """A timeout is not in the retryable set, so one batch is one attempt."""
        index = self._timing_out_index()

        with pytest.raises(PineconeTimeoutError) as excinfo:
            index.upsert_from_dataframe(
                _frame([{"id": "v1", "values": [0.1]}]),
                show_progress=False,
                timeout=42.0,
                on_error="raise",
            )

        message = str(excinfo.value)
        assert "not retried" in message
        assert "single attempt" in message
        assert "max_retries is not the knob" in message

    def test_says_whether_retrying_is_safe(self) -> None:
        index = self._timing_out_index()

        with pytest.raises(PineconeTimeoutError) as excinfo:
            index.upsert_from_dataframe(
                _frame([{"id": "v1", "values": [0.1]}]),
                show_progress=False,
                on_error="raise",
            )

        assert "idempotent" in str(excinfo.value)

    def test_falls_back_to_the_index_level_value(self) -> None:
        index = self._timing_out_index()

        with pytest.raises(PineconeTimeoutError) as excinfo:
            index.upsert_from_dataframe(
                _frame([{"id": "v1", "values": [0.1]}]),
                show_progress=False,
                on_error="raise",
            )

        assert "index-level timeout of 20.0s" in str(excinfo.value)

    def test_the_partial_result_still_rides_along(self) -> None:
        index = self._timing_out_index()

        with pytest.raises(PineconeTimeoutError) as excinfo:
            index.upsert_from_dataframe(
                _frame([{"id": "v1", "values": [0.1]}]),
                show_progress=False,
                on_error="raise",
            )

        assert excinfo.value.response is not None


class TestMessagesAreStable:
    def test_leading_clause_carries_nothing_run_specific(self) -> None:
        """These strings become de facto API once agents match on them."""
        index = _index()

        with pytest.raises(Exception) as excinfo:
            index.upsert_from_dataframe(_frame([{"values": [0.1]}]), show_progress=False)

        leading = str(excinfo.value).split(".")[0]
        assert not re.search(r"0x[0-9a-f]+", leading), "address in the leading clause"
        assert "object at" not in leading

    def test_appended_timeout_guidance_starts_its_own_sentence(self) -> None:
        """Otherwise leading-clause matching swallows the run-specific values."""
        channel = MagicMock()
        channel.upsert.side_effect = PineconeTimeoutError("deadline exceeded")
        index = _index(channel)

        with pytest.raises(PineconeTimeoutError) as excinfo:
            index.upsert_from_dataframe(
                _frame([{"id": "v1", "values": [0.1]}]),
                show_progress=False,
                timeout=42.0,
                on_error="raise",
            )

        leading = str(excinfo.value).split(".")[0]
        assert leading == "deadline exceeded"

    def test_a_punctuated_message_is_not_double_punctuated(self) -> None:
        channel = MagicMock()
        channel.upsert.side_effect = PineconeTimeoutError("deadline exceeded.")
        index = _index(channel)

        with pytest.raises(PineconeTimeoutError) as excinfo:
            index.upsert_from_dataframe(
                _frame([{"id": "v1", "values": [0.1]}]),
                show_progress=False,
                on_error="raise",
            )

        assert ".." not in str(excinfo.value)


class TestTheRenameExampleFixesTheReportedFailure:
    """An agent following the example has to end up with the missing column."""

    @pytest.mark.parametrize(
        ("frame", "missing", "irrelevant"),
        [
            (_frame([{"values": [0.1]}]), "id", "'embedding': 'values'"),
            (_frame([{"id": "v1"}]), "values", "'doc_id': 'id'"),
        ],
        ids=["missing-id", "missing-values"],
    )
    def test_example_renames_the_column_that_is_missing(
        self, frame: Any, missing: str, irrelevant: str
    ) -> None:
        index = _index()

        with pytest.raises(Exception) as excinfo:
            index.upsert_from_dataframe(frame, show_progress=False)

        message = str(excinfo.value)
        example = message.split("df.rename(columns=", 1)[1]
        assert f"'{missing}'" in example
        assert irrelevant not in example
