"""Columnar extraction of upsert records from a pandas DataFrame.

``df.iterrows()`` rebuilds a ``Series`` per row — 144ms per 20k rows against
1.7ms for a columnar zip — and coerces dtypes while doing it, since a row
spanning several columns has to find one dtype to hold them all. Reading each
column once with ``.to_numpy()`` avoids both.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pinecone.errors.exceptions import PineconeValueError

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]

_REQUIRED_COLUMNS = ("id", "values")
# Only to make the rename example concrete; nothing detects these in the frame.
_COMMON_ALIASES = {"id": "doc_id", "values": "embedding"}
_OPTIONAL_COLUMNS = ("sparse_values", "metadata")
_ON_ERROR_VALUES = ("raise", "collect")
_NAN_LIKE_DTYPE_KINDS = frozenset("fMm")


def _na_singletons() -> tuple[Any, ...]:
    """Missing-value markers that can only be detected by identity.

    ``pd.NA != pd.NA`` evaluates to ``pd.NA``, and calling ``bool()`` on that
    raises, so these cannot be compared.
    """
    import pandas as pd

    return (None, pd.NA, pd.NaT)


def _is_missing(value: Any, na: tuple[Any, ...]) -> bool:
    """Whether *value* is a missing-value marker rather than real data.

    Deliberately avoids ``pd.isna``: these columns hold dicts and arrays, which
    it either rejects or answers elementwise.
    """
    for sentinel in na:
        if value is sentinel:
            return True
    if isinstance(value, float):
        return value != value
    # float32, datetime64 and timedelta64 all carry a NaN-like value unequal to
    # itself, and none of them subclass float. ndim keeps arrays out of the
    # comparison, which would answer elementwise.
    if (
        getattr(value, "ndim", None) == 0
        and getattr(getattr(value, "dtype", None), "kind", "") in _NAN_LIKE_DTYPE_KINDS
    ):
        return bool(value != value)
    return False


def _require_columns(df: pd.DataFrame) -> None:
    """Fail on a missing column here, where the frame is still in hand.

    Reaching the column access with one absent raises a bare ``KeyError: 'id'``,
    which names neither the method nor the schema it wanted.
    """
    found = list(df.columns)
    missing = [name for name in _REQUIRED_COLUMNS if name not in found]
    if not missing:
        return
    # The example has to rename the column that is actually missing. A fixed
    # `'vector': 'values'` is a fix for a different failure than the one reported.
    example = ", ".join(f"{_COMMON_ALIASES[name]!r}: {name!r}" for name in missing)
    raise PineconeValueError(
        f"DataFrame is missing required column(s): {missing}. "
        f"upsert_from_dataframe requires {list(_REQUIRED_COLUMNS)} "
        f"and optionally {list(_OPTIONAL_COLUMNS)}; the frame has {found}. "
        f"Rename or add the missing column(s), e.g. df = df.rename(columns={{{example}}})."
    )


def extract_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return one upsert record dict per row of *df*.

    ``id`` and ``values`` are required. ``sparse_values`` and ``metadata`` are
    included when their column exists and the row's value is present.

    A frame built from row dicts where only some rows carry ``metadata`` leaves
    ``NaN`` in the rest, pandas having no other way to fill the gap. Those cells
    count as absent, the same as ``None`` — otherwise ``NaN`` reaches validation
    and surfaces as ``metadata must be a dict, got float``.

    Raises:
        PineconeValueError: If ``id`` or ``values`` is missing.
    """
    _require_columns(df)

    ids = df["id"].to_numpy()
    values = df["values"].to_numpy()

    records: list[dict[str, Any]] = [
        {"id": id_, "values": vals} for id_, vals in zip(ids, values, strict=True)
    ]

    present = [name for name in _OPTIONAL_COLUMNS if name in df.columns]
    if not present:
        return records

    na = _na_singletons()
    for name in present:
        for record, value in zip(records, df[name].to_numpy(), strict=True):
            if not _is_missing(value, na):
                record[name] = value

    return records


def _resolve_on_error(on_error: str | None) -> str:
    """Validate an ``on_error`` argument, treating absence as ``"collect"``."""
    if on_error is None:
        return "collect"
    if on_error not in _ON_ERROR_VALUES:
        raise PineconeValueError(
            f"on_error must be one of {list(_ON_ERROR_VALUES)}, got {on_error!r}"
        )
    return on_error
