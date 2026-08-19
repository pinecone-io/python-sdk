"""Columnar extraction of upsert records from a pandas DataFrame.

``df.iterrows()`` rebuilds a ``Series`` per row — 144ms per 20k rows against
1.7ms for a columnar zip — and coerces dtypes while doing it, since a row
spanning several columns has to find one dtype to hold them all. Reading each
column once with ``.to_numpy()`` avoids both.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]

_OPTIONAL_COLUMNS = ("sparse_values", "metadata")
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


def extract_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return one upsert record dict per row of *df*.

    ``id`` and ``values`` are required. ``sparse_values`` and ``metadata`` are
    included when their column exists and the row's value is present.

    A frame built from row dicts where only some rows carry ``metadata`` leaves
    ``NaN`` in the rest, pandas having no other way to fill the gap. Those cells
    count as absent, the same as ``None`` — otherwise ``NaN`` reaches validation
    and surfaces as ``metadata must be a dict, got float``.

    Raises:
        KeyError: If ``id`` or ``values`` is missing.
    """
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
