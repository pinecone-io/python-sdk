"""Property-based tests for VectorFactory input parsing.

VectorFactory.build accepts three input forms — a Vector (passthrough), a
2/3-element tuple, and a dict — plus any iterable for the dense values. These
tests assert properties across *all* forms: *equivalence* (the same vector
spelled different ways builds into equal Vectors), *passthrough identity*, and
*validation boundary* (bad ids, wrong top-level types, wrong tuple lengths, and
non-dict metadata are rejected regardless of form). Note that the Vector
passthrough does not re-validate the id, so id-rejection is asserted only for
the tuple and dict forms. Complements the example-based cases in
test_vector_factory.py.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from pinecone._internal.vector_factory import VectorFactory
from pinecone.errors.exceptions import PineconeTypeError, PineconeValueError
from pinecone.models.vectors.sparse import SparseValues
from pinecone.models.vectors.vector import Vector

_RECOGNIZED_KEYS = frozenset({"id", "values", "sparse_values", "metadata"})

_ascii_ids = st.text(alphabet=st.characters(min_codepoint=1, max_codepoint=127), max_size=64)
_dense_values = st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=32)
_scalar = st.one_of(
    st.text(max_size=16),
    st.integers(),
    st.booleans(),
    st.floats(allow_nan=False, allow_infinity=False),
)
_metadata = st.dictionaries(st.text(min_size=1, max_size=16), _scalar, max_size=5)


@st.composite
def sparse_values(draw: st.DrawFn) -> SparseValues:
    length = draw(st.integers(min_value=0, max_value=16))
    indices = draw(
        st.lists(st.integers(min_value=0, max_value=10**6), min_size=length, max_size=length)
    )
    values = draw(
        st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=length, max_size=length)
    )
    return SparseValues(indices=indices, values=values)


@st.composite
def valid_vectors(draw: st.DrawFn) -> Vector:
    id = draw(_ascii_ids)
    values = draw(st.lists(st.floats(allow_nan=False, allow_infinity=False), max_size=32))
    sparse = draw(st.one_of(st.none(), sparse_values()))
    metadata = draw(st.one_of(st.none(), _metadata))
    assume(values or sparse is not None)
    return Vector(id=id, values=values, sparse_values=sparse, metadata=metadata)


@given(v=valid_vectors())
def test_vector_input_passes_through_by_identity(v: Vector) -> None:
    assert VectorFactory.build(v) is v


@given(id=_ascii_ids, values=_dense_values)
def test_tuple_dict_vector_dense_forms_are_equivalent(id: str, values: list[float]) -> None:
    from_tuple = VectorFactory.build((id, values))
    from_dict = VectorFactory.build({"id": id, "values": values})
    from_vector = VectorFactory.build(Vector(id=id, values=values))
    assert from_tuple == from_dict == from_vector


@given(id=_ascii_ids, values=_dense_values, metadata=st.one_of(st.none(), _metadata))
def test_tuple_dict_vector_metadata_forms_are_equivalent(
    id: str, values: list[float], metadata: dict[str, object] | None
) -> None:
    from_tuple = VectorFactory.build((id, values, metadata))
    from_dict = VectorFactory.build({"id": id, "values": values, "metadata": metadata})
    from_vector = VectorFactory.build(Vector(id=id, values=values, metadata=metadata))
    assert from_tuple == from_dict == from_vector


@given(id=_ascii_ids, values=_dense_values, sparse=sparse_values())
def test_dict_and_vector_sparse_forms_are_equivalent(
    id: str, values: list[float], sparse: SparseValues
) -> None:
    from_dict = VectorFactory.build(
        {
            "id": id,
            "values": values,
            "sparse_values": {"indices": sparse.indices, "values": sparse.values},
        }
    )
    from_vector = VectorFactory.build(Vector(id=id, values=values, sparse_values=sparse))
    assert from_dict == from_vector


@given(id=_ascii_ids, values=_dense_values)
def test_dense_values_iterable_type_does_not_matter(id: str, values: list[float]) -> None:
    as_list = VectorFactory.build((id, values))
    as_tuple = VectorFactory.build((id, tuple(values)))
    as_dict_tuple = VectorFactory.build({"id": id, "values": tuple(values)})
    assert as_list == as_tuple == as_dict_tuple


@given(
    id=st.text(alphabet=st.characters(min_codepoint=128), min_size=1, max_size=32),
    values=_dense_values,
)
def test_non_ascii_id_rejected_in_tuple_and_dict_forms(id: str, values: list[float]) -> None:
    forms: list[Callable[[], Vector]] = [
        lambda: VectorFactory.build((id, values)),
        lambda: VectorFactory.build((id, values, None)),
        lambda: VectorFactory.build({"id": id, "values": values}),
    ]
    for build_form in forms:
        with pytest.raises(PineconeValueError):
            build_form()


@given(prefix=_ascii_ids, suffix=_ascii_ids, values=_dense_values)
def test_null_char_id_rejected_in_tuple_and_dict_forms(
    prefix: str, suffix: str, values: list[float]
) -> None:
    id = f"{prefix}\x00{suffix}"
    forms: list[Callable[[], Vector]] = [
        lambda: VectorFactory.build((id, values)),
        lambda: VectorFactory.build((id, values, None)),
        lambda: VectorFactory.build({"id": id, "values": values}),
    ]
    for build_form in forms:
        with pytest.raises(PineconeValueError):
            build_form()


@given(
    item=st.one_of(
        st.none(),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(max_size=8),
        st.lists(st.integers(), max_size=4),
        st.booleans(),
    )
)
def test_wrong_top_level_type_raises_type_error(item: object) -> None:
    with pytest.raises(PineconeTypeError):
        VectorFactory.build(item)


@given(elements=st.lists(st.floats(allow_nan=False, allow_infinity=False), max_size=8))
def test_tuple_of_wrong_length_raises_value_error(elements: list[float]) -> None:
    assume(len(elements) not in (2, 3))
    with pytest.raises(PineconeValueError):
        VectorFactory.build(tuple(elements))


@given(
    id=_ascii_ids,
    values=_dense_values,
    bad_metadata=st.one_of(
        st.text(max_size=8),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.lists(st.integers(), max_size=4),
    ),
)
def test_non_dict_metadata_rejected_in_tuple_and_dict_forms(
    id: str, values: list[float], bad_metadata: object
) -> None:
    forms: list[Callable[[], Vector]] = [
        lambda: VectorFactory.build((id, values, bad_metadata)),
        lambda: VectorFactory.build({"id": id, "values": values, "metadata": bad_metadata}),
    ]
    for build_form in forms:
        with pytest.raises(PineconeTypeError):
            build_form()


@given(
    id=_ascii_ids,
    values=_dense_values,
    extra_key=st.text(min_size=1, max_size=16).filter(lambda k: k not in _RECOGNIZED_KEYS),
    extra_value=_scalar,
)
def test_unrecognized_dict_key_is_always_rejected(
    id: str, values: list[float], extra_key: str, extra_value: object
) -> None:
    with pytest.raises(PineconeValueError):
        VectorFactory.build({"id": id, "values": values, extra_key: extra_value})
