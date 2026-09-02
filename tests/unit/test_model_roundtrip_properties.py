"""Property-based round-trip tests for Vector and SparseValues.

For any generated model, serializing with ``to_dict`` and parsing the result
back must reproduce an equal model. Both parsers are exercised: the model's own
``from_dict`` and ``VectorFactory.build``, which accepts the same snake_case
shape. (Note: ``_vector_to_dict`` emits camelCase wire keys and is deliberately
*not* a factory input, so it is not part of this round-trip.)
"""

from __future__ import annotations

from hypothesis import assume, given
from hypothesis import strategies as st

from pinecone._internal.vector_factory import VectorFactory
from pinecone.models.vectors.sparse import SparseValues
from pinecone.models.vectors.vector import Vector

_scalar = st.one_of(
    st.text(max_size=16),
    st.integers(),
    st.booleans(),
    st.floats(allow_nan=False, allow_infinity=False),
)
_metadata = st.dictionaries(st.text(min_size=1, max_size=16), _scalar, max_size=5)
_ascii_ids = st.text(alphabet=st.characters(min_codepoint=1, max_codepoint=127), max_size=64)


@st.composite
def sparse_values(draw: st.DrawFn) -> SparseValues:
    length = draw(st.integers(min_value=0, max_value=16))
    indices = draw(
        st.lists(st.integers(min_value=0, max_value=10**6), min_size=length, max_size=length)
    )
    values = draw(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False),
            min_size=length,
            max_size=length,
        )
    )
    return SparseValues(indices=indices, values=values)


@st.composite
def vectors(draw: st.DrawFn) -> Vector:
    id = draw(_ascii_ids)
    values = draw(st.lists(st.floats(allow_nan=False, allow_infinity=False), max_size=32))
    sparse = draw(st.one_of(st.none(), sparse_values()))
    metadata = draw(st.one_of(st.none(), _metadata))
    assume(values or sparse is not None)
    return Vector(id=id, values=values, sparse_values=sparse, metadata=metadata)


@given(v=vectors())
def test_vector_from_dict_roundtrip(v: Vector) -> None:
    assert Vector.from_dict(v.to_dict()) == v


@given(v=vectors())
def test_vector_factory_build_roundtrip(v: Vector) -> None:
    assert VectorFactory.build(v.to_dict()) == v


@given(sv=sparse_values())
def test_sparse_values_from_dict_roundtrip(sv: SparseValues) -> None:
    assert SparseValues.from_dict(sv.to_dict()) == sv
