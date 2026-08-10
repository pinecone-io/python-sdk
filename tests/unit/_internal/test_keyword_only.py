from __future__ import annotations

import inspect

import pytest

from pinecone._client import Pinecone
from pinecone._internal.keyword_only import keyword_only_methods
from pinecone.async_client.async_index import AsyncIndex
from pinecone.async_client.pinecone import AsyncPinecone
from pinecone.errors.exceptions import PineconeValueError
from pinecone.grpc import GrpcIndex
from pinecone.index import Index


@keyword_only_methods
class _Toy:
    def kw_only(self, *, a: int, b: int = 0) -> int:
        return a + b

    async def async_kw_only(self, *, a: int, b: int = 0) -> int:
        return a + b

    def positional(self, a: int, b: int = 0) -> int:
        return a + b

    def _private_kw_only(self, *, a: int) -> int:
        return a


def test_correct_keyword_call_is_unaffected() -> None:
    assert _Toy().kw_only(a=2, b=3) == 5


async def test_correct_async_keyword_call_is_unaffected() -> None:
    assert await _Toy().async_kw_only(a=2, b=3) == 5


def test_positional_call_raises_pinecone_value_error() -> None:
    with pytest.raises(PineconeValueError) as exc:
        _Toy().kw_only(1, 2)  # type: ignore[misc]
    message = str(exc.value)
    assert "_Toy.kw_only()" in message
    assert "keyword-only" in message
    assert "2 positional arguments" in message
    assert "a, b" in message


def test_single_positional_uses_singular_wording() -> None:
    with pytest.raises(PineconeValueError) as exc:
        _Toy().kw_only(1)  # type: ignore[misc]
    assert "1 positional argument." in str(exc.value)


async def test_async_positional_call_raises_pinecone_value_error() -> None:
    with pytest.raises(PineconeValueError) as exc:
        await _Toy().async_kw_only(1)  # type: ignore[misc]
    assert "_Toy.async_kw_only()" in str(exc.value)
    assert "keyword-only" in str(exc.value)


def test_methods_with_positional_params_are_not_wrapped() -> None:
    assert not hasattr(_Toy.positional, "__wrapped__")
    assert _Toy().positional(1, 2) == 3


def test_private_methods_are_not_wrapped() -> None:
    assert not hasattr(_Toy._private_kw_only, "__wrapped__")


def test_signature_is_preserved_for_type_checkers_and_docs() -> None:
    sig = inspect.signature(_Toy.kw_only)
    assert list(sig.parameters) == ["self", "a", "b"]
    assert sig.parameters["a"].kind is inspect.Parameter.KEYWORD_ONLY
    assert _Toy.kw_only.__name__ == "kw_only"
    assert _Toy.kw_only.__wrapped__.__name__ == "kw_only"  # type: ignore[attr-defined]


@pytest.mark.parametrize("cls", [Index, GrpcIndex])
def test_query_guarded_on_sync_index_clients(cls: type) -> None:
    instance = cls.__new__(cls)  # type: ignore[call-overload]
    with pytest.raises(PineconeValueError) as exc:
        instance.query([0.1, 0.2, 0.3], top_k=10)
    message = str(exc.value)
    assert f"{cls.__name__}.query()" in message
    assert "top_k" in message


async def test_query_guarded_on_async_index_client() -> None:
    instance = AsyncIndex.__new__(AsyncIndex)
    with pytest.raises(PineconeValueError) as exc:
        await instance.query([0.1, 0.2, 0.3], top_k=10)  # type: ignore[misc, arg-type]
    assert "AsyncIndex.query()" in str(exc.value)


def test_keyword_only_guarded_on_sync_control_plane() -> None:
    instance = Pinecone.__new__(Pinecone)
    with pytest.raises(PineconeValueError) as exc:
        instance.create_backup("my-index", "my-backup")  # type: ignore[misc]
    assert "Pinecone.create_backup()" in str(exc.value)


async def test_keyword_only_guarded_on_async_control_plane() -> None:
    instance = AsyncPinecone.__new__(AsyncPinecone)
    with pytest.raises(PineconeValueError) as exc:
        await instance.create_backup("my-index", "my-backup")  # type: ignore[misc]
    assert "AsyncPinecone.create_backup()" in str(exc.value)


def test_real_client_signature_unchanged() -> None:
    sig = inspect.signature(Index.query)
    assert "top_k" in sig.parameters
    assert "vector" in sig.parameters
    assert sig.parameters["top_k"].kind is inspect.Parameter.KEYWORD_ONLY
    assert Index.query.__doc__


def test_non_keyword_only_real_method_is_untouched() -> None:
    assert not hasattr(Index.describe_import, "__wrapped__")


def test_property_is_untouched() -> None:
    assert isinstance(inspect.getattr_static(Index, "host"), property)
