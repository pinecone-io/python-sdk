"""Tests for enumeration models."""

from __future__ import annotations

import pytest

from pinecone.models.enums import (
    CloudProvider,
    DeletionProtection,
    EmbedModel,
    Metric,
    PodType,
    RerankModel,
    VectorType,
)


def _str_eq(enum_member: str, expected: str) -> bool:
    """Compare an enum member (which is a str subclass) to a plain string.

    Wrapped in a helper to avoid mypy comparison-overlap false positives
    with Literal enum types.
    """
    return enum_member == expected


class TestCloudProvider:
    def test_members(self) -> None:
        assert CloudProvider.AWS.value == "aws"
        assert CloudProvider.GCP.value == "gcp"
        assert CloudProvider.AZURE.value == "azure"

    def test_length(self) -> None:
        assert len(CloudProvider) == 3

    def test_string_coercion(self) -> None:
        assert _str_eq(CloudProvider.AWS, "aws")

    def test_construction_from_string(self) -> None:
        assert CloudProvider("aws") == CloudProvider.AWS
        assert CloudProvider("gcp") == CloudProvider.GCP
        assert CloudProvider("azure") == CloudProvider.AZURE

    def test_iterable(self) -> None:
        members = list(CloudProvider)
        assert len(members) == 3
        assert CloudProvider.AWS in members

    def test_invalid_value(self) -> None:
        with pytest.raises(ValueError, match="'invalid' is not a valid CloudProvider"):
            CloudProvider("invalid")


class TestMetric:
    def test_members(self) -> None:
        assert Metric.COSINE.value == "cosine"
        assert Metric.EUCLIDEAN.value == "euclidean"
        assert Metric.DOTPRODUCT.value == "dotproduct"

    def test_length(self) -> None:
        assert len(Metric) == 3

    def test_string_coercion(self) -> None:
        assert _str_eq(Metric.COSINE, "cosine")

    def test_construction_from_string(self) -> None:
        assert Metric("cosine") == Metric.COSINE
        assert Metric("euclidean") == Metric.EUCLIDEAN
        assert Metric("dotproduct") == Metric.DOTPRODUCT

    def test_iterable(self) -> None:
        members = list(Metric)
        assert len(members) == 3

    def test_invalid_value(self) -> None:
        with pytest.raises(ValueError, match="'hamming' is not a valid Metric"):
            Metric("hamming")


class TestDeletionProtection:
    def test_members(self) -> None:
        assert DeletionProtection.ENABLED.value == "enabled"
        assert DeletionProtection.DISABLED.value == "disabled"

    def test_length(self) -> None:
        assert len(DeletionProtection) == 2

    def test_string_coercion(self) -> None:
        assert _str_eq(DeletionProtection.ENABLED, "enabled")

    def test_construction_from_string(self) -> None:
        assert DeletionProtection("enabled") == DeletionProtection.ENABLED
        assert DeletionProtection("disabled") == DeletionProtection.DISABLED

    def test_iterable(self) -> None:
        members = list(DeletionProtection)
        assert len(members) == 2

    def test_invalid_value(self) -> None:
        with pytest.raises(ValueError, match="'maybe' is not a valid DeletionProtection"):
            DeletionProtection("maybe")


class TestVectorType:
    def test_members(self) -> None:
        assert VectorType.DENSE.value == "dense"
        assert VectorType.SPARSE.value == "sparse"

    def test_length(self) -> None:
        assert len(VectorType) == 2

    def test_string_coercion(self) -> None:
        assert _str_eq(VectorType.DENSE, "dense")

    def test_construction_from_string(self) -> None:
        assert VectorType("dense") == VectorType.DENSE
        assert VectorType("sparse") == VectorType.SPARSE

    def test_iterable(self) -> None:
        members = list(VectorType)
        assert len(members) == 2

    def test_invalid_value(self) -> None:
        with pytest.raises(ValueError, match="'hybrid' is not a valid VectorType"):
            VectorType("hybrid")


class TestEmbedModel:
    def test_members(self) -> None:
        assert EmbedModel.Multilingual_E5_Large.value == "multilingual-e5-large"
        assert EmbedModel.Pinecone_Sparse_English_V0.value == "pinecone-sparse-english-v0"
        assert EmbedModel.Llama_Text_Embed_V2.value == "llama-text-embed-v2"
        assert EmbedModel.Pinecone_Sparse_Multilingual_V0.value == "pinecone-sparse-multilingual-v0"

    def test_length(self) -> None:
        assert len(EmbedModel) == 4

    def test_string_coercion(self) -> None:
        assert _str_eq(EmbedModel.Llama_Text_Embed_V2, "llama-text-embed-v2")
        assert _str_eq(
            EmbedModel.Pinecone_Sparse_Multilingual_V0, "pinecone-sparse-multilingual-v0"
        )

    def test_construction_from_string(self) -> None:
        assert EmbedModel("llama-text-embed-v2") == EmbedModel.Llama_Text_Embed_V2
        assert (
            EmbedModel("pinecone-sparse-multilingual-v0")
            == EmbedModel.Pinecone_Sparse_Multilingual_V0
        )

    def test_pre_existing_members_keep_their_positions(self) -> None:
        """The 2026-07 additions are appended, so declaration order stays stable.

        ``EmbedModel`` is public and iterable, so reordering it would silently
        change what ``list(EmbedModel)[0]`` means for callers. This ticket is
        labelled additive; pinning the order keeps it that way.
        """
        assert [m.value for m in EmbedModel] == [
            "multilingual-e5-large",
            "pinecone-sparse-english-v0",
            "llama-text-embed-v2",
            "pinecone-sparse-multilingual-v0",
        ]

    def test_invalid_value(self) -> None:
        with pytest.raises(ValueError, match="'no-such-model' is not a valid EmbedModel"):
            EmbedModel("no-such-model")


class TestRerankModel:
    def test_members(self) -> None:
        assert RerankModel.Bge_Reranker_V2_M3.value == "bge-reranker-v2-m3"
        assert RerankModel.Cohere_Rerank_3_5.value == "cohere-rerank-3.5"
        assert RerankModel.Pinecone_Rerank_V0.value == "pinecone-rerank-v0"

    def test_length(self) -> None:
        assert len(RerankModel) == 3

    def test_deprecated_member_is_still_addressable(self) -> None:
        """``pinecone-rerank-v0`` is deprecated server-side, not removed here.

        Dropping the member would break imports for callers who reference it,
        and projects on the rollout allow-list can still use it, so the
        deprecation lives in the docstring rather than in the enum.
        """
        assert RerankModel("pinecone-rerank-v0") == RerankModel.Pinecone_Rerank_V0

    def test_invalid_value(self) -> None:
        with pytest.raises(ValueError, match="'no-such-model' is not a valid RerankModel"):
            RerankModel("no-such-model")


class TestPodType:
    def test_p1_members(self) -> None:
        assert PodType.P1_X1.value == "p1.x1"
        assert PodType.P1_X2.value == "p1.x2"
        assert PodType.P1_X4.value == "p1.x4"
        assert PodType.P1_X8.value == "p1.x8"

    def test_s1_members(self) -> None:
        assert PodType.S1_X1.value == "s1.x1"
        assert PodType.S1_X2.value == "s1.x2"
        assert PodType.S1_X4.value == "s1.x4"
        assert PodType.S1_X8.value == "s1.x8"

    def test_p2_members(self) -> None:
        assert PodType.P2_X1.value == "p2.x1"
        assert PodType.P2_X2.value == "p2.x2"
        assert PodType.P2_X4.value == "p2.x4"
        assert PodType.P2_X8.value == "p2.x8"

    def test_length(self) -> None:
        assert len(PodType) == 12

    def test_string_coercion(self) -> None:
        assert _str_eq(PodType.P1_X1, "p1.x1")
        assert _str_eq(PodType.S1_X4, "s1.x4")
        assert _str_eq(PodType.P2_X8, "p2.x8")

    def test_construction_from_string(self) -> None:
        assert PodType("p1.x1") == PodType.P1_X1
        assert PodType("s1.x2") == PodType.S1_X2
        assert PodType("p2.x4") == PodType.P2_X4

    def test_iterable(self) -> None:
        members = list(PodType)
        assert len(members) == 12

    def test_invalid_value(self) -> None:
        with pytest.raises(ValueError, match=r"'p3\.x1' is not a valid PodType"):
            PodType("p3.x1")


class TestEnumStringBehavior:
    """Test that str(Enum) mixin makes enums usable as plain strings."""

    def test_enum_equals_string(self) -> None:
        assert _str_eq(Metric.COSINE, "cosine")
        assert _str_eq(CloudProvider.AWS, "aws")
        assert _str_eq(DeletionProtection.ENABLED, "enabled")
        assert _str_eq(VectorType.DENSE, "dense")
        assert _str_eq(PodType.P1_X1, "p1.x1")

    def test_enum_as_dict_key(self) -> None:
        d: dict[str, int] = {Metric.COSINE: 1}
        assert d["cosine"] == 1

    def test_isinstance_str(self) -> None:
        assert isinstance(Metric.COSINE, str)
        assert isinstance(CloudProvider.AWS, str)
        assert isinstance(PodType.P1_X1, str)
