"""Table tests for the pure legacy-to-2026-07 translation module (#498).

Every example here is a row from #498's translation table: the literal the
function must return verbatim for the input on its left. The four validation
messages are pinned as exact strings because they are 9.x wording that user
code may match on.

``TestPurity`` pins the constraint the module exists to satisfy — no imports
from the client packages or the HTTP layer — both in the source and at
runtime, so a transitive import added three modules away is caught too.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys
from typing import Any

import pytest

from pinecone._internal.legacy_index_translation import (
    legacy_pod_scaling,
    legacy_vector_schema,
    spec_to_deployment,
    spec_to_read_capacity,
)
from pinecone.errors.exceptions import PineconeTypeError, PineconeValueError
from pinecone.models.enums import Metric, PodType, VectorType
from pinecone.models.indexes.specs import (
    ByocSpec,
    EmbedConfig,
    IntegratedSpec,
    PodSpec,
    ServerlessSpec,
)

MODULE = "pinecone._internal.legacy_index_translation"
MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[3] / "pinecone/_internal/legacy_index_translation.py"
)

FORBIDDEN_PREFIXES = ("pinecone.client", "pinecone.async_client", "pinecone._internal.http_client")


class TestPurity:
    def test_source_imports_nothing_from_the_client_or_http_layers(self) -> None:
        tree = ast.parse(MODULE_PATH.read_text(), filename=str(MODULE_PATH))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)
                imported.update(f"{node.module}.{alias.name}" for alias in node.names)

        offenders = sorted(n for n in imported if n.startswith(FORBIDDEN_PREFIXES))
        assert offenders == [], f"{MODULE} must not import {offenders}"

    @pytest.mark.timeout(60)
    def test_importing_it_pulls_in_no_client_or_http_module(self) -> None:
        """The source check misses a forbidden module reached through a sibling.

        A fresh interpreter is the only honest place to ask: by the time the
        unit suite runs, pytest collection has already imported the whole SDK,
        so an in-process ``sys.modules`` check would pass vacuously.
        """
        probe = (
            f"import sys; import {MODULE}; "
            f"print([m for m in sys.modules if m.startswith({FORBIDDEN_PREFIXES!r})])"
        )
        proc = subprocess.run(  # noqa: S603 — argv is this repo's own module name
            [sys.executable, "-c", probe],
            cwd=MODULE_PATH.parents[2],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "[]", proc.stdout


class TestSpecToDeployment:
    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            pytest.param(
                ServerlessSpec(cloud="aws", region="us-east-1"),
                {"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
                id="serverless-object",
            ),
            pytest.param(
                {"serverless": {"cloud": "aws", "region": "us-east-1"}},
                {"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
                id="serverless-dict",
            ),
            pytest.param(
                PodSpec(environment="us-east-1-aws", pod_type="p1.x1", replicas=1, shards=1),
                {
                    "deployment_type": "pod",
                    "environment": "us-east-1-aws",
                    "pod_type": "p1.x1",
                    "replicas": 1,
                    "shards": 1,
                },
                id="pod-object",
            ),
            pytest.param(
                {"pod": {"environment": "us-east-1-aws"}},
                {
                    "deployment_type": "pod",
                    "environment": "us-east-1-aws",
                    "pod_type": "p1.x1",
                    "replicas": 1,
                    "shards": 1,
                },
                id="pod-dict-inherits-podspec-defaults",
            ),
            pytest.param(
                ByocSpec(environment="aws-us-east-1-b921"),
                {"deployment_type": "byoc", "environment": "aws-us-east-1-b921"},
                id="byoc-object",
            ),
            pytest.param(
                {"byoc": {"environment": "aws-us-east-1-b921"}},
                {"deployment_type": "byoc", "environment": "aws-us-east-1-b921"},
                id="byoc-dict",
            ),
        ],
    )
    def test_translation_table(self, spec: Any, expected: dict[str, Any]) -> None:
        assert spec_to_deployment(spec) == expected

    def test_pods_is_dropped_rather_than_rejected(self) -> None:
        """Every PodSpec carries pods=1, so rejecting it would reject them all."""
        deployment = spec_to_deployment(PodSpec(environment="us-east-1-aws", pods=4))
        assert "pods" not in deployment

    def test_read_capacity_does_not_leak_into_the_deployment(self) -> None:
        spec = ServerlessSpec(cloud="aws", region="us-east-1", read_capacity={"mode": "OnDemand"})
        assert spec_to_deployment(spec) == {
            "deployment_type": "managed",
            "cloud": "aws",
            "region": "us-east-1",
        }


class TestSpecToReadCapacity:
    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            pytest.param(
                ServerlessSpec(cloud="aws", region="us-east-1", read_capacity={"mode": "OnDemand"}),
                {"mode": "OnDemand"},
                id="serverless-object",
            ),
            pytest.param(
                {
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-east-1",
                        "read_capacity": {"mode": "OnDemand"},
                    }
                },
                {"mode": "OnDemand"},
                id="serverless-dict",
            ),
            pytest.param(
                ByocSpec(environment="aws-us-east-1-b921", read_capacity={"mode": "OnDemand"}),
                {"mode": "OnDemand"},
                id="byoc-object",
            ),
            pytest.param(ServerlessSpec(cloud="aws", region="us-east-1"), None, id="unset"),
            pytest.param(PodSpec(environment="us-east-1-aws"), None, id="pod-has-none"),
            pytest.param({"pod": {"environment": "us-east-1-aws"}}, None, id="pod-dict-has-none"),
        ],
    )
    def test_translation_table(self, spec: Any, expected: dict[str, Any] | None) -> None:
        assert spec_to_read_capacity(spec) == expected

    def test_result_does_not_alias_the_callers_spec(self) -> None:
        spec = ServerlessSpec(
            cloud="aws",
            region="us-east-1",
            read_capacity={"mode": "Dedicated", "dedicated": {"scaling": "Manual"}},
        )
        lifted = spec_to_read_capacity(spec)
        assert lifted is not None
        lifted["dedicated"]["scaling"] = "Auto"
        assert spec.read_capacity == {"mode": "Dedicated", "dedicated": {"scaling": "Manual"}}


class TestSpecRejection:
    def test_malformed_dict_names_the_2026_07_replacement(self) -> None:
        with pytest.raises(PineconeValueError) as exc:
            spec_to_deployment({"managed": {"cloud": "aws"}})
        assert "'serverless', 'pod', or 'byoc'" in str(exc.value)
        assert "deployment=" in str(exc.value)

    def test_non_spec_object_names_the_2026_07_replacement(self) -> None:
        with pytest.raises(PineconeValueError) as exc:
            spec_to_deployment("us-east-1-aws")
        assert "got 'str'" in str(exc.value)
        assert "deployment=" in str(exc.value)

    def test_integrated_spec_points_at_create_for_model(self) -> None:
        spec = IntegratedSpec(
            cloud="aws",
            region="us-east-1",
            embed=EmbedConfig(model="multilingual-e5-large", field_map={"text": "chunk"}),
        )
        with pytest.raises(PineconeValueError, match="create_for_model"):
            spec_to_deployment(spec)

    def test_inner_dict_missing_a_required_key_names_the_struct(self) -> None:
        with pytest.raises(PineconeValueError) as exc:
            spec_to_deployment({"serverless": {"cloud": "aws"}})
        assert "ServerlessSpec" in str(exc.value)
        assert "region" in str(exc.value)

    def test_inner_dict_with_an_unknown_key_is_rejected(self) -> None:
        with pytest.raises(PineconeValueError, match="pod_size"):
            spec_to_deployment({"pod": {"environment": "us-east-1-aws", "pod_size": "large"}})

    def test_a_null_inner_value_reads_as_a_malformed_dict(self) -> None:
        with pytest.raises(PineconeValueError, match="'serverless', 'pod', or 'byoc'"):
            spec_to_read_capacity({"serverless": None})


class TestLegacyVectorSchema:
    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            pytest.param(
                {"dimension": 1536, "metric": "cosine", "vector_type": None},
                {
                    "fields": {
                        "_values": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}
                    }
                },
                id="dense-implicit-vector-type",
            ),
            pytest.param(
                {"dimension": 1536, "metric": None, "vector_type": "dense"},
                {
                    "fields": {
                        "_values": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}
                    }
                },
                id="dense-metric-defaults-to-cosine",
            ),
            pytest.param(
                {"dimension": None, "metric": "dotproduct", "vector_type": "sparse"},
                {"fields": {"_sparse_values": {"type": "sparse_vector"}}},
                id="sparse-drops-metric",
            ),
        ],
    )
    def test_translation_table(self, kwargs: dict[str, Any], expected: dict[str, Any]) -> None:
        assert legacy_vector_schema(**kwargs) == expected

    def test_enum_members_translate_like_their_values(self) -> None:
        assert legacy_vector_schema(
            dimension=8, metric=Metric.EUCLIDEAN, vector_type=VectorType.DENSE
        ) == legacy_vector_schema(dimension=8, metric="euclidean", vector_type="dense")

    @pytest.mark.parametrize(
        ("kwargs", "error", "message"),
        [
            pytest.param(
                {"dimension": None, "metric": "cosine", "vector_type": "dense"},
                PineconeValueError,
                "dimension is required for dense indexes",
                id="dense-without-dimension",
            ),
            pytest.param(
                {"dimension": 1536, "metric": "cosine", "vector_type": "sparse"},
                PineconeValueError,
                "dimension must not be provided for sparse indexes",
                id="sparse-with-dimension",
            ),
            pytest.param(
                {"dimension": 1536, "metric": "manhattan", "vector_type": "dense"},
                PineconeValueError,
                "metric must be one of ['cosine', 'dotproduct', 'euclidean'], got 'manhattan'",
                id="unknown-metric",
            ),
            pytest.param(
                {"dimension": "1536", "metric": "cosine", "vector_type": "dense"},
                PineconeTypeError,
                "dimension must be an integer, got 'str'",
                id="non-integer-dimension",
            ),
            pytest.param(
                {"dimension": 8, "metric": "cosine", "vector_type": "SPARSE"},
                PineconeValueError,
                "vector_type must be one of ['dense', 'sparse'], got 'SPARSE'",
                id="mis-cased-vector-type",
            ),
            pytest.param(
                {"dimension": 8, "metric": "cosine", "vector_type": "sparce"},
                PineconeValueError,
                "vector_type must be one of ['dense', 'sparse'], got 'sparce'",
                id="misspelled-vector-type",
            ),
        ],
    )
    def test_9x_validation_messages_are_verbatim(
        self, kwargs: dict[str, Any], error: type[Exception], message: str
    ) -> None:
        with pytest.raises(error) as exc:
            legacy_vector_schema(**kwargs)
        assert str(exc.value) == message

    def test_metric_is_validated_before_dimension(self) -> None:
        """9.x checked metric first; a call wrong in both ways must not change error."""
        with pytest.raises(PineconeValueError, match="metric must be one of"):
            legacy_vector_schema(dimension="1536", metric="manhattan", vector_type="dense")


class TestLegacyPodScaling:
    @pytest.mark.parametrize(
        ("replicas", "pod_type", "expected"),
        [
            pytest.param(3, None, {"replicas": 3}, id="replicas-only"),
            pytest.param(None, "p1.x2", {"pod_type": "p1.x2"}, id="pod-type-only"),
            pytest.param(3, "p1.x2", {"replicas": 3, "pod_type": "p1.x2"}, id="both"),
            pytest.param(None, None, {}, id="neither"),
        ],
    )
    def test_translation_table(
        self, replicas: int | None, pod_type: str | None, expected: dict[str, Any]
    ) -> None:
        assert legacy_pod_scaling(replicas=replicas, pod_type=pod_type) == expected

    def test_pod_type_enum_resolves_to_its_value(self) -> None:
        scaling = legacy_pod_scaling(replicas=None, pod_type=PodType.P1_X2)
        assert scaling == {"pod_type": "p1.x2"}
        assert type(scaling["pod_type"]) is str
