"""Hypothesis suite for the legacy-to-2026-07 translation module (#498).

Five properties, over every spec variant in both object and dict form and
every ``metric``/``vector_type`` value as both an enum member and a plain
string:

* A translated schema declares exactly one field, and it is the reserved name
  for that vector type — ``_values`` or ``_sparse_values``, never both and
  never a third key.
* An enum member and its ``.value`` string produce identical output, down to
  the error raised when the call is invalid.
* The object form of a spec and its dict equivalent translate identically.
* Every output survives the orjson encoder the HTTP layer uses, unchanged,
  and carries no ``str``-subclass enum member that would go on the wire
  mangled elsewhere.
* ``deployment`` never carries read capacity or ``pods``: read capacity is
  lifted to the top level of the request, and pod capacity is replicas x shards.
"""

from __future__ import annotations

from typing import Any

import orjson
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pinecone._internal.legacy_index_translation import (
    legacy_pod_scaling,
    legacy_vector_schema,
    spec_to_deployment,
    spec_to_read_capacity,
)
from pinecone.errors.exceptions import PineconeError
from pinecone.models.enums import Metric, PodType, VectorType
from pinecone.models.indexes.specs import ByocSpec, PodSpec, ServerlessSpec

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

#: Every metric, twice: as the enum member and as the string it stands for.
METRICS = st.sampled_from([*Metric, *(m.value for m in Metric)])
VECTOR_TYPES = st.sampled_from([*VectorType, *(v.value for v in VectorType)])
POD_TYPES = st.sampled_from([*PodType, *(p.value for p in PodType)])

DIMENSIONS = st.one_of(st.none(), st.integers(min_value=1, max_value=20_000))

SpecPair = tuple[ServerlessSpec | PodSpec | ByocSpec, dict[str, Any]]


def _read_capacities() -> st.SearchStrategy[dict[str, Any]]:
    on_demand = st.just({"mode": "OnDemand"})
    dedicated = st.builds(
        lambda shards, replicas: {
            "mode": "Dedicated",
            "dedicated": {"scaling": "Manual", "manual": {"shards": shards, "replicas": replicas}},
        },
        shards=st.integers(min_value=1, max_value=8),
        replicas=st.integers(min_value=1, max_value=8),
    )
    return st.one_of(on_demand, dedicated)


def _optional(**fields: Any) -> dict[str, Any]:
    """The subset of *fields* the caller actually supplied.

    Both forms of a spec are built from this same subset, so an omitted key
    means "inherit the struct default" on the object side too — which is what
    makes the two forms comparable at all.
    """
    return {key: value for key, value in fields.items() if value is not None}


@st.composite
def serverless_specs(draw: st.DrawFn) -> SpecPair:
    cloud = draw(st.sampled_from(["aws", "gcp", "azure"]))
    region = draw(st.sampled_from(["us-east-1", "eu-west-1", "us-west-2"]))
    rest = _optional(
        read_capacity=draw(st.one_of(st.none(), _read_capacities())),
        schema=draw(st.one_of(st.none(), st.just({"fields": {"genre": {"type": "string"}}}))),
    )
    obj = ServerlessSpec(cloud=cloud, region=region, **rest)
    return obj, {"serverless": {"cloud": cloud, "region": region, **rest}}


@st.composite
def pod_specs(draw: st.DrawFn) -> SpecPair:
    environment = draw(st.sampled_from(["us-east-1-aws", "us-west1-gcp", "eastus-azure"]))
    rest = _optional(
        pod_type=draw(st.one_of(st.none(), st.sampled_from([p.value for p in PodType]))),
        replicas=draw(st.one_of(st.none(), st.integers(min_value=1, max_value=10))),
        shards=draw(st.one_of(st.none(), st.integers(min_value=1, max_value=10))),
        pods=draw(st.one_of(st.none(), st.integers(min_value=1, max_value=10))),
    )
    obj = PodSpec(environment=environment, **rest)
    return obj, {"pod": {"environment": environment, **rest}}


@st.composite
def byoc_specs(draw: st.DrawFn) -> SpecPair:
    environment = draw(st.sampled_from(["aws-us-east-1-b921", "gcp-us-central1-c004"]))
    rest = _optional(read_capacity=draw(st.one_of(st.none(), _read_capacities())))
    obj = ByocSpec(environment=environment, **rest)
    return obj, {"byoc": {"environment": environment, **rest}}


SPECS = st.one_of(serverless_specs(), pod_specs(), byoc_specs())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _outcome(call: Any, /, **kwargs: Any) -> Any:
    """The call's result, or a comparable stand-in for the error it raised."""
    try:
        return call(**kwargs)
    except PineconeError as exc:
        return type(exc), str(exc)


def _assert_plain_json_scalars(value: Any) -> None:
    """No enum member survives into the request body.

    ``Metric.COSINE == "cosine"`` is true, so an equality-only round-trip check
    would wave a leaked member through — and elsewhere in the SDK a member that
    reaches a boundary unresolved goes on the wire as ``"Metric.COSINE"``.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            assert type(key) is str, f"{key!r} is a {type(key).__name__}, not a plain str"
            _assert_plain_json_scalars(item)
    elif isinstance(value, list):
        for item in value:
            _assert_plain_json_scalars(item)
    else:
        assert type(value) in (str, int, float, bool, type(None)), (
            f"{value!r} is a {type(value).__name__}, not a plain JSON scalar"
        )


def _assert_encoder_round_trip(value: Any) -> None:
    _assert_plain_json_scalars(value)
    assert orjson.loads(orjson.dumps(value)) == value


# ---------------------------------------------------------------------------
# Property 1: a schema declares exactly one reserved field
# ---------------------------------------------------------------------------


@given(dimension=DIMENSIONS, metric=st.one_of(st.none(), METRICS), vector_type=VECTOR_TYPES)
def test_schema_declares_exactly_one_reserved_field(
    dimension: int | None, metric: Any, vector_type: Any
) -> None:
    is_sparse = getattr(vector_type, "value", vector_type) == "sparse"
    kwargs = {"dimension": dimension, "metric": metric, "vector_type": vector_type}

    if is_sparse != (dimension is None):
        with pytest.raises(PineconeError):
            legacy_vector_schema(**kwargs)
        return

    schema = legacy_vector_schema(**kwargs)
    assert set(schema) == {"fields"}
    assert set(schema["fields"]) == ({"_sparse_values"} if is_sparse else {"_values"})


@given(dimension=st.integers(min_value=1, max_value=20_000), metric=st.one_of(st.none(), METRICS))
def test_vector_type_defaults_to_dense(dimension: int, metric: Any) -> None:
    assert legacy_vector_schema(
        dimension=dimension, metric=metric, vector_type=None
    ) == legacy_vector_schema(dimension=dimension, metric=metric, vector_type="dense")


# ---------------------------------------------------------------------------
# Property 2: an enum member and its .value string agree
# ---------------------------------------------------------------------------


@given(
    dimension=DIMENSIONS,
    metric=st.sampled_from(list(Metric)),
    vector_type=st.sampled_from(list(VectorType)),
)
def test_enum_input_matches_its_value_string(
    dimension: int | None, metric: Metric, vector_type: VectorType
) -> None:
    as_members = _outcome(
        legacy_vector_schema, dimension=dimension, metric=metric, vector_type=vector_type
    )
    as_strings = _outcome(
        legacy_vector_schema,
        dimension=dimension,
        metric=metric.value,
        vector_type=vector_type.value,
    )
    assert as_members == as_strings


@given(replicas=st.one_of(st.none(), st.integers(min_value=1, max_value=10)), pod_type=POD_TYPES)
def test_pod_type_enum_matches_its_value_string(replicas: int | None, pod_type: Any) -> None:
    resolved = getattr(pod_type, "value", pod_type)
    assert legacy_pod_scaling(replicas=replicas, pod_type=pod_type) == legacy_pod_scaling(
        replicas=replicas, pod_type=resolved
    )


@given(
    replicas=st.one_of(st.none(), st.integers(min_value=1, max_value=10)),
    pod_type=st.one_of(st.none(), POD_TYPES),
)
def test_pod_scaling_carries_only_supplied_keys(replicas: int | None, pod_type: Any) -> None:
    scaling = legacy_pod_scaling(replicas=replicas, pod_type=pod_type)
    expected = ({"replicas"} if replicas is not None else set()) | (
        {"pod_type"} if pod_type is not None else set()
    )
    assert set(scaling) == expected


# ---------------------------------------------------------------------------
# Property 3: the object and dict forms of a spec translate identically
# ---------------------------------------------------------------------------


@given(pair=SPECS)
def test_object_and_dict_spec_forms_agree(pair: SpecPair) -> None:
    obj, as_dict = pair
    assert spec_to_deployment(obj) == spec_to_deployment(as_dict)
    assert spec_to_read_capacity(obj) == spec_to_read_capacity(as_dict)


# ---------------------------------------------------------------------------
# Property 4: every output survives the encoder unchanged
# ---------------------------------------------------------------------------


@given(pair=SPECS)
def test_spec_translations_round_trip_through_the_encoder(pair: SpecPair) -> None:
    for spec in pair:
        _assert_encoder_round_trip(spec_to_deployment(spec))
        _assert_encoder_round_trip(spec_to_read_capacity(spec))


@given(dimension=DIMENSIONS, metric=st.one_of(st.none(), METRICS), vector_type=VECTOR_TYPES)
def test_schemas_round_trip_through_the_encoder(
    dimension: int | None, metric: Any, vector_type: Any
) -> None:
    try:
        schema = legacy_vector_schema(dimension=dimension, metric=metric, vector_type=vector_type)
    except PineconeError:
        return
    _assert_encoder_round_trip(schema)


@given(replicas=st.one_of(st.none(), st.integers(min_value=1, max_value=10)), pod_type=POD_TYPES)
def test_pod_scaling_round_trips_through_the_encoder(replicas: int | None, pod_type: Any) -> None:
    _assert_encoder_round_trip(legacy_pod_scaling(replicas=replicas, pod_type=pod_type))


# ---------------------------------------------------------------------------
# Property 5: deployment carries neither read capacity nor pods
# ---------------------------------------------------------------------------


@given(pair=SPECS)
def test_deployment_omits_lifted_and_dropped_keys(pair: SpecPair) -> None:
    obj, as_dict = pair
    for spec in (obj, as_dict):
        deployment = spec_to_deployment(spec)
        assert "read_capacity" not in deployment
        assert "pods" not in deployment
        assert "schema" not in deployment
        assert deployment["deployment_type"] in {"managed", "pod", "byoc"}


@given(pair=SPECS)
def test_read_capacity_is_lifted_verbatim(pair: SpecPair) -> None:
    obj, _ = pair
    expected = None if isinstance(obj, PodSpec) else obj.read_capacity
    assert spec_to_read_capacity(obj) == expected
