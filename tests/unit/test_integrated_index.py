"""Unit tests for the deprecated integrated-index spec models.

IntegratedSpec/EmbedConfig stay importable for one major release so the
create() legacy-kwarg interceptor can translate real values into guided
error text; integrated index creation itself moved to
Indexes.create_for_model (tests/unit/client/test_indexes_create_for_model.py).
"""

from __future__ import annotations

import pytest

from pinecone.models.enums import Metric
from pinecone.models.indexes.specs import EmbedConfig, IntegratedSpec


def test_embed_config_is_immutable() -> None:
    """EmbedConfig is frozen — attribute assignment raises AttributeError."""
    config = EmbedConfig(model="multilingual-e5-large", field_map={"text": "content"})
    with pytest.raises(AttributeError):
        config.model = "other"  # type: ignore[misc]


def test_integrated_spec_is_immutable() -> None:
    """IntegratedSpec is frozen — attribute assignment raises AttributeError."""
    spec = IntegratedSpec(
        cloud="aws",
        region="us-east-1",
        embed=EmbedConfig(
            model="multilingual-e5-large",
            field_map={"text": "content"},
        ),
    )
    with pytest.raises(AttributeError):
        spec.cloud = "gcp"  # type: ignore[misc]


def test_embed_config_to_dict_basic() -> None:
    """to_dict serializes model, field_map, and defaults for read/write params."""
    config = EmbedConfig(model="multilingual-e5-large", field_map={"text": "content"})
    result = config.to_dict()
    assert result == {
        "model": "multilingual-e5-large",
        "field_map": {"text": "content"},
        "read_parameters": {},
        "write_parameters": {},
    }


def test_embed_config_to_dict_with_enum_metric() -> None:
    """Metric enum values are resolved to their string value in to_dict."""
    config = EmbedConfig(
        model="multilingual-e5-large",
        field_map={"text": "content"},
        metric=Metric.COSINE,
    )
    result = config.to_dict()
    assert result["metric"] == "cosine"
    assert not isinstance(result["metric"], Metric)


def test_embed_config_to_dict_with_read_write_params() -> None:
    """Explicit read/write parameters are included as-is."""
    config = EmbedConfig(
        model="multilingual-e5-large",
        field_map={"text": "content"},
        read_parameters={"k": 10},
        write_parameters={"batch": 32},
    )
    result = config.to_dict()
    assert result["read_parameters"] == {"k": 10}
    assert result["write_parameters"] == {"batch": 32}


def test_embed_config_to_dict_defaults_empty_params() -> None:
    """Omitted read/write parameters default to empty dicts."""
    config = EmbedConfig(model="multilingual-e5-large", field_map={"text": "content"})
    result = config.to_dict()
    assert result["read_parameters"] == {}
    assert result["write_parameters"] == {}
