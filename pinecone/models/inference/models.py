"""Model information response models for the Inference API."""

from __future__ import annotations

from typing import Any, cast

import msgspec
from msgspec import Struct


class ModelInfoSupportedParameter(Struct, kw_only=True):
    """One key a model accepts in a ``parameters`` argument, and its bounds.

    Read these off :class:`ModelInfo`'s ``supported_parameters`` to learn what
    ``parameters=`` will take for a given model, rather than guessing and
    catching the rejection.

    Attributes:
        parameter: The key to use in ``parameters``, e.g. ``"input_type"``.
        type: How the value is constrained (e.g. ``"one_of"`` for a fixed set).
        value_type: The value type (e.g. ``"string"``).
        required: Whether the parameter must be sent.
        allowed_values: The values accepted, when the set is fixed.
        min: Minimum value, for numeric parameters.
        max: Maximum value, for numeric parameters.
        default: What the model uses when the key is omitted.
    """

    parameter: str
    type: str
    value_type: str
    required: bool
    allowed_values: list[str | int] | None = None
    min: int | float | None = None
    max: int | float | None = None
    default: str | int | float | bool | None = None

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. param['parameter'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'parameter' in param``)."""
        return key in self.__struct_fields__


_MODEL_INFO_ALIASES: dict[str, str] = {"name": "model", "description": "short_description"}


class ModelInfo(Struct, kw_only=True):
    """What one inference model is and what it will accept.

    Returned by :meth:`~pinecone.client.inference.Inference.get_model`, and by
    :meth:`~pinecone.client.inference.Inference.list_models` for every model in
    the listing. The embed-only fields below are ``None`` on a reranking model,
    so read ``type`` before relying on them. Bracket access with a field name
    (``info["model"]``) reads the fields too.

    Attributes:
        model: The model identifier — what to pass as ``model=``. Also readable
            as ``name``.
        short_description: A brief description of the model. Also readable as
            ``description``.
        type: ``"embed"`` or ``"rerank"``.
        supported_parameters: The
            :class:`ModelInfoSupportedParameter` entries describing what
            ``parameters=`` will take for this model.
        vector_type: For embedding models, ``"dense"`` or ``"sparse"``.
        default_dimension: For embedding models, the output dimension used when
            none is requested.
        supported_dimensions: For embedding models, every output dimension the
            model can produce.
        modality: The input modality (e.g. ``"text"``).
        max_sequence_length: The longest input the model accepts.
        max_batch_size: The most inputs one request may carry.
        provider_name: Who supplies the model.
        supported_metrics: The similarity metrics an index built on this
            model's vectors can use.

    Examples:
        >>> from pinecone import Pinecone
        >>> pc = Pinecone(api_key="your-api-key")
        >>> info = pc.inference.get_model(model="multilingual-e5-large")
        >>> info.type, info.vector_type, info.default_dimension
        ('embed', 'dense', 1024)
        >>> info.name == info.model
        True
    """

    model: str
    short_description: str
    type: str
    supported_parameters: list[ModelInfoSupportedParameter]
    vector_type: str | None = None
    default_dimension: int | None = None
    supported_dimensions: list[int] | None = None
    modality: str | None = None
    max_sequence_length: int | None = None
    max_batch_size: int | None = None
    provider_name: str | None = None
    supported_metrics: list[str] | None = None

    @property
    def name(self) -> str:
        """Alias for ``model`` — the model identifier."""
        return self.model

    @property
    def description(self) -> str:
        """Alias for ``short_description`` — a brief description of the model."""
        return self.short_description

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. model_info['model'])."""
        key = _MODEL_INFO_ALIASES.get(key, key)
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'model' in model_info``)."""
        if isinstance(key, str):
            key = _MODEL_INFO_ALIASES.get(key, key)
        return key in self.__struct_fields__

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict representation of this object."""
        return cast(dict[str, Any], msgspec.to_builtins(self))

    def __getattr__(self, name: str) -> Any:
        """Legacy alias passthrough and AttributeError for unknown attributes."""
        resolved = _MODEL_INFO_ALIASES.get(name)
        if resolved is not None:
            return getattr(self, resolved)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
