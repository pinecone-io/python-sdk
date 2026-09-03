"""Guards that unit tests cannot observe ambient ``PINECONE_*`` variables (#353).

``tests/unit`` is a CI gate, and 149 of its tests changed result depending on
which ``PINECONE_*`` variables the developer happened to have exported — the
gate failed for exactly the people most likely to have a key exported, the ones
working on the live suites. Asserting "no such variable is set" would pass
vacuously on a clean machine, so these tests plant the variables first and then
assert the scrub in tests/unit/conftest.py removed them.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from pinecone._internal.config import PineconeConfig

_AMBIENT = {
    "PINECONE_API_KEY": "pcsk_ambient_not_a_real_key",
    "PINECONE_CONTROLLER_HOST": "https://ambient.invalid",
    "PINECONE_ADDITIONAL_HEADERS": '{"X-Ambient": "yes"}',
    "PINECONE_CLIENT_ID": "ambient-client-id",
    "PINECONE_CLIENT_SECRET": "ambient-client-secret",
    "PINECONE_PLUGIN_ASSISTANT_CONTROL_HOST": "https://ambient-control.invalid",
    "PINECONE_PLUGIN_ASSISTANT_DATA_HOST": "https://ambient-data.invalid",
    "PINECONE_SDK_ENV_FILE": "/ambient/place/.env",
}


@pytest.fixture(scope="module", autouse=True)
def _ambient_environment() -> Iterator[None]:
    """Stand in for the developer's exported environment.

    pytest builds higher-scoped fixtures first, so a module-scoped fixture is
    already in place when the function-scoped scrub in tests/unit/conftest.py
    runs — the same ordering an exported variable has. A function-scoped
    fixture would run after the scrub and prove nothing.
    """
    with pytest.MonkeyPatch.context() as mp:
        for name, value in _AMBIENT.items():
            mp.setenv(name, value)
        yield


def test_no_pinecone_variable_reaches_the_test() -> None:
    assert [name for name in os.environ if name.startswith("PINECONE_")] == []


def test_config_does_not_inherit_an_ambient_api_key() -> None:
    """The instance of the class: #353's ``test_repr_masks_empty_api_key``."""
    assert PineconeConfig(api_key="").api_key == ""
    assert "api_key='***'" in repr(PineconeConfig(api_key=""))


def test_config_does_not_inherit_an_ambient_host_or_headers() -> None:
    config = PineconeConfig(api_key="pcsk_explicit")
    assert config.host == ""
    assert config.additional_headers == {}
