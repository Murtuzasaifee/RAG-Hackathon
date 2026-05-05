import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_settings():
    from app.core.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def env_settings():
    base = {
        "AZURE_DI_ENDPOINT": "https://example.cognitiveservices.azure.com",
        "AZURE_DI_KEY": "fake-key",
        "OPENAI_API_KEY": "sk-fake",
        "COHERE_API_KEY": "fake-cohere",
        "LOGFIRE_TOKEN": "fake-logfire",
    }
    with patch.dict(os.environ, base, clear=False):
        yield base
