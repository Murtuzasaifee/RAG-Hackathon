import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from rag_hackathon.core.settings import Settings

REQUIRED_ENV = {
    "AZURE_DI_ENDPOINT": "https://example.cognitiveservices.azure.com",
    "AZURE_DI_KEY": "fake-key",
    "OPENAI_API_KEY": "sk-fake",
    "COHERE_API_KEY": "fake-cohere",
    "LOGFIRE_TOKEN": "fake-logfire",
}


class _IsolatedSettings(Settings):
    model_config = SettingsConfigDict(
        env_file=None,
        extra="ignore",
    )


def _make_settings(**overrides: str) -> Settings:
    env = {**REQUIRED_ENV, **overrides}
    with patch.dict(os.environ, env, clear=True):
        return _IsolatedSettings()


def test_settings_loads_from_env():
    s = _make_settings()
    assert s.azure_di_endpoint == "https://example.cognitiveservices.azure.com"
    assert s.qdrant_collection == "documents"


def test_settings_defaults():
    s = _make_settings()
    assert s.embedding_model == "text-embedding-3-small"
    assert s.rrf_k == 60
    assert s.top_k_dense == 20
    assert s.sparse_enabled is True
    assert s.chunk_max_tokens == 512
    assert s.cache_ttl_embed == 86400
    assert s.bifrost_url == "http://localhost:8080"
    assert s.llm_model == "gpt-5.4"


def test_settings_numeric_env_parsed_as_int():
    s = _make_settings(RRF_K="100", TOP_K_DENSE="30")
    assert s.rrf_k == 100
    assert s.top_k_dense == 30


def test_settings_missing_required_raises():
    with patch.dict(os.environ, {}, clear=True), pytest.raises(ValidationError):
        _IsolatedSettings()


def test_settings_custom_overrides():
    s = _make_settings(
        BIFROST_URL="http://custom:9090",
        SPARSE_ENABLED="false",
        QDRANT_COLLECTION="my_docs",
    )
    assert s.bifrost_url == "http://custom:9090"
    assert s.sparse_enabled is False
    assert s.qdrant_collection == "my_docs"
