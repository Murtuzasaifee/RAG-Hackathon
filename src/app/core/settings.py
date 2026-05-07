from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    azure_di_endpoint: str
    azure_di_key: str
    openai_api_key: str
    mesh_api_key: str
    cohere_api_key: str
    logfire_token: str
    huggingface_token: str | None = None

    bifrost_url: str = "http://localhost:8080"
    bifrost_public_url: str | None = None
    logfire_project_url: str = "https://logfire-us.pydantic.dev/msaifee/rag-hackathon"
    llm_guard_url: str = "http://localhost:8001"
    qdrant_url: str = "http://localhost:6333"
    redis_url: str = "redis://localhost:6379/0"

    embedding_model: str = "text-embedding-3-small"
    rerank_model: str = "rerank-english-v3.0"
    llm_model: str = "gpt-5.4"
    llm_provider: str = "meshapi"
    splade_model: str = "naver/splade-v3"

    rrf_k: int = 60
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_n_rerank: int = 5

    chunker_strategy: str = "document_aware"
    sparse_enabled: bool = True
    llm_guard_input_enabled: bool = True
    llm_guard_output_enabled: bool = True

    chunk_max_tokens: int = 512
    chunk_overlap: int = 50

    cache_ttl_embed: int = 86400
    cache_ttl_rerank: int = 21600
    cache_ttl_answer: int = 3600

    qdrant_collection: str = "documents"


@lru_cache
def get_settings() -> Settings:
    return Settings()
