"""Settings: every configurable value in ONE typed class.

pydantic-settings reads values from environment variables and the .env file,
converts them to the right type, and fails at startup if something is invalid.
Field name `llm_model` <-> env var LLM_MODEL (case-insensitive).
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- App ---
    app_name: str = "RAG Backend"
    app_version: str = "0.1.0"
    # If set, every /api/v1 request must send the header  X-API-Key: <value>
    api_key: str | None = None
    # Which websites (frontends) may call this API from a browser
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # --- AI providers (same idea as rag-project/config.py) ---
    llm_provider: str = "google"            # google | openai | ollama
    llm_model: str | None = None            # None = provider's default
    embedding_provider: str = "google"      # google | openai | ollama
    embedding_model: str | None = None
    google_api_key: str | None = None
    openai_api_key: str | None = None
    llm_timeout_seconds: int = 60

    # --- RAG storage ---
    chroma_dir: str = "storage/chroma"
    collection_name: str = "pdf_docs"
    upload_dir: str = "storage/uploads"
    chunk_size: int = 400
    chunk_overlap: int = 50
    retriever_k: int = 3
    max_upload_mb: int = 20


@lru_cache  # read .env once, then reuse the same object everywhere
def get_settings() -> Settings:
    return Settings()
