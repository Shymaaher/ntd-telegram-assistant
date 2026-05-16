# ntd_bot/config.py
from __future__ import annotations

from pathlib import Path
from typing import FrozenSet

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    BOT_TOKEN: str | None = None
    ADMIN_IDS: str = ""
    DOCUMENTS_DIR: Path = Path("./data/documents")
    CHROMA_DIR: Path = Path("./data/chroma")
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"
    RAG_TOP_K: int = 10
    RELEVANCE_THRESHOLD: float = 0.60
    MIN_RELEVANT_CHUNKS: int = 2
    OLLAMA_BASE_URL: str | None = None
    OLLAMA_MODEL: str | None = None

    @property
    def bot_token(self) -> str | None:
        return self.BOT_TOKEN

    @property
    def admin_ids(self) -> FrozenSet[int]:
        return frozenset(
            int(x.strip()) for x in self.ADMIN_IDS.split(",") if x.strip().isdigit()
        )

    @property
    def documents_dir(self) -> Path:
        return self.DOCUMENTS_DIR

    @property
    def chroma_dir(self) -> Path:
        return self.CHROMA_DIR

    @property
    def embedding_model(self) -> str:
        return self.EMBEDDING_MODEL

    @property
    def rag_top_k(self) -> int:
        return self.RAG_TOP_K

    @property
    def relevance_threshold(self) -> float:
        return self.RELEVANCE_THRESHOLD

    @property
    def min_relevant_chunks(self) -> int:
        return self.MIN_RELEVANT_CHUNKS

    @property
    def ollama_base_url(self) -> str | None:
        return self.OLLAMA_BASE_URL

    @property
    def ollama_model(self) -> str | None:
        return self.OLLAMA_MODEL


def load_settings() -> Settings:
    return Settings()