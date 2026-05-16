
from __future__ import annotations

import logging
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

from .config import Settings

logger = logging.getLogger(__name__)

_embeddings: Optional[HuggingFaceEmbeddings] = None
_embeddings_model_name: str | None = None


def get_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    global _embeddings, _embeddings_model_name

    if _embeddings is None or _embeddings_model_name != settings.embedding_model:
        logger.info("Загрузка модели эмбеддингов: %s", settings.embedding_model)
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
        )
        _embeddings_model_name = settings.embedding_model

    return _embeddings