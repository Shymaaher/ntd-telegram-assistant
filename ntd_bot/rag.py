
import logging
import re
import time
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import Settings
from .embeddings import get_embeddings

logger = logging.getLogger(__name__)

_vectorstore: Optional[Chroma] = None
_bm25_index = None
_bm25_docs: list[Document] = []
_llm = None
_llm_settings_key: tuple | None = None

_RRF_THRESHOLD = 0.008


def _init_vectorstore(settings: Settings) -> None:
    global _vectorstore
    if _vectorstore is not None:
        return
    _vectorstore = Chroma(
        persist_directory=str(settings.chroma_dir),
        embedding_function=get_embeddings(settings),
        collection_name="ntd_docs",
        collection_metadata={"hnsw:space": "cosine"},
    )


def _init_bm25() -> None:
    global _bm25_index, _bm25_docs
    if _bm25_index is not None:
        return
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank-bm25 не установлен. pip install rank-bm25")
        return

    t0 = time.perf_counter()
    logger.info("Построение BM25-индекса...")
    raw = _vectorstore.get(include=["documents", "metadatas"])
    _bm25_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]
    tokenized = []
    for doc in _bm25_docs:
        text_tokens = re.split(r"[\s\-–—\.\/]+", doc.page_content.lower())
        name_tokens = re.split(r"[\s\-–—\.\/\_\(\)]+", doc.metadata.get("doc_name", "").lower())
        tokenized.append(text_tokens + name_tokens)
    _bm25_index = BM25Okapi(tokenized)
    logger.info("BM25: %d чанков (%.2fs)", len(_bm25_docs), time.perf_counter() - t0)


def _get_llm(settings: Settings):
    global _llm, _llm_settings_key
    current_key = (settings.ollama_base_url, settings.ollama_model)
    if _llm is None or _llm_settings_key != current_key:
        try:
            from langchain_ollama import OllamaLLM
        except ImportError:
            from langchain_community.llms import Ollama as OllamaLLM  # type: ignore
        _llm = OllamaLLM(model=settings.ollama_model, base_url=settings.ollama_base_url)
        _llm_settings_key = current_key
        logger.info("LLM: %s @ %s", settings.ollama_model, settings.ollama_base_url)
    return _llm

def _vector_search(query: str, k: int) -> list[tuple[Document, float]]:
    return _vectorstore.similarity_search_with_relevance_scores(f"query: {query}", k=k)


def _bm25_search(query: str, k: int) -> list[tuple[Document, float]]:
    if _bm25_index is None:
        return []
    tokens = re.split(r"[\s\-–—\.\/]+", query.lower())
    scores = _bm25_index.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    max_score = scores[top_idx[0]] if top_idx and scores[top_idx[0]] > 0 else 1.0
    return [
        (_bm25_docs[i], float(scores[i] / max_score))
        for i in top_idx if scores[i] > 0
    ]


def _rrf_merge(
    *ranked_lists: list[tuple[Document, float]],
    weights: list[float] | None = None,
    k_rrf: int = 60,
) -> list[tuple[Document, float]]:
    if weights is None:
        weights = [1.0 / len(ranked_lists)] * len(ranked_lists)

    def _key(doc: Document) -> str:
        return doc.page_content[:200]

    scores: dict[str, float] = {}
    docs_map: dict[str, Document] = {}

    for lst, w in zip(ranked_lists, weights):
        for rank, (doc, _) in enumerate(lst):
            key = _key(doc)
            scores[key] = scores.get(key, 0.0) + w / (k_rrf + rank + 1)
            docs_map[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(docs_map[key], score) for key, score in ranked]


def _hybrid_search(original_query: str, rewritten_query: str, k: int) -> list[tuple[Document, float]]:
    vec_rewritten = _vector_search(rewritten_query, k=k * 2)
    vec_original  = _vector_search(original_query,  k=k * 2)
    bm25_original = _bm25_search(original_query,    k=k * 2)

    if not bm25_original:
        logger.debug("BM25 недоступен, два векторных потока")
        return _rrf_merge(vec_rewritten, vec_original, weights=[0.6, 0.4])[:k]

    return _rrf_merge(
        vec_rewritten, vec_original, bm25_original,
        weights=[0.5, 0.2, 0.3],
    )[:k]

def _is_ntd_question(
    settings: Settings,
    question: str,
    history: list[dict] | None = None,
) -> bool:
    if history and len(question.strip()) < 30:
        logger.info("Классификатор: короткий вопрос с историей → ДА")
        return True
    t0 = time.perf_counter()
    prompt = (
        "Определи, относится ли вопрос к нормативно-технической документации, "
        "стандартам, регламентам, правилам безопасности, техническим требованиям "
        "или инженерным вопросам в области промышленности и энергетики.\n\n"
        "Отвечай ТОЛЬКО одним словом: ДА или НЕТ.\n\n"
        f"Вопрос: {question}\n\nОтвет:"
    )
    result = _get_llm(settings).invoke(prompt).strip().upper()
    logger.info("Классификатор: %r → %s (%.2fs)", question[:60], result, time.perf_counter() - t0)
    return "ДА" in result


def _build_context(
        docs_with_score: list[tuple[Document, float]],
        top_k: int,
        threshold: float,
) -> tuple[str, list[str]]:
    context_parts: list[str] = []
    sources: list[str] = []
    for doc, score in docs_with_score[:top_k]:
        if score < threshold:
            continue
        text = doc.page_content.replace("passage: ", "").strip()
        doc_name = doc.metadata.get("doc_name", "Неизвестный документ")
        context_parts.append(f"[{doc_name}]\n{text}")
        if doc_name not in sources:
            sources.append(doc_name)

    # Оставляем только топ-2 источника по количеству чанков
    from collections import Counter
    source_counts = Counter(
        doc.metadata.get("doc_name", "")
        for doc, score in docs_with_score[:top_k]
        if score >= threshold
    )
    top_sources = [s for s, _ in source_counts.most_common(2)]
    sources = [s for s in sources if s in top_sources]

    return "\n\n---\n\n".join(context_parts), sources

_PROMPT_WITH_CONTEXT = """Ты — интеллектуальный ассистент оперативного персонала на объектах атомной энергетики.

ПРАВИЛА:
1. Отвечай на основе предоставленных фрагментов НТД из раздела «КОНТЕКСТ».
2. Если информация есть — структурируй её в виде чётких пронумерованных шагов.
3. Если информация частичная — используй её и дополни логически, явно указывая что полные данные могут отсутствовать.
4. Никогда не выдумывай номера ГОСТов, регламенты, конкретные цифры, номера телефонов и процедур.
5. Пиши ТОЛЬКО на русском языке, никаких английских слов и фраз.
6. Если конкретные данные отсутствуют в контексте — не придумывай их, напиши "согласно установленному регламенту".
7. Учитывай историю диалога для корректной интерпретации уточняющих вопросов.

ИСТОРИЯ ДИАЛОГА:
{history}

КОНТЕКСТ ИЗ НТД:
{context}

ВОПРОС ОПЕРАТОРА:
{question}

ОТВЕТ (на русском языке, чётко, структурированно, пронумерованными шагами):"""


def _ask_with_context(settings: Settings, question: str, context: str, history: list[dict]) -> str:
    history_text = "\n".join(
        f"{'Оператор' if m['role'] == 'user' else 'Ассистент'}: {m['content']}"
        for m in history[:-1]
    ) if len(history) > 1 else "Начало диалога."
    t0 = time.perf_counter()
    result = _get_llm(settings).invoke(_PROMPT_WITH_CONTEXT.format(
        history=history_text, context=context, question=question,
    ))
    logger.info("Генерация: %.2fs", time.perf_counter() - t0)
    return result


def _ask_fallback(question: str) -> str:
    return (
        "Этот вопрос выходит за рамки нормативно-технической документации.\n\n"
        "Я отвечаю только на вопросы по НТД объектов атомной энергетики: "
        "регламенты, стандарты, технические требования, правила безопасности. "
        "Пожалуйста, переформулируйте вопрос."
    )

_DOC_ID_PATTERN = re.compile(
    r"\b(?:ГОСТ\s*Р?\s*|НП-|НРБ-|СТО\s+|ТУ\s+|ОСПОРБ|МУК\s*)[\d\w\-\.\/]+",
    re.IGNORECASE | re.UNICODE,
)

def _rewrite_query(settings: Settings, question: str, history: list[dict] | None = None) -> str:
    try:
        history_text = ""
        if history and len(history) > 1:
            prev = history[-2]
            if prev["role"] == "user":
                history_text = f"Предыдущий вопрос пользователя: {prev['content']}\n\n"

        preserved_ids = _DOC_ID_PATTERN.findall(question)
        preserve_hint = ""
        if preserved_ids:
            ids_str = ", ".join(preserved_ids)
            preserve_hint = (
                f"ВАЖНО: вопрос содержит обозначения документов ({ids_str}). "
                f"Включи их дословно в переформулированный запрос.\n\n"
            )

        t0 = time.perf_counter()
        prompt = (
            "Перефразируй вопрос для поиска в нормативной документации об атомных электростанциях. "
            "Если вопрос является уточнением предыдущего — включи тему предыдущего вопроса. "
            "Убери лишнюю конкретику (номера блоков, имена), оставь суть. "
            "Отвечай ТОЛЬКО на русском языке. "
            "Верни ТОЛЬКО переформулированный вопрос, без пояснений.\n\n"
            f"{preserve_hint}"
            f"{history_text}"
            f"Вопрос: {question}"
        )
        result = _get_llm(settings).invoke(prompt)
        logger.info("Переформулировка: %.2fs → %r", time.perf_counter() - t0, result.strip()[:80])
        return result
    except Exception:
        logger.warning("Ошибка переформулировки, используем оригинал")
        return question

def answer_question(
    settings: Settings,
    question: str,
    history: list[dict] | None = None,
) -> str:
    t_total = time.perf_counter()
    _init_vectorstore(settings)
    _init_bm25()
    history = history or []

    if not question or len(question.strip()) < 3:
        return "Запрос слишком короткий. Уточните вопрос."

    has_llm = bool(settings.ollama_base_url and settings.ollama_model)

    try:
        if has_llm and not _is_ntd_question(settings, question.strip(), history):
            logger.info("→ Классификатор: не НТД, fallback")
            return _ask_fallback(question)

        original = question.strip()
        rewritten = _rewrite_query(settings, original, history) if has_llm else original

        t_search = time.perf_counter()
        docs_with_score = _hybrid_search(original, rewritten.strip(), k=settings.rag_top_k)
        logger.info("Поиск: %d результатов (%.2fs)", len(docs_with_score), time.perf_counter() - t_search)
        for doc, score in docs_with_score:
            mark = "✓" if score >= _RRF_THRESHOLD else "✗"
            logger.info("  %s rrf=%.4f  doc=%s", mark, score, doc.metadata.get("doc_name", "?"))

        relevant_count = sum(1 for _, s in docs_with_score if s >= _RRF_THRESHOLD)
        context, sources = _build_context(docs_with_score, top_k=settings.rag_top_k, threshold=_RRF_THRESHOLD)

        disclaimer = (
            "\n\nПримечание: Подробные регламенты и технические нормативы "
            "объектов атомной энергетики могут быть засекречены или ограничены "
            "в открытом доступе."
        )

        if context and relevant_count >= settings.min_relevant_chunks:
            logger.info("→ Ответ по НТД (%d чанков)", relevant_count)
            answer = _ask_with_context(settings, question, context, history) if has_llm else context

            negative_markers = [
                "не представлена", "не содержится", "не найдено",
                "отсутствует", "не упоминается",
            ]
            has_answer = not any(m in answer.lower() for m in negative_markers)

            if has_answer:
                sources_str = "\n".join(f"  • {s}" for s in sources)
                result = f"{answer}\n\nИсточники:\n{sources_str}{disclaimer}"
            else:
                result = "В загруженных документах НТД информация по данному вопросу не найдена. Попробуйте переформулировать вопрос."
            logger.info("Итого: %.2fs", time.perf_counter() - t_total)
            return result

        logger.info("→ Контекст не найден (%.2fs)", time.perf_counter() - t_total)
        return (
            f"В базе НТД не найдено информации по запросу:\n\n"
            f"«{question}»\n\n"
            f"Возможно, нужный документ не проиндексирован. "
            f"Попробуйте переформулировать вопрос."
        )

    except Exception:
        logger.exception("Ошибка в RAG pipeline")
        return "Произошла ошибка при поиске по документации. Попробуйте позже."