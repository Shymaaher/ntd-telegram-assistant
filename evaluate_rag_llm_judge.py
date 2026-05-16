

import sys
import time
import json
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ntd_bot.config import load_settings
from ntd_bot.rag import (
    answer_question,
    _init_vectorstore,
    _init_bm25,
    _hybrid_search,
    _build_context,
    _get_llm,
    _RRF_THRESHOLD,
)
TEST_QUESTIONS = [
    "Какие марки нержавеющих сталей регламентирует ГОСТ 5632?",
    "Какие технические требования предъявляются к трубам бесшовным из коррозионностойкой стали по ГОСТ 24030-80?",
    "Какие требования к болтам и шпилькам для фланцевых соединений АЭС установлены в ГОСТ 23304-78?",
    "Какой состав и чистота аргона допускается по ГОСТ 10157-2016?",
    "Какие марки деформируемых титановых сплавов указаны в ГОСТ 19807-91?",
    "Какие требования к покрытым металлическим электродам для ручной дуговой сварки установлены в ГОСТ 9466-75?",
    "Что регулирует НП-082-07 в части ядерной безопасности реакторных установок АЭС?",
    "Какие нормы радиационной безопасности установлены в НРБ-99/2009?",
    "Какие требования к трубопроводам атомных станций содержит ГОСТ Р 58328-2018?",
    "Каковы требования к водородной взрывозащите по НП-010-16?",
]

FAITHFULNESS_PROMPT = """Ты — строгий эксперт по оценке качества ответов системы поиска по документам.

Оцени, насколько ОТВЕТ основан на предоставленном КОНТЕКСТЕ.
Критерии оценки:
5 — ответ полностью основан на контексте, нет никаких утверждений вне контекста
4 — ответ преимущественно из контекста, есть незначительные дополнения
3 — ответ частично основан на контексте, есть заметные добавления от себя
2 — ответ слабо связан с контекстом, много информации вне контекста
1 — ответ не основан на контексте, явные галлюцинации

ВОПРОС: {question}

КОНТЕКСТ ИЗ ДОКУМЕНТОВ:
{context}

ОТВЕТ СИСТЕМЫ:
{answer}

Ответь ТОЛЬКО в формате JSON: {{"score": <число от 1 до 5>, "reason": "<одно предложение>"}}"""

RELEVANCE_PROMPT = """Ты — строгий эксперт по оценке качества ответов.

Оцени, насколько ОТВЕТ соответствует ВОПРОСУ и отвечает на него по существу.
Критерии оценки:
5 — ответ полностью и точно отвечает на вопрос
4 — ответ в основном отвечает на вопрос, есть небольшие отклонения
3 — ответ частично отвечает на вопрос
2 — ответ слабо связан с вопросом
1 — ответ не отвечает на вопрос

ВОПРОС: {question}

ОТВЕТ СИСТЕМЫ:
{answer}

Ответь ТОЛЬКО в формате JSON: {{"score": <число от 1 до 5>, "reason": "<одно предложение>"}}"""

CONTEXT_QUALITY_PROMPT = """Ты — строгий эксперт по оценке качества поиска по документам.

Оцени, насколько найденные ФРАГМЕНТЫ ДОКУМЕНТОВ релевантны ВОПРОСУ.
Критерии оценки:
5 — все фрагменты напрямую относятся к вопросу
4 — большинство фрагментов релевантны, есть незначительно нерелевантные
3 — часть фрагментов релевантна, часть нет
2 — большинство фрагментов нерелевантны
1 — фрагменты не относятся к вопросу

ВОПРОС: {question}

НАЙДЕННЫЕ ФРАГМЕНТЫ:
{context}

Ответь ТОЛЬКО в формате JSON: {{"score": <число от 1 до 5>, "reason": "<одно предложение>"}}"""


def parse_judge_response(response: str) -> tuple[float, str]:
    """Извлекает оценку и причину из ответа судьи."""
    try:
        # Ищем JSON в ответе
        match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            score = float(data.get("score", 3))
            reason = data.get("reason", "")
            return min(max(score, 1), 5), reason
    except Exception:
        pass
    # Fallback — ищем просто число
    match = re.search(r'\b([1-5])\b', response)
    if match:
        return float(match.group(1)), "не удалось разобрать причину"
    return 3.0, "ошибка парсинга"


def get_context_for_question(settings, question: str) -> tuple[str, list[str], float]:
    """Возвращает контекст, источники и средний RRF-скор для вопроса."""
    _init_vectorstore(settings)
    _init_bm25()

    docs_with_score = _hybrid_search(question, question, k=settings.rag_top_k)
    relevant = [(d, s) for d, s in docs_with_score if s >= _RRF_THRESHOLD]

    context, sources = _build_context(
        docs_with_score,
        top_k=settings.rag_top_k,
        threshold=_RRF_THRESHOLD,
    )
    avg_score = sum(s for _, s in relevant) / len(relevant) if relevant else 0.0
    return context, sources, avg_score


def evaluate_question(settings, question: str) -> dict:
    """Прогоняет один вопрос через RAG и оценивает тремя метриками."""
    print(f"\n  Получаю ответ...", end=" ", flush=True)
    t0 = time.perf_counter()
    answer = answer_question(settings, question)
    elapsed = time.perf_counter() - t0
    print(f"({elapsed:.1f}s)")

    # Получаем контекст отдельно для передачи судье
    context, sources, avg_rrf = get_context_for_question(settings, question)

    if not context or "не найдено" in answer.lower():
        return {
            "question": question,
            "answer": answer,
            "sources": [],
            "avg_rrf_score": 0.0,
            "faithfulness": 1.0,
            "faithfulness_reason": "контекст не найден",
            "relevance": 1.0,
            "relevance_reason": "контекст не найден",
            "context_quality": 1.0,
            "context_quality_reason": "контекст не найден",
            "elapsed": elapsed,
        }

    llm = _get_llm(settings)

    # Оценка Faithfulness
    print(f"  Оцениваю faithfulness...", end=" ", flush=True)
    faith_resp = llm.invoke(FAITHFULNESS_PROMPT.format(
        question=question,
        context=context[:2000],
        answer=answer[:1000],
    ))
    faith_score, faith_reason = parse_judge_response(faith_resp)
    print(f"{faith_score:.0f}/5")

    # Оценка Answer Relevance
    print(f"  Оцениваю relevance...", end=" ", flush=True)
    rel_resp = llm.invoke(RELEVANCE_PROMPT.format(
        question=question,
        answer=answer[:1000],
    ))
    rel_score, rel_reason = parse_judge_response(rel_resp)
    print(f"{rel_score:.0f}/5")

    # Оценка Context Quality
    print(f"  Оцениваю context quality...", end=" ", flush=True)
    ctx_resp = llm.invoke(CONTEXT_QUALITY_PROMPT.format(
        question=question,
        context=context[:2000],
    ))
    ctx_score, ctx_reason = parse_judge_response(ctx_resp)
    print(f"{ctx_score:.0f}/5")

    return {
        "question": question,
        "answer": answer[:500],
        "sources": sources,
        "avg_rrf_score": avg_rrf,
        "faithfulness": faith_score,
        "faithfulness_reason": faith_reason,
        "relevance": rel_score,
        "relevance_reason": rel_reason,
        "context_quality": ctx_score,
        "context_quality_reason": ctx_reason,
        "elapsed": elapsed,
    }


def main():
    print("=" * 70)
    print("ОЦЕНКА КАЧЕСТВА RAG — LLM-as-a-Judge")
    print("=" * 70)

    settings = load_settings()

    if not settings.ollama_base_url or not settings.ollama_model:
        print("ОШИБКА: не настроен Ollama (OLLAMA_BASE_URL, OLLAMA_MODEL)")
        return

    results = []
    lines = ["РЕЗУЛЬТАТЫ ОЦЕНКИ RAG — LLM-as-a-Judge\n", "=" * 70 + "\n\n"]

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i:02d}/{len(TEST_QUESTIONS)}] {question[:65]}...")
        result = evaluate_question(settings, question)
        results.append(result)

        line = (
            f"[Q{i:02d}] {question}\n"
            f"Faithfulness    : {result['faithfulness']:.0f}/5 — {result['faithfulness_reason']}\n"
            f"Answer Relevance: {result['relevance']:.0f}/5 — {result['relevance_reason']}\n"
            f"Context Quality : {result['context_quality']:.0f}/5 — {result['context_quality_reason']}\n"
            f"Avg RRF score   : {result['avg_rrf_score']:.4f}\n"
            f"Источники: {', '.join(result['sources'][:3])}\n"
            f"Время: {result['elapsed']:.1f}s\n"
            + "-" * 70 + "\n"
        )
        lines.append(line)

    n = len(results)
    avg_faith = sum(r["faithfulness"] for r in results) / n
    avg_rel = sum(r["relevance"] for r in results) / n
    avg_ctx = sum(r["context_quality"] for r in results) / n
    avg_rrf = sum(r["avg_rrf_score"] for r in results) / n
    avg_time = sum(r["elapsed"] for r in results) / n

    # Доля оценок >= 4 (хорошо и отлично)
    faith_good = sum(1 for r in results if r["faithfulness"] >= 4) / n * 100
    rel_good = sum(1 for r in results if r["relevance"] >= 4) / n * 100
    ctx_good = sum(1 for r in results if r["context_quality"] >= 4) / n * 100

    summary = f"""
{'=' * 70}
СВОДНЫЕ РЕЗУЛЬТАТЫ (LLM-as-a-Judge, шкала 1-5)
{'=' * 70}
Вопросов оценено : {n}

Средние оценки:
  Faithfulness     (нет галлюцинаций)  : {avg_faith:.2f} / 5  ({faith_good:.0f}% оценок ≥ 4)
  Answer Relevance (ответ по существу) : {avg_rel:.2f} / 5  ({rel_good:.0f}% оценок ≥ 4)
  Context Quality  (качество контекста): {avg_ctx:.2f} / 5  ({ctx_good:.0f}% оценок ≥ 4)

Средний RRF-score найденных чанков : {avg_rrf:.4f}
Среднее время ответа               : {avg_time:.1f}s
{'=' * 70}
"""
    print(summary)
    lines.append(summary)

    output_path = Path(__file__).parent / "evaluate_rag_llm_judge_results.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Подробные результаты сохранены в: {output_path}")


if __name__ == "__main__":
    main()
