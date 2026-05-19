"""
Оценка качества RETRIEVAL-подсистемы RAG через классические метрики
Information Retrieval: Recall@k и Mean Reciprocal Rank (MRR).

В отличие от LLM-as-a-Judge, этот метод:
- объективен (документ либо нашёлся, либо нет)
- воспроизводим (одинаковый результат при каждом запуске)
- не требует LLM (быстрый прогон, ~1 минута на 10 вопросов)
- использует стандартные метрики из Information Retrieval

Запуск:
    python evaluate_retrieval.py

Результаты сохраняются в evaluate_retrieval_results.txt
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ntd_bot.config import load_settings
from ntd_bot.rag import (
    _init_vectorstore,
    _init_bm25,
    _hybrid_search,
)


TEST_CASES = [
    {
        "qid": "Q01",
        "question": "Какие марки нержавеющих сталей регламентирует ГОСТ 5632-2014?",
        "expected_doc_substring": "ГОСТ 5632-2014",
    },
    {
        "qid": "Q02",
        "question": "Какие технические требования предъявляются к бесшовным холодно- и теплодеформированным трубам из коррозионно-стойкой стали по ГОСТ 9941-81?",
        "expected_doc_substring": "ГОСТ 9941-81",
    },
    {
        "qid": "Q03",
        "question": "Какие технические требования к болтам, шпилькам, гайкам и шайбам для фланцевых соединений по ГОСТ 20700-75?",
        "expected_doc_substring": "ГОСТ 20700-75",
    },
    {
        "qid": "Q04",
        "question": "Какой состав и чистота аргона допускается по ГОСТ 10157-2016?",
        "expected_doc_substring": "ГОСТ 10157-2016",
    },
    {
        "qid": "Q05",
        "question": "Какие марки деформируемых титановых сплавов указаны в ГОСТ 19807-91?",
        "expected_doc_substring": "ГОСТ 19807-91",
    },
    {
        "qid": "Q06",
        "question": "Какие требования к покрытым металлическим электродам для ручной дуговой сварки установлены в ГОСТ 9466-75?",
        "expected_doc_substring": "ГОСТ 9466-75",
    },
    {
        "qid": "Q07",
        "question": "Что регулирует НП-082-07 в части ядерной безопасности реакторных установок АЭС?",
        "expected_doc_substring": "НП-082-07",
    },
    {
        "qid": "Q08",
        "question": "Какие нормы радиационной безопасности установлены в НРБ-99/2009?",
        "expected_doc_substring": "НРБ",
    },
    {
        "qid": "Q09",
        "question": "Какие требования к трубопроводам атомных станций содержит ГОСТ Р 58328-2018?",
        "expected_doc_substring": "ГОСТ Р 58328-2018",
    },
    {
        "qid": "Q10",
        "question": "Каковы требования к водородной взрывозащите по НП-010-16?",
        "expected_doc_substring": "НП-010-16",
    },
]


def find_doc_rank(results, expected_substring: str) -> int:
    """Возвращает позицию (1-based) первого документа, в имени которого
    встречается expected_substring. Если не найден — возвращает 0.
    """
    for rank, (doc, _score) in enumerate(results, start=1):
        doc_name = doc.metadata.get("doc_name", "")
        if expected_substring.lower() in doc_name.lower():
            return rank
    return 0


def evaluate_one(settings, question: str, expected: str, k: int) -> dict:
    """Прогоняет один вопрос через гибридный поиск и возвращает позицию
    ожидаемого документа.
    """
    t0 = time.perf_counter()
    results = _hybrid_search(question, question, k=k)
    elapsed = time.perf_counter() - t0

    rank = find_doc_rank(results, expected)

    found_docs = []
    for doc, score in results[:5]:
        doc_name = doc.metadata.get("doc_name", "?")
        found_docs.append(f"  rank={results.index((doc, score))+1} score={score:.4f} {doc_name}")

    return {
        "rank": rank,
        "elapsed": elapsed,
        "top_docs": found_docs,
        "total_results": len(results),
    }


def main():
    print("=" * 70)
    print("ОЦЕНКА КАЧЕСТВА RETRIEVAL — Recall@k и MRR")
    print("=" * 70)

    settings = load_settings()

    print("Инициализация поиска...")
    _init_vectorstore(settings)
    _init_bm25()
    print("Готово\n")

    K_VALUES = [1, 3, 5, 10]
    K_MAX = max(K_VALUES)

    results = []
    lines = [
        "РЕЗУЛЬТАТЫ ОЦЕНКИ RETRIEVAL — Recall@k и MRR\n",
        "Метрики Information Retrieval: проверка позиции эталонного документа\n",
        "=" * 70 + "\n\n",
    ]

    for case in TEST_CASES:
        qid = case["qid"]
        question = case["question"]
        expected = case["expected_doc_substring"]

        print(f"[{qid}] {question[:60]}...")
        print(f"      Ожидаемый документ: {expected}")

        r = evaluate_one(settings, question, expected, k=K_MAX)

        rank = r["rank"]
        if rank > 0:
            print(f"      Найден на позиции: {rank}  ({r['elapsed']:.1f}s)")
        else:
            print(f"      НЕ НАЙДЕН в топ-{K_MAX}  ({r['elapsed']:.1f}s)")

        line = f"[{qid}] {question}\n"
        line += f"Ожидаемый документ: {expected}\n"
        if rank > 0:
            line += f"Найден на позиции : {rank}\n"
        else:
            line += f"Найден на позиции : НЕ НАЙДЕН в топ-{K_MAX}\n"
        line += f"Время поиска      : {r['elapsed']:.2f}s\n"
        line += f"Топ-5 результатов поиска:\n"
        for doc_info in r["top_docs"]:
            line += f"{doc_info}\n"
        line += "-" * 70 + "\n"
        lines.append(line)

        results.append({
            "qid": qid,
            "rank": rank,
            "elapsed": r["elapsed"],
        })

    n = len(results)

    recall_at = {}
    for k in K_VALUES:
        hits = sum(1 for r in results if 0 < r["rank"] <= k)
        recall_at[k] = hits / n * 100

    reciprocal_ranks = [
        1.0 / r["rank"] if r["rank"] > 0 else 0.0
        for r in results
    ]
    mrr = sum(reciprocal_ranks) / n

    ranks_found = [r["rank"] for r in results if r["rank"] > 0]
    avg_rank = sum(ranks_found) / len(ranks_found) if ranks_found else 0.0
    not_found = sum(1 for r in results if r["rank"] == 0)

    avg_time = sum(r["elapsed"] for r in results) / n

    summary = f"""
{'=' * 70}
СВОДНЫЕ РЕЗУЛЬТАТЫ — RETRIEVAL METRICS
{'=' * 70}
Вопросов оценено : {n}

Recall@k (доля вопросов, где эталонный документ попал в топ-k):
  Recall@1   : {recall_at[1]:.1f}%   ({int(recall_at[1]*n/100)} из {n})
  Recall@3   : {recall_at[3]:.1f}%   ({int(recall_at[3]*n/100)} из {n})
  Recall@5   : {recall_at[5]:.1f}%   ({int(recall_at[5]*n/100)} из {n})
  Recall@10  : {recall_at[10]:.1f}%   ({int(recall_at[10]*n/100)} из {n})

Mean Reciprocal Rank (MRR) : {mrr:.4f}
  Интерпретация: чем ближе к 1.0, тем выше в среднем находится эталонный
  документ. 1.0 = всегда на 1-м месте, 0.5 = в среднем на 2-м месте.

Средняя позиция найденных     : {avg_rank:.2f}
Не найдено в топ-{K_MAX}            : {not_found} из {n}
Среднее время поиска (на запрос) : {avg_time:.2f}s
{'=' * 70}
"""
    print(summary)
    lines.append(summary)

    output_path = Path(__file__).parent / "evaluate_retrieval_results.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Подробные результаты сохранены в: {output_path}")


if __name__ == "__main__":
    main()
