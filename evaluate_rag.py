
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ntd_bot.config import load_settings
from ntd_bot.rag import answer_question


TEST_CASES = [
    # По теме НТД
    ("Q01", "Какие марки нержавеющих сталей регламентирует ГОСТ 5632?",
     "found", "НТД — металлы"),

    ("Q02", "Какие технические требования предъявляются к трубам бесшовным из коррозионностойкой стали по ГОСТ 24030-80?",
     "found", "НТД — трубы"),

    ("Q03", "Какие требования к болтам и шпилькам для фланцевых соединений АЭС установлены в ГОСТ 23304-78?",
     "found", "НТД — крепёж АЭС"),

    ("Q04", "Какой состав и чистота аргона допускается по ГОСТ 10157-2016?",
     "found", "НТД — газы"),

    ("Q05", "Какие марки деформируемых титановых сплавов указаны в ГОСТ 19807-91?",
     "found", "НТД — металлы"),

    ("Q06", "Какие требования к покрытым металлическим электродам для ручной дуговой сварки установлены в ГОСТ 9466-75?",
     "found", "НТД — сварка"),

    ("Q07", "Что регулирует НП-082-07 в части ядерной безопасности реакторных установок АЭС?",
     "found", "НТД — ядерная безопасность"),

    ("Q08", "Какие нормы радиационной безопасности установлены в НРБ-99/2009?",
     "found", "НТД — радиация"),

    ("Q09", "Какие требования к трубопроводам атомных станций содержит ГОСТ Р 58328-2018?",
     "found", "НТД — трубопроводы АЭС"),

    ("Q10", "Какие требования к противопожарной защите АЭС установлены в НП-068-05?",
     "found", "НТД — пожарная безопасность"),

    ("Q11", "Каковы требования к водородной взрывозащите по НП-010-16?",
     "found", "НТД — взрывозащита"),

    ("Q12", "Какие марки латуней обрабатываемых давлением регламентирует ГОСТ 15527-2004?",
     "found", "НТД — металлы"),

    ("Q13", "Какие требования к стальной сварочной проволоке содержит ГОСТ 2246-70?",
     "found", "НТД — сварка"),

    # Граничные случаи
    ("Q14", "Какие требования к безопасности лифтов?",
     "found", "Граничный — ТР ТС 011"),

    ("Q15", "Какие требования к трубам из стали 08Х18Н10Т для трубопроводов АЭС?",
     "found", "Граничный — марка стали"),

    ("Q16", "Какие требования к поковкам из коррозионностойких сталей по ГОСТ 25054-81?",
     "found", "НТД — поковки"),

    # Не по теме — отклонить
    ("Q17", "Какая сегодня погода в Москве?",
     "rejected", "Не НТД — погода"),

    ("Q18", "Кто выиграл чемпионат мира по футболу в 2022 году?",
     "rejected", "Не НТД — спорт"),

    ("Q19", "Как приготовить борщ?",
     "rejected", "Не НТД — кулинария"),

    ("Q20", "Сколько стоит доллар сегодня?",
     "rejected", "Не НТД — курс валют"),
]

# Маркеры отклонённого ответа
REJECTION_MARKERS = [
    "выходит за рамки",
    "нормативно-технической документации",
    "переформулируйте",
    "не относится к нтд",
]

# Маркеры ненайденного ответа
NOT_FOUND_MARKERS = [
    "не найдено информации",
    "не найдено",
    "переформулировать вопрос",
    "не проиндексирован",
]


def classify_answer(answer: str, expected: str) -> tuple[str, bool]:
    lower = answer.lower()

    is_rejected = any(m in lower for m in REJECTION_MARKERS)
    is_not_found = any(m in lower for m in NOT_FOUND_MARKERS)
    has_sources = "источники" in lower or "источник" in lower

    if is_rejected:
        status = "ОТКЛОНЁН"
        correct = (expected == "rejected")
    elif is_not_found:
        status = "НЕ НАЙДЕНО"
        correct = (expected == "rejected")
    elif has_sources:
        status = "ОТВЕТ С ИСТОЧНИКОМ"
        correct = (expected == "found")
    else:
        status = "ОТВЕТ БЕЗ ИСТОЧНИКА"
        correct = (expected == "found")

    return status, correct


def main():
    print("=" * 70)
    print("ОЦЕНКА КАЧЕСТВА RAG-ПАЙПЛАЙНА")
    print("=" * 70)

    settings = load_settings()

    results = []
    total = len(TEST_CASES)
    correct_count = 0
    found_with_source = 0
    correctly_rejected = 0
    false_negatives = 0
    false_positives = 0

    lines = []
    lines.append("РЕЗУЛЬТАТЫ ОЦЕНКИ КАЧЕСТВА RAG-ПАЙПЛАЙНА\n")
    lines.append("=" * 70 + "\n\n")

    for qid, question, expected, category in TEST_CASES:
        print(f"\n[{qid}] {question[:60]}...")
        t0 = time.perf_counter()
        try:
            answer = answer_question(settings, question)
        except Exception as e:
            answer = f"ОШИБКА: {e}"
        elapsed = time.perf_counter() - t0

        status, correct = classify_answer(answer, expected)
        if correct:
            correct_count += 1
        if status == "ОТВЕТ С ИСТОЧНИКОМ":
            found_with_source += 1
        if status == "ОТКЛОНЁН" and expected == "rejected":
            correctly_rejected += 1
        if expected == "found" and status in ("ОТКЛОНЁН", "НЕ НАЙДЕНО"):
            false_negatives += 1
        if expected == "rejected" and status == "ОТВЕТ С ИСТОЧНИКОМ":
            false_positives += 1

        mark = "✓" if correct else "✗"
        print(f"  {mark} [{status}] ({elapsed:.1f}s)")

        line = f"[{qid}] {mark} {category}\n"
        line += f"Вопрос: {question}\n"
        line += f"Статус: {status} | Ожидалось: {expected} | Время: {elapsed:.1f}s\n"
        line += f"Ответ (первые 300 символов):\n{answer[:300]}\n"
        line += "-" * 70 + "\n"
        lines.append(line)

        results.append((qid, correct, status, elapsed))

    ntd_questions = [r for r in results if TEST_CASES[results.index(r)][2] == "found"]
    rejected_questions = [r for r in results if TEST_CASES[results.index(r)][2] == "rejected"]

    accuracy = correct_count / total * 100
    precision_ntd = found_with_source / max(len([t for t in TEST_CASES if t[2] == "found"]), 1) * 100
    classifier_acc = correctly_rejected / max(len([t for t in TEST_CASES if t[2] == "rejected"]), 1) * 100
    avg_time = sum(r[3] for r in results) / total

    summary = f"""
{'=' * 70}
СВОДНЫЕ РЕЗУЛЬТАТЫ
{'=' * 70}
Всего вопросов      : {total}
Корректных ответов  : {correct_count} / {total}  ({accuracy:.1f}%)

По категориям:
  Вопросы по НТД    : {len([t for t in TEST_CASES if t[2]=='found'])} вопросов
    - Ответ с источником : {found_with_source}
    - Ложноотрицательных : {false_negatives} (ожидался ответ, не найдено)

  Не по теме        : {len([t for t in TEST_CASES if t[2]=='rejected'])} вопросов
    - Правильно отклонено : {correctly_rejected}
    - Ложноположительных  : {false_positives} (должен был отклонить)

Точность поиска по НТД  : {precision_ntd:.1f}%
Точность классификатора : {classifier_acc:.1f}%
Общая точность          : {accuracy:.1f}%
Среднее время ответа    : {avg_time:.1f}s
{'=' * 70}
"""
    print(summary)
    lines.append(summary)

    output_path = Path(__file__).parent / "evaluate_rag_results.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\nПодробные результаты сохранены в: {output_path}")


if __name__ == "__main__":
    main()
