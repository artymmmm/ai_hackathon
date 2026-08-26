"""Self-audit loop: анонимизированный текст прогоняется теми же детекторами ещё раз.

Важный нюанс формата этой анонимизации: подстановка **format-preserving** — псевдоним номера
карты снова выглядит как номер карты, псевдо-ФИО снова выглядит как ФИО. Поэтому наивная
проверка "детектор нашёл PII в анонимизированном тексте → утечка" здесь даёт ложную тревогу:
детектор ОБЯЗАН находить что-то похожее на PII в тексте, иначе документ перестал бы быть
правдоподобным. Настоящая утечка — это когда повторный прогон находит **исходное** значение
(а не псевдоним), т.е. подстановка эту сущность не заменила.

Поэтому self-audit сверяет то, что находится в анонимизированном тексте, с множеством исходных
значений (по `original_spans`, нормализованных):

- `true_leak_rate` — доля повторно найденных спанов, чей текст совпадает с ОРИГИНАЛЬНЫМ
  значением (детектор снова нашёл ту же реальную сущность → подстановка её не затронула).
  Целевое значение — 0.
- `literal_leak_rate` — более жёсткая проверка без детектора вообще: ищем оригинальный текст
  каждого span подстрокой в анонимизированном тексте. Ловит и то, что детектор мог пропустить
  при повторном прогоне. Тоже должно быть 0.
- `residual_pii_shaped_rate` — сколько ПОСЛЕ анонимизации всё ещё выглядит как PII (это ожидаемо
  большая доля — так и задумано форматосохраняющей подстановкой), но это FAKE-данные, не утечка.
  Показываем отдельно, чтобы не путать с настоящей утечкой.
"""

from __future__ import annotations

import re

from cases.pii.detectors import detect_pii


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def find_leaks(anonymized_text: str, original_spans: list[dict]) -> dict:
    original_values_norm = {_normalize(sp["text"]) for sp in original_spans if sp["text"]}
    redetected = detect_pii(anonymized_text)
    true_leaks = [sp for sp in redetected if _normalize(sp["text"]) in original_values_norm]
    residual_pii_shaped = [sp for sp in redetected if _normalize(sp["text"]) not in original_values_norm]
    return {"true_leaks": true_leaks, "residual_pii_shaped": residual_pii_shaped}


# Значения короче этого длины (напр. 3-значный CVV) слишком часто встречаются как случайная
# подстрока где-то ещё в тексте (в номере карты, ID и т.п.) — литеральная проверка на таких
# длинах даёт шум, а не сигнал об утечке. Тот же порог использует `aliaser._MIN_REOCCURRENCE_LEN`.
_MIN_LITERAL_CHECK_LEN = 4


def audit(original_text: str, anonymized_text: str, original_spans: list[dict]) -> dict:
    """self-audit по одному документу."""
    detected = find_leaks(anonymized_text, original_spans)
    true_leaks = detected["true_leaks"]
    residual = detected["residual_pii_shaped"]

    literal_leftovers = [
        sp
        for sp in original_spans
        if sp["text"] and len(sp["text"]) >= _MIN_LITERAL_CHECK_LEN and sp["text"] in anonymized_text
    ]

    n = len(original_spans)
    return {
        "n_original_spans": n,
        "n_true_leaks": len(true_leaks),
        "true_leak_rate": round(len(true_leaks) / n, 4) if n else 0.0,
        "true_leaks": true_leaks,
        "n_literal_leftovers": len(literal_leftovers),
        "literal_leak_rate": round(len(literal_leftovers) / n, 4) if n else 0.0,
        "literal_leftovers": literal_leftovers,
        "n_residual_pii_shaped": len(residual),
        "residual_pii_shaped_rate": round(len(residual) / n, 4) if n else 0.0,
    }


def audit_batch(records: list[dict]) -> dict:
    """`records`: список `{'original_text','anonymized_text','spans'}`. Агрегирует по выборке."""
    total_spans = 0
    total_true_leaks = 0
    total_literal = 0
    total_residual = 0
    docs_with_leak = 0

    for rec in records:
        report = audit(rec["original_text"], rec["anonymized_text"], rec["spans"])
        total_spans += report["n_original_spans"]
        total_true_leaks += report["n_true_leaks"]
        total_literal += report["n_literal_leftovers"]
        total_residual += report["n_residual_pii_shaped"]
        if report["n_true_leaks"] or report["n_literal_leftovers"]:
            docs_with_leak += 1

    n_docs = len(records)
    return {
        "n_docs": n_docs,
        "docs_with_leak": docs_with_leak,
        "doc_leak_rate": round(docs_with_leak / n_docs, 4) if n_docs else 0.0,
        "total_spans": total_spans,
        "total_true_leaks": total_true_leaks,
        "true_leak_rate": round(total_true_leaks / total_spans, 4) if total_spans else 0.0,
        "total_literal_leftovers": total_literal,
        "literal_leak_rate": round(total_literal / total_spans, 4) if total_spans else 0.0,
        "total_residual_pii_shaped": total_residual,
        "residual_pii_shaped_rate": round(total_residual / total_spans, 4) if total_spans else 0.0,
    }
