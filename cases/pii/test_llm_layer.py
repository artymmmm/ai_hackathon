"""Синтетические тесты слияния спанов и LLM-слоя. Без pytest (не установлен в venv,
незачем тянуть зависимость ради нескольких assert'ов) — запуск:

    .venv/bin/python cases/pii/test_llm_layer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.llm import LLMClient, LLMConfig  # noqa: E402

from cases.pii.llm_layer import (  # noqa: E402
    _find_occurrences,
    detect_pii_llm,
    merge_spans,
)

_failures: list[str] = []


def check(name: str, cond: bool) -> None:
    status = "ok" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        _failures.append(name)


# ---------------------------------------------------------------- merge_spans

def test_merge_no_overlap():
    detectors = [{"start": 0, "end": 5, "label": "email", "text": "a@b.c"}]
    llm = [{"start": 10, "end": 15, "label": "first_name", "text": "Maria"}]
    merged = merge_spans(detectors, llm)
    check("merge: непересекающиеся спаны оба остаются", merged == sorted(
        detectors + llm, key=lambda s: s["start"]
    ))


def test_merge_detector_wins_on_overlap():
    # LLM ошибочно размечает часть номера телефона как "occupation" — детектор уже занял
    # ровно тот же диапазон номером телефона. Детектор должен победить целиком.
    detectors = [{"start": 5, "end": 20, "label": "phone_number", "text": "+1 415 555 0100"}]
    llm = [{"start": 5, "end": 20, "label": "occupation", "text": "+1 415 555 0100"}]
    merged = merge_spans(detectors, llm)
    check("merge: при полном совпадении диапазона побеждает детектор", merged == detectors)


def test_merge_partial_overlap_drops_llm_span_entirely():
    detectors = [{"start": 10, "end": 20, "label": "ssn", "text": "123-45-6789"}]
    llm = [{"start": 15, "end": 25, "label": "occupation", "text": "6789 analyst"}]
    merged = merge_spans(detectors, llm)
    check(
        "merge: частично пересекающийся LLM-спан отбрасывается целиком, не обрезается",
        merged == detectors,
    )


def test_merge_dedupes_identical_llm_ranges():
    llm = [
        {"start": 0, "end": 5, "label": "first_name", "text": "Maria"},
        {"start": 0, "end": 5, "label": "last_name", "text": "Maria"},  # тот же диапазон, другой лейбл
    ]
    merged = merge_spans([], llm)
    check("merge: дубликат по (start,end) среди LLM-спанов схлопывается в один", len(merged) == 1)
    check("merge: остаётся первый по порядку появления", merged[0]["label"] == "first_name")


def test_merge_preserves_order():
    detectors = [{"start": 20, "end": 25, "label": "email", "text": "a@b.c"}]
    llm = [{"start": 0, "end": 5, "label": "first_name", "text": "Maria"}]
    merged = merge_spans(detectors, llm)
    check("merge: результат отсортирован по start", [s["start"] for s in merged] == [0, 20])


# ------------------------------------------------------------ _find_occurrences

def test_find_occurrences_word_boundary():
    doc = "Ann met Anna at the park with Ann again."
    spans = _find_occurrences(doc, "Ann")
    found = [doc[s:e] for s, e in spans]
    check(
        "_find_occurrences: 'Ann' не матчит подстроку внутри 'Anna' (word boundary)",
        found == ["Ann", "Ann"],
    )


def test_find_occurrences_non_word_edges():
    doc = "Blood type: AB- confirmed by lab."
    spans = _find_occurrences(doc, "AB-")
    check("_find_occurrences: находит значение с небуквенным концом ('AB-')", spans == [(12, 15)])


def test_find_occurrences_not_present():
    check("_find_occurrences: пустой список, если подстрока отсутствует", _find_occurrences("hello world", "Maria") == [])


def test_find_occurrences_multiword():
    doc = "She lives in New York City. New York City is large."
    spans = _find_occurrences(doc, "New York City")
    check("_find_occurrences: находит все вхождения многословной фразы", len(spans) == 2)


# ------------------------------------------------------------------- detect_pii_llm

class _FakeLLM:
    """Минимальная подмена `LLMClient` — только `complete_json`, без сети/dry-run стаба."""

    def __init__(self, payload: dict):
        self._payload = payload

    def complete_json(self, prompt, *, example, system=None, model=None, **kw):
        return self._payload


def test_detect_pii_llm_filters_unknown_label():
    doc = "The patient, Maria Petrov, works as a cashier in Boston."
    fake = _FakeLLM({"entities": [
        {"text": "Maria Petrov", "label": "not_a_real_label"},
        {"text": "cashier", "label": "occupation"},
        {"text": "Boston", "label": "city"},
    ]})
    spans = detect_pii_llm(doc, fake, document_type="note", domain="healthcare")
    labels = {sp["label"] for sp in spans}
    check("detect_pii_llm: неизвестный лейбл отброшен", "not_a_real_label" not in labels)
    check("detect_pii_llm: валидные лейблы найдены", labels == {"occupation", "city"})
    for sp in spans:
        check(
            f"detect_pii_llm: span text совпадает с doc[start:end] ({sp['text']!r})",
            doc[sp["start"]:sp["end"]] == sp["text"],
        )


def test_detect_pii_llm_drops_text_not_in_doc():
    doc = "Employee works remotely."
    fake = _FakeLLM({"entities": [{"text": "Maria Petrov", "label": "first_name"}]})
    spans = detect_pii_llm(doc, fake)
    check("detect_pii_llm: находка с несуществующей подстрокой отброшена", spans == [])


def test_detect_pii_llm_malformed_response_returns_empty():
    fake = _FakeLLM({"not_entities": []})
    spans = detect_pii_llm("some text", fake)
    check("detect_pii_llm: ответ без ключа 'entities' -> пустой список", spans == [])


def test_detect_pii_llm_dry_run_does_not_crash():
    # Реальный контракт: LLMClient в dry-run режиме, без единого сетевого вызова.
    client = LLMClient(LLMConfig(dry_run=True))
    doc = "Patient Maria Petrov, occupation: cashier, lives in Boston, age 41."
    spans = detect_pii_llm(doc, client, document_type="note", domain="healthcare")
    check("detect_pii_llm: dry-run не падает и возвращает list", isinstance(spans, list))
    for sp in spans:
        check(
            "detect_pii_llm(dry-run): любой возвращённый спан валиден (text == doc[start:end])",
            doc[sp["start"]:sp["end"]] == sp["text"],
        )
    client.close()


# ----------------------------------------------------------- интеграция с detectors.py

def test_merge_end_to_end_with_real_detector():
    from cases.pii.detectors import detect_pii

    doc = "Email john@example.com, patient Maria Petrov, occupation: cashier."
    detector_spans = detect_pii(doc)
    llm_spans = [
        {"start": doc.index("Maria Petrov"), "end": doc.index("Maria Petrov") + len("Maria Petrov"),
         "label": "first_name", "text": "Maria Petrov"},
        {"start": doc.index("cashier"), "end": doc.index("cashier") + len("cashier"),
         "label": "occupation", "text": "cashier"},
    ]
    merged = merge_spans(detector_spans, llm_spans)
    labels = {sp["label"] for sp in merged}
    check("end-to-end: email найден детектором", "email" in labels)
    check("end-to-end: first_name/occupation добавлены LLM-слоем поверх детектора", {"first_name", "occupation"} <= labels)
    # непересекаемость финального списка
    spans_sorted = sorted(merged, key=lambda s: s["start"])
    ok = all(spans_sorted[i]["end"] <= spans_sorted[i + 1]["start"] for i in range(len(spans_sorted) - 1))
    check("end-to-end: финальный список спанов непересекающийся", ok)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print()
    if _failures:
        print(f"FAILED: {len(_failures)}/{len(tests)} тестов: {_failures}")
        return 1
    print(f"OK: все {len(tests)} тестов прошли")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
