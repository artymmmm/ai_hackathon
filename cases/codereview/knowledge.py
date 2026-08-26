"""Стек знаний для конфигурации A: CWE-карточки + правила CERT C + вывод flawfinder.

Все три источника — доменное знание, не зависящее от размеченного корпуса кейса 3 (в отличие
от retrieval в конфигурации B, `retrieval.py`) — поэтому переносятся на любой чужой C/C++ код.

Источники:
  - `kb/cwe_cards.json` — карточки CWE, извлечённые из официального каталога MITRE
    (`kb/cwec_v4.20.xml`, 969 записей) скриптом `build_cwe_kb.py`. Отбор: топ по частоте среди
    восстановленных vulnerable-лейблов + explicit confusable-кластер {119,787,125,120,20}
    (заявленная слабость: CWE accuracy 1/6, report/model_benchmark.md).
  - `kb/cert_c_rules.json` — 109 правил SEI CERT C с привязкой к CWE (готовый маппинг
    правило -> список CWE, извлечён из Taxonomy_Mapping каталога).
  - `static_analyzer.flawfinder_prompt_block` — текстовый regex-сканер (см. static_analyzer.py
    за проверкой, что это не компиляция/исполнение). ВАЖНО (измерено, см. improvements.md):
    recall против восстановленных лейблов на eval-150 — 18% (any level) / 4% (error level).
    Это слабый, но НЕНУЛЕВОЙ дополнительный сигнал, а не основа конфигурации A — основной вес
    несут CWE-карточки и CERT-правила, подтягиваемые по категориям `triage.py`, а не только
    по находкам flawfinder.

Подбор релевантного набора для конкретного фрагмента — по категориям сигнатурного триажа
(`triage.score_fragment`), которые не зависят от лейбла и не читают эталон, плюс CWE,
которые прямо назвал flawfinder (если есть).
"""

from __future__ import annotations

import json
from pathlib import Path

from cases.codereview.triage import score_fragment
from cases.codereview.static_analyzer import run_flawfinder

_ROOT = Path(__file__).resolve().parents[2]
_CWE_CARDS = json.loads((_ROOT / "kb" / "cwe_cards.json").read_text(encoding="utf-8"))
_CERT_RULES = json.loads((_ROOT / "kb" / "cert_c_rules.json").read_text(encoding="utf-8"))

# Категория сигнатурного триажа -> CWE, которые она типично сигнализирует. Грубая, но
# осмысленная эвристика (не лейбл, не эталон) — используется только для отбора, КАКИЕ карточки
# показать модели, не для вынесения вердикта.
_CATEGORY_TO_CWE = {
    "unsafe_func": ["120", "119"],
    "memcpy_unchecked": ["787", "125", "119"],
    "format_string": ["134"],
    "unchecked_alloc": ["476"],
    "double_free": ["415"],
    "use_after_free": ["416"],
    "size_int_overflow": ["190"],
}
# Буферный кластер, который модель путает сильнее всего — подтягивается либо когда триаж
# отметил буферную категорию, либо как fallback (доминирующий класс корпуса, см. §5
# research/case3_label_matching.md: CWE-119 — самый частый CWE среди vulnerable).
_CONFUSABLE_CLUSTER = ["119", "787", "125", "120", "20"]
_BUFFER_CATEGORIES = {"unsafe_func", "memcpy_unchecked", "size_int_overflow"}

MAX_CWE_CARDS = 5
MAX_CERT_RULES = 5


def candidate_cwe_ids(code: str) -> list[str]:
    scores = score_fragment(code)
    hit_categories = {c for c in scores.get("categories", "").split(";") if c}

    candidates: list[str] = []
    for cat in hit_categories:
        for cwe in _CATEGORY_TO_CWE.get(cat, []):
            if cwe not in candidates:
                candidates.append(cwe)

    if hit_categories & _BUFFER_CATEGORIES or not hit_categories:
        for cwe in _CONFUSABLE_CLUSTER:
            if cwe not in candidates:
                candidates.append(cwe)

    flaw_hits = run_flawfinder(code)
    for h in flaw_hits:
        for cwe_tag in h.get("cwe_ids", []):
            cwe_num = cwe_tag.removeprefix("CWE-")
            if cwe_num not in candidates:
                candidates.append(cwe_num)

    return candidates


def cwe_cards_block(code: str, max_cards: int = MAX_CWE_CARDS) -> str:
    ids = [c for c in candidate_cwe_ids(code) if c in _CWE_CARDS][:max_cards]
    if not ids:
        return ""
    lines = ["Справочник CWE, релевантных этому фрагменту (официальный каталог MITRE):"]
    for cid in ids:
        lines.append(f"- {_CWE_CARDS[cid]['prompt_text']}")
    return "\n".join(lines)


def cert_rules_block(code: str, max_rules: int = MAX_CERT_RULES) -> str:
    ids = candidate_cwe_ids(code)
    cwe_set = {f"CWE-{c}" for c in ids}
    matched = [
        (rule_id, r) for rule_id, r in _CERT_RULES.items()
        if cwe_set & set(r.get("cwes", []))
    ]
    if not matched:
        # fallback — самые общие и частые правила, если ничего конкретного не подобралось
        fallback_ids = ["ARR30-C", "EXP34-C", "MEM30-C", "STR31-C"]
        matched = [(rid, _CERT_RULES[rid]) for rid in fallback_ids if rid in _CERT_RULES]
    # Более специфичные правила (меньше связанных CWE — точнее применимость) — вперёд;
    # общие правила вида "Understand how arrays work" со списком из многих CWE менее полезны.
    matched.sort(key=lambda pair: len(pair[1].get("cwes", [])))
    matched = matched[:max_rules]
    if not matched:
        return ""
    lines = ["Применимые правила безопасного кодирования (SEI CERT C, привязка к CWE):"]
    for rid, r in matched:
        lines.append(f"- {rid} ({r['name']}) — связан с {', '.join(r.get('cwes', []))}")
    return "\n".join(lines)


def flawfinder_block(code: str) -> str:
    from cases.codereview.static_analyzer import flawfinder_prompt_block
    return flawfinder_prompt_block(code)


def knowledge_stack_block(code: str) -> str:
    """Объединённый блок конфигурации A: CWE-карточки + CERT-правила + flawfinder.
    Пустые под-блоки просто не добавляются (не заполняем промпт нулевым сигналом)."""
    parts = [b for b in (cwe_cards_block(code), cert_rules_block(code), flawfinder_block(code)) if b]
    return "\n\n".join(parts)
