"""Строит компактные карточки CWE из локального официального каталога MITRE
(`kb/cwec_v4.20.xml`, версия 4.20, 969 Weakness-записей) — без сети, без WebFetch по одной
странице (по указанию координатора: каталог уже скачан целиком).

Отбор CWE для карточек:
  - топ по частоте среди восстановленных `vulnerable`-лейблов (`research/case3_recovered_labels.csv`,
    колонка `cwe_id`) — то, что реально встречается в целевом распределении;
  - плюс explicit confusable-кластер {CWE-119, 787, 125, 120, 20} — заявленная слабость ревьюера
    (CWE accuracy 1/6, report/model_benchmark.md), эти CWE специально требуют разграничения.

Каждая карточка: Name, Description (официальный текст MITRE), Common_Consequences (кратко),
Related_Weaknesses (Nature + CWE_ID + имя — это и есть материал для различения смежных CWE),
1-2 Potential_Mitigations (кратко).

Результат: `kb/cwe_cards.json` (полные карточки, машиночитаемо) и
`cases/codereview/out/cwe_cards_prompt.json` (то же, но с уже отформатированным полем `prompt_text`
— готовый компактный текст под вставку в system/user промпт LLM).
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_CWEC_XML = _ROOT / "kb" / "cwec_v4.20.xml"
_GOLD_CSV = _ROOT / "research" / "case3_recovered_labels.csv"
_NS = {"c": "http://cwe.mitre.org/cwe-7"}

# Явный кластер, который модель путает (report/model_benchmark.md: CWE accuracy 1/6).
CONFUSABLE_CLUSTER = ["119", "787", "125", "120", "20"]


def _txt(el) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def top_cwe_ids_from_gold(n: int = 20) -> list[str]:
    df = pd.read_csv(_GOLD_CSV)
    ids = df["cwe_id"].dropna().astype(str).str.extract(r"(\d+)")[0].dropna()
    counts = Counter(ids)
    return [cwe_id for cwe_id, _ in counts.most_common(n)]


def parse_catalog(target_ids: set[str]) -> dict[str, dict]:
    root = ET.parse(_CWEC_XML).getroot()
    cards: dict[str, dict] = {}
    for w in root.findall(".//c:Weakness", _NS):
        wid = w.attrib.get("ID")
        if wid not in target_ids:
            continue
        name = w.attrib.get("Name", "")
        description = _txt(w.find("c:Description", _NS))
        extended = _txt(w.find("c:Extended_Description", _NS))

        consequences = []
        cc = w.find("c:Common_Consequences", _NS)
        if cc is not None:
            for consq in cc.findall("c:Consequence", _NS):
                scope = [_txt(s) for s in consq.findall("c:Scope", _NS)]
                impact = [_txt(s) for s in consq.findall("c:Impact", _NS)]
                consequences.append({"scope": scope, "impact": impact})

        related = []
        rw = w.find("c:Related_Weaknesses", _NS)
        if rw is not None:
            for r in rw.findall("c:Related_Weakness", _NS):
                related.append({"nature": r.attrib.get("Nature"), "cwe_id": r.attrib.get("CWE_ID")})

        mitigations = []
        pm = w.find("c:Potential_Mitigations", _NS)
        if pm is not None:
            for m in pm.findall("c:Mitigation", _NS)[:2]:
                mitigations.append(_txt(m.find("c:Description", _NS))[:220])

        cards[wid] = {
            "cwe_id": f"CWE-{wid}",
            "name": name,
            "description": description,
            "extended_description": extended[:400],
            "consequences": consequences,
            "related_weaknesses": related,
            "mitigations": mitigations,
        }
    return cards


# Ручная сводка различий для самого частого источника путаницы (119 — абстрактный родитель,
# 787/125/120 — конкретные дети). Каталог даёт связи Related_Weaknesses, но не готовое
# сравнительное предложение — это добавлено вручную поверх официальных описаний, коротко.
_MANUAL_DIFFERENTIATION = {
    "119": "Родовая (абстрактная) категория «нарушение границ буфера» — используй, только если "
           "непонятно, чтение это или запись; если понятно — указывай 787 или 125, они точнее.",
    "787": "Конкретно ЗАПИСЬ за пределы буфера (переполнение при записи, out-of-bounds write) — "
           "самый опасный случай, часто ведёт к RCE через перезапись соседней памяти/указателей.",
    "125": "Конкретно ЧТЕНИЕ за пределы буфера (out-of-bounds read) — обычно утечка памяти/инфы "
           "или падение, реже RCE. Не путать с 787: если функция только читает (не пишет) за "
           "границу — это 125, не 787/119.",
    "120": "Частный случай 787/119: конкретно небезопасное КОПИРОВАНИЕ без проверки размера входа "
           "(strcpy/sprintf/strcat без длины) — если видишь именно такой паттерн вызова, "
           "используй 120, а не общий 119.",
    "20": "Ещё более широкий родитель: любая недостаточная валидация входа, которая может "
          "порождать 119/787/125/120 (и не только их — инъекции, path traversal и т.п.). "
          "Используй 20, если проблема в самой проверке входа, а не в конкретном "
          "буферном паттерне ниже по потоку.",
}


def build_prompt_text(card: dict) -> str:
    lines = [f"{card['cwe_id']} — {card['name']}", card["description"]]
    if card.get("related_weaknesses"):
        rel = ", ".join(
            f"{r['nature']} CWE-{r['cwe_id']}" for r in card["related_weaknesses"] if r.get("cwe_id")
        )
        if rel:
            lines.append(f"Связанные CWE: {rel}.")
    manual = _MANUAL_DIFFERENTIATION.get(card["cwe_id"].removeprefix("CWE-"))
    if manual:
        lines.append(f"Отличие от смежных: {manual}")
    if card.get("mitigations"):
        lines.append(f"Типичное исправление: {card['mitigations'][0]}")
    return " ".join(lines)


def main() -> None:
    top_ids = top_cwe_ids_from_gold(20)
    target_ids = set(top_ids) | set(CONFUSABLE_CLUSTER)
    print(f"Целевые CWE ({len(target_ids)}): {sorted(target_ids, key=int)}")

    cards = parse_catalog(target_ids)
    missing = target_ids - set(cards)
    if missing:
        print(f"ВНИМАНИЕ: не найдены в каталоге: {sorted(missing)}")

    for wid, card in cards.items():
        card["prompt_text"] = build_prompt_text(card)

    kb_dir = _ROOT / "kb"
    kb_dir.mkdir(parents=True, exist_ok=True)
    (kb_dir / "cwe_cards.json").write_text(
        json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    compact = {c["cwe_id"]: c["prompt_text"] for c in cards.values()}
    (out_dir / "cwe_cards_prompt.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"{len(cards)} карточек -> kb/cwe_cards.json, out/cwe_cards_prompt.json")
    for wid in sorted(cards, key=int):
        print(f"  CWE-{wid}: {cards[wid]['name']}")


if __name__ == "__main__":
    main()
