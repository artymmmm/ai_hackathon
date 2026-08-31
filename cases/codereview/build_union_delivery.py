"""Собирает поставочную выгрузку кейса 3 в конфигурации ОБЪЕДИНЕНИЯ трёх слоёв.

Вердикт `vulnerable` ставится, если фрагмент пометил хотя бы один слой: LLM-ревьюер,
flawfinder или сигнатурный триаж. Пересечение проверено и заметно хуже (J +0.053 против
+0.187) — слои находят разное, поэтому складываются, а не сверяются.

Фрагментам, которые нашла только статика, CWE/механизм/патч добираются отдельным проходом
(`enrich_static_findings.py`); его несогласие с анализатором сохраняется в колонке
`llm_подтвердил_паттерн`, а не прячется.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.export import to_json, to_xlsx  # noqa: E402
from core.schema import Verdict  # noqa: E402
from cases.codereview import export_columns as base_columns  # noqa: E402

SOURCE_RU = {
    "llm": "LLM-ревьюер",
    "static": "статический анализ",
    "both": "LLM-ревьюер + статический анализ",
}


def export_columns(v: Verdict) -> dict:
    row = base_columns(v)
    a = v.artifacts
    row["источник_вердикта"] = SOURCE_RU.get(a.get("detected_by", ""), a.get("detected_by", ""))
    row["статические_инструменты"] = a.get("static_tools", "")
    row["статические_проверки"] = a.get("static_checks", "")
    row["llm_независимый_вердикт"] = a.get("llm_independent_verdict", "")
    row["llm_подтвердил_паттерн"] = a.get("pattern_confirmed")
    return row


def main(verdicts_path: str, enrichment_path: str, out_dir: str) -> None:
    verdicts = json.loads(Path(verdicts_path).read_text(encoding="utf-8"))
    enrich = {r["unique_id"]: r for r in
              json.loads(Path(enrichment_path).read_text(encoding="utf-8")) if "error" not in r}

    ff = pd.read_csv(ROOT / "cases/codereview/out/flawfinder_full_hits.csv")
    ffh = set(ff[ff["any_hit"]]["unique_id"].astype(int))
    tr = pd.read_csv(ROOT / "cases/codereview/out/triage_scores.csv")
    tr_hit = tr[tr["risk_level"].fillna("none") != "none"]
    trh = set(tr_hit["unique_id"].astype(int))
    tr_cats = dict(zip(tr_hit["unique_id"].astype(int), tr_hit["categories"].fillna("")))

    out: list[Verdict] = []
    stats = {"llm": 0, "static": 0, "both": 0, "не помечен": 0}
    for raw in verdicts:
        v = Verdict.model_validate(raw)
        uid = int(v.doc_id)
        by_llm = v.verdict == "vulnerable"
        by_static = uid in ffh or uid in trh
        a = dict(v.artifacts)
        a["llm_independent_verdict"] = v.verdict
        tools = [t for t, s in (("flawfinder", ffh), ("сигнатурный триаж", trh)) if uid in s]
        a["static_tools"] = ", ".join(tools)
        a["static_checks"] = tr_cats.get(uid, "") if uid in trh else ""

        if by_llm and by_static:
            a["detected_by"] = "both"
        elif by_llm:
            a["detected_by"] = "llm"
        elif by_static:
            a["detected_by"] = "static"
        else:
            a["detected_by"] = ""

        if by_static and not by_llm:
            e = enrich.get(uid)
            if e:
                a.update({k: e[k] for k in
                          ("cwe_id", "exploitation_mechanism", "patched_code", "patch_rationale")})
                a["pattern_confirmed"] = e["pattern_confirmed"]
                a["pattern_note"] = e["pattern_note"]
        elif by_llm:
            a["pattern_confirmed"] = True

        if by_llm or by_static:
            stats[a["detected_by"]] += 1
            rationale = v.rationale
            if a["detected_by"] == "static":
                mark = "подтвердил" if a.get("pattern_confirmed") else "НЕ подтвердил"
                rationale = (f"Найдено статическим слоем ({a['static_tools']}). "
                             f"Независимый разбор LLM: вердикт «{v.verdict}», "
                             f"конструкцию {mark}. {a.get('pattern_note', '')}").strip()
            out.append(v.model_copy(update={
                "verdict": "vulnerable",
                "action": "manual_review",
                "confidence": v.confidence if by_llm else 0.4,
                "rationale": rationale[:1000],
                "artifacts": a,
            }))
        else:
            stats["не помечен"] += 1
            out.append(v.model_copy(update={"artifacts": a}))

    d = Path(out_dir)
    to_xlsx(out, str(d / "case3_verdicts.xlsx"), columns_fn=export_columns)
    to_json(out, str(d / "case3_verdicts.json"))
    n_vuln = sum(1 for v in out if v.verdict == "vulnerable")
    print(f"всего {len(out)}, vulnerable {n_vuln} ({n_vuln/len(out):.1%})")
    print("источник вердикта:", stats)
    print("с CWE:", sum(1 for v in out if v.verdict == "vulnerable" and v.artifacts.get("cwe_id")))
    print("с патчем:", sum(1 for v in out if v.verdict == "vulnerable" and v.artifacts.get("patched_code")))


if __name__ == "__main__":
    main(*sys.argv[1:4])
