"""Сравнение слоёв кейса 3 на ВСЁМ корпусе против восстановленной разметки v4.

Считает LLM (сдано), flawfinder, сигнатурный триаж и их объединения/пересечения.
Сети не требует: все три набора предсказаний уже лежат на диске.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main(verdicts_path: str, out_path: str) -> None:
    lab = pd.read_csv(ROOT / "research" / "case3_recovered_labels_v4.csv")
    gold = dict(zip(lab["unique_id"].astype(int), lab["recovered_label"].astype(int)))
    n, npos = len(gold), sum(gold.values())

    def metrics(ids: set[int]) -> dict:
        tp = sum(1 for i in ids if gold.get(i) == 1)
        fp = len(ids) - tp
        tn, fn = n - npos - fp, npos - tp
        pr = tp / len(ids) if ids else 0.0
        rc = tp / npos
        den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return {"precision": round(pr, 4), "recall": round(rc, 4),
                "f1": round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0,
                "fpr": round(fp / (n - npos), 4), "youden_j": round(rc - fp / (n - npos), 4),
                "mcc": round((tp * tn - fp * fn) / den, 4) if den else 0.0,
                "помечено_доля": round(len(ids) / n, 4), "найдено_уязвимых": tp}

    v = json.loads(Path(verdicts_path).read_text(encoding="utf-8"))
    llm = {int(x["doc_id"]) for x in v if x["verdict"] == "vulnerable"}
    unc = {int(x["doc_id"]) for x in v if x["verdict"] == "uncertain"}
    ff = pd.read_csv(ROOT / "cases/codereview/out/flawfinder_full_hits.csv")
    ffh = set(ff[ff["any_hit"]]["unique_id"].astype(int))
    tr = pd.read_csv(ROOT / "cases/codereview/out/triage_scores.csv")
    trh = set(tr[tr["risk_level"].fillna("none") != "none"]["unique_id"].astype(int))

    combos = {
        "LLM один (поставочная)": llm,
        "flawfinder один": ffh,
        "сигнатурный триаж один": trh,
        "LLM ∪ flawfinder": llm | ffh,
        "LLM ∪ триаж": llm | trh,
        "LLM ∪ flawfinder ∪ триаж": llm | ffh | trh,
        "LLM ∪ (uncertain ∩ статика)": llm | (unc & (ffh | trh)),
        "LLM ∩ flawfinder": llm & ffh,
        "LLM ∩ триаж": llm & trh,
    }
    out = {"эталон": "research/case3_recovered_labels_v4.csv",
           "n": n, "уязвимых": npos, "доля_уязвимых": round(npos / n, 4),
           "конфигурации": {k: metrics(s) for k, s in combos.items()}}
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, r in out["конфигурации"].items():
        print(f'{k:<30} F1 {r["f1"]:.4f}  J {r["youden_j"]:+.4f}  найдено {r["найдено_уязвимых"]:>4}  '
              f'помечено {r["помечено_доля"]:.1%}')


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
