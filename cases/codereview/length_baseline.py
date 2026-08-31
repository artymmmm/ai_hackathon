"""Проверка кейса 3 на confound длины кода — базовая линия, которую обязан бить любой детектор.

Считает: (а) детектор «самый длинный код» при том же бюджете ревью, (б) стратифицированную
по длине оценку вклада каждого слоя (Мантель-Хензель, 20 страт), (в) комбинированное
ранжирование длина + LLM + статика.

Сети не требует. Веса комбинации подбираются на этих же данных — это верхняя оценка,
а не честная оценка на отложенной выборке; в отчёте писать именно так.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.data import load_case3  # noqa: E402


def main(verdicts_path: str, out_path: str) -> None:
    lab = pd.read_csv(ROOT / "research" / "case3_recovered_labels_v4.csv")
    gold = dict(zip(lab["unique_id"].astype(int), lab["recovered_label"].astype(int)))
    df = load_case3()
    df["unique_id"] = df["unique_id"].astype(int)
    length = dict(zip(df["unique_id"], df["code"].str.len()))

    v = json.loads(Path(verdicts_path).read_text(encoding="utf-8"))
    llm = {int(x["doc_id"]) for x in v if x["verdict"] == "vulnerable"}
    ff = pd.read_csv(ROOT / "cases/codereview/out/flawfinder_full_hits.csv")
    ffh = set(ff[ff["any_hit"]]["unique_id"].astype(int))
    tr = pd.read_csv(ROOT / "cases/codereview/out/triage_scores.csv")
    trh = set(tr[tr["risk_level"].fillna("none") != "none"]["unique_id"].astype(int))
    stat = ffh | trh

    ids = sorted(gold)
    n, npos = len(ids), sum(gold.values())
    lenrank = {i: r / n for r, i in enumerate(sorted(ids, key=lambda i: length[i]))}

    def at_budget(order: list[int], k: int) -> dict:
        sel = order[:k]
        tp = sum(gold[i] for i in sel)
        fp, tn, fn = k - tp, n - npos - (k - tp), npos - tp
        pr, rc = tp / k, tp / npos
        fpr = fp / (n - npos)
        den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return {"precision": round(pr, 4), "recall": round(rc, 4),
                "f1": round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0,
                "fpr": round(fpr, 4), "youden_j": round(rc - fpr, 4),
                "mcc": round((tp * tn - fp * fn) / den, 4) if den else 0.0,
                "найдено": tp, "бюджет": round(k / n, 4)}

    def stratified_lift(layer: set[int], strata: int = 20) -> float:
        s = pd.Series({i: length[i] for i in ids})
        q = pd.qcut(s, strata, labels=False, duplicates="drop")
        obs = exp = 0.0
        for d in sorted(set(q)):
            grp = set(q[q == d].index)
            a, b = grp & layer, grp - layer
            if not a or not b:
                continue
            obs += sum(gold[i] for i in a)
            exp += len(a) * (sum(gold[i] for i in b) / len(b))
        return round(obs / exp, 3) if exp else 0.0

    out: dict = {
        "n": n, "уязвимых": npos,
        "стратифицированный_вклад_при_равной_длине": {
            "LLM": stratified_lift(llm),
            "статика (flawfinder ∪ триаж)": stratified_lift(stat),
            "объединение": stratified_lift(llm | stat),
        },
        "при_равном_бюджете": {},
    }
    for budget in (0.10, 0.151, 0.205, 0.30):
        k = int(n * budget)
        row = {
            "только длина": at_budget(sorted(ids, key=lambda i: -lenrank[i]), k),
            "LLM ∪ статика": at_budget(
                sorted(ids, key=lambda i: -((i in llm) * 2 + (i in stat) + lenrank[i] * 0.001)), k),
        }
        best = None
        for w in np.arange(0, 1.01, 0.05):
            order = sorted(ids, key=lambda i: -(lenrank[i] + w * (i in llm) + 0.5 * w * (i in stat)))
            r = at_budget(order, k)
            if best is None or r["youden_j"] > best[1]["youden_j"]:
                best = (round(float(w), 2), r)
        row[f"длина + {best[0]}*LLM + {best[0]/2}*статика"] = best[1]
        out["при_равном_бюджете"][f"{budget:.1%}"] = row

    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["стратифицированный_вклад_при_равной_длине"], ensure_ascii=False))
    for b, row in out["при_равном_бюджете"].items():
        print(f'\nбюджет {b}:')
        for k2, r in row.items():
            print(f'  {k2:<40} F1 {r["f1"]:.4f}  J {r["youden_j"]:+.4f}  найдено {r["найдено"]}')


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
