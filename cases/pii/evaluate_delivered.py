"""Span-level метрики кейса 1 на ФАКТИЧЕСКИ доставленной выгрузке out/case1_verdicts.json.

Отличие от evaluate.py: тот прогоняет пайплайн заново, этот меряет то, что реально сдано.
Определение метрики то же — точное совпадение (start, end, label), жадный первый матч,
плюс span-only recall (нашли факт PII, лейбл не важен).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.data import load_case1  # noqa: E402


def main(verdicts_path: str, split: str, out_path: str) -> None:
    df = load_case1(split=split)
    seen: dict[str, int] = {}
    gold_by_id: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        uid = str(row["uid"])
        seen[uid] = seen.get(uid, 0) + 1
        gold_by_id[f"{uid}#{seen[uid]}"] = row["spans"]

    verdicts = json.loads(Path(verdicts_path).read_text(encoding="utf-8"))
    tp = fp = fn = tp_span = fn_span = 0
    per_type_gold: Counter = Counter()
    per_type_tp: Counter = Counter()
    per_type_pred: Counter = Counter()
    matched_docs = 0

    for v in verdicts:
        gold = gold_by_id.get(v["doc_id"])
        if gold is None:
            continue
        matched_docs += 1
        pred = v["artifacts"]["pii_found"]
        for g in gold:
            per_type_gold[g["label"]] += 1
        for p in pred:
            per_type_pred[p["label"]] += 1

        gold_used = [False] * len(gold)
        pred_used = [False] * len(pred)
        for pi, p in enumerate(pred):
            for gi, g in enumerate(gold):
                if gold_used[gi]:
                    continue
                if p["start"] == g["start"] and p["end"] == g["end"] and p["label"] == g["label"]:
                    tp += 1
                    per_type_tp[g["label"]] += 1
                    gold_used[gi] = pred_used[pi] = True
                    break
        fp += pred_used.count(False)
        fn += gold_used.count(False)

        gold_used2 = [False] * len(gold)
        for p in pred:
            for gi, g in enumerate(gold):
                if gold_used2[gi]:
                    continue
                if p["start"] < g["end"] and g["start"] < p["end"]:
                    tp_span += 1
                    gold_used2[gi] = True
                    break
        fn_span += gold_used2.count(False)

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    per_type = sorted(
        ({"label": l, "gold": per_type_gold[l], "pred": per_type_pred[l],
          "tp": per_type_tp[l], "recall": round(per_type_tp[l] / per_type_gold[l], 4)}
         for l in per_type_gold),
        key=lambda d: -d["gold"],
    )
    result = {
        "источник": verdicts_path, "split": split,
        "документов_в_выгрузке": len(verdicts), "сматчено_с_эталоном": matched_docs,
        "span_exact": {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
                       "tp": tp, "fp": fp, "fn": fn},
        "recall_span_only": round(tp_span / (tp_span + fn_span), 4) if tp_span + fn_span else 0.0,
        "золотых_спанов": sum(per_type_gold.values()),
        "предсказанных_спанов": sum(per_type_pred.values()),
        "по_типам": per_type,
    }
    Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    s = result["span_exact"]
    print(f'документов {matched_docs}, золотых спанов {result["золотых_спанов"]}')
    print(f'span-exact  P {s["precision"]}  R {s["recall"]}  F1 {s["f1"]}')
    print(f'recall без учёта типа: {result["recall_span_only"]}')


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
