"""Метрики кейса 2 на фактически доставленной выгрузке out/case2_verdicts.json (весь test).

doc_id имеет вид `case2-test-<i>`, где i — позиция строки в выборке. Прогон шёл без --sample,
то есть выборка совпадает с исходным parquet построчно (см. ловушку в STATE.md: при --sample
это НЕ так и сопоставление по индексу даёт мусор).
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.data import load_case2  # noqa: E402


def main(verdicts_path: str, split: str, out_path: str) -> None:
    df = load_case2(split=split)
    gold = {f"case2-{split}-{i}": (row["verdict_binary"], int(row["label"]))
            for i, row in df.iterrows()}

    verdicts = json.loads(Path(verdicts_path).read_text(encoding="utf-8"))
    tp = fp = tn = fn = 0
    by_source: Counter = Counter()
    src_correct: Counter = Counter()
    action_dist: Counter = Counter()
    sub_tp = sub_total = 0
    matched = 0
    for v in verdicts:
        g = gold.get(v["doc_id"])
        if g is None:
            continue
        matched += 1
        gold_v, gold_label = g
        pred_v = v["verdict"]
        src = v["artifacts"].get("source", "?")
        by_source[src] += 1
        src_correct[src] += pred_v == gold_v
        action_dist[v["action"]] += 1
        pos_pred = pred_v == "injection_malicious"
        pos_gold = gold_v == "injection_malicious"
        tp += pos_gold and pos_pred; fp += (not pos_gold) and pos_pred
        fn += pos_gold and (not pos_pred); tn += (not pos_gold) and (not pos_pred)
        if pos_gold and pos_pred:
            sub = v["artifacts"].get("subtype")
            if sub in ("masked", "direct"):
                sub_total += 1
                sub_tp += (sub == "masked" and gold_label == 1) or (sub == "direct" and gold_label == 2)

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    out = {
        "источник": verdicts_path, "split": split, "сматчено": matched,
        "бинарно": {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
                     "fpr": round(fpr, 4), "accuracy": round((tp + tn) / matched, 4),
                     "mcc": round((tp * tn - fp * fn) / den, 4) if den else 0.0,
                     "tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "по_слоям": {s: {"n": by_source[s], "accuracy": round(src_correct[s] / by_source[s], 4)}
                       for s in by_source},
        "решения": dict(action_dist),
        "под_тип_masked_vs_direct": {"n": sub_total,
                                       "accuracy": round(sub_tp / sub_total, 4) if sub_total else None},
    }
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
