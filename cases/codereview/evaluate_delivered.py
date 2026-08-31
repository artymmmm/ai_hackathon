"""Метрики кейса 3 на фактически доставленной выгрузке out/case3_verdicts.json.

Эталон — research/case3_recovered_labels_v4.csv (восстановленная разметка, см. STATE.md).
Меряет на ВСЁМ корпусе, где доля уязвимых 5.6%, а не на сбалансированном наборе 600 (33%),
поэтому числа с замером на eval600 напрямую несопоставимы — это и есть смысл замера.

Кроме P/R/F1 считает то, что для несбалансированной задачи осмысленнее: Youden J, MCC,
тривиальные базовые линии и VD-S (recall при фиксированном FPR), рекомендованный разведкой
литературы в research/case3_bigvul_literature.md.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / den) if den else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "fpr": round(fpr, 4), "youden_j": round(rec - fpr, 4), "mcc": round(mcc, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def main(verdicts_path: str, out_path: str) -> None:
    labels = pd.read_csv(ROOT / "research" / "case3_recovered_labels_v4.csv")
    gold = {int(r.unique_id): r.recovered_label for r in labels.itertuples()
            if pd.notna(r.recovered_label) and str(r.recovered_label) in ("0", "1", "0.0", "1.0")}
    gold = {k: int(float(v)) for k, v in gold.items()}
    gold_cwe = {int(r.unique_id): r.cwe_id for r in labels.itertuples() if pd.notna(r.cwe_id)}

    verdicts = json.loads(Path(verdicts_path).read_text(encoding="utf-8"))
    pairs = [(gold[int(v["doc_id"])], v) for v in verdicts if int(v["doc_id"]) in gold]

    schemes = {
        "vulnerable_only (поставочная)": lambda vd: vd == "vulnerable",
        "vulnerable+uncertain (эскалация)": lambda vd: vd in ("vulnerable", "uncertain"),
    }
    out: dict = {
        "источник": verdicts_path,
        "фрагментов_в_выгрузке": len(verdicts),
        "с_восстановленным_лейблом": len(pairs),
        "доля_уязвимых_в_эталоне": round(sum(g for g, _ in pairs) / len(pairs), 4),
        "схемы": {},
    }
    for name, positive in schemes.items():
        tp = fp = tn = fn = 0
        for g, v in pairs:
            p = positive(v["verdict"])
            tp += g == 1 and p; fp += g == 0 and p
            fn += g == 1 and not p; tn += g == 0 and not p
        out["схемы"][name] = _metrics(tp, fp, tn, fn)

    n_pos = sum(g for g, _ in pairs)
    n = len(pairs)
    out["тривиальные_базы"] = {
        "всё vulnerable": _metrics(n_pos, n - n_pos, 0, 0),
        "всё secure": _metrics(0, 0, n - n_pos, n_pos),
    }

    # VD-S: recall при фиксированном FPR. Порог — по confidence внутри позитивов модели.
    scored = sorted(((v["confidence"] if v["verdict"] == "vulnerable" else 0.0, g)
                     for g, v in pairs), key=lambda t: -t[0])
    neg_total = n - n_pos
    vds = {}
    for target_fpr in (0.01, 0.05):
        allowed_fp = int(neg_total * target_fpr)
        fp_seen = tp_seen = 0
        for score, g in scored:
            if score <= 0.0:
                break
            if g == 1:
                tp_seen += 1
            else:
                fp_seen += 1
                if fp_seen > allowed_fp:
                    tp_seen -= 0
                    break
        vds[f"recall@FPR={target_fpr}"] = round(tp_seen / n_pos, 4) if n_pos else 0.0
    out["vd_s"] = vds

    # CWE: доля точных совпадений среди верно найденных уязвимых, у которых есть золотой CWE
    exact = total = 0
    for g, v in pairs:
        if g != 1 or v["verdict"] != "vulnerable":
            continue
        gc = gold_cwe.get(int(v["doc_id"]))
        if not isinstance(gc, str) or not gc.strip():
            continue
        total += 1
        exact += str(v["artifacts"].get("cwe_id", "")).strip().upper() == gc.strip().upper()
    out["cwe"] = {"с_золотым_cwe": total, "точных": exact,
                  "accuracy": round(exact / total, 4) if total else None,
                  "оговорка": "эталон CWE второсортный: 21% — категорийные узлы, запрещённые MITRE "
                              "к маппингу, 43% — широкие классы; см. STATE.md"}

    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
