"""Диагностика ошибок офлайн-слоя (модель (a), см. baseline.py) на полном тесте кейса 2.

Разовый скрипт для локализации потолка F1 — не часть пайплайна `run.py`. Пишет:
  out/guard/case2_errors_full_test.csv   — все FP/FN с текстом, истинным классом,
                                            предсказанием, уверенностью.
  out/guard/case2_errors_summary.json    — сводные числа (пересчитанные, не по памяти).

Запуск: .venv/bin/python -m cases.guard.error_analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.data import load_case2
from cases.guard.baseline import apply_model_a
from cases.guard.model import load_or_train

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "out" / "guard"

LABEL_NAMES = {0: "safe", 1: "injection_masked", 2: "injection_direct"}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading full test split (10000, no sampling — doc order == parquet order)...")
    test = load_case2(split="test")  # n=None -> полный сплит, doc_id = позиция == индекс parquet
    texts = test["text"].tolist()
    y_true = test["label"].to_numpy()

    print("Loading persisted offline model (out/guard/model_a.joblib)...")
    model = load_or_train()
    proba, classes = apply_model_a(model, texts)
    pred_idx = proba.argmax(axis=1)
    y_pred = classes[pred_idx]
    confidence = proba.max(axis=1)

    y_true_bin = (y_true != 0).astype(int)
    y_pred_bin = (y_pred != 0).astype(int)

    tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
    tn = int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
    fp_mask = (y_true_bin == 0) & (y_pred_bin == 1)
    fn_mask = (y_true_bin == 1) & (y_pred_bin == 0)
    fp = int(fp_mask.sum())
    fn = int(fn_mask.sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    print(f"Recomputed full-test binary: P={precision:.4f} R={recall:.4f} F1={f1:.4f} "
          f"FPR={fpr:.4f} (tp={tp} fp={fp} tn={tn} fn={fn})")

    err_mask = fp_mask | fn_mask
    idx = np.where(err_mask)[0]

    rows = []
    for i in idx:
        rows.append({
            "doc_id": f"case2-test-{i}",
            "row_index": int(i),
            "text": texts[i],
            "true_label": int(y_true[i]),
            "true_label_name": LABEL_NAMES[int(y_true[i])],
            "true_verdict_binary": test["verdict_binary"].iloc[i],
            "pred_label": int(y_pred[i]),
            "pred_label_name": LABEL_NAMES[int(y_pred[i])],
            "pred_verdict_binary": "safe" if y_pred[i] == 0 else "injection_malicious",
            "confidence": round(float(confidence[i]), 4),
            "proba_safe": round(float(proba[i][list(classes).index(0)]), 4) if 0 in classes else None,
            "proba_masked": round(float(proba[i][list(classes).index(1)]), 4) if 1 in classes else None,
            "proba_direct": round(float(proba[i][list(classes).index(2)]), 4) if 2 in classes else None,
            "error_type": "FP" if fp_mask[i] else "FN",
        })

    import pandas as pd
    err_df = pd.DataFrame(rows).sort_values("confidence")  # низкая уверенность -> вверху
    csv_path = OUT_DIR / "case2_errors_full_test.csv"
    err_df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(err_df)} rows)")

    # разбивка FN по истинному под-типу (1=masked против 2=direct) — какие сложнее пропустить
    fn_by_subtype = {
        LABEL_NAMES[lbl]: int(((y_true == lbl) & fn_mask).sum()) for lbl in (1, 2)
    }
    # confidence distribution ошибок
    conf_bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    conf_hist = np.histogram(err_df["confidence"], bins=conf_bins)[0].tolist()

    summary = {
        "test_size": len(test),
        "binary_metrics_recomputed": {
            "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        },
        "n_errors_total": int(len(err_df)),
        "n_fp": fp,
        "n_fn": fn,
        "fn_by_true_subtype": fn_by_subtype,
        "error_confidence_histogram": {"bin_edges": conf_bins, "counts": conf_hist},
        "csv_path": str(csv_path.relative_to(ROOT)),
    }
    json_path = OUT_DIR / "case2_errors_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
