"""Кривая «доля документов, отданных LLM -> F1/FPR» для гибрида (офлайн-слой + LLM
серая зона), кейс 2. Порог `route()` сейчас — 0.80 (44 вызова на 1000, см. __init__.py).

Метод: n=1000 test (split='test', seed=42 — тот же сэмпл, на котором уже измерен гибрид
report.md). Уверенность офлайн-слоя считается один раз. Порог верхней границы (0.95) даёт
объединённое множество «серой зоны» по всем порогам сразу (~164 документов, в бюджете
<=300) — каждый документ уходит в LLM только один раз, дальше переиспользуется кеш
(`out/llm_cache_case2_errors.sqlite3` — свой, общий кеш не трогается) для каждой точки
порога отдельно (для меньших порогов берётся подмножество уже посчитанного).

Запуск:
  set -a && . ./.env && set +a && \
  .venv/bin/python -m cases.guard.grey_zone_sweep

Пишет: out/guard/case2_grey_zone_sweep.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.data import load_case2
from core.llm import LLMClient, LLMConfig
from core.pipeline import PipelineContext
from cases.guard.baseline import apply_model_a
from cases.guard.grey_zone import classify_grey_zone
from cases.guard.model import load_or_train

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "out" / "guard"
CACHE_PATH = str(OUT_DIR / "llm_cache_case2_errors.sqlite3")  # свой кеш, тот же что в step 4

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def binary_metrics(y_true_bin, y_pred_bin) -> dict:
    tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
    tn = int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
    fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "fpr": fpr, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def main() -> None:
    print("Loading test sample n=1000, seed=42 (тот же сэмпл, что и уже измеренный гибрид)...")
    df = load_case2(split="test", n=1000, seed=42)
    texts = df["text"].tolist()
    y_true = df["label"].to_numpy()
    y_true_bin = (y_true != 0).astype(int)

    model = load_or_train()
    proba, classes = apply_model_a(model, texts)
    offline_pred = classes[proba.argmax(axis=1)]
    offline_pred_bin = (offline_pred != 0).astype(int)
    confidence = proba.max(axis=1)

    max_thr = max(THRESHOLDS)
    union_mask = confidence < max_thr
    union_idx = np.where(union_mask)[0]
    print(f"Union grey-zone set across all thresholds (<= {max_thr}): {len(union_idx)} docs")

    doc_ids = [f"case2-sweep-{i}" for i in range(len(df))]
    records = [{"doc_id": doc_ids[i], "text": texts[i]} for i in union_idx]

    llm_config = LLMConfig(
        model="deepseek-chat", backend="openai_compat", base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY", temperature=0.0, max_tokens=1024, max_concurrency=8,
        dry_run=False, cache_path=CACHE_PATH,
    )
    llm = LLMClient(llm_config)
    ctx = PipelineContext(case="guard", config={}, llm=llm)
    verdicts = classify_grey_zone(records, ctx)
    llm.close()

    llm_pred_bin_by_idx = {}
    for idx, v in zip(union_idx, verdicts):
        llm_pred_bin_by_idx[int(idx)] = 0 if v.verdict == "safe" else 1

    rows = []
    for thr in THRESHOLDS:
        grey_mask = confidence < thr
        n_grey = int(grey_mask.sum())
        pred_bin = offline_pred_bin.copy()
        for idx in np.where(grey_mask)[0]:
            pred_bin[idx] = llm_pred_bin_by_idx[int(idx)]
        m = binary_metrics(y_true_bin, pred_bin)
        rows.append({
            "threshold": thr,
            "n_total": len(df),
            "n_sent_to_llm": n_grey,
            "grey_zone_share": round(n_grey / len(df), 4),
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
        })
        print(f"thr={thr}: grey_n={n_grey} ({n_grey/len(df)*100:.1f}%) "
              f"F1={m['f1']:.4f} FPR={m['fpr']:.4f}")

    out_df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "case2_grey_zone_sweep.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")
    print(f"LLM usage: {llm.usage_summary()}")


if __name__ == "__main__":
    main()
