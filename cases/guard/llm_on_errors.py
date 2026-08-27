"""Проверка: чинит ли LLM-серая-зона (grey_zone.classify_grey_zone) конкретные ошибки
офлайн-слоя из case2_errors_categorized.csv? Свой кеш, не трогает out/llm_cache.sqlite3.

Требует DEEPSEEK_API_KEY в окружении (грузить через `set -a && . ./.env && set +a`,
этот скрипт .env сам не читает — как и остальные скрипты cases/*).

Запуск:
  set -a && . ./.env && set +a && \
  .venv/bin/python -m cases.guard.llm_on_errors

Пишет: out/guard/case2_llm_on_errors.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core.llm import LLMClient, LLMConfig
from core.pipeline import PipelineContext
from cases.guard.grey_zone import classify_grey_zone

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "out" / "guard"
CACHE_PATH = str(OUT_DIR / "llm_cache_case2_errors.sqlite3")  # свой кеш, см. границы задачи


def main() -> None:
    df = pd.read_csv(OUT_DIR / "case2_errors_categorized.csv")
    print(f"Sending {len(df)} offline-layer errors to LLM grey-zone stage (<=300 budget)...")

    records = [
        {"doc_id": row["doc_id"], "text": row["text"]}
        for _, row in df.iterrows()
    ]

    llm_config = LLMConfig(
        model="deepseek-chat",
        backend="openai_compat",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        temperature=0.0,
        max_tokens=1024,
        max_concurrency=8,
        dry_run=False,
        cache_path=CACHE_PATH,
    )
    llm = LLMClient(llm_config)
    ctx = PipelineContext(case="guard", config={}, llm=llm)

    verdicts = classify_grey_zone(records, ctx)
    llm.close()

    by_doc = {v.doc_id: v for v in verdicts}

    rows = []
    n_fixed = 0
    n_still_wrong = 0
    n_parse_failed = 0
    for _, row in df.iterrows():
        v = by_doc[row["doc_id"]]
        llm_pred_bin = "safe" if v.verdict == "safe" else "injection_malicious"
        true_bin = row["true_verdict_binary"]
        llm_correct = llm_pred_bin == true_bin
        parse_failed = bool(v.artifacts.get("parse_failed"))
        if parse_failed:
            n_parse_failed += 1
        if llm_correct:
            n_fixed += 1
        else:
            n_still_wrong += 1
        rows.append({
            "doc_id": row["doc_id"],
            "error_type_offline": row["error_type"],
            "category": row["category"],
            "true_verdict_binary": true_bin,
            "offline_pred": row["pred_verdict_binary"],
            "llm_pred": llm_pred_bin,
            "llm_confidence": v.confidence,
            "llm_correct": llm_correct,
            "llm_parse_failed": parse_failed,
            "llm_rationale": v.rationale[:300],
        })

    out_df = pd.DataFrame(rows)
    detail_csv = OUT_DIR / "case2_llm_on_errors_detail.csv"
    out_df.to_csv(detail_csv, index=False)

    # разбивка "исправлено" по категории ошибки и по FP/FN
    fixed_by_category = out_df[out_df.llm_correct].groupby("category").size().to_dict()
    fixed_by_error_type = out_df[out_df.llm_correct].groupby("error_type_offline").size().to_dict()
    total_by_category = out_df.groupby("category").size().to_dict()

    summary = {
        "n_sent_to_llm": len(df),
        "n_fixed": n_fixed,
        "n_still_wrong": n_still_wrong,
        "n_llm_parse_failed": n_parse_failed,
        "fix_rate": round(n_fixed / len(df), 4),
        "fixed_by_category": fixed_by_category,
        "total_by_category": total_by_category,
        "fixed_by_offline_error_type": fixed_by_error_type,
        "note": (
            "Вход — только документы, на которых офлайн-слой уже ошибся. 'Ломает' в буквальном "
            "смысле невозможно (двоичная задача, все входы уже неверны): либо LLM исправляет, "
            "либо ошибка остаётся. n_still_wrong — это и есть 'сколько не исправлено'."
        ),
        "llm_usage": llm.usage_summary(),
        "detail_csv": str(detail_csv.relative_to(ROOT)),
    }
    out_path = OUT_DIR / "case2_llm_on_errors.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path}")
    print(f"Wrote {detail_csv}")


if __name__ == "__main__":
    main()
