"""Диагностика LLM-слоя: матрица путаницы меток + причины recall-провалов на пяти целевых
типах (first_name, last_name, company_name, occupation, street_address).

Не часть пайплайна — разовый инструмент анализа для текущей задачи (recall LLM-слоя). Не
трогает `evaluate.py`/`detectors.py`. Использует ту же выборку (`_load`, `_gold_spans`) и то же
span-only сопоставление (`_match_span_only`), что и `evaluate.py`, импортируя их напрямую —
не дублируем логику матчинга.

Запуск (кеш уже прогрет тем же `--model`/`--n`/`--seed`, что и `evaluate.py --ablation`,
поэтому сеть не трогается, пока промпт не менялся):

    set -a && . ./.env && set +a && .venv/bin/python cases/pii/llm_confusion.py \
        --n 200 --split test --seed 42 --model deepseek-chat \
        --base-url https://api.deepseek.com/v1 --api-key-env DEEPSEEK_API_KEY \
        --out-suffix before

Пишет `out/pii/llm_label_confusion<suffix>.json` и `.csv` (матрица путаницы), плюс печатает
список пропущенных (FN) сущностей пяти целевых типов с контекстом документа — сырьё для
ручного разбора причин recall.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.llm import LLMClient, LLMConfig  # noqa: E402

from cases.pii.evaluate import _gold_spans, _load, _match_span_only  # noqa: E402
from cases.pii.llm_layer import detect_pii_llm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "out" / "pii"

TARGET_TYPES = {"first_name", "last_name", "company_name", "occupation", "street_address"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--backend", default="openai_compat")
    ap.add_argument("--base-url", default="https://api.deepseek.com/v1")
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--cache-path", default="out/llm_cache.sqlite3")
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--max-tokens", type=int, default=None,
                     help="прокидывается в detect_pii_llm через LLMConfig.max_tokens")
    args = ap.parse_args()

    df = _load(args.split, args.n, args.seed)

    cfg_kwargs = dict(
        model=args.model, backend=args.backend, base_url=args.base_url,
        api_key_env=args.api_key_env, dry_run=False, cache_path=args.cache_path,
        max_concurrency=args.max_workers,
    )
    if args.max_tokens is not None:
        cfg_kwargs["max_tokens"] = args.max_tokens
    llm = LLMClient(LLMConfig(**cfg_kwargs))

    def _predict(row):
        return detect_pii_llm(
            row["text"], llm,
            document_type=row.get("document_type") or "",
            domain=row.get("domain") or "",
            model=args.model,
        )

    rows = [row for _, row in df.iterrows()]
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        preds = list(ex.map(_predict, rows))

    confusion: Counter = Counter()  # (gold_label, pred_label) -> count, только span-match с разными label
    correct: Counter = Counter()  # gold_label -> count span-match с тем же label
    fn_examples: list[dict] = []  # пропущенные (FN) сущности целевых типов, с контекстом
    per_type_tp = Counter()
    per_type_gold = Counter()
    per_type_pred = Counter()
    per_type_fp = Counter()

    for row, pred in zip(rows, preds):
        gold = _gold_spans(row)
        text = row["text"]

        for g in gold:
            per_type_gold[g["label"]] += 1
        for p in pred:
            per_type_pred[p["label"]] += 1

        gold_used = [False] * len(gold)
        pred_used = [False] * len(pred)

        # span-only matching, тот же порядок, что evaluate._match_span_only использует внутри evaluate()
        for pi, p in enumerate(pred):
            for gi, g in enumerate(gold):
                if gold_used[gi]:
                    continue
                if _match_span_only(p, g):
                    gold_used[gi] = True
                    pred_used[pi] = True
                    if p["label"] == g["label"]:
                        correct[g["label"]] += 1
                        per_type_tp[g["label"]] += 1
                    else:
                        confusion[(g["label"], p["label"])] += 1
                    break

        for pi, p in enumerate(pred):
            if not pred_used[pi]:
                per_type_fp[p["label"]] += 1

        for gi, g in enumerate(gold):
            if not gold_used[gi] and g["label"] in TARGET_TYPES:
                start, end = g["start"], g["end"]
                ctx_lo = max(0, start - 60)
                ctx_hi = min(len(text), end + 60)
                fn_examples.append({
                    "uid": row.get("uid"),
                    "label": g["label"],
                    "text": text[start:end],
                    "doc_len": len(text),
                    "n_gold_entities": len(gold),
                    "n_pred_entities": len(pred),
                    "context": text[ctx_lo:ctx_hi].replace("\n", " "),
                })

    # top pairs
    top_pairs = confusion.most_common(20)

    result = {
        "n_docs": len(rows),
        "model": args.model,
        "confusion_pairs_total": sum(confusion.values()),
        "correct_span_matches_total": sum(correct.values()),
        "top_confusion_pairs": [
            {"gold_label": g, "pred_label": p, "count": c} for (g, p), c in top_pairs
        ],
        "per_type_target": {
            label: {
                "gold": per_type_gold[label],
                "tp_exact_via_span_match": per_type_tp[label],
                "fp": per_type_fp[label],
                "n_fn_examples_captured": sum(1 for e in fn_examples if e["label"] == label),
            }
            for label in sorted(TARGET_TYPES)
        },
        "llm_usage": llm.usage_summary(),
    }
    llm.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    out_json = OUT_DIR / f"llm_label_confusion{suffix}.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    out_csv = OUT_DIR / f"llm_label_confusion{suffix}.csv"
    with out_csv.open("w") as f:
        f.write("gold_label,pred_label,count\n")
        for (g, p), c in confusion.most_common():
            f.write(f"{g},{p},{c}\n")

    fn_path = OUT_DIR / f"llm_fn_examples{suffix}.json"
    fn_path.write_text(json.dumps(fn_examples, ensure_ascii=False, indent=2))

    print(f"n_docs={len(rows)} confusion_pairs_total={result['confusion_pairs_total']} "
          f"correct_span_matches={result['correct_span_matches_total']}")
    print("\nтоп путаниц (gold -> pred : count):")
    for pair in result["top_confusion_pairs"][:15]:
        print(f"  {pair['gold_label']:20s} -> {pair['pred_label']:20s} : {pair['count']}")
    print(f"\nusage: {json.dumps(result['llm_usage'], ensure_ascii=False)}")
    print(f"сохранено: {out_json}, {out_csv}, {fn_path}")


if __name__ == "__main__":
    main()
