"""Span-level метрики детекторов против эталонной разметки `case 1/data/*.parquet`.

Считает precision/recall/F1 по точному совпадению span (start, end, label) и отдельно —
recall по каждому типу сущности (таблица «где мы слабы»), плюс recall без учёта типа
(«нашли сам факт PII, лейбл неважен» — честная метрика для детектора, который не обязан
отличать account_number от customer_id без контекста).

Запуск (детекторы, поведение не менялось): `.venv/bin/python cases/pii/evaluate.py [--n 1500]
[--split test] [--seed 42]`.

Ablation-таблица трёх конфигураций (детекторы / LLM / гибрид): `--ablation` (см. `evaluate_ablation`
ниже). Без `--dry-run` и без ключа попытка живого вызова упадёт с понятной ошибкой из
`core/llm.py` — это ожидаемо, а не баг: см. CLAUDE.md, «не делать живые вызовы без явной команды».
"""

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.llm import LLMClient, LLMConfig  # noqa: E402

from cases.pii.detectors import detect_pii  # noqa: E402
from cases.pii.llm_layer import detect_pii_llm, merge_spans  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "case 1" / "data"
OUT_DIR = ROOT / "out" / "pii"


def _load(split: str, n: int, seed: int) -> pd.DataFrame:
    path = DATA_DIR / f"{split}-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


def _gold_spans(row) -> list[dict]:
    return ast.literal_eval(row["spans"])


def _match_exact(pred: dict, gold: dict) -> bool:
    return pred["start"] == gold["start"] and pred["end"] == gold["end"] and pred["label"] == gold["label"]


def _match_span_only(pred: dict, gold: dict) -> bool:
    """Пересечение диапазонов без учёта типа — «нашли ли сам факт PII»."""
    return pred["start"] < gold["end"] and gold["start"] < pred["end"]


def evaluate(df: pd.DataFrame, predict_fn: Callable[[pd.Series], list[dict]] | None = None) -> dict:
    """`predict_fn(row) -> spans` — по умолчанию чистые детекторы (поведение не менялось).
    Используется также `evaluate_ablation()` ниже с llm-only/hybrid predict_fn."""
    predict_fn = predict_fn or (lambda row: detect_pii(row["text"]))
    tp_exact = fp_exact = fn_exact = 0
    tp_span = fn_span = 0
    per_type_tp: Counter = Counter()
    per_type_gold: Counter = Counter()
    per_type_pred: Counter = Counter()
    per_type_fp: Counter = Counter()

    for _, row in df.iterrows():
        gold = _gold_spans(row)
        pred = predict_fn(row)

        for g in gold:
            per_type_gold[g["label"]] += 1
        for p in pred:
            per_type_pred[p["label"]] += 1

        gold_used = [False] * len(gold)
        pred_used = [False] * len(pred)

        # точное совпадение (start, end, label)
        for pi, p in enumerate(pred):
            for gi, g in enumerate(gold):
                if gold_used[gi]:
                    continue
                if _match_exact(p, g):
                    tp_exact += 1
                    per_type_tp[g["label"]] += 1
                    gold_used[gi] = True
                    pred_used[pi] = True
                    break
        fp_exact += pred_used.count(False)
        fn_exact += gold_used.count(False)
        for pi, p in enumerate(pred):
            if not pred_used[pi]:
                per_type_fp[p["label"]] += 1

        # span-only (детектировали факт PII, лейбл не важен) — для честной оценки покрытия
        gold_used2 = [False] * len(gold)
        pred_used2 = [False] * len(pred)
        for pi, p in enumerate(pred):
            for gi, g in enumerate(gold):
                if gold_used2[gi]:
                    continue
                if _match_span_only(p, g):
                    tp_span += 1
                    gold_used2[gi] = True
                    pred_used2[pi] = True
                    break
        fn_span += gold_used2.count(False)

    precision = tp_exact / (tp_exact + fp_exact) if (tp_exact + fp_exact) else 0.0
    recall = tp_exact / (tp_exact + fn_exact) if (tp_exact + fn_exact) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    recall_span_only = tp_span / (tp_span + fn_span) if (tp_span + fn_span) else 0.0

    per_type = []
    for label in sorted(per_type_gold, key=lambda l: -per_type_gold[l]):
        g = per_type_gold[label]
        tp = per_type_tp[label]
        pr = per_type_pred[label]
        fp = per_type_fp[label]
        rec = tp / g if g else 0.0
        prec = tp / pr if pr else 0.0
        per_type.append(
            {
                "label": label,
                "gold_count": g,
                "pred_count": pr,
                "tp": tp,
                "fp": fp,
                "recall": round(rec, 4),
                "precision": round(prec, 4),
                "covered_by_design": _covered_by_design(label),
            }
        )

    return {
        "n_docs": len(df),
        "overall": {
            "precision_exact": round(precision, 4),
            "recall_exact": round(recall, 4),
            "f1_exact": round(f1, 4),
            "recall_span_only": round(recall_span_only, 4),
            "tp_exact": tp_exact,
            "fp_exact": fp_exact,
            "fn_exact": fn_exact,
        },
        "per_type": per_type,
    }


# Типы, которые детекторы принципиально не пытаются ловить (свободный текст без формата:
# имена, гео-названия, демография, свободные должности) — размечено сюда честно, чтобы в
# таблице recall=0 по ним не выглядело как баг, а явно читалось как "зона LLM-слоя".
NOT_ATTEMPTED = {
    "first_name",
    "last_name",
    "user_name",
    "company_name",
    "occupation",
    "country",
    "city",
    "state",
    "county",
    "street_address",
    "employment_status",
    "education_level",
    "race_ethnicity",
    "language",
    "gender",
    "political_view",
    "religious_belief",
    "sexuality",
    "blood_type",
    "age",
}


def _covered_by_design(label: str) -> bool:
    return label not in NOT_ATTEMPTED


# --------------------------------------------------------------- ablation: 3 конфигурации

def evaluate_ablation(df: pd.DataFrame, llm_client: LLMClient, model: str | None = None) -> dict:
    """Одна и та же выборка, три `predict_fn` для `evaluate()`: только детекторы (не менялось,
    те же цифры, что и в `evaluate(df)`), только LLM-слой, гибрид (`llm_layer.merge_spans`).

    `llm_client` может быть в `dry_run` (умолчание всего проекта, ключей нет) — тогда `llm`/
    `hybrid` дают заглушечные, не содержательные числа (см. докстринг модуля и
    `prompts/context_entities.md`, раздел «Ablation»): это единственное честное поведение без
    сети, но код идентичен тому, что даст настоящую таблицу сразу после появления ключа.
    """

    def _detector_pred(row: pd.Series) -> list[dict]:
        return detect_pii(row["text"])

    def _llm_pred(row: pd.Series) -> list[dict]:
        return detect_pii_llm(
            row["text"], llm_client,
            document_type=row.get("document_type") or "",
            domain=row.get("domain") or "",
            model=model,
        )

    def _hybrid_pred(row: pd.Series) -> list[dict]:
        return merge_spans(_detector_pred(row), _llm_pred(row))

    configs: dict[str, Callable[[pd.Series], list[dict]]] = {
        "detectors_only": _detector_pred,
        "llm_only": _llm_pred,
        "hybrid": _hybrid_pred,
    }
    per_config = {name: evaluate(df, predict_fn=fn) for name, fn in configs.items()}
    return {
        "n_docs": len(df),
        "dry_run": llm_client.config.dry_run,
        "overall_by_config": {name: m["overall"] for name, m in per_config.items()},
        "per_type_by_config": {name: m["per_type"] for name, m in per_config.items()},
        "llm_usage": llm_client.usage_summary(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--ablation", action="store_true",
        help="считать 3 конфигурации (детекторы/LLM/гибрид) вместо одиночной таблицы детекторов",
    )
    ap.add_argument("--dry-run", action="store_true", help="только вместе с --ablation")
    ap.add_argument("--model", default="openai/gpt-4o-mini", help="только вместе с --ablation")
    ap.add_argument("--backend", default="openai_compat", choices=["openai_compat", "anthropic", "gigachat"])
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--cache-path", default="out/llm_cache.sqlite3")
    # reasoning-модели тратят весь бюджет на внутренние рассуждения: при 1024 они не успевают
    # выдать ответ и падают с пустым content (см. core/llm.py).
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-concurrency", type=int, default=4)
    # Те же флаги, что у run.py: bench.sh прокидывает их всем трём кейсам одинаково.
    ap.add_argument("--provider", default=None, help="OpenRouter: фиксация провайдера-исполнителя")
    ap.add_argument("--allow-fallbacks", action="store_true")
    ap.add_argument("--price-in", type=float, default=None)
    ap.add_argument("--price-out", type=float, default=None)
    args = ap.parse_args()

    df = _load(args.split, args.n, args.seed)

    if args.ablation:
        llm_client = LLMClient(LLMConfig(
            model=args.model, backend=args.backend, base_url=args.base_url,
            api_key_env=args.api_key_env, dry_run=args.dry_run, cache_path=args.cache_path,
            max_tokens=args.max_tokens, max_concurrency=args.max_concurrency,
            price_per_1m_input=args.price_in, price_per_1m_output=args.price_out,
            provider_order=tuple(x.strip() for x in args.provider.split(",")) if args.provider else None,
            allow_fallbacks=args.allow_fallbacks,
        ))
        result = evaluate_ablation(df, llm_client, model=args.model)
        llm_client.close()

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / "ablation_metrics.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

        print(f"n_docs={result['n_docs']}  dry_run={result['dry_run']}")
        print(f"{'config':16s} {'precision':>10s} {'recall':>8s} {'f1':>8s}")
        for name, overall in result["overall_by_config"].items():
            print(
                f"{name:16s} {overall['precision_exact']:10.3f} "
                f"{overall['recall_exact']:8.3f} {overall['f1_exact']:8.3f}"
            )
        print(f"\nllm usage: {json.dumps(result['llm_usage'], ensure_ascii=False)}")
        print(f"сохранено: {out_path}")
        if args.dry_run:
            print(
                "\n(dry-run: числа конфигураций llm/hybrid — заглушки core.llm, не реальное "
                "качество; см. CLAUDE.md — живые вызовы запрещены без ключа)"
            )
        return

    metrics = evaluate(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "detectors_metrics.json"
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    print(f"n_docs={metrics['n_docs']}")
    print(json.dumps(metrics["overall"], ensure_ascii=False, indent=2))
    print()
    print(f"{'label':32s} {'gold':>6s} {'pred':>6s} {'tp':>6s} {'fp':>6s} {'recall':>8s} {'prec':>8s}  design")
    for row in metrics["per_type"]:
        flag = "" if row["covered_by_design"] else "  (не пытаемся — зона LLM)"
        print(
            f"{row['label']:32s} {row['gold_count']:6d} {row['pred_count']:6d} {row['tp']:6d} "
            f"{row['fp']:6d} {row['recall']:8.3f} {row['precision']:8.3f}{flag}"
        )
    print(f"\nсохранено: {out_path}")


if __name__ == "__main__":
    main()
