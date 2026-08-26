"""ШАГ 3: ансамбль logreg (TF-IDF, cases/codereview/knn_baseline.py) + LLM-ревьюер.

Оба слоя уже посчитаны и закешированы — этот скрипт НЕ делает новых сетевых вызовов:
  - LLM: `out/bench/case3_<tag>.json` (полный прогон 150 eval id, kлюч оговорён в
    `report/model_benchmark.md`).
  - logreg: пересчитывается здесь заново (быстро, офлайн, ~2 мин, TF-IDF fit только на
    train pool, eval — только .transform, та же дисциплина утечки, что в `knn_baseline.py`).

Стратегии объединения (см. improvements.md шаг 3):
  - union        — vulnerable, если LLM=vulnerable ИЛИ logreg=1 (порог 0.5). Растит recall.
  - intersection — vulnerable, если ОБА согласны. Растит precision.
  - weighted     — скор = w*llm_score + (1-w)*logreg_proba, порог 0.5, где llm_score =
                    confidence при verdict=vulnerable, 1-confidence при verdict=secure,
                    0.5 при uncertain. w подбирается перебором на этом же eval-наборе
                    (это НЕ CV-честный подбор порога, а прямая демонстрация верхней границы
                    потенциала объединения — так и помечается в отчёте).

CWE берётся из LLM-стороны (logreg не предсказывает CWE); если сработал только logreg
(union, LLM сказал secure) — CWE не сравнивается (нет источника).

Запуск:
    .venv/bin/python cases/codereview/ensemble.py --llm-verdicts out/bench/case3_deepseek-chat.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cases.codereview.cwe_map import normalize_cwe  # noqa: E402
from cases.codereview.evaluate import load_gold  # noqa: E402
from cases.codereview.features import CodeFeaturizer  # noqa: E402
from cases.codereview.knn_baseline import fit_final_model, load_pool_and_eval  # noqa: E402
from core.schema import Verdict  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_OUT_DIR = Path(__file__).resolve().parent / "out"
_LOGREG_BEST = {"model": "logreg", "params": {"C": 1.0}}  # cv_selection.best, step1_knn_results.json


def compute_logreg_scores() -> pd.DataFrame:
    """unique_id -> logreg proba на eval-150, обученный на train pool (без eval)."""
    pool_df, eval_df = load_pool_and_eval()
    featurizer = CodeFeaturizer()
    X_pool = featurizer.fit_transform(pool_df["code"].tolist())
    X_eval = featurizer.transform(eval_df["code"].tolist())
    clf = fit_final_model(_LOGREG_BEST, X_pool, pool_df["label"].to_numpy())
    proba = clf.predict_proba(X_eval)[:, 1]
    return pd.DataFrame({
        "unique_id": eval_df["unique_id"].astype(str),
        "logreg_proba": proba,
        "gold_label_int": eval_df["label"].to_numpy(),
    })


def llm_score(v: Verdict) -> float:
    """confidence -> P(vulnerable), 0.5 для uncertain (максимальная неопределённость)."""
    if v.verdict == "vulnerable":
        return v.confidence
    if v.verdict == "secure":
        return 1.0 - v.confidence
    return 0.5


def _metrics(y_true: list[int], y_pred: list[int]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "fpr": round(fpr, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _cwe_accuracy(rows: list[dict], gold: dict) -> tuple[int, int]:
    correct, comparable = 0, 0
    for r in rows:
        if r["gold_label"] != 1 or r["ensemble_pred"] != 1:
            continue
        g = gold.get(r["doc_id"])
        if g is None or not g["cwe_id"]:
            continue
        if r["source_of_positive"] == "logreg_only":
            continue  # logreg не предсказывает CWE
        pred_cwe = normalize_cwe(r.get("llm_cwe_raw"))
        if pred_cwe is None:
            continue
        comparable += 1
        if pred_cwe == g["cwe_id"]:
            correct += 1
    return correct, comparable


def run(llm_verdicts_path: Path, tag: str) -> dict:
    gold = load_gold()
    llm_data = json.loads(llm_verdicts_path.read_text(encoding="utf-8"))
    llm_verdicts = {d["doc_id"]: Verdict(**d) for d in llm_data}

    print("Пересчёт logreg на eval-150 (offline, ~1-2 мин)...")
    logreg_df = compute_logreg_scores()
    logreg_by_id = {row.unique_id: row for row in logreg_df.itertuples()}

    rows = []
    for doc_id, v in llm_verdicts.items():
        g = gold.get(doc_id)
        if g is None or g["label"] is None:
            continue
        lr = logreg_by_id.get(doc_id)
        if lr is None:
            continue
        gold_label_int = 1 if g["label"] == "vulnerable" else 0
        llm_pred = 1 if v.verdict == "vulnerable" else 0
        logreg_pred = 1 if lr.logreg_proba >= 0.5 else 0
        rows.append({
            "doc_id": doc_id,
            "gold_label": gold_label_int,
            "llm_verdict": v.verdict,
            "llm_pred": llm_pred,
            "llm_confidence": v.confidence,
            "llm_score": llm_score(v),
            "llm_cwe_raw": v.artifacts.get("cwe_id_raw") if v.artifacts else None,
            "logreg_proba": float(lr.logreg_proba),
            "logreg_pred": logreg_pred,
        })

    y_true = [r["gold_label"] for r in rows]

    # --- union: OR ---
    for r in rows:
        r["union_pred"] = 1 if (r["llm_pred"] == 1 or r["logreg_pred"] == 1) else 0
        r["union_source"] = (
            "both" if (r["llm_pred"] == 1 and r["logreg_pred"] == 1) else
            "llm_only" if r["llm_pred"] == 1 else
            "logreg_only" if r["logreg_pred"] == 1 else "neither"
        )
    union_metrics = _metrics(y_true, [r["union_pred"] for r in rows])
    union_rows = [{**r, "ensemble_pred": r["union_pred"],
                   "source_of_positive": r["union_source"]} for r in rows]
    union_cwe = _cwe_accuracy(union_rows, gold)

    # --- intersection: AND ---
    for r in rows:
        r["intersection_pred"] = 1 if (r["llm_pred"] == 1 and r["logreg_pred"] == 1) else 0
    intersection_metrics = _metrics(y_true, [r["intersection_pred"] for r in rows])
    intersection_rows = [{**r, "ensemble_pred": r["intersection_pred"],
                           "source_of_positive": "both"} for r in rows]
    intersection_cwe = _cwe_accuracy(intersection_rows, gold)

    # --- weighted vote: подбор веса w на этом же eval-наборе (верхняя граница потенциала,
    # не CV-честная оценка — явно помечено в отчёте) ---
    best_w, best_f1, best_pred = 0.5, -1.0, None
    for w in np.arange(0.0, 1.01, 0.05):
        scores = [w * r["llm_score"] + (1 - w) * r["logreg_proba"] for r in rows]
        pred = [1 if s >= 0.5 else 0 for s in scores]
        m = _metrics(y_true, pred)
        if m["f1"] > best_f1:
            best_f1, best_w, best_pred = m["f1"], float(w), pred
    weighted_metrics = _metrics(y_true, best_pred)
    weighted_metrics["best_w_llm"] = round(best_w, 2)
    weighted_rows = [{**r, "ensemble_pred": p, "source_of_positive": "both" if p else "neither"}
                      for r, p in zip(rows, best_pred)]
    weighted_cwe = _cwe_accuracy(weighted_rows, gold)

    # --- baselines для сравнения (пересчитаны на том же пересечении rows, не полном 150) ---
    llm_only_metrics = _metrics(y_true, [r["llm_pred"] for r in rows])
    logreg_only_metrics = _metrics(y_true, [r["logreg_pred"] for r in rows])

    result = {
        "tag": tag,
        "n": len(rows),
        "n_vulnerable_gold": sum(y_true),
        "llm_only": llm_only_metrics,
        "logreg_only": logreg_only_metrics,
        "union": {**union_metrics, "cwe_correct": union_cwe[0], "cwe_comparable": union_cwe[1]},
        "intersection": {**intersection_metrics, "cwe_correct": intersection_cwe[0],
                          "cwe_comparable": intersection_cwe[1]},
        "weighted": {**weighted_metrics, "cwe_correct": weighted_cwe[0],
                     "cwe_comparable": weighted_cwe[1]},
    }

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / f"ensemble_results_{tag}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(_OUT_DIR / f"ensemble_rows_{tag}.csv", index=False)

    print(f"\n=== Ансамбль ({tag}), n={len(rows)}, vulnerable_gold={sum(y_true)} ===")
    for name in ("llm_only", "logreg_only", "union", "intersection", "weighted"):
        m = result[name]
        print(f"{name:15s} precision={m['precision']:.3f} recall={m['recall']:.3f} "
              f"f1={m['f1']:.3f} fpr={m['fpr']:.3f}")
    print(f"weighted: best_w_llm={weighted_metrics['best_w_llm']} "
          "(вес LLM-скора; 1-w — вес logreg)")
    print(f"\n-> {out_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-verdicts", type=Path,
                         default=_ROOT / "out" / "bench" / "case3_deepseek-chat.json")
    parser.add_argument("--tag", type=str, default="deepseek-chat")
    args = parser.parse_args()
    run(args.llm_verdicts, args.tag)


if __name__ == "__main__":
    main()
