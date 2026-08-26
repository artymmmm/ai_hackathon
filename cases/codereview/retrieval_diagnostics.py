"""Диагностика retrieval ПЕРЕД тем, как доверять ему как компоненту (конфигурация B).

Два вопроса, оба чисто офлайн (без LLM):

1. Near-duplicate: насколько высокое сходство top-1/top-5 соседей на eval-150? BigVul/
   DiverseVul собраны из немногих крупных репозиториев (Chrome, linux) — велик риск, что
   "похожий пример" на самом деле почти дубликат той же функции из соседнего коммита/файла,
   и retrieval выигрывает за счёт сопоставления дубликатов, а не переноса знания.

2. Кросс-проектный перенос: пул поиска строится ТОЛЬКО из проектов, отличных от проекта
   оцениваемого фрагмента (пул = не-Chrome, eval = Chrome-фрагменты из eval-150). Если
   качество внутри распределения сильно выше кросс-проектного — это прямое доказательство,
   что выигрыш держится на близости в пределах одного проекта/кодовой базы, а не на переносимом
   сигнале.

Использует kNN/LogReg-классификатор (не LLM) как измерительный инструмент — то же самое,
что и в `knn_baseline.py`, просто с двумя разными пулами поиска. Это дешёвый прокси для
вопроса "сколько сигнала в retrieval вообще есть", до того как тратить LLM-бюджет на
конфигурацию B.

Запуск:
    .venv/bin/python cases/codereview/retrieval_diagnostics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cases.codereview.knn_baseline import load_pool_and_eval, _metrics_at_threshold  # noqa: E402
from cases.codereview.retrieval import NearestExampleRetriever  # noqa: E402
from cases.codereview.features import CodeFeaturizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

_OUT_DIR = Path(__file__).resolve().parent / "out"


def near_duplicate_diagnostic(pool_df: pd.DataFrame, eval_df: pd.DataFrame) -> dict:
    retriever = NearestExampleRetriever(pool_df)
    sims = retriever.query_many_similarities(eval_df, k=5)
    sims.to_csv(_OUT_DIR / "retrieval_neighbor_similarities.csv", index=False)

    def _stats(s: pd.Series) -> dict:
        return {
            "mean": round(float(s.mean()), 4), "median": round(float(s.median()), 4),
            "p90": round(float(s.quantile(0.9)), 4), "max": round(float(s.max()), 4),
            "min": round(float(s.min()), 4),
            "frac_above_0.9": round(float((s >= 0.9).mean()), 4),
            "frac_above_0.7": round(float((s >= 0.7).mean()), 4),
            "frac_above_0.5": round(float((s >= 0.5).mean()), 4),
        }

    result = {
        "n_eval": len(sims),
        "top1_similarity": _stats(sims["top1_similarity"]),
        "topk_mean_similarity_k5": _stats(sims["topk_mean_similarity"]),
        "same_project_as_top1_neighbor_share": round(
            float((sims["own_project"] == sims["top1_neighbor_project"]).mean()), 4
        ),
    }
    return result


def _fit_eval_logreg(pool_df: pd.DataFrame, eval_df: pd.DataFrame) -> dict:
    featurizer = CodeFeaturizer()
    X_pool = featurizer.fit_transform(pool_df["code"].tolist())
    X_eval = featurizer.transform(eval_df["code"].tolist())
    y_pool = pool_df["label"].to_numpy()
    y_eval = eval_df["label"].to_numpy()

    clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, solver="liblinear")
    clf.fit(X_pool, y_pool)
    proba = clf.predict_proba(X_eval)[:, 1]
    # Порог 0.5 как нейтральная точка сравнения (это диагностика/сравнение пулов, не
    # претензия на лучшую итоговую модель — та зафиксирована в knn_baseline.py).
    return _metrics_at_threshold(y_eval, proba, 0.5) | {
        "n_pool": len(pool_df), "n_pool_vulnerable": int(y_pool.sum()),
        "n_eval": len(eval_df), "n_eval_vulnerable": int(y_eval.sum()),
    }


def cross_project_diagnostic(full_pool_df: pd.DataFrame, eval_df: pd.DataFrame) -> dict:
    eval_chrome = eval_df[eval_df["source_project"] == "Chrome"].reset_index(drop=True)
    if len(eval_chrome) < 5:
        return {"error": f"слишком мало Chrome-фрагментов в eval для честного замера: {len(eval_chrome)}"}

    # (a) В пределах распределения: пул = весь train pool (все проекты) минус eval, eval = Chrome-подмножество.
    in_distribution = _fit_eval_logreg(full_pool_df, eval_chrome)

    # (b) Кросс-проектно: пул строго БЕЗ Chrome (Chrome полностью исключён из обучающего пула).
    pool_no_chrome = full_pool_df[full_pool_df["source_project"] != "Chrome"].reset_index(drop=True)
    cross_project = _fit_eval_logreg(pool_no_chrome, eval_chrome)

    return {
        "eval_chrome_n": len(eval_chrome),
        "eval_chrome_n_vulnerable": int(eval_chrome["label"].sum()),
        "pool_no_chrome_n": len(pool_no_chrome),
        "in_distribution_pool_all_projects": in_distribution,
        "cross_project_pool_excludes_chrome": cross_project,
    }


def main() -> None:
    print("Загрузка train pool / eval (та же дисциплина утечки, что и knn_baseline.py)...")
    pool_df, eval_df = load_pool_and_eval()

    # source_project нужен для обеих диагностик — подтягиваем из recovered_labels.csv напрямую
    # (load_pool_and_eval этого не делает, т.к. knn_baseline не нуждается в этой колонке).
    _ROOT = Path(__file__).resolve().parents[2]
    gold_full = pd.read_csv(_ROOT / "research" / "case3_recovered_labels.csv")
    gold_full["unique_id"] = gold_full["unique_id"].astype(int)
    proj_map = gold_full.set_index("unique_id")["source_project"]
    pool_df = pool_df.assign(source_project=pool_df["unique_id"].map(proj_map))
    eval_df = eval_df.assign(source_project=eval_df["unique_id"].map(proj_map))

    print("\n=== 1. Near-duplicate диагностика (топ-5 соседей на eval-150) ===")
    ndup = near_duplicate_diagnostic(pool_df, eval_df)
    print(json.dumps(ndup, ensure_ascii=False, indent=2))

    print("\n=== 2. Кросс-проектная диагностика (пул без Chrome vs eval=Chrome) ===")
    cross = cross_project_diagnostic(pool_df, eval_df)
    print(json.dumps(cross, ensure_ascii=False, indent=2))

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"near_duplicate": ndup, "cross_project": cross}
    (_OUT_DIR / "retrieval_diagnostics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nполный результат -> {_OUT_DIR / 'retrieval_diagnostics.json'}")


if __name__ == "__main__":
    main()
