"""ШАГ 1: kNN/логрег-baseline без LLM, поверх TF-IDF признаков кода.

Строго офлайн, без сети, без LLM. Цель — понять, сколько сигнала есть в самом коде
(эмбеддинги/лексика) до подключения LLM, и сравнить с LLM-ревьюером (0.538/0.140/0.222,
report/model_benchmark.md) и с baseline «всё secure».

Дисциплина утечки:
  - Обучающий пул = весь `research/case3_recovered_labels.csv` с recovered_label in {0,1},
    ЗА ВЫЧЕТОМ 150 id из `out/bench/case3_eval_ids.txt` (читается, не перезаписывается).
  - Оценка — строго на этих же 150 id (100 secure + 50 vulnerable), чтобы числа были
    сравнимы 1-в-1 с таблицей LLM в report/model_benchmark.md.
  - Подбор гиперпараметров/порога решения — ТОЛЬКО через k-fold CV на обучающем пуле
    (eval-выборка ни разу не участвует в выборе модели или порога, иначе это утечка через
    многократный подгляд).
  - TfidfVectorizer.fit вызывается только на обучающем пуле (eval только .transform).

Запуск:
    .venv/bin/python cases/codereview/knn_baseline.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.data import load_case3  # noqa: E402
from core.eval import false_positive_rate  # noqa: E402
from cases.codereview.features import CodeFeaturizer  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_GOLD_CSV = _ROOT / "research" / "case3_recovered_labels.csv"
_EVAL_IDS_TXT = _ROOT / "out" / "bench" / "case3_eval_ids.txt"
_OUT_DIR = Path(__file__).resolve().parent / "out"

LLM_BASELINE = {"precision": 0.538, "recall": 0.140, "f1": 0.222, "fpr": 0.060}


def load_pool_and_eval() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает (train_pool, eval_df), обе с колонками unique_id, code, label(0/1)."""
    corpus = load_case3()
    corpus["unique_id"] = corpus["unique_id"].astype(int)

    gold = pd.read_csv(_GOLD_CSV)
    gold["unique_id"] = gold["unique_id"].astype(int)
    gold = gold[gold["recovered_label"].isin([0, 1, "0", "1"])].copy()
    gold["label"] = gold["recovered_label"].astype(int)

    df = corpus.merge(gold[["unique_id", "label"]], on="unique_id", how="inner")

    eval_ids = {int(x) for x in _EVAL_IDS_TXT.read_text().split() if x.strip()}
    eval_df = df[df["unique_id"].isin(eval_ids)].reset_index(drop=True)
    pool_df = df[~df["unique_id"].isin(eval_ids)].reset_index(drop=True)

    # Явная проверка нулевого пересечения (дисциплина утечки — не молчаливое допущение).
    overlap = set(pool_df["unique_id"]) & set(eval_df["unique_id"])
    assert not overlap, f"УТЕЧКА: {len(overlap)} id пересекаются между train pool и eval"
    assert len(eval_df) == len(eval_ids), (
        f"eval_df содержит {len(eval_df)} строк, ожидалось {len(eval_ids)} "
        "(часть eval id не нашлась в размеченном подмножестве — не должно случиться, "
        "т.к. заранее проверено вручную)"
    )
    return pool_df, eval_df


def _metrics_at_threshold(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    y_pred = (proba >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1,
            "fpr": fpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _best_threshold_by_f1(y_true: np.ndarray, proba: np.ndarray) -> float:
    candidates = np.unique(np.round(proba, 4))
    candidates = np.concatenate([candidates, [0.5]])
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        m = _metrics_at_threshold(y_true, proba, t)
        if m["f1"] > best_f1:
            best_f1, best_t = m["f1"], t
    return float(best_t)


def cv_select_model(X_pool, y_pool: np.ndarray, seed: int = 42) -> dict:
    """5-fold Stratified CV на обучающем пуле. Для каждого кандидата: cross_val_predict
    (proba), затем порог, максимизирующий F1, тоже подобранный ВНУТРИ CV (на конкатенации
    out-of-fold предсказаний, что честно, т.к. каждая точка предсказана моделью, её не
    видевшей). Eval-выборка здесь нигде не участвует.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    candidates = []

    for C in (0.1, 1.0, 10.0):
        clf = LogisticRegression(class_weight="balanced", C=C, max_iter=2000, solver="liblinear")
        proba = cross_val_predict(clf, X_pool, y_pool, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
        t = _best_threshold_by_f1(y_pool, proba)
        m = _metrics_at_threshold(y_pool, proba, t)
        candidates.append({"model": "logreg", "params": {"C": C}, "cv": m})

    for k in (5, 15, 31, 51):
        clf = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
        proba = cross_val_predict(clf, X_pool, y_pool, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
        t = _best_threshold_by_f1(y_pool, proba)
        m = _metrics_at_threshold(y_pool, proba, t)
        candidates.append({"model": "knn", "params": {"k": k}, "cv": m})

    candidates.sort(key=lambda c: c["cv"]["f1"], reverse=True)
    return {"all_candidates": candidates, "best": candidates[0]}


def fit_final_model(best: dict, X_pool, y_pool: np.ndarray):
    if best["model"] == "logreg":
        clf = LogisticRegression(class_weight="balanced", C=best["params"]["C"],
                                  max_iter=2000, solver="liblinear")
    else:
        clf = KNeighborsClassifier(n_neighbors=best["params"]["k"], metric="cosine", weights="distance")
    clf.fit(X_pool, y_pool)
    return clf


def main() -> None:
    t0 = time.time()
    print("Загрузка корпуса и восстановленных лейблов...")
    pool_df, eval_df = load_pool_and_eval()
    print(f"train pool: {len(pool_df)} (vulnerable={int(pool_df['label'].sum())}), "
          f"eval: {len(eval_df)} (vulnerable={int(eval_df['label'].sum())})")

    print("Векторизация (TF-IDF char+token, fit только на train pool)...")
    featurizer = CodeFeaturizer()
    X_pool = featurizer.fit_transform(pool_df["code"].tolist())
    X_eval = featurizer.transform(eval_df["code"].tolist())
    y_pool = pool_df["label"].to_numpy()
    y_eval = eval_df["label"].to_numpy()
    print(f"X_pool shape={X_pool.shape}, X_eval shape={X_eval.shape}")

    print("CV-подбор модели и порога (5-fold, на train pool, eval не используется)...")
    selection = cv_select_model(X_pool, y_pool)
    for c in selection["all_candidates"]:
        print(f"  {c['model']:6s} {c['params']} -> CV f1={c['cv']['f1']:.3f} "
              f"precision={c['cv']['precision']:.3f} recall={c['cv']['recall']:.3f} "
              f"fpr={c['cv']['fpr']:.3f} thr={c['cv']['threshold']:.3f}")
    best = selection["best"]
    print(f"Лучший по CV: {best['model']} {best['params']} "
          f"(CV f1={best['cv']['f1']:.3f}, порог={best['cv']['threshold']:.3f})")

    print("Финальное обучение на всём train pool, ОДНОРАЗОВОЕ применение к eval-150...")
    final_clf = fit_final_model(best, X_pool, y_pool)
    eval_proba = final_clf.predict_proba(X_eval)[:, 1]

    frozen_threshold = best["cv"]["threshold"]  # порог зафиксирован по CV, до eval
    eval_metrics_frozen = _metrics_at_threshold(y_eval, eval_proba, frozen_threshold)
    eval_metrics_default = _metrics_at_threshold(y_eval, eval_proba, 0.5)

    # Baseline "всё secure"
    n_vuln_eval = int(y_eval.sum())
    baseline_all_secure_acc = (len(y_eval) - n_vuln_eval) / len(y_eval)

    # Дополнительно: второй кандидат (второй лучший по CV) тоже применяем к eval —
    # для устойчивости вывода (не одна случайная точка).
    second_best = selection["all_candidates"][1]
    second_clf = fit_final_model(second_best, X_pool, y_pool)
    second_proba = second_clf.predict_proba(X_eval)[:, 1]
    second_metrics = _metrics_at_threshold(y_eval, second_proba, second_best["cv"]["threshold"])

    result = {
        "n_train_pool": len(pool_df),
        "n_train_pool_vulnerable": int(y_pool.sum()),
        "n_eval": len(eval_df),
        "n_eval_vulnerable": n_vuln_eval,
        "feature_shape_pool": list(X_pool.shape),
        "feature_shape_eval": list(X_eval.shape),
        "cv_selection": selection,
        "best_model": {"model": best["model"], "params": best["params"]},
        "eval_metrics_frozen_threshold": eval_metrics_frozen,
        "eval_metrics_default_threshold_0.5": eval_metrics_default,
        "second_best_model_eval": {
            "model": second_best["model"], "params": second_best["params"],
            "metrics": second_metrics,
        },
        "baseline_all_secure_accuracy": baseline_all_secure_acc,
        "llm_deepseek_chat_baseline": LLM_BASELINE,
        "elapsed_s": round(time.time() - t0, 1),
    }

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / "step1_knn_results.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== ИТОГ ШАГА 1 (на 150 eval id, порог зафиксирован по CV на train pool) ===")
    m = eval_metrics_frozen
    print(f"{best['model']} {best['params']}, threshold={m['threshold']:.3f}")
    print(f"precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f} fpr={m['fpr']:.3f}")
    print(f"tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']}")
    print(f"\nLLM (deepseek-chat) для сравнения: precision={LLM_BASELINE['precision']} "
          f"recall={LLM_BASELINE['recall']} f1={LLM_BASELINE['f1']} fpr={LLM_BASELINE['fpr']}")
    print(f"baseline «всё secure»: accuracy={baseline_all_secure_acc:.3f} (precision/recall/f1=0 для vulnerable)")
    print(f"\nполный результат -> {out_path}")


if __name__ == "__main__":
    main()
