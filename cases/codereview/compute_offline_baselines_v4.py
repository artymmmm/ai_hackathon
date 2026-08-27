"""Пересчёт офлайн-базлайнов кейса 3 против v4-разметки (100% покрытие корпуса).

Три базлайна, каждый на двух наборах — эталонные 150 (`out/bench/case3_eval_ids.txt`) и
весь корпус (18 864 фрагмента):

  1. flawfinder (`out/flawfinder_eval_hits.csv` для 150, `out/flawfinder_full_hits.csv`
     для полного корпуса — строится `run_flawfinder_full.py`). Предсказание "vulnerable" =
     `any_hit` (хотя бы один хит любого уровня) — воспроизводит старые числа R=0.18/FPR=0.05
     на eval150, проверено эмпирически при разработке этого скрипта.
  2. Сигнатурный триаж (`out/triage_scores.csv`, уже на полном корпусе). Предсказание
     "vulnerable" = `risk_level != "none"` — определение из report/journal.md §7
     (recall 10.5% на прежней разметке).
  3. logreg на TF-IDF (`features.CodeFeaturizer`) — обучается заново здесь же, тем же
     способом, что `knn_baseline.py` (class_weight=balanced, C=1.0, solver=liblinear,
     random_state=42 везде, где есть стохастика), но на v4-лейблах.

Схема валидации logreg (чтобы обучение и оценка не пересекались по фрагментам):

  - **eval150**: train pool = весь корпус МИНУС эти 150 id (v4-лейблы). Внутри train pool —
    5-fold StratifiedKFold CV (cross_val_predict) только для подбора порога, максимизирующего
    F1 на out-of-fold вероятностях (сама eval150 в этом подборе не участвует). Финальная модель
    обучается на всём train pool один раз и применяется к eval150 — ровно методология
    knn_baseline.py, один в один, только лейблы v4.
  - **full corpus (18 864)**: одноуровневая 5-fold StratifiedKFold CV (cross_val_predict) по
    ВСЕМУ корпусу — каждый фрагмент получает вероятность от модели, которая его не видела при
    обучении (честно, без утечки фрагментов между train/test). Порог 0.5 — фиксированный,
    без подгонки. "Калиброванный" порог для полного корпуса выбирается максимизацией F1 на этих
    же out-of-fold вероятностях (иного варианта без отдельного held-out на всём корпусе нет);
    это ЕДИНСТВЕННОЕ место с мягкой оптимистичностью в отчёте (модель ничего не утекает —
    каждая точка предсказана моделью, её не видевшей, — но сам порог как скаляр подобран по тем
    же точкам, на которых считается метрика). Явно помечено в выводе как `threshold_selection:
    "in-sample on OOF probabilities"`.
  - VD-S (recall при фиксированном FPR 1%/5%) считается на тех же out-of-fold вероятностях
    полного корпуса — так же определяется в литературе по обнаружению уязвимостей
    (VulDeePecker и др.): порог ищется по ROC-кривой того же набора, на котором меряется recall.

ВАЖНО про честность чисел: v4-лейблы получены сопоставлением с внешним датасетом
(LineVul_Test_Dataset), а не размечены человеком вручную. Обучение logreg на v4 и оценка на
v4 же — легитимно (единственная разметка, которая есть), но это оценка НА ВОССТАНОВЛЕННОЙ
разметке, не на золотом стандарте, размеченном руками.

Запуск:
    .venv/bin/python cases/codereview/compute_offline_baselines_v4.py
Требует (уже посчитано отдельно):
    cases/codereview/out/flawfinder_full_hits.csv  (см. run_flawfinder_full.py)
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

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
from core.data import load_case3  # noqa: E402
from cases.codereview.features import CodeFeaturizer  # noqa: E402

_V4_CSV = _ROOT / "research" / "case3_recovered_labels_v4.csv"
_EVAL_IDS_TXT = _ROOT / "out" / "bench" / "case3_eval_ids.txt"
_FLAWFINDER_EVAL_CSV = _ROOT / "cases" / "codereview" / "out" / "flawfinder_eval_hits.csv"
_FLAWFINDER_FULL_CSV = _ROOT / "cases" / "codereview" / "out" / "flawfinder_full_hits.csv"
_TRIAGE_CSV = _ROOT / "cases" / "codereview" / "out" / "triage_scores.csv"
_OUT_DIR = _ROOT / "cases" / "codereview" / "out"
_OUT_CSV = _OUT_DIR / "offline_baselines_v4.csv"
_OUT_JSON = _OUT_DIR / "offline_baselines_v4.json"

SEED = 42
LOGREG_C = 1.0  # закреплено по прежнему CV-подбору (knn_baseline.py: logreg C=1.0 победил
# knn при k in {5,15,31,51} и C in {0.1,1,10}); здесь не переподбирается заново по времени,
# так как гиперпараметр TF-IDF-логрег не специфичен к конкретной версии лейблов.


def metrics_at(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
            "n": int(len(y_true)), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    candidates = np.unique(np.round(proba, 4))
    candidates = np.concatenate([candidates, [0.5]])
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        pred = (proba >= t).astype(int)
        f1 = metrics_at(y_true, pred)["f1"]
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def recall_at_fpr(y_true: np.ndarray, proba: np.ndarray, target_fpr: float) -> dict:
    """VD-S: recall при наибольшем пороге, дающем FPR <= target_fpr на этих же вероятностях."""
    order = np.unique(proba)[::-1]  # от высокого к низкому порогу
    best = {"recall": 0.0, "fpr": 0.0, "threshold": 1.0}
    for t in order:
        pred = (proba >= t).astype(int)
        m = metrics_at(y_true, pred)
        if m["fpr"] <= target_fpr:
            if m["recall"] >= best["recall"]:
                best = {"recall": m["recall"], "fpr": m["fpr"], "threshold": float(t)}
        else:
            break  # fpr монотонно растёт по мере понижения порога
    return best


def load_v4() -> pd.DataFrame:
    df = pd.read_csv(_V4_CSV)
    df["unique_id"] = df["unique_id"].astype(int)
    df["label"] = df["recovered_label"].astype(int)
    return df[["unique_id", "label"]]


def load_eval_ids() -> set[int]:
    return {int(x) for x in _EVAL_IDS_TXT.read_text().split() if x.strip()}


# ---------------------------------------------------------------------------
# 1. flawfinder
# ---------------------------------------------------------------------------

def eval_flawfinder(v4: pd.DataFrame, eval_ids: set[int]) -> list[dict]:
    rows = []
    ff_eval = pd.read_csv(_FLAWFINDER_EVAL_CSV)[["unique_id", "any_hit"]]
    ff_eval["unique_id"] = ff_eval["unique_id"].astype(int)
    m = ff_eval.merge(v4, on="unique_id", how="inner")
    assert len(m) == len(eval_ids), f"flawfinder eval150: смёржилось {len(m)}, ожидалось {len(eval_ids)}"
    y_true = m["label"].to_numpy()
    y_pred = m["any_hit"].astype(int).to_numpy()
    rows.append({"baseline": "flawfinder", "набор": "eval150",
                 **metrics_at(y_true, y_pred),
                 "заметка": "предсказание = any_hit (хотя бы один хит любого уровня)"})

    if not _FLAWFINDER_FULL_CSV.exists():
        rows.append({"baseline": "flawfinder", "набор": "full",
                     "precision": None, "recall": None, "f1": None, "fpr": None, "n": 0,
                     "заметка": f"НЕ ПОСЧИТАНО: {_FLAWFINDER_FULL_CSV.name} отсутствует "
                                "(run_flawfinder_full.py не завершён/не запущен)"})
        return rows

    ff_full = pd.read_csv(_FLAWFINDER_FULL_CSV)[["unique_id", "any_hit"]]
    ff_full["unique_id"] = ff_full["unique_id"].astype(int)
    mf = ff_full.merge(v4, on="unique_id", how="inner")
    assert len(mf) == len(v4), f"flawfinder full: смёржилось {len(mf)}, ожидалось {len(v4)}"
    y_true_f = mf["label"].to_numpy()
    y_pred_f = mf["any_hit"].astype(int).to_numpy()
    rows.append({"baseline": "flawfinder", "набор": "full",
                 **metrics_at(y_true_f, y_pred_f),
                 "заметка": "предсказание = any_hit; весь корпус, впервые возможно при 100% покрытии v4"})
    return rows


# ---------------------------------------------------------------------------
# 2. сигнатурный триаж
# ---------------------------------------------------------------------------

def eval_triage(v4: pd.DataFrame, eval_ids: set[int]) -> list[dict]:
    rows = []
    triage = pd.read_csv(_TRIAGE_CSV)[["unique_id", "risk_level"]]
    triage["unique_id"] = triage["unique_id"].astype(int)
    triage["pred"] = (triage["risk_level"] != "none").astype(int)

    m_full = triage.merge(v4, on="unique_id", how="inner")
    assert len(m_full) == len(v4), f"триаж full: смёржилось {len(m_full)}, ожидалось {len(v4)}"
    y_true_f = m_full["label"].to_numpy()
    y_pred_f = m_full["pred"].to_numpy()
    rows.append({"baseline": "триаж (сигнатурный)", "набор": "full",
                 **metrics_at(y_true_f, y_pred_f),
                 "заметка": "предсказание = risk_level != 'none'; было посчитано и раньше (весь корпус без разметки не мешал: у триажа лейблы не нужны для прогона, только для метрики)"})

    m_eval = m_full[m_full["unique_id"].isin(eval_ids)]
    assert len(m_eval) == len(eval_ids), f"триаж eval150: смёржилось {len(m_eval)}, ожидалось {len(eval_ids)}"
    y_true_e = m_eval["label"].to_numpy()
    y_pred_e = m_eval["pred"].to_numpy()
    rows.append({"baseline": "триаж (сигнатурный)", "набор": "eval150",
                 **metrics_at(y_true_e, y_pred_e),
                 "заметка": "предсказание = risk_level != 'none'"})
    return rows


# ---------------------------------------------------------------------------
# 3. logreg TF-IDF
# ---------------------------------------------------------------------------

def eval_logreg(v4: pd.DataFrame, eval_ids: set[int]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    extra: dict = {}

    corpus = load_case3()
    corpus["unique_id"] = corpus["unique_id"].astype(int)
    df = corpus.merge(v4, on="unique_id", how="inner")
    assert len(df) == len(v4), f"logreg: смёржилось {len(df)} против v4={len(v4)}"

    # --- eval150: train pool = всё минус eval150 ---
    eval_df = df[df["unique_id"].isin(eval_ids)].reset_index(drop=True)
    pool_df = df[~df["unique_id"].isin(eval_ids)].reset_index(drop=True)
    overlap = set(pool_df["unique_id"]) & set(eval_df["unique_id"])
    assert not overlap, f"УТЕЧКА eval150: {len(overlap)} общих id"
    assert len(eval_df) == len(eval_ids), f"eval_df={len(eval_df)}, ожидалось {len(eval_ids)}"

    t0 = time.time()
    feat_pool = CodeFeaturizer()
    X_pool = feat_pool.fit_transform(pool_df["code"].tolist())
    X_eval = feat_pool.transform(eval_df["code"].tolist())
    y_pool = pool_df["label"].to_numpy()
    y_eval = eval_df["label"].to_numpy()
    print(f"  [logreg/eval150] TF-IDF fit+transform: {time.time()-t0:.1f}s, "
          f"X_pool={X_pool.shape}, X_eval={X_eval.shape}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clf_cv = LogisticRegression(class_weight="balanced", C=LOGREG_C, max_iter=2000, solver="liblinear")
    t0 = time.time()
    oof_proba_pool = cross_val_predict(clf_cv, X_pool, y_pool, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
    print(f"  [logreg/eval150] 5-fold OOF на train pool (для порога): {time.time()-t0:.1f}s")
    calibrated_threshold = best_f1_threshold(y_pool, oof_proba_pool)

    final_clf = LogisticRegression(class_weight="balanced", C=LOGREG_C, max_iter=2000, solver="liblinear")
    final_clf.fit(X_pool, y_pool)
    eval_proba = final_clf.predict_proba(X_eval)[:, 1]

    for thr_name, thr in [("порог 0.5", 0.5), ("откалиброванный порог", calibrated_threshold)]:
        pred = (eval_proba >= thr).astype(int)
        rows.append({"baseline": f"logreg TF-IDF ({thr_name})", "набор": "eval150",
                     **metrics_at(y_eval, pred),
                     "заметка": f"threshold={thr:.4f}; train pool = corpus минус eval150 (v4-лейблы), "
                                "порог подобран 5-fold CV на train pool (eval150 не участвует)"
                                if thr_name != "порог 0.5" else f"threshold={thr:.4f}; фиксированный, без подбора"})

    # --- full corpus: 5-fold OOF по всему корпусу ---
    t0 = time.time()
    feat_full = CodeFeaturizer()
    X_full = feat_full.fit_transform(df["code"].tolist())
    y_full = df["label"].to_numpy()
    print(f"  [logreg/full] TF-IDF fit_transform: {time.time()-t0:.1f}s, X_full={X_full.shape}")

    skf_full = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clf_full = LogisticRegression(class_weight="balanced", C=LOGREG_C, max_iter=2000, solver="liblinear")
    t0 = time.time()
    oof_proba_full = cross_val_predict(clf_full, X_full, y_full, cv=skf_full, method="predict_proba", n_jobs=-1)[:, 1]
    print(f"  [logreg/full] 5-fold OOF по всему корпусу: {time.time()-t0:.1f}s")

    calibrated_threshold_full = best_f1_threshold(y_full, oof_proba_full)
    for thr_name, thr in [("порог 0.5", 0.5), ("откалиброванный порог", calibrated_threshold_full)]:
        pred = (oof_proba_full >= thr).astype(int)
        note = (f"threshold={thr:.4f}; 5-fold StratifiedKFold OOF по всему корпусу (18864), "
                f"каждый фрагмент предсказан моделью, его не видевшей при обучении")
        if thr_name != "порог 0.5":
            note += "; порог подобран максимизацией F1 на ЭТИХ ЖЕ OOF-вероятностях (см. докстринг файла — единственное место с мягкой оптимистичностью)"
        else:
            note += "; фиксированный, без подбора"
        rows.append({"baseline": f"logreg TF-IDF ({thr_name})", "набор": "full",
                     **metrics_at(y_full, pred), "заметка": note})

    # --- VD-S на full corpus OOF proba ---
    vds_1 = recall_at_fpr(y_full, oof_proba_full, 0.01)
    vds_5 = recall_at_fpr(y_full, oof_proba_full, 0.05)
    extra["vds"] = {
        "fpr_target_0.01": vds_1,
        "fpr_target_0.05": vds_5,
        "note": "recall при пороге, дающем FPR<=target на OOF-вероятностях 5-fold CV по всему "
                "корпусу (18864); порог ищется по той же ROC-кривой, что и recall — стандартное "
                "определение VD-S в литературе по обнаружению уязвимостей, не generalization-оценка",
    }
    extra["logreg_model"] = {"C": LOGREG_C, "class_weight": "balanced", "solver": "liblinear",
                              "seed": SEED}
    extra["calibrated_threshold_eval150_pool_cv"] = calibrated_threshold
    extra["calibrated_threshold_full_corpus_oof"] = calibrated_threshold_full
    extra["n_pool"] = int(len(pool_df))
    extra["n_pool_vulnerable"] = int(y_pool.sum())
    extra["n_full"] = int(len(df))
    extra["n_full_vulnerable"] = int(y_full.sum())
    return rows, extra


def main() -> None:
    t_start = time.time()
    v4 = load_v4()
    eval_ids = load_eval_ids()
    print(f"v4: {len(v4)} строк, vulnerable={int(v4['label'].sum())} ({v4['label'].mean():.4f})")
    print(f"eval_ids: {len(eval_ids)}")

    all_rows: list[dict] = []
    print("\n[1/3] flawfinder...")
    all_rows += eval_flawfinder(v4, eval_ids)
    print("[2/3] триаж...")
    all_rows += eval_triage(v4, eval_ids)
    print("[3/3] logreg TF-IDF (это займёт время: TF-IDF fit + 2x 5-fold CV)...")
    logreg_rows, logreg_extra = eval_logreg(v4, eval_ids)
    all_rows += logreg_rows

    out_df = pd.DataFrame(all_rows)
    csv_cols = ["baseline", "набор", "precision", "recall", "f1", "fpr", "n", "заметка"]
    out_df_csv = out_df[csv_cols].copy()
    for c in ("precision", "recall", "f1", "fpr"):
        out_df_csv[c] = out_df_csv[c].apply(lambda x: round(x, 4) if x is not None else None)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df_csv.to_csv(_OUT_CSV, index=False, encoding="utf-8-sig")

    out_json = {
        "gold": str(_V4_CSV.relative_to(_ROOT)),
        "n_gold": int(len(v4)),
        "n_gold_vulnerable": int(v4["label"].sum()),
        "base_rate": round(float(v4["label"].mean()), 4),
        "n_eval_ids": len(eval_ids),
        "rows": all_rows,
        "logreg_extra": logreg_extra,
        "elapsed_s": round(time.time() - t_start, 1),
    }
    _OUT_JSON.write_text(json.dumps(out_json, ensure_ascii=False, indent=2, default=lambda o: None),
                          encoding="utf-8")

    print(f"\n-> {_OUT_CSV}")
    print(f"-> {_OUT_JSON}")
    print("\n" + out_df_csv.to_string(index=False))
    print("\nVD-S:", json.dumps(logreg_extra["vds"], ensure_ascii=False, indent=2))
    print(f"\nвсего времени: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
