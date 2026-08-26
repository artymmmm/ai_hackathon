"""Офлайн-слой классификатора кейса 2: обучение и честная оценка на test.

Два варианта, оба scikit-learn, оба целиком офлайн (без единого сетевого вызова):

  (a) TF-IDF (word + char n-grams) + LogisticRegression — чистый текстовый линейный baseline.
  (b) TF-IDF -> TruncatedSVD (100 компонент) + ручные признаки (features.py) ->
      HistGradientBoostingClassifier — ансамбль на бустинге с гибридным представлением.

Обе модели учатся как 3-классовые (0=safe, 1=замаскированная инъекция, 2=прямой вредоносный
запрос), чтобы:
  - бинарная метрика задания (safe vs injection-and-malicious) получалась схлопыванием {1,2};
  - различение 1 vs 2 мерилось честно, а не терялось в усреднении.

Запуск:  .venv/bin/python -m cases.guard.baseline
Пишет:   out/guard/metrics.json  (все числа отчёта report.md посчитаны отсюда)
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from cases.guard.features import extract_features_batch

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "case 2" / "prompt-injection-safety" / "data"
OUT_DIR = ROOT / "out" / "guard"

LABEL_NAMES = {0: "safe", 1: "injection_masked", 2: "injection_direct"}
CONF_THRESHOLDS = [0.34, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"{name}-00000-of-00001.parquet")


def to_binary(y: np.ndarray) -> np.ndarray:
    """0 -> safe(0), {1,2} -> injection_and_malicious(1)."""
    return (y != 0).astype(int)


def binary_metrics(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> dict:
    """Precision/recall/F1 по позитивному классу (injection) + FPR на safe."""
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="binary", pos_label=1, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "fpr": fpr,
        "accuracy": accuracy,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = [0, 1, 2]
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        LABEL_NAMES[lbl]: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, lbl in enumerate(labels)
    }


def subtype_1_vs_2(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Качество различения 1 (замаскированная) vs 2 (прямая), только среди истинных {1,2}."""
    mask = np.isin(y_true, [1, 2])
    yt, yp = y_true[mask], y_pred[mask]
    # среди тех, что модель тоже отнесла к {1,2} (иначе "1 vs 2" не определено)
    mask_pred_injection = np.isin(yp, [1, 2])
    yt2, yp2 = yt[mask_pred_injection], yp[mask_pred_injection]
    if len(yt2) == 0:
        return {"n_evaluated": 0}
    correct_subtype = int((yt2 == yp2).sum())
    cm = confusion_matrix(yt2, yp2, labels=[1, 2])
    return {
        "n_true_injection": int(mask.sum()),
        "n_classified_as_injection_by_model": int(mask_pred_injection.sum()),
        "n_evaluated": int(len(yt2)),
        "subtype_accuracy": correct_subtype / len(yt2),
        "confusion_matrix_1v2": cm.tolist(),  # rows=true[1,2], cols=pred[1,2]
    }


def coverage_curve(y_true_bin: np.ndarray, proba: np.ndarray) -> list[dict]:
    """Доля потока, уверенно решаемого офлайн-слоем, vs доля в 'серую зону' по порогу уверенности."""
    confidence = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    pred_bin = (pred != 0).astype(int)
    n = len(y_true_bin)
    rows = []
    for thr in CONF_THRESHOLDS:
        mask = confidence >= thr
        coverage = mask.sum() / n
        if mask.sum() == 0:
            rows.append({"threshold": thr, "coverage": 0.0, "grey_zone_share": 1.0})
            continue
        m = binary_metrics(y_true_bin[mask], pred_bin[mask])
        rows.append(
            {
                "threshold": thr,
                "coverage": float(coverage),
                "grey_zone_share": float(1 - coverage),
                "n_resolved_offline": int(mask.sum()),
                **{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in m.items()},
            }
        )
    return rows


def build_model_a(X_train_text, y_train):
    """(a) TF-IDF (word 1-2gram + char 3-5gram) + LogisticRegression, multinomial, class_weight balanced."""
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2), max_features=30000, min_df=2, sublinear_tf=True
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), max_features=20000, min_df=2, sublinear_tf=True
    )
    from scipy.sparse import hstack

    Xw = word_vec.fit_transform(X_train_text)
    Xc = char_vec.fit_transform(X_train_text)
    X = hstack([Xw, Xc]).tocsr()
    # sklearn >=1.7: multinomial выбирается автоматически при multi-class y и solver='lbfgs',
    # отдельный параметр multi_class удалён.
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=3.0, solver="lbfgs")
    clf.fit(X, y_train)
    return {"word_vec": word_vec, "char_vec": char_vec, "clf": clf}


def apply_model_a(model, texts):
    from scipy.sparse import hstack

    Xw = model["word_vec"].transform(texts)
    Xc = model["char_vec"].transform(texts)
    X = hstack([Xw, Xc]).tocsr()
    return model["clf"].predict_proba(X), model["clf"].classes_


def build_model_b(X_train_text, y_train):
    """(b) TF-IDF -> SVD(100) + ручные признаки -> HistGradientBoostingClassifier."""
    tfidf = TfidfVectorizer(ngram_range=(1, 1), max_features=40000, min_df=2, sublinear_tf=True)
    Xt = tfidf.fit_transform(X_train_text)
    svd = TruncatedSVD(n_components=100, random_state=0)
    Xsvd = svd.fit_transform(Xt)
    Xhand = extract_features_batch(X_train_text).to_numpy(dtype=np.float32)
    X = np.hstack([Xsvd, Xhand])
    clf = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, learning_rate=0.1, random_state=0, class_weight="balanced"
    )
    clf.fit(X, y_train)
    return {"tfidf": tfidf, "svd": svd, "clf": clf}


def apply_model_b(model, texts):
    Xt = model["tfidf"].transform(texts)
    Xsvd = model["svd"].transform(Xt)
    Xhand = extract_features_batch(texts).to_numpy(dtype=np.float32)
    X = np.hstack([Xsvd, Xhand])
    return model["clf"].predict_proba(X), model["clf"].classes_


def proba_to_pred(proba, classes):
    idx = proba.argmax(axis=1)
    return classes[idx]


def evaluate_model(name, y_true, proba, classes, timings: dict) -> dict:
    pred = proba_to_pred(proba, classes)
    y_true_bin = to_binary(y_true)
    pred_bin = to_binary(pred)
    result = {
        "name": name,
        "binary": binary_metrics(y_true_bin, pred_bin),
        "per_class": per_class_metrics(y_true, pred),
        "confusion_matrix_3class": confusion_matrix(y_true, pred, labels=[0, 1, 2]).tolist(),
        "subtype_1_vs_2": subtype_1_vs_2(y_true, pred),
        "coverage_curve": coverage_curve(y_true_bin, proba),
        "timings_sec": timings,
    }
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data...")
    train = load_split("train")
    test = load_split("test")
    X_train_text = train["text"].tolist()
    y_train = train["label"].to_numpy()
    X_test_text = test["text"].tolist()
    y_test = test["label"].to_numpy()

    results = {"train_size": len(train), "test_size": len(test)}

    print("Training model A (TF-IDF word+char + LogisticRegression)...")
    t0 = time.perf_counter()
    model_a = build_model_a(X_train_text, y_train)
    fit_a = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba_a, classes_a = apply_model_a(model_a, X_test_text)
    infer_a = time.perf_counter() - t0
    results["model_a"] = evaluate_model(
        "tfidf_logreg",
        y_test,
        proba_a,
        classes_a,
        {"fit": fit_a, "infer_test": infer_a, "infer_per_1000": infer_a / len(test) * 1000},
    )
    print(f"  fit={fit_a:.1f}s infer={infer_a:.2f}s")

    print("Training model B (TF-IDF SVD + hand features + HistGradientBoosting)...")
    t0 = time.perf_counter()
    model_b = build_model_b(X_train_text, y_train)
    fit_b = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba_b, classes_b = apply_model_b(model_b, X_test_text)
    infer_b = time.perf_counter() - t0
    results["model_b"] = evaluate_model(
        "svd_handfeat_histgb",
        y_test,
        proba_b,
        classes_b,
        {"fit": fit_b, "infer_test": infer_b, "infer_per_1000": infer_b / len(test) * 1000},
    )
    print(f"  fit={fit_b:.1f}s infer={infer_b:.2f}s")

    out_path = OUT_DIR / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")

    for key in ("model_a", "model_b"):
        b = results[key]["binary"]
        print(
            f"{key}: P={b['precision']:.3f} R={b['recall']:.3f} F1={b['f1']:.3f} "
            f"FPR={b['fpr']:.3f} Acc={b['accuracy']:.3f}"
        )


if __name__ == "__main__":
    main()
