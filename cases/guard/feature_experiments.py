"""Эксперименты с дешёвыми офлайн-приёмами против конкретных 99 FN (замаскированные инъекции)
кейса 2, найденными в `case2_errors_full_test.csv` (см. `case2_notes.md` §2 и
`injection_detection_methods.md` §4,6): 36% FN содержат код-подобный синтаксис (function-call
камуфляж вида `fetch_secret_keycode()`), ~10% — посимвольная перестановка букв внутри слов.

Гипотезы (выведены из данных, не из статей):
  A. identifier_split — snake_case/camelCase разбивается на отдельные слова перед векторизацией
     (word-токенизатор sklearn берёт `fetch_secret_keycode` целиком как один редкий токен;
     после сплита это три частых токена fetch/secret/keycode).
  B. handfeat_concat — ручные структурные признаки (`features.py`: entropy, non_ascii_ratio,
     has_code_block, etc.) конкатенированы к TF-IDF-представлению модели (a), а не заменяют его
     (в отличие от model_b, который на этих признаках без word-TF-IDF уже хуже baseline —
     metrics.json: F1=0.9654 < 0.9844).
  C. combined — A + B вместе.

Baseline = модель (a) из baseline.py, обучена на ПОЛНОМ train (не сэмпл), оценена на ПОЛНОМ
test (10000) — те же числа, что уже в `case2_errors_summary.json`: P=0.9871 R=0.9818 F1=0.9844
FPR=0.0154. Критерий успеха приёма: F1 выше 0.9844 при FPR <= 0.0154 на том же полном тесте.

Запуск: .venv/bin/python -m cases.guard.feature_experiments
Пишет:  out/guard/case2_feature_exp_identifier_split.json
        out/guard/case2_feature_exp_handfeat_concat.json
        out/guard/case2_feature_exp_combined.json
        out/guard/case2_feature_exp_summary.json
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from cases.guard.baseline import load_split
from cases.guard.features import extract_features_batch

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "out" / "guard"

BASELINE = {"precision": 0.9870705578130772, "recall": 0.9818115010104722,
            "f1": 0.9844340057106015, "fpr": 0.015360983102918587}


def split_identifiers(text: str) -> str:
    """snake_case/camelCase -> отдельные слова. fetch_secret_keycode -> fetch secret keycode."""
    text = text.replace("_", " ")
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return text


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


def fit_tfidf(train_texts, char_analyzer_texts=None):
    word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, min_df=2, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=20000,
                                min_df=2, sublinear_tf=True)
    Xw = word_vec.fit_transform(train_texts)
    Xc = char_vec.fit_transform(train_texts)
    return word_vec, char_vec, hstack([Xw, Xc]).tocsr()


def apply_tfidf(word_vec, char_vec, texts):
    Xw = word_vec.transform(texts)
    Xc = char_vec.transform(texts)
    return hstack([Xw, Xc]).tocsr()


def run_variant(name: str, X_train, y_train, X_test, y_test) -> dict:
    t0 = time.perf_counter()
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=3.0, solver="lbfgs")
    clf.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    proba = clf.predict_proba(X_test)
    pred = clf.classes_[proba.argmax(axis=1)]
    y_true_bin = (y_test != 0).astype(int)
    y_pred_bin = (pred != 0).astype(int)
    m = binary_metrics(y_true_bin, y_pred_bin)
    beats_baseline = m["f1"] > BASELINE["f1"] and m["fpr"] <= BASELINE["fpr"]
    result = {
        "name": name,
        "binary_metrics": m,
        "baseline": BASELINE,
        "delta_f1": m["f1"] - BASELINE["f1"],
        "beats_baseline_criterion": bool(beats_baseline),
        "fit_seconds": round(fit_s, 1),
    }
    print(f"[{name}] P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
          f"FPR={m['fpr']:.4f} delta_f1={result['delta_f1']:+.4f} "
          f"beats_baseline={beats_baseline}")
    out_path = OUT_DIR / f"case2_feature_exp_{name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  wrote {out_path}")
    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading full train/test (no sampling)...")
    train = load_split("train")
    test = load_split("test")
    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()

    results = {}

    # --- A: identifier_split ---
    print("\n=== A: identifier_split ===")
    train_texts_a = [split_identifiers(t) for t in train["text"].tolist()]
    test_texts_a = [split_identifiers(t) for t in test["text"].tolist()]
    word_vec, char_vec, X_train_a = fit_tfidf(train_texts_a)
    X_test_a = apply_tfidf(word_vec, char_vec, test_texts_a)
    results["identifier_split"] = run_variant("identifier_split", X_train_a, y_train, X_test_a, y_test)

    # --- B: handfeat_concat (на исходном, не split-нутом тексте — тот же baseline TF-IDF) ---
    print("\n=== B: handfeat_concat ===")
    word_vec_b, char_vec_b, X_train_tfidf = fit_tfidf(train["text"].tolist())
    X_test_tfidf = apply_tfidf(word_vec_b, char_vec_b, test["text"].tolist())
    scaler = StandardScaler()
    Xh_train = scaler.fit_transform(extract_features_batch(train["text"]).to_numpy(dtype=np.float64))
    Xh_test = scaler.transform(extract_features_batch(test["text"]).to_numpy(dtype=np.float64))
    X_train_b = hstack([X_train_tfidf, csr_matrix(Xh_train)]).tocsr()
    X_test_b = hstack([X_test_tfidf, csr_matrix(Xh_test)]).tocsr()
    results["handfeat_concat"] = run_variant("handfeat_concat", X_train_b, y_train, X_test_b, y_test)

    # --- C: combined (identifier_split + handfeat_concat) ---
    print("\n=== C: combined ===")
    X_train_c = hstack([X_train_a, csr_matrix(Xh_train)]).tocsr()
    X_test_c = hstack([X_test_a, csr_matrix(Xh_test)]).tocsr()
    results["combined"] = run_variant("combined", X_train_c, y_train, X_test_c, y_test)

    summary_path = OUT_DIR / "case2_feature_exp_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"baseline": BASELINE, "variants": results}, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
