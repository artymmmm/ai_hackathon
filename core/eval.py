"""Метрики: precision/recall/F1/FPR, confusion matrix, кривая эскалации, ablation-таблицы."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from core.schema import Verdict


def verdicts_to_arrays(verdicts: Sequence[Verdict], gold: dict[str, str]) -> tuple[list[str], list[str]]:
    """Сопоставляет вердикты с эталоном по doc_id. Id, которых нет в gold, пропускаются."""
    y_true, y_pred = [], []
    for v in verdicts:
        if v.doc_id in gold:
            y_true.append(gold[v.doc_id])
            y_pred.append(v.verdict)
    return y_true, y_pred


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str],
                            labels: list[str] | None = None) -> dict:
    """precision/recall/f1/support по классам + macro/weighted avg (обёртка sklearn)."""
    return classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)


def confusion_matrix_table(y_true: Sequence[str], y_pred: Sequence[str],
                            labels: list[str] | None = None) -> pd.DataFrame:
    labels = labels or sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])


def false_positive_rate(y_true: Sequence[str], y_pred: Sequence[str], positive_label: str) -> float:
    """FPR для `positive_label`: доля ложных срабатываний среди истинно-отрицательных примеров."""
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive_label and p == positive_label)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t != positive_label and p != positive_label)
    return fp / (fp + tn) if (fp + tn) else 0.0


def escalation_curve(confidence: Sequence[float], correct: Sequence[bool],
                      thresholds: Sequence[float] | None = None) -> pd.DataFrame:
    """Кривая «доля эскалации vs точность на автоматически решённых».

    Для каждого порога t: то, что confidence < t, уходит в ручную проверку (эскалация);
    точность считается среди оставшегося (confidence >= t).
    """
    thresholds = thresholds if thresholds is not None else [i / 20 for i in range(21)]
    confidence = list(confidence)
    correct = list(correct)
    n = len(confidence)
    rows = []
    for t in thresholds:
        auto_idx = [i for i in range(n) if confidence[i] >= t]
        escalated = n - len(auto_idx)
        auto_accuracy = (sum(correct[i] for i in auto_idx) / len(auto_idx)) if auto_idx else None
        rows.append({
            "threshold": t,
            "escalation_rate": escalated / n if n else 0.0,
            "auto_count": len(auto_idx),
            "auto_accuracy": auto_accuracy,
        })
    return pd.DataFrame(rows)


def ablation_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """results: {имя_конфигурации: {метрика: значение}} → одна таблица для сравнения."""
    return pd.DataFrame(results).T
