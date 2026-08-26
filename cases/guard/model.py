"""Персистентность быстрого офлайн-слоя (модель (a) из `baseline.py`) для пайплайна.

`baseline.py` — обучение и офлайн-оценка (научная честность, метрики в report.md).
Этот модуль — тонкая обёртка сверху: обучить один раз, сохранить на диск, переиспользовать
между запусками `run.py --case 2`, чтобы `route()` не переобучался на каждый вызов CLI.
"""

from __future__ import annotations

from pathlib import Path

import joblib

from cases.guard.baseline import apply_model_a, build_model_a, load_split

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "out" / "guard" / "model_a.joblib"


def train_and_save(model_path: Path = MODEL_PATH) -> dict:
    train = load_split("train")
    model = build_model_a(train["text"].tolist(), train["label"].to_numpy())
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model


def load_or_train(model_path: Path = MODEL_PATH) -> dict:
    if model_path.exists():
        return joblib.load(model_path)
    return train_and_save(model_path)


def predict_proba(model: dict, texts: list[str]):
    """Возвращает (proba [n, n_classes], classes_)."""
    return apply_model_a(model, texts)
