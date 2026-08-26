"""Загрузчики трёх датасетов хакатона. Пути и схемы — см. CLAUDE.md."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

CASE1_DIR = _PROJECT_ROOT / "case 1" / "data"
CASE2_DIR = _PROJECT_ROOT / "case 2" / "prompt-injection-safety" / "data"
CASE3_CSV = _PROJECT_ROOT / "case 3" / "3 кейс_датасет.csv"


def _stratified_sample(df: pd.DataFrame, label_col: str, n: int, seed: int) -> pd.DataFrame:
    """Сэмплирует ~n строк, сохраняя пропорции классов `label_col` (минимум 1 на класс)."""
    if n >= len(df):
        return df
    frac = n / len(df)
    parts = []
    for _, group in df.groupby(label_col, group_keys=False):
        k = max(1, round(len(group) * frac))
        parts.append(group.sample(n=min(k, len(group)), random_state=seed))
    sampled = pd.concat(parts)
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed)
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_case1(split: str = "train", n: int | None = None, seed: int = 42,
               data_dir: Path | None = None) -> pd.DataFrame:
    """Кейс 1 (PII): uid, domain, document_type, document_description, document_format,
    locale, text, spans (list[{start,end,text,label}]), text_tagged.

    Стратификация выборки — по `domain` (55+ доменов документов).
    """
    data_dir = data_dir or CASE1_DIR
    path = data_dir / f"{split}-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    df["spans"] = df["spans"].apply(ast.literal_eval)  # хранится как repr-строка Python, не JSON
    if n is not None:
        df = _stratified_sample(df, "domain", n, seed)
    return df.reset_index(drop=True)


def load_case2(split: str = "train", n: int | None = None, seed: int = 42,
               data_dir: Path | None = None) -> pd.DataFrame:
    """Кейс 2 (guard): text, label (0 safe / 1 замаскированная инъекция / 2 прямой вред).

    Добавляет `verdict_binary`: 'safe' при label==0, иначе 'injection_malicious' —
    маппинг из задания; исходный `label` сохраняется как под-тип (1 vs 2).
    Стратификация — по `label` (класс 2 в ~6 раз меньше остальных).
    """
    data_dir = data_dir or CASE2_DIR
    path = data_dir / f"{split}-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    df["verdict_binary"] = df["label"].apply(lambda v: "safe" if v == 0 else "injection_malicious")
    if n is not None:
        df = _stratified_sample(df, "label", n, seed)
    return df.reset_index(drop=True)


def load_case3(n: int | None = None, seed: int = 42, csv_path: Path | None = None) -> pd.DataFrame:
    """Кейс 3 (code review): unique_id, code (C/C++). Лейблов нет — стратификация невозможна,
    сэмплирование простое случайное.
    """
    path = csv_path or CASE3_CSV
    df = pd.read_csv(path)
    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=seed)
    return df.reset_index(drop=True)
