"""Ручные признаки для baseline-классификатора кейса 2.

Быстрые (без сети, без токенизации LLM) числовые признаки текста: статистика
символов, энтропия, следы обфускации, счётчики тегов таксономии. Используются
как вход модели (б) в baseline.py и как самостоятельный сигнал для отчёта.
"""

import math
import re
from collections import Counter

import pandas as pd

from cases.guard.taxonomy import TAGS, tag_text

_URL_RE = re.compile(r"https?://\S+")
_HEX_RUN_RE = re.compile(r"\b(?:0x)?[0-9a-fA-F]{16,}\b")
_EXCLAIM_RE = re.compile(r"!")
_ZERO_WIDTH_RE = re.compile(r"[​‌‍﻿]")
_CURLY_RE = re.compile(r"[{}]")

FEATURE_NAMES: list[str] = [
    "char_len",
    "word_count",
    "avg_word_len",
    "shannon_entropy",
    "non_ascii_ratio",
    "digit_ratio",
    "upper_ratio",
    "special_char_ratio",
    "hex_run_count",
    "has_code_block",
    "curly_brace_ratio",
    "exclaim_count",
    "url_count",
    "zero_width_count",
] + [f"tag_{tag}" for tag in TAGS]


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def extract_features(text: str) -> dict[str, float]:
    """Признаки одного текста. Не зависит от корпуса (без fit)."""
    text = text or ""
    n = len(text)
    words = text.split()
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    digits = sum(1 for ch in text if ch.isdigit())
    upper = sum(1 for ch in text if ch.isupper())
    special = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())

    feats: dict[str, float] = {
        "char_len": n,
        "word_count": len(words),
        "avg_word_len": (sum(len(w) for w in words) / len(words)) if words else 0.0,
        "shannon_entropy": _shannon_entropy(text),
        "non_ascii_ratio": non_ascii / n if n else 0.0,
        "digit_ratio": digits / n if n else 0.0,
        "upper_ratio": upper / n if n else 0.0,
        "special_char_ratio": special / n if n else 0.0,
        "hex_run_count": len(_HEX_RUN_RE.findall(text)),
        "has_code_block": 1.0 if "```" in text else 0.0,
        "curly_brace_ratio": len(_CURLY_RE.findall(text)) / n if n else 0.0,
        "exclaim_count": len(_EXCLAIM_RE.findall(text)),
        "url_count": len(_URL_RE.findall(text)),
        "zero_width_count": len(_ZERO_WIDTH_RE.findall(text)),
    }
    tags = set(tag_text(text))
    for tag in TAGS:
        feats[f"tag_{tag}"] = 1.0 if tag in tags else 0.0
    return feats


def extract_features_batch(texts) -> pd.DataFrame:
    """Признаки для списка/Series текстов, в фиксированном порядке колонок."""
    rows = [extract_features(t) for t in texts]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)
