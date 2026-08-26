"""Офлайн-признаки кода для kNN/логрег-baseline и RAG-поиска соседей (шаги 1 и 3).

Без LLM, без сети. Два TF-IDF представления, склеенные в одну разреженную матрицу:
- символьные n-граммы (char_wb, 3-5) — ловят синтаксис/лексику независимо от токенизации
  (скобки, `->`, `[i]`, конкретные имена опасных функций типа `strcpy`, `memcpy`);
- "словесные" n-граммы по C-токенам (идентификаторы, операторы, числа) — ловят паттерны
  уровня вызовов функций/API, менее чувствительны к пробелам/форматированию.

`FEATURE_CHAR_CAP` обрезает код перед векторизацией (не для промпта LLM — отдельный
механизм от `reviewer.MAX_CODE_CHARS`, здесь только чтобы TF-IDF не тратил время на
редкие гигантские фрагменты; p99 длины корпуса — 7276 символов, см. cases/codereview/
improvements.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

FEATURE_CHAR_CAP = 8000

# Простой C/C++-токенизатор: идентификаторы/числа, либо один из типовых
# многосимвольных операторов, либо один прочий не-пробельный символ.
_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*|\d+\.?\d*|->|\+\+|--|==|!=|<=|>=|&&|\|\||::|[^\sA-Za-z0-9_]"
)


def truncate(code: str, cap: int = FEATURE_CHAR_CAP) -> str:
    return code[:cap] if len(code) > cap else code


def tokenize_code(code: str) -> list[str]:
    return _TOKEN_RE.findall(truncate(code))


def _token_analyzer(code: str) -> list[str]:
    return tokenize_code(code)


@dataclass
class CodeFeaturizer:
    """Обёртка над двумя TF-IDF векторизаторами. `fit` только на обучающем пуле."""

    char_ngram: tuple[int, int] = (3, 5)
    token_ngram: tuple[int, int] = (1, 2)
    char_max_features: int = 60000
    token_max_features: int = 40000

    def __post_init__(self) -> None:
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=self.char_ngram,
            max_features=self.char_max_features, sublinear_tf=True, min_df=2,
        )
        # analyzer='word' + tokenizer=... (не analyzer=callable!) — иначе sklearn игнорирует
        # ngram_range и отдаёт только униграммы (проверено: было тихой потерей биграмм).
        self.token_vec = TfidfVectorizer(
            analyzer="word", tokenizer=_token_analyzer, token_pattern=None,
            ngram_range=self.token_ngram,
            max_features=self.token_max_features, sublinear_tf=True, min_df=2,
        )

    def fit(self, codes: list[str]) -> "CodeFeaturizer":
        texts = [truncate(c) for c in codes]
        self.char_vec.fit(texts)
        self.token_vec.fit(texts)
        return self

    def transform(self, codes: list[str]) -> sp.csr_matrix:
        texts = [truncate(c) for c in codes]
        a = self.char_vec.transform(texts)
        b = self.token_vec.transform(texts)
        return sp.hstack([a, b], format="csr")

    def fit_transform(self, codes: list[str]) -> sp.csr_matrix:
        self.fit(codes)
        return self.transform(codes)
