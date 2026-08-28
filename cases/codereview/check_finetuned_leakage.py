"""Проверка утечки: сколько из 600 эталонных фрагментов кейса 3 модель
`mahdin70/CodeBERT-PrimeVul-BigVul` видела при обучении.

Обучающий сплит модели — `mahdin70/balanced_merged_bigvul_primevul` (train, 124 780 строк,
колонка `func`), см. карточку модели: https://huggingface.co/mahdin70/CodeBERT-PrimeVul-BigVul
(README: "Train: 124,780 samples" на `mahdin70/balanced_merged_bigvul_primevul`).
Датасет скачан локально: cases/codereview/out/hf_dataset_bigvul_primevul/data/train-*.parquet.

Нормализация — как в PrimeVul dedup: убрать комментарии (// и /* */), схлопнуть все
пробельные символы, привести к нижнему регистру не требуется (код регистрозависим,
PrimeVul сравнивает как есть) — здесь оставляем регистр, схлопываем только whitespace
и вырезаем комментарии, затем считаем MD5.

Запуск:
    .venv/bin/python cases/codereview/check_finetuned_leakage.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.data import load_case3  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_EVAL600_IDS = _ROOT / "out" / "bench" / "case3_eval600_ids.txt"
_TRAIN_SPLIT = _ROOT / "cases" / "codereview" / "out" / "hf_dataset_bigvul_primevul" / "data" / "train-00000-of-00001.parquet"
_OUT_JSON = _ROOT / "out" / "bench" / "case3_finetuned_leakage.json"

_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_WS_RE = re.compile(r"\s+")


def normalize(code: str) -> str:
    code = _COMMENT_RE.sub(" ", code)
    code = _WS_RE.sub(" ", code)
    return code.strip()


def md5_of(code: str) -> str:
    return hashlib.md5(normalize(code).encode("utf-8", errors="ignore")).hexdigest()


def main() -> None:
    eval_ids = {x.strip() for x in _EVAL600_IDS.read_text().split() if x.strip()}
    corpus = load_case3()
    corpus["unique_id"] = corpus["unique_id"].astype(str)
    ours = corpus[corpus["unique_id"].isin(eval_ids)].reset_index(drop=True)
    print(f"эталон 600: id в файле {len(eval_ids)}, найдено в корпусе {len(ours)}")

    ours["md5"] = ours["code"].apply(md5_of)

    train = pd.read_parquet(_TRAIN_SPLIT)
    train_hashes = set(train["func"].apply(md5_of))
    print(f"обучающий сплит модели: {len(train)} строк, {len(train_hashes)} уникальных хешей")

    ours["in_train"] = ours["md5"].isin(train_hashes)
    n_checked = len(ours)
    n_matched = int(ours["in_train"].sum())
    frac = n_matched / n_checked if n_checked else 0.0

    # Справочно (не влияет на основной вывод): val/test сплиты того же датасета —
    # не участвовали в обновлении весов, но показывают общее пересечение корпусов.
    val_path = _TRAIN_SPLIT.parent / "validation-00000-of-00001.parquet"
    test_path = _TRAIN_SPLIT.parent / "test-00000-of-00001.parquet"
    val_hashes = set(pd.read_parquet(val_path)["func"].apply(md5_of))
    test_hashes = set(pd.read_parquet(test_path)["func"].apply(md5_of))
    n_matched_val = int(ours["md5"].isin(val_hashes).sum())
    n_matched_test = int(ours["md5"].isin(test_hashes).sum())

    result = {
        "n_checked": n_checked,
        "n_matched_in_train_split": n_matched,
        "fraction_matched": round(frac, 4),
        "train_split_source": "mahdin70/balanced_merged_bigvul_primevul (train, HF datasets)",
        "train_split_local_path": str(_TRAIN_SPLIT.relative_to(_ROOT)),
        "train_split_rows": int(len(train)),
        "model_card_url": "https://huggingface.co/mahdin70/CodeBERT-PrimeVul-BigVul",
        "normalization": "убраны // и /* */ комментарии, схлопнуты пробельные символы, MD5 нормализованного тела",
        "auxiliary_val_test_not_used_in_training": {
            "n_matched_validation_split": n_matched_val,
            "n_matched_test_split": n_matched_test,
            "note": "val/test того же датасета не входили в обновление весов модели; приведены только для картины общего пересечения корпусов, в основной вывод об утечке не входят",
        },
        "note": (
            "Сплит опубликован автором модели как отдельный датасет (та же карточка модели "
            "прямо ссылается на mahdin70/balanced_merged_bigvul_primevul как train). "
            "Это train-часть, на которой модель реально обновляла веса — не held-out. "
            "Совпадение по нормализованному MD5 означает, что модель видела этот же фрагмент "
            "кода (возможно, под другим unique_id/CVE) во время обучения."
        ),
    }
    _OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    _OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n-> {_OUT_JSON}")


if __name__ == "__main__":
    main()
