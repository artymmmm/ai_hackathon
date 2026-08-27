"""Строит research/case3_recovered_labels_v4.csv из найденного ключа разметки кейса 3.

Ключ: датасет `realvul/LineVul_Test_Dataset` (Hugging Face) — колонка `target` побайтово
совпадает по `unique_id`/`code` с датасетом кейса 3 (см. research/case3_label_key_v4.md).

v4 = v3 со схемой (unique_id, recovered_label, match_source, cwe_id, source_project), но:
- recovered_label берётся из `target` ключа для всех 18864 строк;
- match_source = 'linevul_key' для всех строк;
- cwe_id, source_project переносятся из v3 там, где они были (ключ их не содержит).

v3 не модифицируется, только читается.

Запуск: .venv/bin/python research/build_case3_labels_v4.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_KEY_PARQUET = Path(
    "/private/tmp/claude-501/-Users-artemmartynov-claude-ai-hackathon/"
    "717ab4fa-30bc-488b-bc75-2b5c1e3ccf8e/scratchpad/linevul_realvul/data/"
    "test-00000-of-00001.parquet"
)
_V3_CSV = _ROOT / "research" / "case3_recovered_labels_v3.csv"
_V4_CSV = _ROOT / "research" / "case3_recovered_labels_v4.csv"
_EVAL_IDS_TXT = _ROOT / "out" / "bench" / "case3_eval_ids.txt"
_CHECKS_JSON = _ROOT / "research" / "case3_labels_v4_checks.json"


def main() -> None:
    key = pd.read_parquet(_KEY_PARQUET, columns=["unique_id", "target"])
    assert key["unique_id"].duplicated().sum() == 0, "дубликаты unique_id в ключе"
    key = key.rename(columns={"target": "recovered_label"})

    v3 = pd.read_csv(_V3_CSV)
    assert v3["unique_id"].duplicated().sum() == 0, "дубликаты unique_id в v3"

    assert set(key["unique_id"]) == set(v3["unique_id"]), "множества unique_id не совпадают"

    v4 = key.merge(
        v3[["unique_id", "cwe_id", "source_project", "recovered_label", "match_source"]]
        .rename(columns={"recovered_label": "v3_label", "match_source": "v3_match_source"}),
        on="unique_id",
        how="left",
    )
    v4["match_source"] = "linevul_key"
    v4 = v4[["unique_id", "recovered_label", "match_source", "cwe_id", "source_project", "v3_label", "v3_match_source"]]

    assert len(v4) == 18864
    assert v4["recovered_label"].isna().sum() == 0

    diff_mask = v4["recovered_label"] != v4["v3_label"]
    n_diff = int(diff_mask.sum())
    diff_by_v3_source = v4.loc[diff_mask, "v3_match_source"].value_counts().to_dict()

    eval_ids = [int(x) for x in _EVAL_IDS_TXT.read_text().split()]
    eval_mask = v4["unique_id"].isin(eval_ids)
    eval_diff_mask = eval_mask & diff_mask
    eval_diff_ids = v4.loc[eval_diff_mask, "unique_id"].tolist()

    checks = {
        "n_rows": int(len(v4)),
        "label_balance_v4": v4["recovered_label"].value_counts().to_dict(),
        "n_diff_v3_vs_v4": n_diff,
        "diff_by_v3_match_source": diff_by_v3_source,
        "n_eval_ids": len(eval_ids),
        "eval_label_balance_v3": v4.loc[eval_mask, "v3_label"].value_counts().to_dict(),
        "eval_label_balance_v4": v4.loc[eval_mask, "recovered_label"].value_counts().to_dict(),
        "eval_diff_ids": eval_diff_ids,
        "n_eval_diff": int(eval_diff_mask.sum()),
    }

    # финальный v4 в целевой схеме (без служебных v3_* колонок)
    v4_out = v4[["unique_id", "recovered_label", "match_source", "cwe_id", "source_project"]]
    v4_out.to_csv(_V4_CSV, index=False)

    _CHECKS_JSON.write_text(json.dumps(checks, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"v4 -> {_V4_CSV} ({len(v4_out)} строк)")
    print(f"баланс v4: {checks['label_balance_v4']}")
    print(f"расхождений v3 vs v4: {n_diff}")
    print(f"расхождения по source: {diff_by_v3_source}")
    print(f"эталонные 150: v3={checks['eval_label_balance_v3']} v4={checks['eval_label_balance_v4']}")
    print(f"расхождений в эталонных 150: {checks['n_eval_diff']} -> {eval_diff_ids}")
    print(f"проверки -> {_CHECKS_JSON}")


if __name__ == "__main__":
    main()
