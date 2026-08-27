"""Прогон flawfinder на ВСЕХ 18864 фрагментах кейса 3 (офлайн, только чтение текста,
никакой компиляции/исполнения — см. cases/codereview/static_analyzer.py).

Раньше flawfinder гонялся только на 150 eval-фрагментах (`out/flawfinder_eval_hits.csv`).
Этот скрипт делает то же самое (`run_flawfinder`, признак предсказания — `any_hit`, то есть
хотя бы один хит любого уровня — так воспроизводятся старые числа R=0.18/FPR=0.05 на eval150)
на всём корпусе, чтобы посчитать метрики против `research/case3_recovered_labels_v4.csv`
(100% покрытие) в `compute_offline_baselines_v4.py`.

Запуск: .venv/bin/python cases/codereview/run_flawfinder_full.py
Выход: cases/codereview/out/flawfinder_full_hits.csv (unique_id, n_hits, any_hit, has_error_level)
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
from core.data import load_case3  # noqa: E402
from cases.codereview.static_analyzer import run_flawfinder  # noqa: E402

_OUT = _ROOT / "cases" / "codereview" / "out" / "flawfinder_full_hits.csv"


def score_one(item) -> dict:
    uid, code = item
    hits = run_flawfinder(code)
    has_error = any(h.get("level") == "error" for h in hits)
    return {"unique_id": int(uid), "n_hits": len(hits), "any_hit": len(hits) > 0,
            "has_error_level": has_error}


def main() -> None:
    df = load_case3()
    print(f"фрагментов: {len(df)}")
    t0 = time.time()
    items = list(zip(df["unique_id"].tolist(), df["code"].tolist()))
    results = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        for i, r in enumerate(ex.map(score_one, items)):
            results.append(r)
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(items)} ({time.time()-t0:.0f}s)")
    out = pd.DataFrame(results)
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(_OUT, index=False)
    print(f"готово за {time.time()-t0:.0f}s -> {_OUT}")
    print(out["any_hit"].value_counts())


if __name__ == "__main__":
    main()
