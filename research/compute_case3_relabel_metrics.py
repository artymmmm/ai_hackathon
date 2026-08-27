"""Пересчёт P/R/F1/FPR всех прогонов кейса 3 против v3- и v4-разметки (см. case3_label_key_v4.md).

Берёт каждый out/bench/case3_*.json, если это список вердиктов (doc_id/verdict/...),
и считает метрики класса "vulnerable" против research/case3_recovered_labels_v3.csv и
research/case3_recovered_labels_v4.csv. `uncertain` бинаризуется как "secure" — то же
соглашение, что в cases/codereview/evaluate.py. Файлы, которые не являются списком
вердиктов (агрегаты/summary), попадают в `skipped` с причиной, а не молча пропускаются.

nemotron-ultra исключён явно: там сейчас идёт фоновый прогон, файл трогать нельзя.

Запуск: .venv/bin/python research/compute_case3_relabel_metrics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from core.eval import classification_metrics, false_positive_rate  # noqa: E402

_BENCH_DIR = _ROOT / "out" / "bench"
_V3_CSV = _ROOT / "research" / "case3_recovered_labels_v3.csv"
_V4_CSV = _ROOT / "research" / "case3_recovered_labels_v4.csv"
_OUT_CSV = _BENCH_DIR / "case3_relabel_v4_metrics.csv"
_OUT_JSON = _BENCH_DIR / "case3_relabel_v4_metrics.json"

_LABELS = ["secure", "vulnerable"]
_RAW_TO_NAME = {0: "secure", 1: "vulnerable"}
_EXCLUDE_SUBSTR = ("nemotron-ultra",)


def load_gold(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    return {str(int(r.unique_id)): _RAW_TO_NAME[int(r.recovered_label)] for r in df.itertuples()}


def load_verdicts_or_none(path: Path) -> list[dict] | None:
    """Возвращает список вердиктов, либо None, если формат не подходит (не список / нет полей)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"не парсится как JSON: {e}") from e
    if not isinstance(data, list):
        raise ValueError(f"не список вердиктов (top-level тип: {type(data).__name__})")
    if not data:
        raise ValueError("пустой список")
    if not all(isinstance(d, dict) and "doc_id" in d and "verdict" in d for d in data):
        raise ValueError("элементы списка не похожи на Verdict (нет doc_id/verdict)")
    return data


def compute_metrics(verdicts: list[dict], gold: dict[str, str]) -> dict:
    matched = [(v, gold[v["doc_id"]]) for v in verdicts if v["doc_id"] in gold]
    n = len(matched)
    if n == 0:
        return {"n": 0, "p": None, "r": None, "f1": None, "fpr": None}
    y_true = [g for _, g in matched]
    y_pred = [(v["verdict"] if v["verdict"] in _LABELS else "secure") for v, _ in matched]
    report = classification_metrics(y_true, y_pred, labels=_LABELS)
    fpr = false_positive_rate(y_true, y_pred, positive_label="vulnerable")
    return {
        "n": n,
        "p": round(report["vulnerable"]["precision"], 4),
        "r": round(report["vulnerable"]["recall"], 4),
        "f1": round(report["vulnerable"]["f1-score"], 4),
        "fpr": round(fpr, 4),
    }


def main() -> None:
    gold_v3 = load_gold(_V3_CSV)
    gold_v4 = load_gold(_V4_CSV)

    files = sorted(
        p for p in _BENCH_DIR.glob("case3_*.json")
        if not any(sub in p.name for sub in _EXCLUDE_SUBSTR)
    )

    rows: list[dict] = []
    skipped: list[dict] = []

    for path in files:
        try:
            verdicts = load_verdicts_or_none(path)
        except ValueError as e:
            skipped.append({"file": path.name, "reason": str(e)})
            continue

        m3 = compute_metrics(verdicts, gold_v3)
        m4 = compute_metrics(verdicts, gold_v4)
        if m3["n"] == 0 or m4["n"] == 0:
            skipped.append({"file": path.name, "reason": "0 doc_id пересеклись с gold"})
            continue

        delta_f1 = round(m4["f1"] - m3["f1"], 4)
        rows.append({
            "file": path.name,
            "n": m3["n"],
            "p_v3": m3["p"], "r_v3": m3["r"], "f1_v3": m3["f1"], "fpr_v3": m3["fpr"],
            "p_v4": m4["p"], "r_v4": m4["r"], "f1_v4": m4["f1"], "fpr_v4": m4["fpr"],
            "delta_f1": delta_f1,
        })

    df = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
    df.to_csv(_OUT_CSV, index=False)

    out_json = {
        "gold_v3": str(_V3_CSV.relative_to(_ROOT)),
        "gold_v4": str(_V4_CSV.relative_to(_ROOT)),
        "n_files_scored": len(rows),
        "n_files_skipped": len(skipped),
        "skipped": skipped,
        "results": rows,
    }
    _OUT_JSON.write_text(json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"посчитано файлов: {len(rows)}; пропущено: {len(skipped)}")
    for s in skipped:
        print(f"  SKIP {s['file']}: {s['reason']}")
    print(f"-> {_OUT_CSV}")
    print(f"-> {_OUT_JSON}")

    print("\nтоп-5 по f1_v3:")
    print(df.sort_values("f1_v3", ascending=False).head(5)[["file", "f1_v3", "f1_v4", "delta_f1"]].to_string(index=False))
    print("\nтоп-5 по f1_v4:")
    print(df.sort_values("f1_v4", ascending=False).head(5)[["file", "f1_v3", "f1_v4", "delta_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
