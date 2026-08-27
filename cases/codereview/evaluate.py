"""Измерение LLM-ревьюера кейса 3 против восстановленных лейблов.

Ground truth: `research/case3_recovered_labels.csv` (см. `research/case3_label_matching.md`) —
восстановлен матчингом с BigVul/DiverseVul для 90.77% корпуса (17123 / 18864 фрагментов).
`recovered_label` ∈ {'0' (secure), '1' (vulnerable), 'conflict' (два источника разошлись — не
доверять), NaN (не найдено ни в одном источнике)}. В метрики идут только фрагменты с
`recovered_label ∈ {'0','1'}` — `conflict` и unmatched исключаются явно (см. `research/
case3_label_matching.md` §7), с раздельным подсчётом, сколько исключено и почему.

Если прогнанная выборка (`--verdicts`) меньше полного корпуса — метрики считаются по пересечению
(что реально прогнано) ∩ (что реально размечено), это не ошибка, а нормальный режим работы на
`--sample N`.

Baseline «всё secure»: среди размеченных 4.1-4.5% vulnerable (в зависимости от знаменателя) →
тривиальный классификатор «всегда secure» даёт ~95-96% accuracy. Печатаем эту цифру рядом с
нашей всегда — иначе accuracy как метрика вводит в заблуждение (сильный дисбаланс классов).

`verdict == "uncertain"` в бинарных precision/recall/F1 трактуется как предсказание "secure" —
это честная НИЖНЯЯ оценка полностью автоматической системы (без учёта того, что часть потока
уходит на ручную проверку). Отдельно считается `escalation_rate` и кривая confidence→точность
(`core.eval.escalation_curve`) — так эскалация видна как настраиваемый SLA, а не спрятана
внутри одной агрегированной цифры (PLAN.md §6).

CWE accuracy считается ТОЛЬКО на истинных vulnerable, которые модель тоже назвала vulnerable
(true positives), и только там, где известен gold CWE (578 из 774 — `research/
case3_label_matching.md` §5). Это не полная картина CWE-качества (модель могла быть права по
CWE даже без исходного совпадения по gold_cwe, если тот не заполнен), а нижняя консервативная
оценка на пересечении, где есть с чем сравнивать.

Запуск:
    .venv/bin/python cases/codereview/evaluate.py --verdicts out/case3_verdicts.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.eval import classification_metrics, confusion_matrix_table, escalation_curve, false_positive_rate  # noqa: E402
from core.schema import Verdict  # noqa: E402
from cases.codereview.cwe_map import normalize_cwe  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent.parent
_RECOVERED_LABELS_CSV = _ROOT / "research" / "case3_recovered_labels_v4.csv"
_LABELS = ["secure", "vulnerable"]
_RAW_LABEL_TO_NAME = {"0": "secure", "1": "vulnerable"}


def load_gold(path: Path = _RECOVERED_LABELS_CSV) -> dict[str, dict]:
    """unique_id (как строка, совпадает с Verdict.doc_id) -> {'label', 'raw_label', 'cwe_id'}.

    `label` — 'secure'/'vulnerable' для однозначно размеченных, `None` для 'conflict' и unmatched
    (эти два случая различаются в `raw_label`: 'conflict' против отсутствия значения).
    """
    df = pd.read_csv(path)
    gold: dict[str, dict] = {}
    for _, row in df.iterrows():
        raw_label = str(row["recovered_label"]) if pd.notna(row["recovered_label"]) else None
        gold[str(int(row["unique_id"]))] = {
            "label": _RAW_LABEL_TO_NAME.get(raw_label),
            "raw_label": raw_label,
            "cwe_id": row["cwe_id"] if pd.notna(row["cwe_id"]) else None,
        }
    return gold


def load_verdicts(path: Path) -> list[Verdict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Verdict(**d) for d in data]


def evaluate(verdicts: list[Verdict], gold: dict[str, dict]) -> dict:
    n_run = len(verdicts)
    matched = [(v, gold[v.doc_id]) for v in verdicts if v.doc_id in gold]
    n_gold_found = len(matched)
    n_conflict_excluded = sum(1 for _, g in matched if g["raw_label"] == "conflict")
    n_unmatched_excluded = sum(1 for _, g in matched if g["raw_label"] is None)

    scoreable = [(v, g) for v, g in matched if g["label"] is not None]
    n_scoreable = len(scoreable)

    result: dict = {
        "n_run": n_run,
        "n_gold_found": n_gold_found,
        "n_scoreable": n_scoreable,
        "n_conflict_excluded": n_conflict_excluded,
        "n_unmatched_excluded": n_unmatched_excluded,
    }
    if n_scoreable == 0:
        result["error"] = "нет пересечения прогона с однозначно размеченным подмножеством"
        return result

    y_true = [g["label"] for _, g in scoreable]
    # uncertain -> "secure" в бинарных метриках: честная нижняя оценка полностью автоматического
    # решения, без учёта того, что uncertain реально уходит на ручную проверку (см. докстринг).
    y_pred = [(v.verdict if v.verdict in _LABELS else "secure") for v, _ in scoreable]

    report = classification_metrics(y_true, y_pred, labels=_LABELS)
    cm = confusion_matrix_table(y_true, y_pred, labels=_LABELS).astype(int)
    fpr = false_positive_rate(y_true, y_pred, positive_label="vulnerable")

    n_true_vulnerable = sum(1 for t in y_true if t == "vulnerable")
    baseline_accuracy = (n_scoreable - n_true_vulnerable) / n_scoreable  # «всё secure»

    n_uncertain = sum(1 for v, _ in scoreable if v.verdict == "uncertain")

    cwe_known_pairs = [
        (normalize_cwe(v.artifacts.get("cwe_id")), g["cwe_id"])
        for v, g in scoreable
        if g["label"] == "vulnerable" and v.verdict == "vulnerable" and g["cwe_id"]
    ]
    cwe_comparable = [(p, gcwe) for p, gcwe in cwe_known_pairs if p is not None]
    cwe_correct = sum(1 for p, gcwe in cwe_comparable if p == gcwe)
    cwe_accuracy = (cwe_correct / len(cwe_comparable)) if cwe_comparable else None

    confidence = [v.confidence for v, _ in scoreable]
    correct = [
        (v.verdict == "vulnerable" and g["label"] == "vulnerable")
        or (v.verdict == "secure" and g["label"] == "secure")
        for v, g in scoreable
    ]
    curve = escalation_curve(confidence, correct)

    result.update({
        "confusion_matrix": cm.to_dict(),
        "precision_vulnerable": round(report["vulnerable"]["precision"], 4),
        "recall_vulnerable": round(report["vulnerable"]["recall"], 4),
        "f1_vulnerable": round(report["vulnerable"]["f1-score"], 4),
        "fpr_vulnerable": round(fpr, 4),
        "accuracy": round(report["accuracy"], 4),
        "baseline_all_secure_accuracy": round(baseline_accuracy, 4),
        "beats_baseline": report["accuracy"] > baseline_accuracy,
        "n_true_vulnerable_in_scoreable": n_true_vulnerable,
        "n_uncertain_verdicts": n_uncertain,
        "escalation_rate": round(n_uncertain / n_scoreable, 4),
        "cwe_pairs_comparable": len(cwe_comparable),
        "cwe_correct": cwe_correct,
        "cwe_accuracy": round(cwe_accuracy, 4) if cwe_accuracy is not None else None,
        "classification_report": report,
        "escalation_curve": curve.to_dict(orient="records"),
    })
    return result


def _print_summary(metrics: dict) -> None:
    print(
        f"Прогон: {metrics['n_run']} вердиктов; найдено в gold: {metrics['n_gold_found']}; "
        f"размечено однозначно (secure/vulnerable): {metrics['n_scoreable']} "
        f"(conflict исключено: {metrics['n_conflict_excluded']}, "
        f"unmatched исключено: {metrics['n_unmatched_excluded']})"
    )
    if "error" in metrics:
        print(f"  {metrics['error']}")
        return
    print(
        f"precision(vulnerable)={metrics['precision_vulnerable']:.3f}  "
        f"recall(vulnerable)={metrics['recall_vulnerable']:.3f}  "
        f"f1(vulnerable)={metrics['f1_vulnerable']:.3f}  "
        f"fpr={metrics['fpr_vulnerable']:.3f}  accuracy={metrics['accuracy']:.3f}"
    )
    verdict_note = "мы выше" if metrics["beats_baseline"] else "!! МЫ НЕ БЬЁМ БАЗОВУЮ ЛИНИЮ !!"
    print(
        f"baseline «всё secure»: accuracy={metrics['baseline_all_secure_accuracy']:.3f} "
        f"на {metrics['n_true_vulnerable_in_scoreable']} истинных vulnerable из "
        f"{metrics['n_scoreable']} ({verdict_note})"
    )
    print(
        f"escalation_rate (verdict=uncertain): {metrics['escalation_rate']:.3f} "
        f"({metrics['n_uncertain_verdicts']} фрагментов)"
    )
    if metrics["cwe_accuracy"] is not None:
        print(
            f"CWE accuracy на пойманных true positive vulnerable: {metrics['cwe_accuracy']:.3f} "
            f"(верно {metrics['cwe_correct']} из {metrics['cwe_pairs_comparable']} сравнимых пар)"
        )
    else:
        print("CWE accuracy: нет сравнимых пар (ни одного истинного vulnerable, пойманного "
              "моделью, с известным gold CWE в этом прогоне)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verdicts", type=Path, default=Path("out/case3_verdicts.json"))
    parser.add_argument("--gold", type=Path, default=_RECOVERED_LABELS_CSV)
    parser.add_argument("--output", type=Path, default=Path("cases/codereview/out/eval_metrics.json"))
    args = parser.parse_args()

    verdicts = load_verdicts(args.verdicts)
    gold = load_gold(args.gold)
    metrics = evaluate(verdicts, gold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_summary(metrics)
    print(f"\nполные метрики -> {args.output}")


if __name__ == "__main__":
    main()
