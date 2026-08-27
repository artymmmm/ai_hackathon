"""Цель 3 (см. промпт координатора): сводная таблица всех конфигураций кейса 3 на eval-150.

Числа P/R/F1/FPR/tp/fp/fn/tn/эскалация берутся ИСКЛЮЧИТЕЛЬНО из уже посчитанных
`cases/codereview/out/eval_metrics_*.json` (результат `evaluate.py`), не переписываются руками.

Число вызовов LLM — не метрика, а структурная величина: считается по размеру множеств
(корпус=150, uncertain/secure-корзины из фактических файлов вердиктов), см. `_calls_*` ниже.

Операционные величины ("ВАЖНО ПРО ОПЕРАЦИОННОЕ ПРОЧТЕНИЕ" в промпте координатора) — сколько
фрагментов реально уходит человеку (vulnerable+uncertain), какая доля истинных уязвимых до него
доходит, какая доля чистого кода зря попадает в очередь — пересчитываются здесь напрямую по
файлам вердиктов + gold (та же логика сопоставления, что в `evaluate.load_gold`), а не берутся
из `evaluate.py` (там uncertain сознательно занижается до secure, см. докстринг evaluate.py).

НИКОГДА не исполнять и не компилировать код из датасета — только статический анализ (CLAUDE.md).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cases.codereview.evaluate import load_gold  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_BENCH = _ROOT / "out" / "bench"
_OUT = _ROOT / "cases" / "codereview" / "out"


def _load_verdicts(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _human_queue_stats(verdicts_path: Path, gold: dict[str, dict]) -> dict:
    rows = _load_verdicts(verdicts_path)
    n_total = len(rows)
    to_human = [r for r in rows if r["verdict"] in ("vulnerable", "uncertain")]
    n_to_human = len(to_human)

    true_vuln_ids = {i for i, g in gold.items() if g["label"] == "vulnerable"}
    true_sec_ids = {i for i, g in gold.items() if g["label"] == "secure"}
    to_human_ids = {r["doc_id"] for r in to_human}

    n_true_vuln_in_run = len({r["doc_id"] for r in rows} & true_vuln_ids)
    n_true_sec_in_run = len({r["doc_id"] for r in rows} & true_sec_ids)
    vuln_reaching_human = len(to_human_ids & true_vuln_ids)
    clean_in_queue = len(to_human_ids & true_sec_ids)

    return {
        "n_to_human": n_to_human,
        "human_queue_share": round(n_to_human / n_total, 4) if n_total else None,
        "vulnerable_reaching_human_of_true_vulnerable": (
            round(vuln_reaching_human / n_true_vuln_in_run, 4) if n_true_vuln_in_run else None
        ),
        "clean_code_in_queue_of_true_secure": (
            round(clean_in_queue / n_true_sec_in_run, 4) if n_true_sec_in_run else None
        ),
    }


def _row_from_metrics(name: str, metrics_path: Path, verdicts_path: Path, n_calls: int,
                       gold: dict[str, dict]) -> dict:
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    cm = m["confusion_matrix"]
    tp = cm["pred_vulnerable"]["true_vulnerable"]
    fp = cm["pred_vulnerable"]["true_secure"]
    fn = cm["pred_secure"]["true_vulnerable"]
    tn = cm["pred_secure"]["true_secure"]
    hq = _human_queue_stats(verdicts_path, gold)
    return {
        "конфигурация": name,
        "P": m["precision_vulnerable"],
        "R": m["recall_vulnerable"],
        "F1": m["f1_vulnerable"],
        "FPR": m["fpr_vulnerable"],
        "эскалация": m["escalation_rate"],
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "число_вызовов_LLM": n_calls,
        "к_человеку_n": hq["n_to_human"],
        "к_человеку_доля_корпуса": hq["human_queue_share"],
        "уязвимых_доходит_до_человека": hq["vulnerable_reaching_human_of_true_vulnerable"],
        "чистого_кода_в_очереди": hq["clean_code_in_queue_of_true_secure"],
        "файл_вердиктов": str(verdicts_path.relative_to(_ROOT)),
    }


def main() -> None:
    gold = load_gold()

    # ---- размеры множеств из фактических файлов (не из промпта) ----
    cert_only = _load_verdicts(_BENCH / "case3_deepseek-chat_cert_only.json")
    n_cert_uncertain = sum(1 for d in cert_only if d["verdict"] == "uncertain")  # 89
    n_cert_secure = sum(1 for d in cert_only if d["verdict"] == "secure")  # 29

    combo_stats = json.loads((_OUT / "combo_stats.json").read_text(encoding="utf-8"))
    n_union_uncertain_raw = combo_stats["union_uncertain_raw_count"]  # 99
    n_new_for_combo_cascade = len(combo_stats["new_ids_for_stage2_not_in_cert_only_89"])  # 10

    secure_lowconf = json.loads((_BENCH / "case3_stage2_secure_lowconf_k1of5.json").read_text())
    # восстановить размер lowconf-множества из самого набора эскалации не тривиально из файла
    # вердиктов напрямую — берём из run_secure_stage2_merge.py логики через cert confidence:
    n_secure_lowconf = sum(1 for d in cert_only if d["verdict"] == "secure" and d["confidence"] < 0.9)  # 2

    calls_150 = 150
    calls_cascade_b = 150 + n_cert_uncertain * 5  # 595
    calls_combo_union = 150 + 150  # оба stage1: screener + cert_only = 300
    calls_combo_cascade = 300 + n_union_uncertain_raw * 5  # 300 + 495 = 795
    calls_secure_all = 150 + n_cert_uncertain * 5 + n_cert_secure * 5  # 740
    calls_secure_lowconf = 150 + n_cert_uncertain * 5 + n_secure_lowconf * 5  # 605

    rows = []

    rows.append(_row_from_metrics(
        "голая LLM", _OUT / "eval_metrics_deepseek_bare.json",
        _BENCH / "case3_deepseek-chat.json", calls_150, gold))
    rows.append(_row_from_metrics(
        "скринер без знаний (SENSITIVE)", _OUT / "eval_metrics_sensitive_full.json",
        _BENCH / "case3_deepseek-chat_sensitive.json", calls_150, gold))
    rows.append(_row_from_metrics(
        "конфигурация A (CWE+CERT+flawfinder)", _OUT / "eval_metrics_configA_full.json",
        _BENCH / "case3_deepseek-chat_configA.json", calls_150, gold))
    rows.append(_row_from_metrics(
        "cwe_only", _OUT / "eval_metrics_cwe_only_full.json",
        _BENCH / "case3_deepseek-chat_cwe_only.json", calls_150, gold))
    rows.append(_row_from_metrics(
        "cert_only", _OUT / "eval_metrics_cert_only_full.json",
        _BENCH / "case3_deepseek-chat_cert_only.json", calls_150, gold))
    for k in (1, 2, 3, 4):
        rows.append(_row_from_metrics(
            f"каскад cert_only+ступень2(uncertain) k>={k}/5",
            _OUT / f"eval_metrics_cascade_B_k{k}of5.json",
            _BENCH / f"case3_deepseek-chat_cascade_B_k{k}of5.json", calls_cascade_b, gold))

    # ---- Цель 1: офлайн-комбинации screener+cert_only ----
    rows.append(_row_from_metrics(
        "combo: union(vulnerable у любого)", _OUT / "eval_metrics_combo_union.json",
        _BENCH / "case3_combo_union.json", calls_combo_union, gold))
    rows.append(_row_from_metrics(
        "combo: intersection(vulnerable у обоих)", _OUT / "eval_metrics_combo_intersection.json",
        _BENCH / "case3_combo_intersection.json", calls_combo_union, gold))
    for k in (1, 2, 3, 4):
        rows.append(_row_from_metrics(
            f"combo: cascade на union(uncertain обоих) k>={k}/5",
            _OUT / f"eval_metrics_combo_cascade_union_k{k}of5.json",
            _BENCH / f"case3_combo_cascade_union_k{k}of5.json", calls_combo_cascade, gold))

    # ---- Цель 2: расширение ступени 2 на secure-корзину cert_only ----
    for k in (1, 2, 3, 4):
        rows.append(_row_from_metrics(
            f"cert_only+ступень2(uncertain∪secure_all) k>={k}/5",
            _OUT / f"eval_metrics_stage2_secure_all_k{k}of5.json",
            _BENCH / f"case3_stage2_secure_all_k{k}of5.json", calls_secure_all, gold))
    for k in (1, 2, 3, 4):
        rows.append(_row_from_metrics(
            f"cert_only+ступень2(uncertain∪secure_lowconf) k>={k}/5",
            _OUT / f"eval_metrics_stage2_secure_lowconf_k{k}of5.json",
            _BENCH / f"case3_stage2_secure_lowconf_k{k}of5.json", calls_secure_lowconf, gold))

    out_path = _OUT / "configs_comparison.csv"
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} строк -> {out_path}")

    best = max(rows, key=lambda r: r["F1"])
    print(f"\nлучший F1: {best['конфигурация']} F1={best['F1']} FPR={best['FPR']}")
    ok = [r for r in rows if r["F1"] > 0.575 and r["FPR"] <= 0.12]
    print(f"конфигураций с F1>0.575 и FPR<=0.12: {len(ok)}")
    for r in ok:
        print(f"  {r['конфигурация']}: F1={r['F1']} FPR={r['FPR']}")


if __name__ == "__main__":
    main()
