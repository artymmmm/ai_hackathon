"""Статическая диагностика "стека знаний" конфигурации A (knowledge.py) — БЕЗ единого
вызова LLM. Отвечает на вопрос: почему cwe_cards_only F1=0.444, cert_only F1=0.600,
flawfinder_only F1=0.370 по отдельности, а весь стек (config_A) — F1=0.261 вместе
(cases/codereview/out/component_decomposition_results.json,
cases/codereview/out/config_experiment_results.json).

Меряется на 150 фрагментах out/bench/case3_eval_ids.txt (эталонный набор кейса 3), только
статическим разбором строк — никакой компиляции/исполнения кода датасета (CLAUDE.md, запрет).

Пять срезов (см. постановку задачи в cases/codereview/findings.md):
  1. Покрытие блоков: доля непустых cwe/cert/flawfinder-блоков + доля fallback в cert_rules_block.
  2. Размеры: длины блоков и фрагмента (симв./токены~=симв/4), доля знаний в промпте.
  3. Разнообразие: сколько разных наборов CWE-карточек / CERT-правил на 150 фрагментов.
  4. Связь с истиной: candidate_cwe_ids(code) против истинного cwe_id (research/case3_recovered_labels.csv).
  5. Flawfinder отдельно: хитов на фрагмент, доля ложных тревог на истинно secure.

Запуск:
    .venv/bin/python cases/codereview/knowledge_diagnostics.py
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from core.data import load_case3
from cases.codereview.knowledge import (
    candidate_cwe_ids, cwe_cards_block, cert_rules_block, flawfinder_block,
    knowledge_stack_block, _CERT_RULES,
)
from cases.codereview.static_analyzer import run_flawfinder

_ROOT = Path(__file__).resolve().parents[2]
_EVAL_IDS_PATH = _ROOT / "out" / "bench" / "case3_eval_ids.txt"
_RECOVERED_LABELS_CSV = _ROOT / "research" / "case3_recovered_labels.csv"
_OUT_JSON = Path(__file__).resolve().parent / "out" / "knowledge_diagnostics.json"
_OUT_CSV = Path(__file__).resolve().parent / "out" / "knowledge_diagnostics_per_fragment.csv"

_FALLBACK_CERT_IDS = {"ARR30-C", "EXP34-C", "MEM30-C", "STR31-C"}

# Совпадает по определению с _CATEGORY_TO_CWE ∪ _CONFUSABLE_CLUSTER в knowledge.py —
# используется только чтобы отдельно посчитать "matched" (не-fallback) набор правил
# так же, как это делает cert_rules_block, без дублирования его внутренней логики
# (просто вызываем cert_rules_block и cwe_cards_block напрямую, ничего не переизобретаем).


def _tokens_est(s: str) -> float:
    """Грубая оценка токенов — символы / 4. Явно назван метод, никакого токенизатора."""
    return len(s) / 4


def _median_p90(values: list[float]) -> dict:
    if not values:
        return {"median": None, "p90": None}
    s = sorted(values)
    median = statistics.median(s)
    idx = min(len(s) - 1, int(round(0.9 * (len(s) - 1))))
    p90 = s[idx]
    return {"median": median, "p90": p90}


def cert_rules_is_fallback(code: str, max_rules: int = 5) -> bool:
    """Повторяет решение cert_rules_block: fallback срабатывает, когда ни одно правило
    из _CERT_RULES не пересекается по CWE с candidate_cwe_ids(code)."""
    ids = candidate_cwe_ids(code)
    cwe_set = {f"CWE-{c}" for c in ids}
    matched = [
        rule_id for rule_id, r in _CERT_RULES.items()
        if cwe_set & set(r.get("cwes", []))
    ]
    return len(matched) == 0


def cert_rule_ids_used(code: str, max_rules: int = 5) -> tuple[str, ...]:
    """Те же rule_id, что попадут в текст cert_rules_block (после сортировки и обрезки),
    для подсчёта разнообразия наборов."""
    ids = candidate_cwe_ids(code)
    cwe_set = {f"CWE-{c}" for c in ids}
    matched = [
        (rule_id, r) for rule_id, r in _CERT_RULES.items()
        if cwe_set & set(r.get("cwes", []))
    ]
    if not matched:
        matched = [(rid, _CERT_RULES[rid]) for rid in
                   ["ARR30-C", "EXP34-C", "MEM30-C", "STR31-C"] if rid in _CERT_RULES]
    matched.sort(key=lambda pair: len(pair[1].get("cwes", [])))
    return tuple(rid for rid, _ in matched[:max_rules])


def cwe_card_ids_used(code: str, max_cards: int = 5) -> tuple[str, ...]:
    from cases.codereview.knowledge import _CWE_CARDS
    return tuple(c for c in candidate_cwe_ids(code) if c in _CWE_CARDS)[:max_cards]


def main() -> None:
    ids_raw = [line.strip() for line in _EVAL_IDS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(i) for i in ids_raw]
    print(f"eval ids: {len(ids)}")

    df = load_case3()
    df["unique_id"] = df["unique_id"].astype(str)
    df = df[df["unique_id"].isin(ids)].drop_duplicates(subset="unique_id").set_index("unique_id")
    missing = [i for i in ids if i not in df.index]
    if missing:
        print(f"ВНИМАНИЕ: {len(missing)} id из eval-набора не найдены в датасете: {missing[:10]}")
    ids = [i for i in ids if i in df.index]
    print(f"фрагментов с кодом: {len(ids)}")

    gold_df = pd.read_csv(_RECOVERED_LABELS_CSV)
    gold_df["unique_id"] = gold_df["unique_id"].astype(str)
    gold = {}
    for _, row in gold_df.iterrows():
        raw = row["recovered_label"] if pd.notna(row["recovered_label"]) else None
        label = {"0": "secure", "1": "vulnerable"}.get(raw)
        gold[row["unique_id"]] = {
            "label": label,
            "cwe_id": row["cwe_id"] if pd.notna(row["cwe_id"]) else None,
        }

    per_fragment = []

    for uid in ids:
        code = df.loc[uid, "code"]
        code = "" if not isinstance(code, str) else code

        cwe_block = cwe_cards_block(code)
        cert_block = cert_rules_block(code)
        flaw_block = flawfinder_block(code)
        stack_block = knowledge_stack_block(code)

        cert_fallback = cert_rules_is_fallback(code)
        cert_rule_ids = cert_rule_ids_used(code)
        cwe_ids_shown = cwe_card_ids_used(code)
        cand_ids = candidate_cwe_ids(code)

        flaw_hits = run_flawfinder(code)

        g = gold.get(uid, {"label": None, "cwe_id": None})
        true_cwe = g["cwe_id"]
        true_cwe_num = true_cwe.removeprefix("CWE-") if isinstance(true_cwe, str) else None
        cwe_rank = None
        if g["label"] == "vulnerable" and true_cwe_num:
            if true_cwe_num in cand_ids:
                cwe_rank = cand_ids.index(true_cwe_num) + 1  # 1-based

        row = {
            "unique_id": uid,
            "true_label": g["label"],
            "true_cwe": true_cwe,
            "code_len_chars": len(code),
            "cwe_block_len": len(cwe_block),
            "cert_block_len": len(cert_block),
            "flaw_block_len": len(flaw_block),
            "stack_block_len": len(stack_block),
            "cwe_block_nonempty": bool(cwe_block),
            "cert_block_nonempty": bool(cert_block),
            "flaw_block_nonempty": bool(flaw_block),
            "cert_is_fallback": cert_fallback,
            "cert_rule_ids": ";".join(cert_rule_ids),
            "cwe_ids_shown": ";".join(cwe_ids_shown),
            "candidate_cwe_ids": ";".join(cand_ids),
            "n_flaw_hits": len(flaw_hits),
            "true_cwe_in_candidates": cwe_rank is not None,
            "true_cwe_candidate_rank": cwe_rank,
        }
        per_fragment.append(row)

    n = len(per_fragment)

    # --- 1. Покрытие блоков ---
    coverage = {
        "n_fragments": n,
        "cwe_cards_block_nonempty_frac": round(sum(r["cwe_block_nonempty"] for r in per_fragment) / n, 4),
        "cert_rules_block_nonempty_frac": round(sum(r["cert_block_nonempty"] for r in per_fragment) / n, 4),
        "flawfinder_block_nonempty_frac": round(sum(r["flaw_block_nonempty"] for r in per_fragment) / n, 4),
        "cert_rules_fallback_frac": round(sum(r["cert_is_fallback"] for r in per_fragment) / n, 4),
    }

    # --- 2. Размеры ---
    code_chars = [r["code_len_chars"] for r in per_fragment]
    cwe_chars = [r["cwe_block_len"] for r in per_fragment]
    cert_chars = [r["cert_block_len"] for r in per_fragment]
    flaw_chars = [r["flaw_block_len"] for r in per_fragment]
    stack_chars = [r["stack_block_len"] for r in per_fragment]

    def stats_block(chars: list[int]) -> dict:
        mp = _median_p90([float(c) for c in chars])
        tok_mp = _median_p90([c / 4 for c in chars])
        return {
            "chars_median": mp["median"], "chars_p90": mp["p90"],
            "tokens_est_median": tok_mp["median"], "tokens_est_p90": tok_mp["p90"],
        }

    sizes = {
        "token_estimate_method": "len(text) / 4 (грубая оценка символы->токены, не настоящий токенизатор)",
        "code_fragment": stats_block(code_chars),
        "cwe_cards_block": stats_block(cwe_chars),
        "cert_rules_block": stats_block(cert_chars),
        "flawfinder_block": stats_block(flaw_chars),
        "knowledge_stack_block": stats_block(stack_chars),
    }

    # доля знаниевого блока в общей длине промпта (код + блок) — для knowledge_stack_block и
    # отдельно для cert_rules_block (лучший одиночный блок по F1)
    stack_share = []
    cert_share = []
    for r in per_fragment:
        total_stack = r["code_len_chars"] + r["stack_block_len"]
        total_cert = r["code_len_chars"] + r["cert_block_len"]
        stack_share.append(r["stack_block_len"] / total_stack if total_stack else 0.0)
        cert_share.append(r["cert_block_len"] / total_cert if total_cert else 0.0)
    sizes["knowledge_share_of_prompt"] = {
        "knowledge_stack_block_variant": _median_p90(stack_share),
        "cert_rules_block_variant": _median_p90(cert_share),
        "definition": "block_len / (code_len + block_len), медиана и p90 по 150 фрагментам",
    }

    # --- 3. Разнообразие ---
    cwe_sets = {r["cwe_ids_shown"] for r in per_fragment}
    cert_sets = {r["cert_rule_ids"] for r in per_fragment}
    from collections import Counter
    cwe_set_counts = Counter(r["cwe_ids_shown"] for r in per_fragment)
    cert_set_counts = Counter(r["cert_rule_ids"] for r in per_fragment)
    diversity = {
        "n_distinct_cwe_card_sets": len(cwe_sets),
        "n_distinct_cert_rule_sets": len(cert_sets),
        "most_common_cwe_card_set": cwe_set_counts.most_common(1)[0] if cwe_set_counts else None,
        "most_common_cwe_card_set_frac": round(cwe_set_counts.most_common(1)[0][1] / n, 4) if cwe_set_counts else None,
        "most_common_cert_rule_set": cert_set_counts.most_common(1)[0] if cert_set_counts else None,
        "most_common_cert_rule_set_frac": round(cert_set_counts.most_common(1)[0][1] / n, 4) if cert_set_counts else None,
        "top5_cwe_card_sets": cwe_set_counts.most_common(5),
        "top5_cert_rule_sets": cert_set_counts.most_common(5),
    }

    # --- 4. Связь с истиной ---
    vuln_with_cwe = [r for r in per_fragment if r["true_label"] == "vulnerable" and r["true_cwe"]]
    hit = [r for r in vuln_with_cwe if r["true_cwe_in_candidates"]]
    ranks = [r["true_cwe_candidate_rank"] for r in hit]
    truth_link = {
        "n_true_vulnerable_with_known_cwe": len(vuln_with_cwe),
        "n_true_cwe_in_candidates": len(hit),
        "hit_rate": round(len(hit) / len(vuln_with_cwe), 4) if vuln_with_cwe else None,
        "rank_distribution": dict(Counter(ranks)),
        "rank_median": statistics.median(ranks) if ranks else None,
        "note": "candidate_cwe_ids(code) — полный список кандидатов до обрезки MAX_CWE_CARDS=5 "
                "в cwe_cards_block; rank учитывает порядок появления в candidate_cwe_ids",
    }
    # также: сколько из этих hit реально попало в топ-5 (то, что физически видит модель в cwe_cards_block)
    truth_link["n_true_cwe_in_shown_top5"] = sum(
        1 for r in vuln_with_cwe
        if r["true_cwe_candidate_rank"] is not None and r["true_cwe_candidate_rank"] <= 5
    )
    truth_link["hit_rate_top5_shown"] = round(
        truth_link["n_true_cwe_in_shown_top5"] / len(vuln_with_cwe), 4
    ) if vuln_with_cwe else None

    # --- 5. Flawfinder отдельно ---
    hits_per_fragment = [r["n_flaw_hits"] for r in per_fragment]
    true_secure = [r for r in per_fragment if r["true_label"] == "secure"]
    secure_with_hits = [r for r in true_secure if r["n_flaw_hits"] > 0]
    flawfinder_stats = {
        "avg_hits_per_fragment": round(sum(hits_per_fragment) / n, 4),
        "median_hits_per_fragment": statistics.median(hits_per_fragment),
        "frac_fragments_with_any_hit": round(sum(1 for h in hits_per_fragment if h > 0) / n, 4),
        "n_true_secure": len(true_secure),
        "n_true_secure_with_flawfinder_hit": len(secure_with_hits),
        "false_alarm_rate_on_true_secure": round(len(secure_with_hits) / len(true_secure), 4) if true_secure else None,
    }

    result = {
        "n_eval_ids": len(ids_raw),
        "n_fragments_scored": n,
        "1_coverage": coverage,
        "2_sizes": sizes,
        "3_diversity": diversity,
        "4_truth_link_candidate_cwe": truth_link,
        "5_flawfinder": flawfinder_stats,
    }

    _OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    _OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"написано {_OUT_JSON}")

    fieldnames = list(per_fragment[0].keys())
    with _OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_fragment)
    print(f"написано {_OUT_CSV}")

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
