"""ШАГИ 2а и 3: прогон и сравнение промпт-конфигураций на малых выборках (бюджет),
с обязательным кешем (`core/llm.py`, SQLite) и честным исключением eval из пулов поиска.

Конфигурации:
  bare        — оригинальный промпт reviewer.py. НЕ вызывается заново — метрики берутся из уже
                посчитанного out/bench/case3_deepseek-chat.json (нулевая стоимость, честно то же
                самое, т.к. промпт идентичен и кеш точно совпал бы).
  sensitive   — SYSTEM_PROMPT_SENSITIVE (высокочувствительный скринер, шаг 2а), без знаний/retrieval.
  config_A    — sensitive + knowledge_stack_block (CWE-карточки + CERT + flawfinder, шаг 3).
  config_B    — config_A + retrieval-соседи (k=5, пул = train pool БЕЗ eval-150).

Дополнительно — кросс-проектный тест конфигурации B на Chrome-подвыборке:
  config_B_in_distribution   — пул поиска включает все проекты (кроме eval).
  config_B_cross_project     — пул поиска ИСКЛЮЧАЕТ Chrome целиком; eval — только Chrome-фрагменты.

Выборки (см. `select_experiment_samples()`):
  sample_mixed  — 20 vulnerable + 20 secure из eval-150 (любые проекты) — для bare/sensitive/A/B.
  sample_chrome — 10 vulnerable (все, что есть) + 14 secure, ВСЕ Chrome, из eval-150 — для
                  in-distribution vs cross-project теста конфигурации B.

Запуск (нужен .env с DEEPSEEK_API_KEY):
    set -a && . ./.env && set +a && .venv/bin/python cases/codereview/run_config_experiment.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.llm import LLMClient, LLMConfig  # noqa: E402
from core.pipeline import PipelineContext  # noqa: E402

from cases.codereview.evaluate import load_gold  # noqa: E402
from cases.codereview.knn_baseline import load_pool_and_eval  # noqa: E402
from cases.codereview.knowledge import knowledge_stack_block  # noqa: E402
from cases.codereview.retrieval import NearestExampleRetriever, format_neighbors_block  # noqa: E402
from cases.codereview.reviewer_configs import SYSTEM_PROMPT_SENSITIVE, review_one  # noqa: E402
from cases.codereview.reviewer import SYSTEM_PROMPT as SYSTEM_PROMPT_BASELINE  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_OUT_DIR = Path(__file__).resolve().parent / "out"
_BASELINE_VERDICTS = _ROOT / "out" / "bench" / "case3_deepseek-chat.json"
_LABELS = ["secure", "vulnerable"]


def _load_env() -> None:
    f = _ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def select_experiment_samples(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, eval_df = load_pool_and_eval()  # eval_df: unique_id, code, label (0/1) — уже исключён из пула
    gold_full = pd.read_csv(_ROOT / "research" / "case3_recovered_labels.csv")
    gold_full["unique_id"] = gold_full["unique_id"].astype(int)
    proj_map = gold_full.set_index("unique_id")["source_project"]
    eval_df = eval_df.assign(source_project=eval_df["unique_id"].map(proj_map))

    vuln = eval_df[eval_df.label == 1]
    secure = eval_df[eval_df.label == 0]
    sample_mixed = pd.concat([
        vuln.sample(n=min(20, len(vuln)), random_state=seed),
        secure.sample(n=min(20, len(secure)), random_state=seed),
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)

    chrome = eval_df[eval_df.source_project == "Chrome"]
    chrome_vuln = chrome[chrome.label == 1]
    chrome_secure = chrome[chrome.label == 0].sample(n=min(14, len(chrome[chrome.label == 0])), random_state=seed)
    sample_chrome = pd.concat([chrome_vuln, chrome_secure]).sample(frac=1, random_state=seed).reset_index(drop=True)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    (_OUT_DIR / "sample_mixed_ids.txt").write_text("\n".join(str(x) for x in sample_mixed.unique_id))
    (_OUT_DIR / "sample_chrome_ids.txt").write_text("\n".join(str(x) for x in sample_chrome.unique_id))
    return sample_mixed, sample_chrome


def _metrics(y_true: list[str], y_pred: list[str]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "vulnerable" and p == "vulnerable")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == "secure" and p == "vulnerable")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "vulnerable" and p != "vulnerable")
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == "secure" and p != "vulnerable")
    n_uncertain = sum(1 for p in y_pred if p == "uncertain")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "fpr": round(fpr, 3), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n": len(y_true), "n_uncertain": n_uncertain}


def _cwe_accuracy(verdicts: list, gold: dict) -> tuple[int, int]:
    from cases.codereview.cwe_map import normalize_cwe
    correct, comparable = 0, 0
    for v in verdicts:
        g = gold.get(v.doc_id)
        if g is None or g["label"] != "vulnerable" or v.verdict != "vulnerable" or not g["cwe_id"]:
            continue
        pred = normalize_cwe(v.artifacts.get("cwe_id"))
        if pred is None:
            continue
        comparable += 1
        if pred == g["cwe_id"]:
            correct += 1
    return correct, comparable


def run_config(sample_df: pd.DataFrame, ctx: PipelineContext, *, system_prompt: str,
               sensitive_schema: bool, use_knowledge: bool,
               retriever: NearestExampleRetriever | None, k: int = 5) -> list:
    verdicts = []
    for _, row in sample_df.iterrows():
        doc_id = str(int(row["unique_id"]))
        code = row["code"]
        knowledge_block = knowledge_stack_block(code) if use_knowledge else ""
        neighbors_block = ""
        if retriever is not None:
            neighbors = retriever.query(code, k=k, exclude_unique_id=int(row["unique_id"]))
            neighbors_block = format_neighbors_block(neighbors)
        v = review_one(
            doc_id, code, ctx, system_prompt=system_prompt,
            use_json_example_sensitive=sensitive_schema,
            knowledge_block=knowledge_block, neighbors_block=neighbors_block,
        )
        verdicts.append(v)
    return verdicts


def eval_against_gold(verdicts: list, gold: dict) -> dict:
    y_true, y_pred, y_raw = [], [], []
    for v in verdicts:
        g = gold.get(v.doc_id)
        if g is None or g["label"] is None:
            continue
        y_true.append(g["label"])
        y_pred.append(v.verdict if v.verdict in _LABELS else "secure")  # uncertain->secure, как evaluate.py
        y_raw.append(v.verdict)
    m = _metrics(y_true, y_pred)
    # БАГ (исправлено): _metrics считает n_uncertain по y_pred, где "uncertain" уже заменён на
    # "secure" строкой выше — там его не может быть по построению, счётчик всегда был 0.
    # Настоящая эскалация — по сырым вердиктам ДО замены.
    m["n_uncertain"] = sum(1 for r in y_raw if r == "uncertain")
    m["escalation_rate"] = round(m["n_uncertain"] / len(y_raw), 3) if y_raw else 0.0
    correct, comparable = _cwe_accuracy(verdicts, gold)
    m["cwe_correct"] = correct
    m["cwe_comparable"] = comparable
    m["cwe_accuracy"] = round(correct / comparable, 3) if comparable else None
    return m


def baseline_metrics_on_subset(ids: list[str], gold: dict) -> dict:
    data = json.loads(_BASELINE_VERDICTS.read_text(encoding="utf-8"))
    from core.schema import Verdict
    verdicts = [Verdict(**d) for d in data if d["doc_id"] in set(ids)]
    return eval_against_gold(verdicts, gold)


def main() -> None:
    _load_env()
    t0 = time.time()
    gold = load_gold()

    print("Отбор выборок для эксперимента (детерминировано, seed=42)...")
    sample_mixed, sample_chrome = select_experiment_samples()
    print(f"sample_mixed: {len(sample_mixed)} (vulnerable={int(sample_mixed.label.sum())})")
    print(f"sample_chrome: {len(sample_chrome)} (vulnerable={int(sample_chrome.label.sum())})")

    print("\nПостроение retrieval-пулов (train pool БЕЗ eval-150, дисциплина утечки — см. "
          "knn_baseline.load_pool_and_eval)...")
    pool_df, _ = load_pool_and_eval()
    gold_full = pd.read_csv(_ROOT / "research" / "case3_recovered_labels.csv")
    gold_full["unique_id"] = gold_full["unique_id"].astype(int)
    proj_map = gold_full.set_index("unique_id")["source_project"]
    pool_df = pool_df.assign(source_project=pool_df["unique_id"].map(proj_map))
    cwe_map = gold_full.set_index("unique_id")["cwe_id"]
    pool_df = pool_df.assign(cwe_id=pool_df["unique_id"].map(cwe_map))

    retriever_full = NearestExampleRetriever(pool_df)
    pool_no_chrome = pool_df[pool_df.source_project != "Chrome"].reset_index(drop=True)
    retriever_no_chrome = NearestExampleRetriever(pool_no_chrome)
    print(f"retriever_full pool={len(pool_df)}, retriever_no_chrome pool={len(pool_no_chrome)}")

    llm_config = LLMConfig(
        model="deepseek-chat", backend="openai_compat", base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY", temperature=0.0, max_tokens=2048, max_concurrency=6,
        dry_run=False, cache_path="out/llm_cache.sqlite3",
    )
    llm = LLMClient(llm_config)
    ctx = PipelineContext(case="codereview", config={}, llm=llm)

    results: dict[str, dict] = {}
    all_verdicts: dict[str, list] = {}

    print("\n=== bare (из уже посчитанного case3_deepseek-chat.json, 0 новых вызовов) ===")
    ids_mixed = [str(int(x)) for x in sample_mixed.unique_id]
    results["bare"] = baseline_metrics_on_subset(ids_mixed, gold)
    print(results["bare"])

    print("\n=== sensitive (новый промпт, без знаний/retrieval) ===")
    v = run_config(sample_mixed, ctx, system_prompt=SYSTEM_PROMPT_SENSITIVE,
                    sensitive_schema=True, use_knowledge=False, retriever=None)
    all_verdicts["sensitive"] = v
    results["sensitive"] = eval_against_gold(v, gold)
    print(results["sensitive"])

    print("\n=== config_A (sensitive + CWE/CERT/flawfinder) ===")
    v = run_config(sample_mixed, ctx, system_prompt=SYSTEM_PROMPT_SENSITIVE,
                    sensitive_schema=True, use_knowledge=True, retriever=None)
    all_verdicts["config_A"] = v
    results["config_A"] = eval_against_gold(v, gold)
    print(results["config_A"])

    print("\n=== config_B (config_A + retrieval k=5, пул = все проекты кроме eval) ===")
    v = run_config(sample_mixed, ctx, system_prompt=SYSTEM_PROMPT_SENSITIVE,
                    sensitive_schema=True, use_knowledge=True, retriever=retriever_full, k=5)
    all_verdicts["config_B"] = v
    results["config_B"] = eval_against_gold(v, gold)
    print(results["config_B"])

    print("\n=== config_B_in_distribution (Chrome-выборка, пул включает Chrome) ===")
    v = run_config(sample_chrome, ctx, system_prompt=SYSTEM_PROMPT_SENSITIVE,
                    sensitive_schema=True, use_knowledge=True, retriever=retriever_full, k=5)
    all_verdicts["config_B_in_distribution"] = v
    results["config_B_in_distribution"] = eval_against_gold(v, gold)
    print(results["config_B_in_distribution"])

    print("\n=== config_B_cross_project (Chrome-выборка, пул БЕЗ Chrome) ===")
    v = run_config(sample_chrome, ctx, system_prompt=SYSTEM_PROMPT_SENSITIVE,
                    sensitive_schema=True, use_knowledge=True, retriever=retriever_no_chrome, k=5)
    all_verdicts["config_B_cross_project"] = v
    results["config_B_cross_project"] = eval_against_gold(v, gold)
    print(results["config_B_cross_project"])

    llm.close()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    (_OUT_DIR / "config_experiment_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, verdicts in all_verdicts.items():
        (_OUT_DIR / f"config_experiment_verdicts_{name}.json").write_text(
            json.dumps([v.model_dump() for v in verdicts], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"\n=== ИТОГОВАЯ ТАБЛИЦА (elapsed {round(time.time() - t0, 1)}s, "
          f"llm usage: {llm.usage.as_dict()}) ===")
    print(f"{'config':30s} {'precision':>10s} {'recall':>8s} {'f1':>6s} {'fpr':>6s} "
          f"{'n_unc':>6s} {'cwe_acc':>8s}")
    for name, m in results.items():
        cwe_acc = f"{m['cwe_accuracy']:.3f}" if m["cwe_accuracy"] is not None else "n/a"
        print(f"{name:30s} {m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1']:>6.3f} "
              f"{m['fpr']:>6.3f} {m['n_uncertain']:>6d} {cwe_acc:>8s}")
    print(f"\n-> {_OUT_DIR / 'config_experiment_results.json'}")


if __name__ == "__main__":
    main()
