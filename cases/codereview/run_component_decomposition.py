"""Разложение конфигурации A по компонентам (координатор, п.5): CWE-карточки, CERT-правила,
flawfinder — по отдельности поверх screener-промпта (sensitive), на том же sample_mixed (40,
детерминированный, sample_mixed_ids.txt), что и исходный config_A/B эксперимент — прямое
сравнение с уже посчитанными bare/sensitive/config_A метриками из config_experiment_results.json.

НЕ пересчитывает bare/sensitive/config_A/B заново — берёт их из уже посчитанного файла.
"""
from __future__ import annotations
import json, sys, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from core.llm import LLMClient, LLMConfig
from core.pipeline import PipelineContext
from cases.codereview.reviewer_configs import SYSTEM_PROMPT_SENSITIVE, review_one
from cases.codereview.knowledge import cwe_cards_block, cert_rules_block, flawfinder_block
from cases.codereview.evaluate import load_gold
from cases.codereview.run_config_experiment import eval_against_gold, select_experiment_samples

_ROOT = Path(__file__).resolve().parents[2]
_OUT_DIR = Path(__file__).resolve().parent / "out"

def _load_env():
    f = _ROOT / ".env"
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))

def run_config(sample_df, ctx, *, block_fn):
    verdicts = []
    for _, row in sample_df.iterrows():
        doc_id = str(int(row["unique_id"]))
        code = row["code"]
        kb = block_fn(code) if block_fn else ""
        v = review_one(doc_id, code, ctx, system_prompt=SYSTEM_PROMPT_SENSITIVE,
                        use_json_example_sensitive=True, knowledge_block=kb)
        verdicts.append(v)
    return verdicts

def main():
    _load_env()
    t0 = time.time()
    gold = load_gold()
    sample_mixed, _ = select_experiment_samples()  # тот же seed=42 -> тот же sample_mixed

    llm_config = LLMConfig(
        model="deepseek-chat", backend="openai_compat", base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY", temperature=0.0, max_tokens=2048, max_concurrency=8,
        dry_run=False, cache_path="out/llm_cache.sqlite3",
    )
    llm = LLMClient(llm_config)
    ctx = PipelineContext(case="codereview", config={}, llm=llm)

    results = {}
    for name, fn in [("cwe_cards_only", cwe_cards_block),
                      ("cert_only", cert_rules_block),
                      ("flawfinder_only", flawfinder_block)]:
        print(f"\n=== {name} ===")
        v = run_config(sample_mixed, ctx, block_fn=fn)
        m = eval_against_gold(v, gold)
        results[name] = m
        print(m)

    llm.close()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / "component_decomposition_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    base = json.loads((_OUT_DIR / "config_experiment_results.json").read_text())
    print(f"\n=== СВОДКА (n=40, elapsed {round(time.time()-t0,1)}s, usage {llm.usage.as_dict()}) ===")
    print(f"{'config':20s} {'precision':>10s} {'recall':>8s} {'f1':>6s} {'fpr':>6s} {'escal':>6s}")
    for name in ("bare", "sensitive"):
        m = base[name]
        print(f"{name:20s} {m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1']:>6.3f} {m['fpr']:>6.3f} {m['escalation_rate']:>6.3f}")
    for name, m in results.items():
        print(f"{name:20s} {m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1']:>6.3f} {m['fpr']:>6.3f} {m['escalation_rate']:>6.3f}")
    m = base["config_A"]
    print(f"{'config_A(all 3)':20s} {m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1']:>6.3f} {m['fpr']:>6.3f} {m['escalation_rate']:>6.3f}")
    print(f"\n-> {out_path}")

if __name__ == "__main__":
    main()
