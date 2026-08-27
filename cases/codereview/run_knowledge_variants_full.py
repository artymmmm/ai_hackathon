"""Проверка разложения конфигурации A на полных 150 eval id (координатор, п.4).

На n=40 отдельные компоненты обогнали полный стек: cert_only F1 0.600 против config_A 0.261.
Гипотеза: склейка трёх блоков знаний размывает внимание модели, а шум даёт flawfinder.
Здесь те же варианты гоняются на эталонном наборе 150 — единственном, с которым сравнимы
остальные строки out/bench/results.csv.

Варианты: cert_only, cwe_only, cwe_cert (конфигурация A без flawfinder).
Каждый — поверх SYSTEM_PROMPT_SENSITIVE, как в run_component_decomposition.py.
"""
from __future__ import annotations
import json, sys, time, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.data import load_case3
from core.llm import LLMClient, LLMConfig
from core.pipeline import PipelineContext
from cases.codereview.reviewer_configs import SYSTEM_PROMPT_SENSITIVE, review_one
from cases.codereview.knowledge import cwe_cards_block, cert_rules_block

_ROOT = Path(__file__).resolve().parents[2]
_EVAL_IDS_TXT = _ROOT / "out" / "bench" / "case3_eval_ids.txt"
_BENCH_DIR = _ROOT / "out" / "bench"


def _load_env():
    for line in (_ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def _cwe_cert(code: str) -> str:
    parts = [b for b in (cwe_cards_block(code), cert_rules_block(code)) if b]
    return "\n\n".join(parts)


VARIANTS = {
    "cert_only": cert_rules_block,
    "cwe_only": cwe_cards_block,
    "cwe_cert": _cwe_cert,
}


def main():
    _load_env()
    names = sys.argv[1:] or list(VARIANTS)
    eval_ids = {x.strip() for x in _EVAL_IDS_TXT.read_text().split() if x.strip()}
    corpus = load_case3()
    corpus["unique_id"] = corpus["unique_id"].astype(str)
    sub = corpus[corpus["unique_id"].isin(eval_ids)].reset_index(drop=True)
    print(f"eval ids: {len(eval_ids)}, найдено в корпусе: {len(sub)}")

    llm = LLMClient(LLMConfig(
        model="deepseek-chat", backend="openai_compat", base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY", temperature=0.0, max_tokens=2048, max_concurrency=8,
        dry_run=False, cache_path="out/llm_cache.sqlite3",
    ))
    ctx = PipelineContext(case="codereview", config={}, llm=llm)

    for name in names:
        block_fn = VARIANTS[name]
        t0 = time.time()
        print(f"\n=== {name} (n={len(sub)}) ===")

        def _one(i_row, block_fn=block_fn):
            i, row = i_row
            v = review_one(str(row["unique_id"]), row["code"], ctx,
                           system_prompt=SYSTEM_PROMPT_SENSITIVE,
                           use_json_example_sensitive=True,
                           knowledge_block=block_fn(row["code"]))
            return i, v

        results: dict[int, object] = {}
        done = 0
        with ThreadPoolExecutor(max_workers=llm.config.max_concurrency) as ex:
            for i, v in ex.map(_one, sub.iterrows()):
                results[i] = v
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{len(sub)}")
        verdicts = [results[i] for i in range(len(sub))]
        out_path = _BENCH_DIR / f"case3_deepseek-chat_{name}.json"
        out_path.write_text(json.dumps([v.model_dump() for v in verdicts],
                                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"elapsed={round(time.time()-t0,1)}s -> {out_path}")

    llm.close()
    print(f"usage={llm.usage.as_dict()}")


if __name__ == "__main__":
    main()
