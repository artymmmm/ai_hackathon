"""Финальный замер конфигурации A (screener sensitive + CWE/CERT/flawfinder, БЕЗ retrieval —
переносимая на чужой код версия) на полных 150 eval id. Аналог run_sensitive_full.py, добавляет
knowledge_stack_block. Использует кеш — часть из 150 уже посчитана в run_config_experiment.py
(n=40 sample_mixed)."""
from __future__ import annotations
import json, sys, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.data import load_case3
from core.llm import LLMClient, LLMConfig
from core.pipeline import PipelineContext
from cases.codereview.reviewer_configs import SYSTEM_PROMPT_SENSITIVE, review_one
from cases.codereview.knowledge import knowledge_stack_block

_ROOT = Path(__file__).resolve().parents[2]
_EVAL_IDS_TXT = _ROOT / "out" / "bench" / "case3_eval_ids.txt"
_OUT_PATH = _ROOT / "out" / "bench" / "case3_deepseek-chat_configA.json"

def _load_env():
    f = _ROOT / ".env"
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))

def main():
    _load_env()
    t0 = time.time()
    eval_ids = [x.strip() for x in _EVAL_IDS_TXT.read_text().split() if x.strip()]
    corpus = load_case3()
    corpus["unique_id"] = corpus["unique_id"].astype(str)
    id_set = set(eval_ids)
    sub = corpus[corpus["unique_id"].isin(id_set)].reset_index(drop=True)
    print(f"eval ids: {len(eval_ids)}, найдено в корпусе: {len(sub)}")

    llm_config = LLMConfig(
        model="deepseek-chat", backend="openai_compat", base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY", temperature=0.0, max_tokens=2048, max_concurrency=8,
        dry_run=False, cache_path="out/llm_cache.sqlite3",
    )
    llm = LLMClient(llm_config)
    ctx = PipelineContext(case="codereview", config={}, llm=llm)

    verdicts = []
    for i, row in sub.iterrows():
        doc_id = str(row["unique_id"])
        code = row["code"]
        kb = knowledge_stack_block(code)
        v = review_one(doc_id, code, ctx, system_prompt=SYSTEM_PROMPT_SENSITIVE,
                        use_json_example_sensitive=True, knowledge_block=kb)
        verdicts.append(v)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(sub)}")

    llm.close()
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUT_PATH.write_text(json.dumps([v.model_dump() for v in verdicts], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"elapsed={round(time.time()-t0,1)}s usage={llm.usage.as_dict()}")
    print(f"-> {_OUT_PATH}")

if __name__ == "__main__":
    main()
