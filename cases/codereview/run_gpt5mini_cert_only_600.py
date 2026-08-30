"""`cert_only` (SYSTEM_PROMPT_SENSITIVE + cert_rules_block) на eval600, модель openai/gpt-5-mini
через OpenRouter — задача координатора, дополнение к шагам 0-3 (см. run_binary_single_600.py).

Зачем: базовый промпт на старом наборе 150 дал gpt-5-mini F1 0.494 против deepseek 0.237 —
разрыв нужно проверить на честном наборе eval600 в конфигурации, идентичной deepseek-версии
(out/bench/case3_cert_only_600.json) и qwen-версии (run_qwen_cert_only_600.py). Ничего не
подстраивать под gpt-5-mini, сравнение должно быть чистым.

Промпт и знания переиспользованы как есть: SYSTEM_PROMPT_SENSITIVE (reviewer_configs.py) +
cert_rules_block (knowledge.py), через review_one — тот же вызов, что и в
run_qwen_cert_only_600.py / run_configA_full.py для deepseek cert_only.

Кеш — отдельный файл (out/llm_cache_case3_gpt5mini.sqlite3), max_concurrency=64.

ОТКЛОНЕНИЕ от deepseek/qwen-конфигурации: max_tokens поднят с 2048 до 4096. gpt-5-mini —
reasoning-модель и на дымовом прогоне (20 id) один вызов ушёл в finish_reason=length
(весь max_tokens потрачен на рассуждения, content пустой) при max_tokens=2048. temperature=0.0
OpenRouter принял без ошибки — это отклонение не понадобилось.

Поддерживает переменную окружения CASE3_LIMIT для дымового прогона на первых N id
(например CASE3_LIMIT=20) — не меняет основной путь при её отсутствии.

НИКОГДА не исполнять и не компилировать код из датасета — только статический анализ (см. CLAUDE.md).
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.data import load_case3
from core.llm import LLMClient, LLMConfig
from core.pipeline import PipelineContext
from cases.codereview.reviewer_configs import SYSTEM_PROMPT_SENSITIVE, review_one
from cases.codereview.knowledge import cert_rules_block

_ROOT = Path(__file__).resolve().parents[2]
_EVAL600_IDS_TXT = _ROOT / "out" / "bench" / "case3_eval600_ids.txt"
_BENCH_DIR = _ROOT / "out" / "bench"
_CACHE_PATH = "out/llm_cache_case3_gpt5mini.sqlite3"
_OUT_VERDICTS = _BENCH_DIR / "case3_gpt5-mini_cert_only_600.json"
_OUT_USAGE = _BENCH_DIR / "case3_gpt5-mini_cert_only_600_usage.json"


def _load_env():
    for line in (_ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def main():
    _load_env()
    ids = [x.strip() for x in _EVAL600_IDS_TXT.read_text().split() if x.strip()]
    seen = set()
    ids = [i for i in ids if not (i in seen or seen.add(i))]
    print(f"eval600 ids: {len(ids)}")
    assert len(ids) == 600, f"ожидали 600, получили {len(ids)}"

    limit = os.environ.get("CASE3_LIMIT")
    if limit:
        ids = ids[: int(limit)]
        print(f"CASE3_LIMIT={limit} -> используем первые {len(ids)} id")

    corpus = load_case3()
    corpus["unique_id"] = corpus["unique_id"].astype(str)
    code_by_id = dict(zip(corpus["unique_id"], corpus["code"]))
    missing = [i for i in ids if i not in code_by_id]
    assert not missing, f"doc_id не найдены в корпусе: {missing[:10]}"

    llm = LLMClient(LLMConfig(
        model="openai/gpt-5-mini", backend="openai_compat",
        base_url="https://openrouter.ai/api/v1", api_key_env="OPENROUTER_API_KEY",
        temperature=0.0, max_tokens=4096, max_concurrency=64,
        dry_run=False, cache_path=_CACHE_PATH,
    ))
    ctx = PipelineContext(case="codereview", config={}, llm=llm)

    print("\n=== gpt-5-mini cert_only (SYSTEM_PROMPT_SENSITIVE + cert_rules_block, k=1, temp=0.0) ===")
    t0 = time.time()
    results: dict[str, dict] = {}
    done = 0

    def _one(doc_id: str):
        code = code_by_id[doc_id]
        v = review_one(doc_id, code, ctx, system_prompt=SYSTEM_PROMPT_SENSITIVE,
                        use_json_example_sensitive=True, knowledge_block=cert_rules_block(code))
        return doc_id, v

    with ThreadPoolExecutor(max_workers=llm.config.max_concurrency) as ex:
        for doc_id, v in ex.map(_one, ids):
            results[doc_id] = v.model_dump()
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(ids)}")

    elapsed = round(time.time() - t0, 1)
    print(f"elapsed={elapsed}s")

    verdicts_out = [results[i] for i in ids]
    _OUT_VERDICTS.write_text(json.dumps(verdicts_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"verdicts -> {_OUT_VERDICTS}")

    usage = llm.usage.as_dict()
    _OUT_USAGE.write_text(json.dumps({
        "model": "openai/gpt-5-mini", "backend": "openai_compat", "n_fragments": len(ids),
        "elapsed_seconds": elapsed, "usage": usage,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"usage -> {_OUT_USAGE}: {usage}")

    llm.close()


if __name__ == "__main__":
    main()
