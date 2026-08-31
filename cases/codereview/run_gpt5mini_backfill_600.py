"""Добор проваленных вызовов из case3_gpt5-mini_cert_only_600.json.

Вчерашний прогон run_gpt5mini_cert_only_600.py (max_concurrency=64) сорвался: 313/600
получили живой ответ, 287 упали (270 HTTP 402 in_flight_budget_exhausted — OpenRouter
резервирует кредиты под все запросы в полёте при высоком параллелизме, 11 непарсящихся
JSON, 6 finish_reason=length). У упавших в поле `rationale` подстрока `llm_call_failed`.

Этот скрипт:
- читает существующий out/bench/case3_gpt5-mini_cert_only_600.json;
- отбирает doc_id с `llm_call_failed` в rationale;
- прогоняет только их в ТОЧНО той же конфигурации (модель, промпт, temperature=0.0,
  max_tokens=4096 — совпадает с оригиналом ради ключа кеша out/llm_cache_case3_gpt5mini.sqlite3,
  313 успешных вызовов оттуда не переоплачиваются), но max_concurrency=2 (было 64) —
  единственное отличие, устраняющее причину 402;
- сливает новые вердикты со старыми успешными и перезаписывает тот же файл, сохраняя
  порядок doc_id из out/bench/case3_eval600_ids.txt;
- пишет usage добора в out/bench/case3_gpt5-mini_backfill_usage.json.

Переменная окружения CASE3_BACKFILL_LIMIT=N — для дымового прогона на первых N провалившихся
id (не меняет основной путь при отсутствии).

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
_OUT_USAGE = _BENCH_DIR / "case3_gpt5-mini_backfill_usage.json"


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
    assert len(ids) == 600, f"ожидали 600, получили {len(ids)}"

    existing = json.loads(_OUT_VERDICTS.read_text(encoding="utf-8"))
    by_id = {v["doc_id"]: v for v in existing}
    assert set(by_id) == set(ids), "doc_id в существующем файле и eval600_ids.txt расходятся"

    failed_ids = [i for i in ids if "llm_call_failed" in (by_id[i].get("rationale") or "")]
    print(f"всего 600, успешных {len(ids) - len(failed_ids)}, к добору {len(failed_ids)}")

    only_ids = os.environ.get("CASE3_BACKFILL_IDS")
    if only_ids:
        wanted = set(only_ids.split(","))
        failed_ids = [i for i in failed_ids if i in wanted]
        print(f"CASE3_BACKFILL_IDS -> добираем только {failed_ids}")

    limit = os.environ.get("CASE3_BACKFILL_LIMIT")
    if limit:
        failed_ids = failed_ids[: int(limit)]
        print(f"CASE3_BACKFILL_LIMIT={limit} -> добираем только первые {len(failed_ids)}")

    corpus = load_case3()
    corpus["unique_id"] = corpus["unique_id"].astype(str)
    code_by_id = dict(zip(corpus["unique_id"], corpus["code"]))
    missing = [i for i in failed_ids if i not in code_by_id]
    assert not missing, f"doc_id не найдены в корпусе: {missing[:10]}"

    llm = LLMClient(LLMConfig(
        model="openai/gpt-5-mini", backend="openai_compat",
        base_url="https://openrouter.ai/api/v1", api_key_env="OPENROUTER_API_KEY",
        temperature=0.0, max_tokens=4096, max_concurrency=2,
        dry_run=False, cache_path=_CACHE_PATH,
    ))
    ctx = PipelineContext(case="codereview", config={}, llm=llm)

    print("\n=== backfill gpt-5-mini cert_only (max_concurrency=2) ===")
    t0 = time.time()
    new_results: dict[str, dict] = {}
    done = 0

    def _one(doc_id: str):
        code = code_by_id[doc_id]
        v = review_one(doc_id, code, ctx, system_prompt=SYSTEM_PROMPT_SENSITIVE,
                        use_json_example_sensitive=True, knowledge_block=cert_rules_block(code))
        return doc_id, v

    with ThreadPoolExecutor(max_workers=llm.config.max_concurrency) as ex:
        for doc_id, v in ex.map(_one, failed_ids):
            new_results[doc_id] = v.model_dump()
            done += 1
            if done % 20 == 0 or done == len(failed_ids):
                print(f"  {done}/{len(failed_ids)}")

    elapsed = round(time.time() - t0, 1)
    print(f"elapsed={elapsed}s")

    still_failed = [i for i in failed_ids if "llm_call_failed" in (new_results[i].get("rationale") or "")]
    print(f"после добора остались провалы: {len(still_failed)}/{len(failed_ids)}")

    merged = dict(by_id)
    merged.update(new_results)
    verdicts_out = [merged[i] for i in ids]
    _OUT_VERDICTS.write_text(json.dumps(verdicts_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"verdicts -> {_OUT_VERDICTS}")

    usage = llm.usage.as_dict()
    _OUT_USAGE.write_text(json.dumps({
        "model": "openai/gpt-5-mini", "backend": "openai_compat",
        "n_backfilled": len(failed_ids), "n_still_failed": len(still_failed),
        "elapsed_seconds": elapsed, "usage": usage,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"usage -> {_OUT_USAGE}: {usage}")

    llm.close()


if __name__ == "__main__":
    main()
