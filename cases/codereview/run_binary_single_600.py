"""Форсированно бинарный `cert_only` как ЕДИНСТВЕННЫЙ проход (1 вызов на фрагмент) на eval600.

Контекст (задача координатора): `cert_only` (SYSTEM_PROMPT_SENSITIVE + cert_rules_block,
трёхклассовый vulnerable/secure/uncertain) на eval600 даёт P 0.540 / R 0.300 / F1 0.386 —
recall упирается в корзину uncertain (359 из 600), которая в метриках считается как secure.

Здесь та же комбинация знаний (`cert_rules_block`), но СИСТЕМНЫЙ ПРОМПТ — форсированный бинарный
(`SYSTEM_PROMPT_FORCED_BINARY` из `run_cascade_stage2.py`, переиспользован как есть, без
изменений) — запрещает verdict=uncertain как третий исход. Прогоняется как единственный проход
по ВСЕМ 600 фрагментам (не только по бывшей корзине uncertain), одним сэмплом (k=1, temp=0.0) —
та же цена вызовов, что у cert_only.

Кеш — отдельный файл (out/llm_cache_case3_binary.sqlite3), чтобы не конфликтовать с другими
агентами/прогонами, пишущими в out/llm_cache.sqlite3 или out/llm_cache_case3_cascade.sqlite3.

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
from cases.codereview.run_cascade_stage2 import review_forced

_ROOT = Path(__file__).resolve().parents[2]
_EVAL600_IDS_TXT = _ROOT / "out" / "bench" / "case3_eval600_ids.txt"
_BENCH_DIR = _ROOT / "out" / "bench"
_CACHE_PATH = "out/llm_cache_case3_binary.sqlite3"
_OUT_VERDICTS = _BENCH_DIR / "case3_binary_cert_only_600.json"
_OUT_USAGE = _BENCH_DIR / "case3_binary_cert_only_600_usage.json"


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

    corpus = load_case3()
    corpus["unique_id"] = corpus["unique_id"].astype(str)
    code_by_id = dict(zip(corpus["unique_id"], corpus["code"]))
    missing = [i for i in ids if i not in code_by_id]
    assert not missing, f"doc_id не найдены в корпусе: {missing[:10]}"

    llm = LLMClient(LLMConfig(
        model="deepseek-chat", backend="openai_compat", base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY", temperature=0.0, max_tokens=2048, max_concurrency=16,
        dry_run=False, cache_path=_CACHE_PATH,
    ))
    ctx = PipelineContext(case="codereview", config={}, llm=llm)

    print("\n=== binary_cert_only_single (форсированный бинарный, cert_rules_block, k=1, temp=0.0) ===")
    t0 = time.time()
    results: dict[str, dict] = {}
    done = 0

    def _one(doc_id: str):
        v, _raw = review_forced(doc_id, code_by_id[doc_id], ctx, temperature=0.0, use_cache=True)
        return doc_id, v

    with ThreadPoolExecutor(max_workers=llm.config.max_concurrency) as ex:
        for doc_id, v in ex.map(_one, ids):
            results[doc_id] = v.model_dump()
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(ids)}")

    elapsed = round(time.time() - t0, 1)
    print(f"elapsed={elapsed}s")

    n_disobeyed = sum(1 for r in results.values() if r["artifacts"].get("disobeyed_binary_instruction"))
    print(f"disobeyed_binary_instruction: {n_disobeyed} / {len(results)} "
          f"({round(100 * n_disobeyed / len(results), 1)}%)")

    verdicts_out = [results[i] for i in ids]
    _OUT_VERDICTS.write_text(json.dumps(verdicts_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"verdicts -> {_OUT_VERDICTS}")

    usage = llm.usage.as_dict()
    _OUT_USAGE.write_text(json.dumps({
        "model": "deepseek-chat", "backend": "openai_compat", "n_fragments": len(ids),
        "elapsed_seconds": elapsed, "n_disobeyed_binary_instruction": n_disobeyed,
        "usage": usage,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"usage -> {_OUT_USAGE}: {usage}")

    llm.close()


if __name__ == "__main__":
    main()
