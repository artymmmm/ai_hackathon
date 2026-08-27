"""Новые вызовы LLM для Целей 1 и 2 (см. промпт координатора). Переиспользует форсированный
бинарный промпт и функцию `review_forced` из `run_cascade_stage2.py` — код промпта не дублируется.

Кто сюда попадает (union двух множеств, каждый id вызывается ОДИН раз, а не дважды):
- Цель 1: 10 id, которые uncertain у скринера (SYSTEM_PROMPT_SENSITIVE), но НЕ входят в 89
  uncertain у cert_only (для тех уже есть готовая ступень 2 в cascade_B_k*of5.json).
- Цель 2: 29 id, у которых cert_only вердикт == "secure" (сейчас ступень 2 на них никогда не
  запускается — расширяем область применения).
Пересечение — 3 id, считаются один раз.

Для каждого id: 5 сэмплов, temperature=0.7, use_cache=False (см. докстринг run_cascade_stage2.py
про схлопывание кеша при одинаковом промпте/параметрах) — свой кеш `out/llm_cache_case3_cascade.sqlite3`.

Пишет:
- `cases/codereview/out/stage2_extended_samples.json` — компактные сэмплы (verdict/confidence/
  cwe_id/hash сырого текста) по каждому новому id, в том же формате, что и существующий
  `cascade_stage2_variant_b_samples.json`, для переиспользования комбинациями (`combo_analysis.py`,
  `run_secure_stage2_merge.py`).
- `cases/codereview/out/extended_stage2_diversity_check.json` — проверка на схлопывание сэмплов.
- `cases/codereview/out/extended_stage2_usage.json` — токены/стоимость этого прогона.

НИКОГДА не исполнять и не компилировать код из датасета — только статический анализ (CLAUDE.md).
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.data import load_case3
from core.llm import LLMClient, LLMConfig
from core.pipeline import PipelineContext
from cases.codereview.run_cascade_stage2 import review_forced, _load_env

_ROOT = Path(__file__).resolve().parents[2]
_BENCH = _ROOT / "out" / "bench"
_OUT = _ROOT / "cases" / "codereview" / "out"
_CACHE_PATH = "out/llm_cache_case3_cascade.sqlite3"


def _load(path: Path) -> dict[str, dict]:
    return {d["doc_id"]: d for d in json.loads(path.read_text(encoding="utf-8"))}


def main() -> None:
    _load_env()

    sensitive = _load(_BENCH / "case3_deepseek-chat_sensitive.json")
    cert = _load(_BENCH / "case3_deepseek-chat_cert_only.json")
    ids = sorted(sensitive)
    assert ids == sorted(cert)

    c_unc = {i for i in ids if cert[i]["verdict"] == "uncertain"}
    s_unc = {i for i in ids if sensitive[i]["verdict"] == "uncertain"}
    obj1_new = sorted(s_unc - c_unc)
    obj2_secure = sorted(i for i in ids if cert[i]["verdict"] == "secure")
    target_ids = sorted(set(obj1_new) | set(obj2_secure))

    print(f"obj1_new (screener-uncertain, не в cert_only-89) = {len(obj1_new)}")
    print(f"obj2_secure (cert_only secure bucket) = {len(obj2_secure)}")
    print(f"union новых id для ступени 2 = {len(target_ids)}")

    corpus = load_case3()
    corpus["unique_id"] = corpus["unique_id"].astype(str)
    code_by_id = dict(zip(corpus["unique_id"], corpus["code"]))
    missing = [i for i in target_ids if i not in code_by_id]
    assert not missing, f"doc_id не найдены в корпусе: {missing}"

    llm = LLMClient(LLMConfig(
        model="deepseek-chat", backend="openai_compat", base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY", temperature=0.0, max_tokens=2048, max_concurrency=8,
        dry_run=False, cache_path=_CACHE_PATH,
    ))
    ctx = PipelineContext(case="codereview", config={}, llm=llm)

    samples: dict[str, list[tuple]] = {i: [] for i in target_ids}
    jobs = [(doc_id, s) for doc_id in target_ids for s in range(5)]
    print(f"\n=== {len(jobs)} вызовов (temp=0.7, k=5 сэмплов на {len(target_ids)} id) ===")
    t0 = time.time()

    def _run(job):
        doc_id, s = job
        v, raw_text = review_forced(doc_id, code_by_id[doc_id], ctx, temperature=0.7,
                                     use_cache=False)
        return doc_id, s, v, raw_text

    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for doc_id, s, v, raw_text in ex.map(_run, jobs):
            samples[doc_id].append((v, raw_text))
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(jobs)}")
    print(f"elapsed={round(time.time()-t0,1)}s")

    # ---- проверка на схлопывание ----
    collapsed = []
    unique_counts = []
    for doc_id, s_list in samples.items():
        texts = [t for _, t in s_list]
        n_unique = len(set(texts))
        unique_counts.append(n_unique)
        if n_unique == 1:
            collapsed.append(doc_id)
    avg_unique = sum(unique_counts) / len(unique_counts)
    print(f"\nСхлопывание: среднее уникальных ответов на fragment = {avg_unique:.2f}/5 "
          f"({len(target_ids)} фрагментов). Идентичных 5/5: {len(collapsed)}.")
    (_OUT / "extended_stage2_diversity_check.json").write_text(json.dumps({
        "avg_unique_responses_per_fragment_of_5": round(avg_unique, 3),
        "fragments_with_all_5_identical": len(collapsed),
        "collapsed_doc_ids": collapsed,
        "n_fragments": len(target_ids),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- компактные сэмплы для переиспользования ----
    compact = {
        doc_id: [
            {"verdict": v.verdict, "confidence": v.confidence, "cwe_id": v.artifacts.get("cwe_id"),
             "text_sha256_short": _short_hash(raw_text)}
            for v, raw_text in s_list
        ]
        for doc_id, s_list in samples.items()
    }
    (_OUT / "stage2_extended_samples.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsamples -> {_OUT / 'stage2_extended_samples.json'}")

    # ---- полные Verdict-объекты по каждому порогу k=1..4, отдельно на всём target_ids ----
    # (используются напрямую сборщиками комбинаций objective 1 / objective 2)
    full_by_k: dict[int, dict[str, dict]] = {k: {} for k in (1, 2, 3, 4)}
    for doc_id, s_list in samples.items():
        for k in (1, 2, 3, 4):
            votes = sum(1 for v, _ in s_list if v.verdict == "vulnerable")
            final_verdict = "vulnerable" if votes >= k else "secure"
            agreeing = [v for v, _ in s_list if v.verdict == final_verdict]
            rep = agreeing[0] if agreeing else s_list[0][0]
            full_by_k[k][doc_id] = {
                **rep.model_dump(),
                "verdict": final_verdict,
                "confidence": round(votes / 5, 2),
                "action": "block" if final_verdict == "vulnerable" else "pass",
                "artifacts": {**rep.artifacts, "vote_count_vulnerable_of_5": votes,
                              "vote_threshold_k": k, "source": "llm_reviewer_extended_stage2"},
            }
    (_OUT / "stage2_extended_full_by_k.json").write_text(
        json.dumps(full_by_k, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"full_by_k -> {_OUT / 'stage2_extended_full_by_k.json'}")

    llm.close()
    usage = llm.usage.as_dict()
    print(f"\nusage={usage}")
    (_OUT / "extended_stage2_usage.json").write_text(json.dumps(usage, ensure_ascii=False, indent=2),
                                                       encoding="utf-8")


def _short_hash(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


if __name__ == "__main__":
    main()
