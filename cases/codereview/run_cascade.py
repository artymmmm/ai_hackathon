"""Сквозной запускаемый каскад кейса 3: ступень 1 (`cert_only`) + ступень 2 (форсированный
бинарный вердикт, k сэмплов, голосование по порогу) — обе ступени МОГУТ идти на разных
моделях и разных бэкендах (`core/llm.py`).

Контекст и промпты не переизобретаются, а переиспользуются:
  - ступень 1 — `cert_only` из `run_knowledge_variants_full.py`: `SYSTEM_PROMPT_SENSITIVE`
    (`reviewer_configs.py`) + `cert_rules_block` (`knowledge.py`), вызов через `review_one`.
  - ступень 2 — форсированный бинарный промпт из `run_cascade_stage2.py`: `review_forced`
    (см. его докстринг за деталями рамки промпта). Как и там, ступень 2 идёт с `use_cache=False`:
    k сэмплов на одном промпте при одной температуре дают идентичный ключ кеша
    (`(промпт, модель, параметры)`) и схлопнулись бы в один ответ.

Отличия от `run_cascade_stage2.py` (который не удаляется — на его артефакты ссылаются
измерения):
  - ступень 1 не прибита к готовому файлу deepseek — можно прогнать заново, на своей
    модели/бэкенде, либо передать готовые вердикты через `--stage1-verdicts`;
  - k и порог голосования — параметры (`--k`, `--k-threshold`), а не хардкод 5/3;
  - ступень 1 и ступень 2 — независимые `LLMClient` с собственными конфигами (модель, бэкенд,
    base_url, api_key_env, цены) — план координатора: ступень 1 на GigaChat-3-Ultra бесплатно,
    ступень 2 на дешёвой модели через OpenRouter;
  - usage/стоимость считаются и сохраняются РАЗДЕЛЬНО по ступеням (`*_usage.json`).

Проверка на схлопывание сэмплов (см. докстринг `run_cascade_stage2.py`) сохранена как
`_check_diversity` и попадает в артефакт `*_diversity.json`.

НИКОГДА не исполнять и не компилировать код из датасета — только статический анализ (см. CLAUDE.md).
"""

from __future__ import annotations

import argparse
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
from core.schema import Verdict
from cases.codereview.reviewer_configs import SYSTEM_PROMPT_SENSITIVE, review_one
from cases.codereview.knowledge import cert_rules_block
from cases.codereview.run_cascade_stage2 import review_forced, _short_hash

_ROOT = Path(__file__).resolve().parents[2]
_EVAL_IDS_TXT = _ROOT / "out" / "bench" / "case3_eval_ids.txt"
_DEFAULT_OUT_DIR = _ROOT / "cases" / "codereview" / "out"


def _load_env():
    env_path = _ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def _load_ids(ids_file: Path) -> list[str]:
    seen = set()
    ids = []
    for tok in ids_file.read_text(encoding="utf-8").split():
        tok = tok.strip()
        if tok and tok not in seen:
            seen.add(tok)
            ids.append(tok)
    return ids


def _build_llm(*, model: str, backend: str, base_url: str | None, api_key_env: str | None,
                temperature: float, max_concurrency: int, cache_path: str, dry_run: bool,
                price_in: float | None, price_out: float | None) -> LLMClient:
    kwargs: dict = dict(
        model=model, backend=backend, temperature=temperature, max_tokens=2048,
        max_concurrency=max_concurrency, dry_run=dry_run, cache_path=cache_path,
        price_per_1m_input=price_in, price_per_1m_output=price_out,
    )
    if base_url:
        kwargs["base_url"] = base_url
    if api_key_env:
        kwargs["api_key_env"] = api_key_env
    return LLMClient(LLMConfig(**kwargs))


def run_stage1(ids: list[str], code_by_id: dict[str, str], llm: LLMClient,
               max_concurrency: int) -> list[dict]:
    """`cert_only`: SYSTEM_PROMPT_SENSITIVE + cert_rules_block, один вызов на фрагмент."""
    ctx = PipelineContext(case="codereview", config={}, llm=llm)

    def _one(doc_id: str) -> tuple[str, Verdict]:
        code = code_by_id[doc_id]
        v = review_one(doc_id, code, ctx, system_prompt=SYSTEM_PROMPT_SENSITIVE,
                        use_json_example_sensitive=True, knowledge_block=cert_rules_block(code))
        return doc_id, v

    results: dict[str, Verdict] = {}
    with ThreadPoolExecutor(max_workers=max_concurrency) as ex:
        for doc_id, v in ex.map(_one, ids):
            results[doc_id] = v
    return [results[i].model_dump() for i in ids]


def _check_diversity(samples_by_id: dict[str, list[str]]) -> dict:
    """Доля фрагментов, где все k сэмплов ступени 2 побайтово совпали текстом ответа —
    признак того, что кеш/провайдер незаметно схлопнул сэмплирование в один ответ."""
    unique_counts = []
    collapsed = []
    for doc_id, texts in samples_by_id.items():
        n_unique = len(set(texts))
        unique_counts.append(n_unique)
        if n_unique == 1 and len(texts) > 1:
            collapsed.append(doc_id)
    avg_unique = sum(unique_counts) / len(unique_counts) if unique_counts else 0.0
    return {
        "k": len(next(iter(samples_by_id.values()), [])),
        "avg_unique_responses_per_fragment": round(avg_unique, 3),
        "fragments_with_all_k_identical": len(collapsed),
        "collapsed_doc_ids": collapsed,
    }


def run_stage2(uncertain_ids: list[str], code_by_id: dict[str, str], llm: LLMClient,
               max_concurrency: int, k: int, k_threshold: int) -> tuple[dict[str, Verdict], dict]:
    """k сэмплов temperature=0.7, use_cache=False, голосование по порогу k_threshold из k."""
    ctx = PipelineContext(case="codereview", config={}, llm=llm)
    samples_by_id: dict[str, list[tuple[Verdict, str]]] = {i: [] for i in uncertain_ids}
    jobs = [(doc_id, s) for doc_id in uncertain_ids for s in range(k)]

    def _run(job):
        doc_id, _s = job
        v, raw_text = review_forced(doc_id, code_by_id[doc_id], ctx, temperature=0.7,
                                     use_cache=False)
        return doc_id, v, raw_text

    with ThreadPoolExecutor(max_workers=max_concurrency) as ex:
        for doc_id, v, raw_text in ex.map(_run, jobs):
            samples_by_id[doc_id].append((v, raw_text))

    diversity = _check_diversity({i: [t for _, t in s] for i, s in samples_by_id.items()})

    stage2_by_id: dict[str, Verdict] = {}
    for doc_id, samples in samples_by_id.items():
        votes = sum(1 for v, _ in samples if v.verdict == "vulnerable")
        final_verdict = "vulnerable" if votes >= k_threshold else "secure"
        agreeing = [v for v, _ in samples if v.verdict == final_verdict]
        rep = agreeing[0] if agreeing else samples[0][0]
        merged_artifacts = {**rep.artifacts, f"vote_count_vulnerable_of_{k}": votes,
                             "vote_threshold_k": k_threshold, "k_samples": k}
        stage2_by_id[doc_id] = Verdict(
            doc_id=doc_id, verdict=final_verdict, confidence=round(votes / k, 2) if k else 0.0,
            action=("block" if final_verdict == "vulnerable" else "pass"),
            evidence=rep.evidence, rationale=rep.rationale, artifacts=merged_artifacts,
        )
    return stage2_by_id, diversity


def merge(stage1: list[dict], stage2_by_id: dict[str, Verdict]) -> list[dict]:
    out = []
    for d in stage1:
        doc_id = d["doc_id"]
        out.append(stage2_by_id[doc_id].model_dump() if doc_id in stage2_by_id else d)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids-file", type=Path, default=_EVAL_IDS_TXT,
                         help="файл с doc_id (по одному/через пробел), по умолчанию эталонные 150")
    parser.add_argument("--all", action="store_true",
                         help="весь корпус (18864 фрагмента) вместо --ids-file")

    parser.add_argument("--stage1-verdicts", type=Path, default=None,
                         help="готовые вердикты ступени 1 (json список Verdict) — не перегонять заново")
    parser.add_argument("--stage1-model", default="deepseek-chat")
    parser.add_argument("--stage1-backend", default="openai_compat",
                         choices=["openai_compat", "anthropic", "gigachat"])
    parser.add_argument("--stage1-base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--stage1-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--stage1-price-in", type=float, default=None)
    parser.add_argument("--stage1-price-out", type=float, default=None)

    parser.add_argument("--stage2-model", default="deepseek-chat")
    parser.add_argument("--stage2-backend", default="openai_compat",
                         choices=["openai_compat", "anthropic", "gigachat"])
    parser.add_argument("--stage2-base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--stage2-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--stage2-price-in", type=float, default=None)
    parser.add_argument("--stage2-price-out", type=float, default=None)

    parser.add_argument("--k", type=int, default=3, help="сэмплов на фрагмент ступени 2")
    parser.add_argument("--k-threshold", type=int, default=2,
                         help="голосов 'vulnerable' из k для финального verdict=vulnerable")

    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--cache-path", default="out/llm_cache_case3_cascade.sqlite3",
                         help="кеш ступени 1 (ступень 2 без кеша, см. докстринг модуля)")
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default="cascade",
                         help="префикс имён выходных файлов")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.k_threshold > args.k:
        parser.error("--k-threshold не может быть больше --k")

    _load_env()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    corpus = load_case3()
    corpus["unique_id"] = corpus["unique_id"].astype(str)
    code_by_id = dict(zip(corpus["unique_id"], corpus["code"]))

    if args.all:
        ids = list(corpus["unique_id"])
    else:
        ids = _load_ids(args.ids_file)
    missing = [i for i in ids if i not in code_by_id]
    assert not missing, f"doc_id не найдены в корпусе: {missing[:10]}"
    print(f"ids: {len(ids)}")

    # ---- ступень 1 ----
    stage1_llm = None
    if args.stage1_verdicts:
        stage1 = json.loads(args.stage1_verdicts.read_text(encoding="utf-8"))
        stage1_by_id = {d["doc_id"]: d for d in stage1}
        missing1 = [i for i in ids if i not in stage1_by_id]
        assert not missing1, f"doc_id из --ids-file не найдены в --stage1-verdicts: {missing1[:10]}"
        stage1 = [stage1_by_id[i] for i in ids]
        stage1_usage = None
        print(f"stage1: переиспользованы готовые вердикты из {args.stage1_verdicts} ({len(stage1)})")
    else:
        stage1_llm = _build_llm(
            model=args.stage1_model, backend=args.stage1_backend, base_url=args.stage1_base_url,
            api_key_env=args.stage1_api_key_env, temperature=0.0,
            max_concurrency=args.max_concurrency, cache_path=args.cache_path, dry_run=args.dry_run,
            price_in=args.stage1_price_in, price_out=args.stage1_price_out,
        )
        t0 = time.time()
        stage1 = run_stage1(ids, code_by_id, stage1_llm, args.max_concurrency)
        print(f"stage1: {len(stage1)} вердиктов за {round(time.time()-t0,1)}s "
              f"(model={args.stage1_model}, backend={args.stage1_backend})")
        stage1_usage = stage1_llm.usage.as_dict()
        stage1_llm.close()

    uncertain_ids = [d["doc_id"] for d in stage1 if d["verdict"] == "uncertain"]
    print(f"uncertain: {len(uncertain_ids)} из {len(stage1)}")

    # ---- ступень 2 ----
    stage2_llm = _build_llm(
        model=args.stage2_model, backend=args.stage2_backend, base_url=args.stage2_base_url,
        api_key_env=args.stage2_api_key_env, temperature=0.7,
        max_concurrency=args.max_concurrency, cache_path=args.cache_path, dry_run=args.dry_run,
        price_in=args.stage2_price_in, price_out=args.stage2_price_out,
    )
    t0 = time.time()
    stage2_by_id, diversity = run_stage2(
        uncertain_ids, code_by_id, stage2_llm, args.max_concurrency, args.k, args.k_threshold,
    )
    n_calls_stage2 = len(uncertain_ids) * args.k
    print(f"stage2: {n_calls_stage2} вызовов ({len(uncertain_ids)} фрагментов x k={args.k}) "
          f"за {round(time.time()-t0,1)}s (model={args.stage2_model}, backend={args.stage2_backend}, "
          f"k_threshold={args.k_threshold})")
    stage2_usage = stage2_llm.usage.as_dict()
    stage2_llm.close()

    # ---- финальные вердикты ----
    final = merge(stage1, stage2_by_id)
    verdicts_path = args.out_dir / f"{args.tag}_verdicts.json"
    verdicts_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"verdicts -> {verdicts_path} ({len(final)})")

    diversity_path = args.out_dir / f"{args.tag}_diversity.json"
    diversity_path.write_text(json.dumps(diversity, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"diversity -> {diversity_path}: {diversity}")

    usage_path = args.out_dir / f"{args.tag}_usage.json"
    usage_path.write_text(json.dumps({
        "stage1": {"model": args.stage1_model, "backend": args.stage1_backend, "usage": stage1_usage},
        "stage2": {"model": args.stage2_model, "backend": args.stage2_backend, "usage": stage2_usage,
                   "k": args.k, "k_threshold": args.k_threshold, "n_calls": n_calls_stage2},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"usage -> {usage_path}")


if __name__ == "__main__":
    main()
