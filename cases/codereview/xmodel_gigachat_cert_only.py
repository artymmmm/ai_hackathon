"""cert_only на GigaChat-2-Max, полные 150 eval id (см. STATE.md, задача агента xmodel).

Зеркало `run_knowledge_variants_full.py` (не трогать — чужая зона), но:
  - backend=gigachat, свой кеш (`out/llm_cache_gigachat.sqlite3`, чтобы не делить sqlite
    с другими агентами, пишущими в `out/llm_cache.sqlite3`);
  - только один вариант, cert_only (SYSTEM_PROMPT_SENSITIVE + cert_rules_block);
  - max_concurrency по умолчанию 1 (GigaChat, по опыту прошлого прогона, не любит параллелизм).

Запуск:
    set -a && . ./.env && set +a && \
    .venv/bin/python cases/codereview/xmodel_gigachat_cert_only.py [--max-concurrency N]
"""
from __future__ import annotations
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.data import load_case3
from core.llm import LLMClient, LLMConfig
from core.pipeline import PipelineContext
from cases.codereview.reviewer_configs import SYSTEM_PROMPT_SENSITIVE, review_one
from cases.codereview.knowledge import cert_rules_block

_ROOT = Path(__file__).resolve().parents[2]
_EVAL_IDS_TXT = _ROOT / "out" / "bench" / "case3_eval_ids.txt"
_OUT_PATH = _ROOT / "out" / "bench" / "case3_gc-max_cert_only.json"


def _load_env():
    for line in (_ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-concurrency", type=int, default=1)
    args = ap.parse_args()

    _load_env()
    eval_ids = {x.strip() for x in _EVAL_IDS_TXT.read_text().split() if x.strip()}
    corpus = load_case3()
    corpus["unique_id"] = corpus["unique_id"].astype(str)
    sub = corpus[corpus["unique_id"].isin(eval_ids)].reset_index(drop=True)
    print(f"eval ids: {len(eval_ids)}, найдено в корпусе: {len(sub)}, max_concurrency={args.max_concurrency}")

    llm = LLMClient(LLMConfig(
        model="GigaChat-2-Max", backend="gigachat",
        api_key_env="GIGACHAT_AUTH_KEY", temperature=0.0, max_tokens=2048,
        max_concurrency=args.max_concurrency,
        dry_run=False, cache_path="out/llm_cache_gigachat.sqlite3",
    ))
    ctx = PipelineContext(case="codereview", config={}, llm=llm)

    t0 = time.time()

    def _one(i_row):
        i, row = i_row
        v = review_one(str(row["unique_id"]), row["code"], ctx,
                       system_prompt=SYSTEM_PROMPT_SENSITIVE,
                       use_json_example_sensitive=True,
                       knowledge_block=cert_rules_block(row["code"]))
        return i, v

    results: dict[int, object] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=llm.config.max_concurrency) as ex:
        for i, v in ex.map(_one, sub.iterrows()):
            results[i] = v
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(sub)}  elapsed={round(time.time()-t0,1)}s")
    verdicts = [results[i] for i in range(len(sub))]
    _OUT_PATH.write_text(json.dumps([v.model_dump() for v in verdicts],
                                     ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"elapsed={round(time.time()-t0,1)}s -> {_OUT_PATH}")
    llm.close()
    print(f"usage={llm.usage.as_dict()}")


if __name__ == "__main__":
    main()
