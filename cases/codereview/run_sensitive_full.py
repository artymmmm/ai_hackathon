"""Финальный замер лучшей ПЕРЕНОСИМОЙ (без корпуса) LLM-конфигурации — sensitive-промпт —
на полных 150 eval id (improvements.md шаг 1/2, `configs_comparison.csv`).

`sensitive` выбран как кандидат вместо config_A/config_B, потому что на выборке n=40
(`config_experiment_results.json`) она дала лучший прирост recall на вызов знаний БЕЗ корпуса
(config_A с CWE/CERT картами не улучшила recall относительно sensitive, config_B зависит от
retrieval-пула — не переносится, шаг 0). Использует кеш (`out/llm_cache.sqlite3`) — 40 из 150
уже посчитаны в `run_config_experiment.py`, досчитываются оставшиеся 110.

Пишет `out/bench/case3_deepseek-chat_sensitive.json` в формате `Verdict` (совместим с
`evaluate.py` и `ensemble.py`).

Запуск:
    set -a && . ./.env && set +a && .venv/bin/python cases/codereview/run_sensitive_full.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd  # noqa: E402

from core.data import load_case3  # noqa: E402
from core.llm import LLMClient, LLMConfig  # noqa: E402
from core.pipeline import PipelineContext  # noqa: E402

from cases.codereview.reviewer_configs import SYSTEM_PROMPT_SENSITIVE, review_one  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_EVAL_IDS_TXT = _ROOT / "out" / "bench" / "case3_eval_ids.txt"
_OUT_PATH = _ROOT / "out" / "bench" / "case3_deepseek-chat_sensitive.json"


def _load_env() -> None:
    f = _ROOT / ".env"
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        import os
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def main() -> None:
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
        v = review_one(doc_id, code, ctx, system_prompt=SYSTEM_PROMPT_SENSITIVE,
                        use_json_example_sensitive=True)
        verdicts.append(v)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(sub)}")

    llm.close()
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUT_PATH.write_text(
        json.dumps([v.model_dump() for v in verdicts], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"elapsed={round(time.time() - t0, 1)}s usage={llm.usage.as_dict()}")
    print(f"-> {_OUT_PATH}")


if __name__ == "__main__":
    main()
