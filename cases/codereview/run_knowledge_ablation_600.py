"""Аблация блока знаний cert_only на 600 фрагментах (координатор, ablation cert_only).

cert_only (SYSTEM_PROMPT_SENSITIVE + cert_rules_block) даёт F1 0.386 против 0.265 у базового
промпта на n=600. Но cert_rules_block на этих 600 фрагментах вырождается в почти константный
блок (8 различных значений, один покрывает 96.3% — измерено координатором). Вопрос: откуда
берётся прирост +0.121 F1 — из содержания блока или из самого факта его присутствия ("рамка")?

Четыре варианта, различается ТОЛЬКО knowledge_block (SYSTEM_PROMPT_SENSITIVE и
use_json_example_sensitive=True — общие для всех, как в run_knowledge_variants_full.py):

  - sensitive_none  — knowledge_block="" (контроль: даёт ли CERT-блок вообще что-то).
  - cert_hardcoded  — одна и та же строка из 5 реальных CERT-правил для всех 600 фрагментов
                       (контроль воспроизводимости cert_only без per-фрагментного подбора).
  - decoy_style      — блок той же формы/длины, но из 5 правил оформления/стиля, не относящихся
                       к уязвимостям памяти.
  - nudge_only       — одна фраза-рамка без единого правила.
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

_ROOT = Path(__file__).resolve().parents[2]
_EVAL_IDS_TXT = _ROOT / "out" / "bench" / "case3_eval600_ids.txt"
_BENCH_DIR = _ROOT / "out" / "bench"


def _load_env():
    for line in (_ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_CERT_HARDCODED = """\
Применимые правила безопасного кодирования (SEI CERT C, привязка к CWE):
- ENV01-C (Do not make assumptions about the size of an environment variable) — связан с CWE-119
- MEM10-C (Define and use a pointer validation function) — связан с CWE-20
- ARR00-C (Understand how arrays work) — связан с CWE-119, CWE-129
- ERR07-C (Prefer functions that support error checking over equivalent functions that don't) — связан с CWE-20, CWE-676
- FIO30-C (Exclude user input from format strings) — связан с CWE-134, CWE-20"""

_DECOY_STYLE = """\
Применимые правила безопасного кодирования (SEI CERT C, привязка к CWE):
- DCL02-C (Use visually distinct identifiers) — связан с оформлением имён переменных
- MSC01-C (Strive for logical completeness in comments) — связан с оформлением комментариев
- MSC04-C (Use comments consistently and in a readable style) — связан с длиной и стилем строк
- PRE08-C (Guarantee that header file names are unique) — связан с порядком include-директив
- DCL00-C (Const-qualify immutable objects) — связан с единым стилем отступов в блоке"""

_NUDGE_ONLY = "Перед вердиктом сверься с типовыми классами дефектов безопасного кодирования."

VARIANTS = {
    "sensitive_none": lambda code: "",
    "cert_hardcoded": lambda code: _CERT_HARDCODED,
    "decoy_style": lambda code: _DECOY_STYLE,
    "nudge_only": lambda code: _NUDGE_ONLY,
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
        api_key_env="DEEPSEEK_API_KEY", temperature=0.0, max_tokens=2048, max_concurrency=64,
        dry_run=False, cache_path="out/llm_cache_case3_ablation.sqlite3",
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
                if done % 100 == 0:
                    print(f"  {done}/{len(sub)}")
        verdicts = [results[i] for i in range(len(sub))]
        out_path = _BENCH_DIR / f"case3_ablation_{name}_600.json"
        out_path.write_text(json.dumps([v.model_dump() for v in verdicts],
                                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"elapsed={round(time.time()-t0,1)}s -> {out_path}")

    usage = llm.usage.as_dict()
    llm.close()
    print(f"usage={usage}")
    usage_path = _BENCH_DIR / "case3_ablation_usage.json"
    prior = {}
    if usage_path.exists():
        prior = json.loads(usage_path.read_text(encoding="utf-8"))
    prior.update(usage)
    usage_path.write_text(json.dumps(prior, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
