"""Потолок пользы от подбора знаний: `cert_rules_block`, построенный по ЗОЛОТОМУ CWE, а не по
сигнатурному триажу (`triage.score_fragment`). Заведомо читерская конфигурация — в продакшн не
идёт, её единственная цель — верхняя граница для вывода "содержание блока знаний безразлично"
(см. docstring `knowledge.py` и результаты `cert_only` на eval600, F1=0.386).

Дыра в прежнем выводе: триаж молчит у 96.3% корпуса и почти все фрагменты получают один и тот же
фолбэк `_CONFUSABLE_CLUSTER` = CWE 119/787/125/120/20 — прежний эксперимент сравнивал
«нерелевантный блок А» с «нерелевантным блоком Б». Здесь вместо триажного подбора в
`cert_rules_block` (knowledge.py) подставляется ЗОЛОТОЙ CWE эталона (для vulnerable) или CWE,
случайно выбранный из эмпирического распределения золотых CWE (для secure — иначе сам факт
"похож ли блок на настоящий" выдаёт метку).

Подвыборка — 516 фрагментов из eval600 (`out/bench/case3_eval600_ids.txt`):
  - 116 vulnerable, у которых golden cwe_id известен в `research/case3_recovered_labels_v4.csv`
    (84 vulnerable без golden CWE исключены — оракул для них не определён);
  - все 400 secure.

Форма блока — один в один как `cert_rules_block` (тот же MAX_CERT_RULES=5, тот же фолбэк на
["ARR30-C","EXP34-C","MEM30-C","STR31-C"], та же сортировка по специфичности), единственное
отличие — источник кандидатного CWE: не `candidate_cwe_ids(code)` (триаж + flawfinder), а ровно
один переданный извне CWE (золотой/случайный). Карточки CWE (`cwe_cards_block`) НЕ используются —
production-конфигурация `cert_only` (см. `run_cascade.run_stage1`, `build_cert_only_600.py`) их
тоже не использует, сравнение идёт с тем же самым семейством блока.

Модель — DeepSeek напрямую (НЕ OpenRouter): model="deepseek-chat",
base_url="https://api.deepseek.com/v1", api_key_env="DEEPSEEK_API_KEY". Кеш и выходные файлы —
только `oracle_`-префикс / отдельный sqlite, чтобы не задевать параллельный прогон gpt5-mini на
том же наборе (см. запрет в промпте задачи).

НИКОГДА не исполнять и не компилировать код из датасета — только статический анализ (см. CLAUDE.md).
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.data import load_case3
from core.llm import LLMClient, LLMConfig
from core.pipeline import PipelineContext
from cases.codereview.reviewer_configs import SYSTEM_PROMPT_SENSITIVE, review_one
from cases.codereview.knowledge import _CERT_RULES, MAX_CERT_RULES  # noqa: F401 (реюз той же базы правил)

_ROOT = Path(__file__).resolve().parents[2]
_EVAL600_IDS_TXT = _ROOT / "out" / "bench" / "case3_eval600_ids.txt"
_GOLD_CSV = _ROOT / "research" / "case3_recovered_labels_v4.csv"
_BENCH_DIR = _ROOT / "out" / "bench"
_OUT_DIR = _ROOT / "cases" / "codereview" / "out"
_CACHE_PATH = "out/llm_cache_case3_oracle.sqlite3"
_OUT_VERDICTS = _BENCH_DIR / "case3_oracle_cwe_516.json"
_OUT_USAGE = _BENCH_DIR / "case3_oracle_cwe_516_usage.json"
_OUT_ASSIGNMENT = _OUT_DIR / "oracle_cwe_assignment_516.json"

_FALLBACK_RULE_IDS = ["ARR30-C", "EXP34-C", "MEM30-C", "STR31-C"]


def _load_env():
    for line in (_ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def oracle_cert_rules_block(cwe_id: str, max_rules: int = MAX_CERT_RULES) -> tuple[str, bool]:
    """Форма идентична `knowledge.cert_rules_block`, но кандидатный CWE — ровно один, переданный
    аргументом (золотой/случайный), а не список от `candidate_cwe_ids` (триаж + flawfinder).
    Возвращает (блок, used_fallback) — used_fallback=True, если для cwe_id не нашлось ни одного
    правила и подставился общий фолбэк-набор (как и в продакшн-версии).
    """
    cwe_set = {cwe_id}
    matched = [
        (rule_id, r) for rule_id, r in _CERT_RULES.items()
        if cwe_set & set(r.get("cwes", []))
    ]
    used_fallback = False
    if not matched:
        used_fallback = True
        matched = [(rid, _CERT_RULES[rid]) for rid in _FALLBACK_RULE_IDS if rid in _CERT_RULES]
    matched.sort(key=lambda pair: len(pair[1].get("cwes", [])))
    matched = matched[:max_rules]
    if not matched:
        return "", used_fallback
    lines = ["Применимые правила безопасного кодирования (SEI CERT C, привязка к CWE):"]
    for rid, r in matched:
        lines.append(f"- {rid} ({r['name']}) — связан с {', '.join(r.get('cwes', []))}")
    return "\n".join(lines), used_fallback


def build_subset() -> tuple[list[str], dict[str, str], dict[str, bool]]:
    """Возвращает (subset_ids в порядке eval600, doc_id -> назначенный oracle CWE,
    doc_id -> is_vulnerable_gold)."""
    ids = [x.strip() for x in _EVAL600_IDS_TXT.read_text().split() if x.strip()]
    seen = set()
    ids = [i for i in ids if not (i in seen or seen.add(i))]
    assert len(ids) == 600, f"ожидали 600, получили {len(ids)}"

    gold = pd.read_csv(_GOLD_CSV)
    gold["unique_id"] = gold["unique_id"].astype(str)
    gold_by_id = gold.set_index("unique_id").to_dict("index")

    vuln_with_cwe = [
        i for i in ids
        if gold_by_id.get(i, {}).get("recovered_label") == 1
        and pd.notna(gold_by_id.get(i, {}).get("cwe_id"))
    ]
    safe = [i for i in ids if gold_by_id.get(i, {}).get("recovered_label") == 0]
    assert len(vuln_with_cwe) == 116, f"ожидали 116 vulnerable с golden CWE, получили {len(vuln_with_cwe)}"
    assert len(safe) == 400, f"ожидали 400 secure, получили {len(safe)}"

    golden_pool = [gold_by_id[i]["cwe_id"] for i in vuln_with_cwe]  # 116 значений с частотами

    random.seed(42)
    oracle_cwe: dict[str, str] = {}
    is_vuln_gold: dict[str, bool] = {}
    for i in vuln_with_cwe:
        oracle_cwe[i] = gold_by_id[i]["cwe_id"]
        is_vuln_gold[i] = True
    for i in safe:
        oracle_cwe[i] = random.choice(golden_pool)
        is_vuln_gold[i] = False

    subset_set = set(vuln_with_cwe) | set(safe)
    subset_ids = [i for i in ids if i in subset_set]
    assert len(subset_ids) == 516, f"ожидали 516, получили {len(subset_ids)}"
    return subset_ids, oracle_cwe, is_vuln_gold


def main():
    _load_env()
    subset_ids, oracle_cwe, is_vuln_gold = build_subset()
    print(f"subset: {len(subset_ids)} (vulnerable-с-golden-CWE: {sum(is_vuln_gold.values())}, "
          f"secure: {sum(not v for v in is_vuln_gold.values())})")

    corpus = load_case3()
    corpus["unique_id"] = corpus["unique_id"].astype(str)
    code_by_id = dict(zip(corpus["unique_id"], corpus["code"]))
    missing = [i for i in subset_ids if i not in code_by_id]
    assert not missing, f"doc_id не найдены в корпусе: {missing[:10]}"

    llm = LLMClient(LLMConfig(
        model="deepseek-chat", backend="openai_compat",
        base_url="https://api.deepseek.com/v1", api_key_env="DEEPSEEK_API_KEY",
        temperature=0.0, max_tokens=2048, max_concurrency=32,
        dry_run=False, cache_path=_CACHE_PATH,
    ))
    ctx = PipelineContext(case="codereview", config={}, llm=llm)

    used_fallback_by_id: dict[str, bool] = {}

    def _one(doc_id: str):
        code = code_by_id[doc_id]
        block, used_fallback = oracle_cert_rules_block(oracle_cwe[doc_id])
        used_fallback_by_id[doc_id] = used_fallback
        v = review_one(doc_id, code, ctx, system_prompt=SYSTEM_PROMPT_SENSITIVE,
                        use_json_example_sensitive=True, knowledge_block=block)
        return doc_id, v

    print("\n=== oracle_cwe_516 (SYSTEM_PROMPT_SENSITIVE + cert_rules_block(golden/random CWE), "
          "k=1, temp=0.0) ===")
    t0 = time.time()
    results: dict[str, dict] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=llm.config.max_concurrency) as ex:
        for doc_id, v in ex.map(_one, subset_ids):
            results[doc_id] = v.model_dump()
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(subset_ids)}")

    elapsed = round(time.time() - t0, 1)
    print(f"elapsed={elapsed}s")

    verdicts_out = [results[i] for i in subset_ids]
    _OUT_VERDICTS.write_text(json.dumps(verdicts_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"verdicts -> {_OUT_VERDICTS}")

    n_fallback_vuln = sum(1 for i in subset_ids if is_vuln_gold[i] and used_fallback_by_id[i])
    n_fallback_total = sum(1 for i in subset_ids if used_fallback_by_id[i])
    assignment_out = {
        i: {
            "is_vulnerable_gold": is_vuln_gold[i],
            "oracle_cwe": oracle_cwe[i],
            "used_fallback_rule": used_fallback_by_id[i],
        }
        for i in subset_ids
    }
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _OUT_ASSIGNMENT.write_text(json.dumps(assignment_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"assignment -> {_OUT_ASSIGNMENT}")
    print(f"golden CWE без правила в базе (фолбэк сработал): {n_fallback_vuln} из 116 vulnerable, "
          f"{n_fallback_total} из {len(subset_ids)} всего")

    usage = llm.usage.as_dict()
    _OUT_USAGE.write_text(json.dumps({
        "model": "deepseek-chat", "backend": "openai_compat", "n_fragments": len(subset_ids),
        "elapsed_seconds": elapsed, "usage": usage,
        "n_fallback_rule_vulnerable": n_fallback_vuln, "n_fallback_rule_total": n_fallback_total,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"usage -> {_OUT_USAGE}: {usage}")

    llm.close()


if __name__ == "__main__":
    main()
