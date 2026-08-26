"""Отдельный эксперимент: CWE-классификация НАПРЯМУЮ (не детекция уязвимости), по запросу
координатора — прежняя CWE accuracy считалась только на пойманных true positive (1-6 пар,
статистически бессмысленно) и смешивала два разных вопроса (детекция vs типизация).

Здесь: все фрагменты eval-150 с gold_label=vulnerable И известным gold cwe_id (41 из 50,
~82% покрытие — лучше, чем совместная метрика). Модели прямо сообщается, что фрагмент содержит
уязвимость (вопрос "есть ли уязвимость" не задаётся), задача — определить ТИП по CWE.
Три варианта промпта:
  no_cards   — только код, без справочника.
  with_cards — + CWE-карточки (candidate_cwe_ids по триажу, kb/cwe_cards.json), включая
               смежный кластер 119/787/125/120/20 для различения.
  with_cert  — + CWE-карточки + правила CERT C (cert_c_rules.json).

Кандидатные карточки подбираются той же эвристикой, что в knowledge.py (по категориям
сигнатурного триажа кода, НЕ по gold CWE) — не даёт модели прямой подсказки ответа, только
контекст. Если сред кандидатов не оказалось верного CWE — карточка с верным ответом просто не
показывается, это не гарантированная утечка.

Запуск:
    set -a && . ./.env && set +a && .venv/bin/python cases/codereview/cwe_experiment.py
"""
from __future__ import annotations
import json, sys, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from core.llm import LLMClient, LLMConfig
from core.pipeline import PipelineContext
from cases.codereview.knowledge import cwe_cards_block, cert_rules_block
from cases.codereview.cwe_map import normalize_cwe

_ROOT = Path(__file__).resolve().parents[2]
_EVAL_IDS_TXT = _ROOT / "out" / "bench" / "case3_eval_ids.txt"
_GOLD_CSV = _ROOT / "research" / "case3_recovered_labels.csv"
_OUT_DIR = Path(__file__).resolve().parent / "out"
_MAX_CODE_CHARS = 12000

SYSTEM_PROMPT = """\
Ты — эксперт по классификации уязвимостей безопасности C/C++ кода по MITRE CWE. Тебе НЕ нужно
решать, есть ли в коде уязвимость — тебе уже сказано, что фрагмент содержит подтверждённую
эксплуатируемую уязвимость. Твоя ЕДИНСТВЕННАЯ задача — определить наиболее вероятный конкретный
номер CWE этой уязвимости, глядя на код статически (не компилировать, не исполнять).

Отвечай ТОЛЬКО JSON:
{"cwe_id": "CWE-<номер>", "reasoning": "<1 предложение, что в коде указывает на этот CWE>"}
"""

USER_TEMPLATE = """\
Этот фрагмент C/C++ кода СОДЕРЖИТ подтверждённую уязвимость безопасности. Определи её тип
по CWE.
{knowledge_block}
<code_fragment>
{code}
</code_fragment>

Ответь JSON согласно system-инструкции.
"""

_EXAMPLE = {"cwe_id": "CWE-119", "reasoning": ""}


def _load_env() -> None:
    f = _ROOT / ".env"
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def select_fragments() -> pd.DataFrame:
    from core.data import load_case3
    corpus = load_case3()
    corpus["unique_id"] = corpus["unique_id"].astype(str)
    gold = pd.read_csv(_GOLD_CSV)
    gold["unique_id"] = gold["unique_id"].astype(str)
    eval_ids = {x.strip() for x in _EVAL_IDS_TXT.read_text().split() if x.strip()}
    gold = gold[gold["unique_id"].isin(eval_ids)]
    gold = gold[gold["recovered_label"].isin([1, "1"])]
    gold = gold[gold["cwe_id"].notna()]
    df = corpus.merge(gold[["unique_id", "cwe_id"]], on="unique_id", how="inner")
    return df.reset_index(drop=True)


def _wrap(text: str, label: str) -> str:
    return f"\n{label}:\n{text}\n" if text else ""


def run_variant(df: pd.DataFrame, ctx: PipelineContext, *, use_cards: bool, use_cert: bool) -> dict:
    correct = 0
    rows = []
    for _, row in df.iterrows():
        code = row["code"][:_MAX_CODE_CHARS]
        kb = ""
        if use_cards:
            kb += _wrap(cwe_cards_block(row["code"]), "Справочник CWE (кандидаты по триажу)")
        if use_cert:
            kb += _wrap(cert_rules_block(row["code"]), "Правила CERT C")
        prompt = USER_TEMPLATE.format(knowledge_block=kb, code=code)
        try:
            parsed = ctx.llm.complete_json(prompt, example=_EXAMPLE, system=SYSTEM_PROMPT)
            pred = normalize_cwe(parsed.get("cwe_id"))
        except Exception as e:
            pred = None
        gold_cwe = row["cwe_id"]
        is_correct = pred == gold_cwe
        correct += int(is_correct)
        rows.append({"unique_id": row["unique_id"], "gold_cwe": gold_cwe, "pred_cwe": pred,
                      "correct": is_correct})
    n = len(df)
    return {"n": n, "correct": correct, "accuracy": round(correct / n, 3) if n else None, "rows": rows}


def main() -> None:
    _load_env()
    t0 = time.time()
    df = select_fragments()
    print(f"Фрагментов для CWE-эксперимента (vulnerable + известный gold cwe_id): {len(df)}")

    llm_config = LLMConfig(
        model="deepseek-chat", backend="openai_compat", base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY", temperature=0.0, max_tokens=512, max_concurrency=8,
        dry_run=False, cache_path="out/llm_cache.sqlite3",
    )
    llm = LLMClient(llm_config)
    ctx = PipelineContext(case="codereview", config={}, llm=llm)

    results = {}
    print("\n=== no_cards ===")
    results["no_cards"] = run_variant(df, ctx, use_cards=False, use_cert=False)
    print(f"accuracy={results['no_cards']['accuracy']} ({results['no_cards']['correct']}/{results['no_cards']['n']})")

    print("\n=== with_cards ===")
    results["with_cards"] = run_variant(df, ctx, use_cards=True, use_cert=False)
    print(f"accuracy={results['with_cards']['accuracy']} ({results['with_cards']['correct']}/{results['with_cards']['n']})")

    print("\n=== with_cards_and_cert ===")
    results["with_cards_and_cert"] = run_variant(df, ctx, use_cards=True, use_cert=True)
    print(f"accuracy={results['with_cards_and_cert']['accuracy']} ({results['with_cards_and_cert']['correct']}/{results['with_cards_and_cert']['n']})")

    llm.close()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / "cwe_experiment_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nelapsed={round(time.time()-t0,1)}s usage={llm.usage.as_dict()}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
