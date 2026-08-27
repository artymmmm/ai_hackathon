"""Прогон состязательного «суда» (прокурор → защита → судья) на всех 150 эталонных фрагментах
кейса 3 и сборка четырёх вариантов сравнения (задание координатора):

  - court_base            — судья, k=1 сэмпл, temperature=0.0.
  - court_vote            — судья, k=5 сэмплов при temperature=0.7, голосование по порогам
                             k>=1..4 из 5. Обвинение и защита считаются один раз (temp=0.0) и
                             переиспользуются для всех 5 сэмплов судьи.
  - court_on_uncertain    — суд применяется только к 89 фрагментам корзины uncertain из
                             cert_only (out/bench/case3_deepseek-chat_cert_only.json), решения
                             ступени 1 по остальным 61 фрагменту заморожены. Не требует новых
                             вызовов LLM — 89 из этих фрагментов уже посчитаны в полном прогоне
                             court_base/court_vote на 150.
  - court_no_defense      — абляция: прокурор + судья без участия защиты (свой системный
                             промпт судьи, не видящий защиту вообще).

Свой кеш `out/llm_cache_case3_court.sqlite3` (не конфликтует с другими агентами).
Модель deepseek-chat, base_url https://api.deepseek.com/v1, ключ DEEPSEEK_API_KEY из .env.

НИКОГДА не исполнять и не компилировать код из датасета — только статический анализ (CLAUDE.md).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.data import load_case3
from core.llm import LLMClient, LLMConfig
from core.schema import Verdict
from cases.codereview import court_reviewer as court
from cases.codereview.reviewer import _fallback as _judge_fallback

_ROOT = Path(__file__).resolve().parents[2]
_BENCH_DIR = _ROOT / "out" / "bench"
_OUT_DIR = _ROOT / "cases" / "codereview" / "out"
_STAGE1_PATH = _BENCH_DIR / "case3_deepseek-chat_cert_only.json"
_EVAL_IDS_TXT = _BENCH_DIR / "case3_eval_ids.txt"
_CACHE_PATH = "out/llm_cache_case3_court.sqlite3"

_VOTE_THRESHOLDS = (1, 2, 3, 4)
_N_VOTE_SAMPLES = 5


def _load_env():
    for line in (_ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def main() -> None:
    _load_env()

    eval_ids = [x.strip() for x in _EVAL_IDS_TXT.read_text().split() if x.strip()]
    assert len(eval_ids) == 150, f"ожидали 150 eval id, получили {len(eval_ids)}"

    corpus = load_case3()
    corpus["unique_id"] = corpus["unique_id"].astype(str)
    code_by_id = dict(zip(corpus["unique_id"], corpus["code"]))
    missing = [i for i in eval_ids if i not in code_by_id]
    assert not missing, f"eval id не найдены в корпусе: {missing}"

    stage1 = json.loads(_STAGE1_PATH.read_text(encoding="utf-8"))
    stage1_by_id = {d["doc_id"]: d for d in stage1}
    uncertain_ids = {d["doc_id"] for d in stage1 if d["verdict"] == "uncertain"}
    print(f"stage1 (cert_only): {len(stage1)} вердиктов, uncertain={len(uncertain_ids)}")

    llm = LLMClient(LLMConfig(
        model="deepseek-chat", backend="openai_compat", base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY", temperature=0.0, max_tokens=2048, max_concurrency=8,
        dry_run=False, cache_path=_CACHE_PATH,
    ))

    prepared_by_id = {}
    for doc_id in eval_ids:
        code = code_by_id[doc_id]
        prepared, truncated, original_length = court.prepare(doc_id, code)
        prepared_by_id[doc_id] = (code, prepared, truncated, original_length)

    # ---- 1. Прокурор, все 150, temp=0.0, кеш обычный ----
    print(f"\n=== Прокурор ({len(eval_ids)} фрагментов) ===")
    t0 = time.time()
    prosecution: dict[str, tuple[dict, str]] = {}

    def _run_pros(doc_id):
        code, prepared, truncated, original_length = prepared_by_id[doc_id]
        try:
            parsed, raw = court.run_prosecutor(llm, doc_id, code, prepared, truncated,
                                                original_length, temperature=0.0, use_cache=True)
        except Exception as e:
            parsed = {"can_build_case": False, "cwe_id": "", "specific_input": "",
                      "program_state": "", "violation_mechanism": "", "argument": "",
                      "honesty_note": f"llm_call_failed:{e}"}
            raw = ""
        return doc_id, parsed, raw

    with ThreadPoolExecutor(max_workers=8) as ex:
        for doc_id, parsed, raw in ex.map(_run_pros, eval_ids):
            prosecution[doc_id] = (parsed, raw)
    n_no_case = sum(1 for p, _ in prosecution.values() if not p.get("can_build_case"))
    print(f"  elapsed={round(time.time()-t0,1)}s; can_build_case=false: {n_no_case}/{len(eval_ids)}")

    # ---- 2. Защита, все 150, temp=0.0, кеш обычный ----
    print(f"\n=== Защита ({len(eval_ids)} фрагментов) ===")
    t0 = time.time()
    defense: dict[str, tuple[dict, str]] = {}

    def _run_def(doc_id):
        code, prepared, truncated, original_length = prepared_by_id[doc_id]
        pros_parsed, _ = prosecution[doc_id]
        try:
            parsed, raw = court.run_defense(llm, doc_id, prepared, truncated, original_length,
                                             pros_parsed.get("argument", ""), temperature=0.0,
                                             use_cache=True)
        except Exception as e:
            parsed = {"can_rebut": False, "existing_check": "", "unstated_assumption": "",
                      "type_size_argument": "", "rebuttal": "",
                      "honesty_note": f"llm_call_failed:{e}"}
            raw = ""
        return doc_id, parsed, raw

    with ThreadPoolExecutor(max_workers=8) as ex:
        for doc_id, parsed, raw in ex.map(_run_def, eval_ids):
            defense[doc_id] = (parsed, raw)
    n_no_rebut = sum(1 for d, _ in defense.values() if not d.get("can_rebut"))
    print(f"  elapsed={round(time.time()-t0,1)}s; can_rebut=false: {n_no_rebut}/{len(eval_ids)}")

    def _to_verdict(doc_id, judge_parsed, judge_raw, *, with_defense: bool) -> Verdict:
        code, prepared, truncated, original_length = prepared_by_id[doc_id]
        pros_parsed, pros_raw = prosecution[doc_id]
        if with_defense:
            def_parsed, def_raw = defense[doc_id]
        else:
            def_parsed, def_raw = None, None
        return court.judge_to_verdict(
            doc_id, judge_parsed, full_code=code, truncated=truncated,
            original_length=original_length, prosecution=pros_parsed, defense=def_parsed,
            prosecution_raw=pros_raw, defense_raw=def_raw, judge_raw=judge_raw,
        )

    # ---- 3. Судья, court_base: k=1, temp=0.0, кеш обычный, с защитой ----
    print(f"\n=== Судья court_base (k=1, temp=0.0, с защитой) ===")
    t0 = time.time()
    court_base: dict[str, Verdict] = {}

    def _run_judge_base(doc_id):
        code, prepared, truncated, original_length = prepared_by_id[doc_id]
        pros_parsed, _ = prosecution[doc_id]
        def_parsed, _ = defense[doc_id]
        try:
            parsed, raw = court.run_judge(llm, doc_id, prepared, truncated, original_length,
                                           pros_parsed.get("argument", ""), def_parsed.get("rebuttal", ""),
                                           temperature=0.0, use_cache=True)
            v = _to_verdict(doc_id, parsed, raw, with_defense=True)
        except Exception as e:
            v = _judge_fallback(doc_id, f"llm_call_failed:{e}", code=code)
        return doc_id, v

    with ThreadPoolExecutor(max_workers=8) as ex:
        for doc_id, v in ex.map(_run_judge_base, eval_ids):
            court_base[doc_id] = v
    print(f"  elapsed={round(time.time()-t0,1)}s")

    # ---- 4. Судья, court_no_defense: k=1, temp=0.0, кеш обычный, без защиты ----
    print(f"\n=== Судья court_no_defense (k=1, temp=0.0, без защиты) ===")
    t0 = time.time()
    court_no_defense: dict[str, Verdict] = {}

    def _run_judge_nodef(doc_id):
        code, prepared, truncated, original_length = prepared_by_id[doc_id]
        pros_parsed, _ = prosecution[doc_id]
        try:
            parsed, raw = court.run_judge(llm, doc_id, prepared, truncated, original_length,
                                           pros_parsed.get("argument", ""), None,
                                           temperature=0.0, use_cache=True)
            v = _to_verdict(doc_id, parsed, raw, with_defense=False)
        except Exception as e:
            v = _judge_fallback(doc_id, f"llm_call_failed:{e}", code=code)
        return doc_id, v

    with ThreadPoolExecutor(max_workers=8) as ex:
        for doc_id, v in ex.map(_run_judge_nodef, eval_ids):
            court_no_defense[doc_id] = v
    print(f"  elapsed={round(time.time()-t0,1)}s")

    # ---- 5. Судья, court_vote: k=5 сэмплов, temp=0.7, кеш ВЫКЛЮЧЕН, с защитой ----
    print(f"\n=== Судья court_vote ({_N_VOTE_SAMPLES} сэмплов, temp=0.7, без кеша, с защитой) ===")
    t0 = time.time()
    vote_samples: dict[str, list[tuple[Verdict, str]]] = {i: [] for i in eval_ids}
    jobs = [(doc_id, s) for doc_id in eval_ids for s in range(_N_VOTE_SAMPLES)]

    def _run_judge_vote(job):
        doc_id, s = job
        code, prepared, truncated, original_length = prepared_by_id[doc_id]
        pros_parsed, _ = prosecution[doc_id]
        def_parsed, _ = defense[doc_id]
        try:
            parsed, raw = court.run_judge(llm, doc_id, prepared, truncated, original_length,
                                           pros_parsed.get("argument", ""), def_parsed.get("rebuttal", ""),
                                           temperature=0.7, use_cache=False)
            v = _to_verdict(doc_id, parsed, raw, with_defense=True)
        except Exception as e:
            v = _judge_fallback(doc_id, f"llm_call_failed:{e}", code=code)
            raw = f"[error sample {s}: {e}]"
        return doc_id, v, raw

    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for doc_id, v, raw in ex.map(_run_judge_vote, jobs):
            vote_samples[doc_id].append((v, raw))
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)}")
    print(f"  elapsed={round(time.time()-t0,1)}s")

    # ---- проверка схлопывания сэмплов (temp=0.7, кеш выключен — проверяем факт, не верим на слово) ----
    unique_counts = []
    collapsed_docs = []
    for doc_id, samples in vote_samples.items():
        texts = [t for _, t in samples]
        n_unique = len(set(texts))
        unique_counts.append(n_unique)
        if n_unique == 1:
            collapsed_docs.append(doc_id)
    avg_unique = sum(unique_counts) / len(unique_counts)
    print(f"\nПроверка схлопывания сэмплов court_vote: среднее число уникальных ответов на "
          f"фрагмент = {avg_unique:.2f} из {_N_VOTE_SAMPLES} ({len(eval_ids)} фрагментов). "
          f"Фрагментов с {_N_VOTE_SAMPLES} идентичными текстами ответа: {len(collapsed_docs)}.")
    diversity_report = {
        "avg_unique_responses_per_fragment": round(avg_unique, 3),
        "n_samples_per_fragment": _N_VOTE_SAMPLES,
        "n_fragments": len(eval_ids),
        "fragments_with_all_identical": len(collapsed_docs),
        "collapsed_doc_ids": collapsed_docs,
    }
    (_OUT_DIR / "court_vote_diversity_check.json").write_text(
        json.dumps(diversity_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # сырые сэмплы court_vote — для аудита
    vote_raw = {
        doc_id: [
            {"verdict": v.verdict, "confidence": v.confidence, "cwe_id": v.artifacts.get("cwe_id"),
             "text_sha256_short": _short_hash(raw)}
            for v, raw in samples
        ]
        for doc_id, samples in vote_samples.items()
    }
    (_OUT_DIR / "court_vote_samples.json").write_text(
        json.dumps(vote_raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- сохранить сырые тексты обвинения/защиты отдельно (иллюстрация для отчёта) ----
    raw_texts = {
        doc_id: {
            "prosecution_argument": prosecution[doc_id][0].get("argument", ""),
            "prosecution_can_build_case": prosecution[doc_id][0].get("can_build_case"),
            "prosecution_honesty_note": prosecution[doc_id][0].get("honesty_note", ""),
            "defense_rebuttal": defense[doc_id][0].get("rebuttal", ""),
            "defense_can_rebut": defense[doc_id][0].get("can_rebut"),
            "defense_honesty_note": defense[doc_id][0].get("honesty_note", ""),
        }
        for doc_id in eval_ids
    }
    (_OUT_DIR / "court_prosecution_defense_texts.json").write_text(
        json.dumps(raw_texts, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- собрать финальные списки вердиктов, в порядке eval_ids ----

    def _dump(verdicts_by_id: dict[str, Verdict], path: Path) -> None:
        out = [verdicts_by_id[doc_id].model_dump() for doc_id in eval_ids]
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  -> {path}")

    print("\n=== Сохранение вердиктов ===")
    _dump(court_base, _BENCH_DIR / "case3_court_base.json")
    _dump(court_no_defense, _BENCH_DIR / "case3_court_no_defense.json")

    vote_by_k: dict[int, dict[str, Verdict]] = {}
    for k in _VOTE_THRESHOLDS:
        by_id: dict[str, Verdict] = {}
        for doc_id, samples in vote_samples.items():
            votes = sum(1 for v, _ in samples if v.verdict == "vulnerable")
            final_verdict = "vulnerable" if votes >= k else "secure"
            agreeing = [v for v, _ in samples if v.verdict == final_verdict]
            rep = agreeing[0] if agreeing else samples[0][0]
            merged_artifacts = {**rep.artifacts, "vote_count_vulnerable_of_5": votes,
                                 "vote_threshold_k": k}
            by_id[doc_id] = Verdict(
                doc_id=doc_id, verdict=final_verdict, confidence=round(votes / _N_VOTE_SAMPLES, 2),
                action=("block" if final_verdict == "vulnerable" else "pass"),
                evidence=rep.evidence, rationale=rep.rationale, artifacts=merged_artifacts,
            )
        vote_by_k[k] = by_id
        _dump(by_id, _BENCH_DIR / f"case3_court_vote_k{k}of5.json")

    # court_on_uncertain: заморозить stage1 везде, кроме 89 uncertain — там результат суда.
    def _on_uncertain(court_by_id: dict[str, Verdict]) -> list[dict]:
        out = []
        for doc_id in eval_ids:
            if doc_id in uncertain_ids:
                out.append(court_by_id[doc_id].model_dump())
            else:
                out.append(stage1_by_id[doc_id])
        return out

    path = _BENCH_DIR / "case3_court_on_uncertain_base.json"
    path.write_text(json.dumps(_on_uncertain(court_base), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {path}")
    for k in _VOTE_THRESHOLDS:
        path = _BENCH_DIR / f"case3_court_on_uncertain_vote_k{k}of5.json"
        path.write_text(json.dumps(_on_uncertain(vote_by_k[k]), ensure_ascii=False, indent=2),
                         encoding="utf-8")
        print(f"  -> {path}")

    llm.close()
    usage = llm.usage.as_dict()
    print(f"\nusage={usage}")
    (_OUT_DIR / "court_usage.json").write_text(json.dumps(usage, ensure_ascii=False, indent=2),
                                                encoding="utf-8")


if __name__ == "__main__":
    main()
