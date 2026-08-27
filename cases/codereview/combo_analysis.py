"""Цель 1 (см. промпт координатора): офлайн-комбинаторика скринер + cert_only, БЕЗ единого
вызова LLM для основной части — только пересборка уже посчитанных вердиктов
(`out/bench/case3_deepseek-chat_sensitive.json`, `out/bench/case3_deepseek-chat_cert_only.json`)
и уже посчитанной ступени 2 (`out/bench/case3_deepseek-chat_cascade_B_k{1..4}of5.json`).

Строит:
- `case3_combo_union.json`   — verdict=vulnerable, если vulnerable у ЛЮБОГО из двух; иначе
  uncertain, если uncertain у любого; иначе secure. (recall-максимизирующая комбинация)
- `case3_combo_intersection.json` — verdict=vulnerable только если ОБА vulnerable; secure только
  если ОБА secure; иначе (разногласие) — синтетический uncertain.
- `case3_combo_cascade_union_k{1..4}of5.json` — база = combo_union; корзина эскалации = union
  «сырых» uncertain-корзин скринера (84) и cert_only (89) = 99 id; для id внутри cert_only-корзины
  (89) переиспользуются уже посчитанные `case3_deepseek-chat_cascade_B_k{k}of5.json`; для новых
  10 id, которых в ступени 2 ещё не было, используются вердикты из отдельного файла
  `cases/codereview/out/stage2_extended_samples.json`, посчитанного `run_extended_stage2.py`
  (тот же форсированный бинарный промпт, k=5 сэмплов, голосование).

Плюс числовая сводка (печатается и пишется в `cases/codereview/out/combo_stats.json`):
объединение/пересечение позитивов, «vulnerable у одного + uncertain у другого», сколько истинных
уязвимых ловит хотя бы один детектор и сколько — ни один.

НИКОГДА не исполнять и не компилировать код из датасета — только статический анализ (CLAUDE.md).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cases.codereview.evaluate import load_gold  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_BENCH = _ROOT / "out" / "bench"
_OUT = _ROOT / "cases" / "codereview" / "out"

_SENSITIVE_PATH = _BENCH / "case3_deepseek-chat_sensitive.json"
_CERT_PATH = _BENCH / "case3_deepseek-chat_cert_only.json"
_EXT_SAMPLES_PATH = _OUT / "stage2_extended_samples.json"


def _load(path: Path) -> dict[str, dict]:
    return {d["doc_id"]: d for d in json.loads(path.read_text(encoding="utf-8"))}


def pick_union(s: dict, c: dict) -> dict:
    """vulnerable у любого -> vulnerable (приоритет объекту cert_only, он точнее); иначе
    uncertain у любого -> uncertain (приоритет cert_only); иначе secure (объект cert_only)."""
    if c["verdict"] == "vulnerable":
        return c
    if s["verdict"] == "vulnerable":
        return s
    if c["verdict"] == "uncertain":
        return c
    if s["verdict"] == "uncertain":
        return s
    return c


def pick_intersection(s: dict, c: dict) -> dict:
    """vulnerable только если оба vulnerable; secure только если оба secure; иначе синтетический
    uncertain (разногласие -> человеку)."""
    if s["verdict"] == "vulnerable" and c["verdict"] == "vulnerable":
        return c
    if s["verdict"] == "secure" and c["verdict"] == "secure":
        return c
    base = c if c["verdict"] == "vulnerable" else (s if s["verdict"] == "vulnerable" else c)
    return {
        **base,
        "verdict": "uncertain",
        "action": "manual_review",
        "confidence": round((s["confidence"] + c["confidence"]) / 2, 2),
        "rationale": (
            f"screener/cert_only не согласны: screener={s['verdict']}, cert_only={c['verdict']}"
        ),
        "artifacts": {**base.get("artifacts", {}), "source": "combo_intersection_disagreement",
                      "screener_verdict": s["verdict"], "cert_only_verdict": c["verdict"]},
    }


def main() -> None:
    sensitive = _load(_SENSITIVE_PATH)
    cert = _load(_CERT_PATH)
    ids = sorted(sensitive)
    assert ids == sorted(cert), "doc_id наборы скринера и cert_only не совпадают"
    assert len(ids) == 150

    gold = load_gold()
    gold_ids = {i: gold[i]["label"] for i in ids if i in gold and gold[i]["label"] is not None}
    assert len(gold_ids) == 150, f"ожидали 150 однозначно размеченных id, получили {len(gold_ids)}"

    # ---- статистика по позициям ----
    s_v = {i: sensitive[i]["verdict"] for i in ids}
    c_v = {i: cert[i]["verdict"] for i in ids}

    union_positives = {i for i in ids if s_v[i] == "vulnerable" or c_v[i] == "vulnerable"}
    intersection_positives = {i for i in ids if s_v[i] == "vulnerable" and c_v[i] == "vulnerable"}
    vuln_one_uncertain_other = {
        i for i in ids
        if (s_v[i] == "vulnerable" and c_v[i] == "uncertain")
        or (c_v[i] == "vulnerable" and s_v[i] == "uncertain")
    }
    true_vulnerable = {i for i in ids if gold_ids[i] == "vulnerable"}
    caught_by_at_least_one = {i for i in true_vulnerable if i in union_positives}
    caught_by_none = true_vulnerable - caught_by_at_least_one

    s_unc = {i for i in ids if s_v[i] == "uncertain"}
    c_unc = {i for i in ids if c_v[i] == "uncertain"}
    union_uncertain_raw = s_unc | c_unc

    stats = {
        "n_total": len(ids),
        "n_true_vulnerable": len(true_vulnerable),
        "n_true_secure": len(ids) - len(true_vulnerable),
        "union_positives_count": len(union_positives),
        "intersection_positives_count": len(intersection_positives),
        "vuln_at_one_uncertain_at_other_count": len(vuln_one_uncertain_other),
        "caught_by_at_least_one_of_true_vulnerable": len(caught_by_at_least_one),
        "caught_by_none_of_true_vulnerable": len(caught_by_none),
        "caught_by_none_ids": sorted(caught_by_none),
        "screener_uncertain_count": len(s_unc),
        "cert_only_uncertain_count": len(c_unc),
        "union_uncertain_raw_count": len(union_uncertain_raw),
        "intersection_uncertain_count": len(s_unc & c_unc),
        "new_ids_for_stage2_not_in_cert_only_89": sorted(union_uncertain_raw - c_unc),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    (_OUT / "combo_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                            encoding="utf-8")

    # ---- combo_union / combo_intersection файлы вердиктов ----
    combo_union = [pick_union(sensitive[i], cert[i]) for i in ids]
    combo_intersection = [pick_intersection(sensitive[i], cert[i]) for i in ids]

    (_BENCH / "case3_combo_union.json").write_text(
        json.dumps(combo_union, ensure_ascii=False, indent=2), encoding="utf-8")
    (_BENCH / "case3_combo_intersection.json").write_text(
        json.dumps(combo_intersection, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ncombo_union -> {_BENCH / 'case3_combo_union.json'}")
    print(f"combo_intersection -> {_BENCH / 'case3_combo_intersection.json'}")

    # ---- combo_cascade_union_k{1..4}of5: база=combo_union, эскалация=union_uncertain_raw(99) ----
    combo_union_by_id = {d["doc_id"]: d for d in combo_union}

    if not _EXT_SAMPLES_PATH.exists():
        print(f"\n{_EXT_SAMPLES_PATH} ещё не существует — запусти run_extended_stage2.py, "
              "прежде чем строить combo_cascade_union_k*of5. Пропускаю эту часть.")
        return

    ext_samples = json.loads(_EXT_SAMPLES_PATH.read_text(encoding="utf-8"))
    new_ids = stats["new_ids_for_stage2_not_in_cert_only_89"]
    missing_ext = [i for i in new_ids if i not in ext_samples]
    assert not missing_ext, f"нет свежих сэмплов ступени 2 для id: {missing_ext}"

    for k in (1, 2, 3, 4):
        cascade_b_k = _load(_BENCH / f"case3_deepseek-chat_cascade_B_k{k}of5.json")
        out_rows = []
        for i in ids:
            if i in union_uncertain_raw:
                if i in c_unc:
                    out_rows.append(cascade_b_k[i])
                else:
                    votes = sum(1 for s in ext_samples[i] if s["verdict"] == "vulnerable")
                    final_verdict = "vulnerable" if votes >= k else "secure"
                    rep = combo_union_by_id[i]
                    out_rows.append({
                        **rep,
                        "verdict": final_verdict,
                        "confidence": round(votes / 5, 2),
                        "action": "block" if final_verdict == "vulnerable" else "pass",
                        "artifacts": {**rep.get("artifacts", {}), "source": "combo_cascade_stage2_new",
                                      "vote_count_vulnerable_of_5": votes, "vote_threshold_k": k},
                    })
            else:
                out_rows.append(combo_union_by_id[i])
        path = _BENCH / f"case3_combo_cascade_union_k{k}of5.json"
        path.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"combo_cascade_union k>={k}/5 -> {path}")


if __name__ == "__main__":
    main()
