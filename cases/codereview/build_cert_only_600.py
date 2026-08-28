"""Восстанавливает вердикты `cert_only` (ступень 1 каскада) на eval600 из готового
`out/bench/cascade600_verdicts.json`, без единого сетевого вызова.

Логика: `run_cascade.py` эскалирует в ступень 2 ВСЕ doc_id, у которых ступень 1 вернула
verdict="uncertain" (см. `run_cascade.main`: `uncertain_ids = [... if d["verdict"]=="uncertain"]`).
Поэтому в `cascade600_verdicts.json`:
  - запись с `artifacts.source == "llm_reviewer"` — вердикт ступени 1 как есть, ступень 1
    не была "uncertain", эскалации не было;
  - запись с `artifacts.source == "llm_reviewer_cascade_stage2"` — вердикт ступени 2, а
    ступень 1 для этого doc_id была "uncertain" (это и есть причина эскалации).

Восстановление: для первой группы — вердикт как есть; для второй — verdict="uncertain",
confidence=0.0, action="manual_review", остальные поля — плейсхолдер (для eval это не важно:
`evaluate.py` смотрит только verdict/confidence/artifacts.cwe_id, а uncertain в бинарных
метриках всегда трактуется как secure независимо от confidence).

НИКОГДА не исполнять и не компилировать код из датасета — только реконструкция json (см. CLAUDE.md).
"""
from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "out" / "bench" / "cascade600_verdicts.json"
_OUT = _ROOT / "out" / "bench" / "case3_cert_only_600.json"


def main():
    data = json.loads(_SRC.read_text(encoding="utf-8"))
    out = []
    n_asis, n_reconstructed = 0, 0
    for d in data:
        if d["artifacts"].get("source") == "llm_reviewer":
            out.append(d)
            n_asis += 1
        else:
            assert d["artifacts"].get("source") == "llm_reviewer_cascade_stage2", d["artifacts"].get("source")
            out.append({
                "doc_id": d["doc_id"],
                "verdict": "uncertain",
                "confidence": 0.0,
                "action": "manual_review",
                "evidence": [],
                "rationale": "",
                "artifacts": {
                    "source": "reconstructed_cert_only_stage1_uncertain",
                    "note": "ступень 1 (cert_only) вернула uncertain для этого doc_id -> "
                             "эскалирован в ступень 2 каскада; исходные поля ступени 1 (кроме "
                             "verdict=uncertain) не сохранились отдельно, восстановлен только verdict.",
                },
            })
            n_reconstructed += 1
    print(f"as-is (llm_reviewer): {n_asis}")
    print(f"reconstructed uncertain (was cascade_stage2): {n_reconstructed}")
    assert n_asis + n_reconstructed == len(data) == 600
    _OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {_OUT}")


if __name__ == "__main__":
    main()
