"""Цель 2 (см. промпт координатора): расширить область применения ступени 2 с "uncertain" (89 id)
также на "secure"-корзину cert_only (29 id), офлайн-сборка поверх уже посчитанных вердиктов —
новые LLM-вызовы для этого сделаны заранее `run_extended_stage2.py`
(`cases/codereview/out/stage2_extended_full_by_k.json`, `stage2_extended_samples.json`).

Два варианта:
- secure_all: эскалация = cert_only uncertain(89) ∪ cert_only secure(29) = 118 id.
- secure_lowconf: эскалация = cert_only uncertain(89) ∪ {secure с confidence < 0.9} (2 id, у
  cert_only секьюр-корзины confidence квантована в {0.8, 0.9, 0.95} — порог "низкая" даёт всего
  2 id, это тоже РЕЗУЛЬТАТ, а не баг: сигнал по этому полю почти отсутствует).

Для обоих: id из uncertain(89) берутся из уже готовых `case3_deepseek-chat_cascade_B_k{k}of5.json`
(там уже смёрджена ступень 1 + ступень 2 для этой корзины); id из secure-корзины (29 либо 2) берут
готовый Verdict из `stage2_extended_full_by_k.json[k]`, посчитанный `run_extended_stage2.py` тем же
промптом/голосованием. Остаток корпуса (vulnerable-корзина cert_only, 32 id) не трогается.

НИКОГДА не исполнять и не компилировать код из датасета — только статический анализ (CLAUDE.md).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_ROOT = Path(__file__).resolve().parents[2]
_BENCH = _ROOT / "out" / "bench"
_OUT = _ROOT / "cases" / "codereview" / "out"


def _load(path: Path) -> dict[str, dict]:
    return {d["doc_id"]: d for d in json.loads(path.read_text(encoding="utf-8"))}


def main() -> None:
    cert = _load(_BENCH / "case3_deepseek-chat_cert_only.json")
    ids = sorted(cert)
    assert len(ids) == 150

    c_unc = {i for i in ids if cert[i]["verdict"] == "uncertain"}
    c_sec = {i for i in ids if cert[i]["verdict"] == "secure"}
    assert len(c_unc) == 89 and len(c_sec) == 29

    lowconf_sec = {i for i in c_sec if cert[i]["confidence"] < 0.9}
    print(f"uncertain(cert_only)={len(c_unc)}  secure(cert_only)={len(c_sec)}  "
          f"secure_lowconf(confidence<0.9)={len(lowconf_sec)}: {sorted(lowconf_sec)}")

    full_by_k = json.loads((_OUT / "stage2_extended_full_by_k.json").read_text(encoding="utf-8"))

    variants = {
        "secure_all": c_sec,
        "secure_lowconf": lowconf_sec,
    }

    for name, extra_ids in variants.items():
        escalate = c_unc | extra_ids
        for k in (1, 2, 3, 4):
            cascade_b_k = _load(_BENCH / f"case3_deepseek-chat_cascade_B_k{k}of5.json")
            extended_k = full_by_k[str(k)]
            out_rows = []
            for i in ids:
                if i in c_unc:
                    out_rows.append(cascade_b_k[i])
                elif i in extra_ids:
                    out_rows.append(extended_k[i])
                else:
                    out_rows.append(cert[i])
            path = _BENCH / f"case3_stage2_{name}_k{k}of5.json"
            path.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{name} k>={k}/5 (escalated={len(escalate)}) -> {path}")


if __name__ == "__main__":
    main()
