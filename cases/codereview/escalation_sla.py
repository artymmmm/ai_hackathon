"""ШАГ 2б: кривая эскалации как настраиваемый SLA — «при бюджете ручной проверки X% ловим Y%
уязвимостей», а не сухая пара precision/recall.

Использует УЖЕ ПОСЧИТАННЫЙ прогон deepseek-chat на eval-150 (`out/bench/case3_deepseek-chat.json`)
— НОВЫХ вызовов LLM здесь нет, это переупаковка существующих данных под другой вопрос.

Политика ранжирования для эскалации: сортируем ВСЕ 150 вердиктов по возрастанию `confidence`
(не только те, что сама модель пометила `uncertain` — низкая уверенность в "vulnerable"/"secure"
тоже кандидат на ручную проверку, это и даёт непрерывную настраиваемую ручку вместо
фиксированного 3-way решения). При бюджете X% в ручную проверку уходит `ceil(X% * N)`
фрагментов с САМОЙ НИЗКОЙ confidence.

"Поймано" (caught) — истинно vulnerable фрагмент, который либо:
  (a) отправлен на ручную проверку (предполагаем, что человек находит реальную уязвимость —
      это одно из объявленных допущений отчёта, не измеренный факт), либо
  (b) НЕ отправлен на проверку, но сама модель дала verdict="vulnerable" (поймано автоматически).
Не поймано — vulnerable фрагмент, который не эскалирован и получил verdict="secure"/"uncertain"
(тут uncertain при confidence выше отсечки формально не эскалирован в этой политике и
трактуется как пропуск — честно, не смягчаем).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cases.codereview.evaluate import load_gold, load_verdicts  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_OUT_DIR = Path(__file__).resolve().parent / "out"


def build_sla_table(verdicts_path: Path, budgets: list[float] | None = None) -> pd.DataFrame:
    budgets = budgets if budgets is not None else [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                                                     0.35, 0.40, 0.50, 0.60, 0.80, 1.0]
    verdicts = load_verdicts(verdicts_path)
    gold = load_gold()

    rows = []
    for v in verdicts:
        g = gold.get(v.doc_id)
        if g is None or g["label"] is None:
            continue
        rows.append({"doc_id": v.doc_id, "confidence": v.confidence, "verdict": v.verdict,
                      "gold_label": g["label"]})
    df = pd.DataFrame(rows).sort_values("confidence", ascending=True).reset_index(drop=True)
    n = len(df)
    n_vuln_total = int((df["gold_label"] == "vulnerable").sum())

    table = []
    for budget in budgets:
        n_escalated = math.ceil(budget * n)
        escalated_idx = set(df.index[:n_escalated])  # самые низкие confidence уходят первыми
        caught = 0
        for i, row in df.iterrows():
            if row["gold_label"] != "vulnerable":
                continue
            if i in escalated_idx or row["verdict"] == "vulnerable":
                caught += 1
        table.append({
            "review_budget_pct": round(budget * 100, 1),
            "n_escalated_to_human": n_escalated,
            "escalated_share_pct": round(n_escalated / n * 100, 1) if n else 0.0,
            "vulnerabilities_caught": caught,
            "vulnerabilities_total": n_vuln_total,
            "catch_rate_pct": round(caught / n_vuln_total * 100, 1) if n_vuln_total else 0.0,
        })
    return pd.DataFrame(table)


def main() -> None:
    verdicts_path = _ROOT / "out" / "bench" / "case3_deepseek-chat.json"
    table = build_sla_table(verdicts_path)

    print("SLA-таблица: бюджет ручной проверки -> доля пойманных уязвимостей "
          f"(источник: {verdicts_path.name}, N=150, из них 50 истинно vulnerable)\n")
    print(table.to_string(index=False))

    # Точка сравнения с текущей 3-way политикой (verdict=uncertain -> эскалация):
    # escalation_rate=23.3%, к человеку попадает 40% (совпадает с report/model_benchmark.md).
    current_policy_note = (
        "Текущая 3-way политика (эскалация только verdict=uncertain): "
        "escalation_rate=23.3%, catch_rate=40.0% (7 vulnerable+13 uncertain из 50) — "
        "см. report/model_benchmark.md. Непрерывная ручка по confidence при том же бюджете "
        "~23% даёт catch_rate, показанный в таблице выше на ближайшей строке."
    )
    print(f"\n{current_policy_note}")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / "escalation_sla_table.json"
    out_path.write_text(
        json.dumps({"table": table.to_dict(orient="records"), "note": current_policy_note},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    table.to_csv(_OUT_DIR / "escalation_sla_table.csv", index=False)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
