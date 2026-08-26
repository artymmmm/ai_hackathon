"""Плагин кейса 3 (CI-ревьюер кода). Экспортирует `PLUGIN: CasePlugin` (конвенция — см.
докстринг `core/pipeline.py`), подключаемый через `run.py --case 3`.

Стадии:
  load       — core.data.load_case3 (весь корпус, без built-in сэмплирования), обогащённый:
               * подсказкой сигнатурного триажа (triage.py, cases/codereview/out/triage_scores.csv,
                 если посчитан заранее) — деривативный признак самого кода, идёт В ПРОМПТ как
                 необязательная подсказка (reviewer.py, _hint_block) — не решающий фактор;
               * восстановленным лейблом (research/case3_recovered_labels.csv, если файл есть) —
                 используется ТОЛЬКО для (а) стратифицированного --sample, чтобы мини-прогон не
                 терял редкие vulnerable (4.1% корпуса — research/case3_label_matching.md), и
                 (б) evaluate.py. НИКОГДА не читается в reviewer.py/patch_check.py и не попадает
                 в промпт LLM — иначе ревью перестало бы быть независимым от эталона, которым же
                 его потом и меряют.
  llm        — cases.codereview.reviewer.review_fragments; единственная точка сетевого вызова,
               и та — через ctx.llm (core/llm.py), по умолчанию dry_run=True (без сети).
  validate   — patch_check.check_patch + patch_check.second_opinion: для verdict=="vulnerable"
               с непустым patched_code — три статические проверки патча (сигнатура, наличие
               функциональности, исчез ли паттерн триажа) плюс независимый второй вызов LLM
               «содержит ли ЭТОТ код уязвимость?». Расхождение уводит вердикт в manual_review
               и понижает confidence — находка не отбрасывается молча, а помечается недоверенной.
  export_columns — плоские колонки xlsx/json: unique_id, фрагмент кода, вердикт, CWE-ID, патч
               (PLAN.md §5 «Формат сдачи»), плюс поля проверки патча и второго мнения.

prefilter/route сознательно не реализованы: recall сигнатурного триажа против восстановленных
лейблов — всего 10.5% (research/case3_label_matching.md), использовать его как фильтр-отсев перед
LLM означало бы терять ~90% реальных уязвимостей. Риск-скор используется только как
дополнительный признак в промпте, LLM видит весь прогоняемый корпус/выборку без предварительного
отсева. Отсутствующие стадии — passthrough (докстринг `core/pipeline.py`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.data import load_case3
from core.pipeline import CasePlugin, PipelineContext, Record
from core.schema import Verdict

from cases.codereview.patch_check import check_patch, second_opinion
from cases.codereview.reviewer import review_fragments

_ROOT = Path(__file__).resolve().parent.parent.parent
_RECOVERED_LABELS_CSV = _ROOT / "research" / "case3_recovered_labels.csv"
_TRIAGE_SCORES_CSV = Path(__file__).resolve().parent / "out" / "triage_scores.csv"


def _load_recovered_labels() -> pd.DataFrame | None:
    if not _RECOVERED_LABELS_CSV.exists():
        return None
    return pd.read_csv(_RECOVERED_LABELS_CSV)


def _load_triage_scores() -> pd.DataFrame | None:
    if not _TRIAGE_SCORES_CSV.exists():
        return None
    return pd.read_csv(_TRIAGE_SCORES_CSV)


def _stratified_sample_by_recovered_label(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Пропорциональная по recovered_label выборка — чтобы --sample N не терял редкие vulnerable
    (4.1% корпуса) при малых N. НЕ эталон для итоговой оценки: evaluate.py всегда меряет качество
    по факту прогнанного подмножества с recovered_label ∈ {0,1}, а не по этой пропорции — это
    только способ не потерять сигнал в маленьком демо-прогоне. Страты: '0.0' (secure), '1.0'
    (vulnerable), 'conflict', 'unmatched' (не нашлось лейбла).
    """
    if n >= len(df):
        return df
    strata = df["recovered_label"].fillna("unmatched").astype(str)
    frac = n / len(df)
    parts = []
    for _, group in df.groupby(strata, group_keys=False):
        k = max(1, round(len(group) * frac))
        parts.append(group.sample(n=min(k, len(group)), random_state=seed))
    sampled = pd.concat(parts)
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed)
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


def load(ctx: PipelineContext) -> list[Record]:
    n = ctx.config.get("sample")
    seed = ctx.config.get("seed", 42)

    df = load_case3()  # весь корпус; своя стратификация ниже вместо простого случайного сэмпла
    df["unique_id"] = df["unique_id"].astype(int)

    labels = _load_recovered_labels()
    if labels is not None:
        df = df.merge(labels, on="unique_id", how="left")
    else:
        for col in ("recovered_label", "match_source", "cwe_id", "source_project"):
            df[col] = None

    triage = _load_triage_scores()
    if triage is not None:
        df = df.merge(triage[["unique_id", "risk_level", "categories"]], on="unique_id", how="left")
    else:
        df["risk_level"] = None
        df["categories"] = None

    # Фиксированный набор фрагментов для сравнения моделей между собой: все прогоны
    # должны идти по одним и тем же id, иначе метрики моделей несопоставимы.
    ids_file = ctx.config.get("ids_file")
    if ids_file:
        wanted = [int(x) for x in Path(ids_file).read_text().split() if x.strip()]
        df = df[df["unique_id"].isin(wanted)]
    elif n is not None and n < len(df):
        df = _stratified_sample_by_recovered_label(df, n, seed)

    records: list[Record] = []
    for _, row in df.iterrows():
        records.append({
            "doc_id": str(int(row["unique_id"])),
            "unique_id": int(row["unique_id"]),
            "code": row["code"],
            "triage_risk_level": row.get("risk_level"),
            "triage_categories": row.get("categories"),
            # Только для evaluate.py — НЕ читается reviewer.py/patch_check.py, в промпт не попадает.
            "gold_label": row.get("recovered_label"),
            "gold_match_source": row.get("match_source"),
            "gold_cwe_id": row.get("cwe_id"),
            "gold_source_project": row.get("source_project"),
        })
    return records


def validate(verdicts: list[Verdict], ctx: PipelineContext) -> list[Verdict]:
    """Проверка патча для каждого verdict=="vulnerable" с непустым patched_code (PLAN.md §5,
    «Проверка патча — наша отличительная фича»). secure/uncertain не трогаем — патча нет."""
    out = []
    for v in verdicts:
        a = v.artifacts
        if v.verdict != "vulnerable" or not a.get("patched_code"):
            out.append(v)
            continue

        original_code = a.get("code", "")
        patched_code = a["patched_code"]
        static_check = check_patch(original_code, patched_code)
        opinion = second_opinion(patched_code, ctx)

        new_artifacts = {**a, "patch_check": static_check, "second_opinion": opinion}

        # Патч не прошёл статическую проверку ИЛИ второе независимое мнение всё ещё видит
        # уязвимость → не заявляем «уязвимость закрыта» без проверки: уводим на ручную проверку
        # и понижаем уверенность, но саму находку не отбрасываем.
        still_vulnerable = opinion.get("contains_vulnerability") is True
        patch_untrusted = not static_check["passed"] or still_vulnerable
        confidence = min(v.confidence, 0.4) if patch_untrusted else v.confidence
        action = "manual_review" if patch_untrusted else v.action

        out.append(v.model_copy(update={"confidence": confidence, "action": action, "artifacts": new_artifacts}))
    return out


def export_columns(v: Verdict) -> dict:
    a = v.artifacts
    pc = a.get("patch_check") or {}
    so = a.get("second_opinion") or {}
    return {
        "unique_id": int(v.doc_id),
        "verdict": v.verdict,
        "confidence": v.confidence,
        "action": v.action,
        "cwe_id": a.get("cwe_id"),
        "cwe_name": a.get("cwe_name"),
        "code": a.get("code", ""),
        "exploitation_mechanism": a.get("exploitation_mechanism", ""),
        "patched_code": a.get("patched_code", ""),
        "patch_rationale": a.get("patch_rationale", ""),
        "patch_signature_status": pc.get("signature", {}).get("status"),
        "patch_functionality_status": pc.get("functionality", {}).get("status"),
        "patch_vulnerable_pattern_status": pc.get("vulnerable_pattern", {}).get("status"),
        "patch_check_passed": pc.get("passed"),
        "second_opinion_contains_vulnerability": so.get("contains_vulnerability"),
        "second_opinion_confidence": so.get("confidence"),
        "evidence": "; ".join(v.evidence),
        "rationale": v.rationale,
        "uncertain_reason": a.get("uncertain_reason", ""),
        "truncated": a.get("truncated", False),
    }


PLUGIN = CasePlugin(
    name="codereview",
    load=load,
    llm=review_fragments,
    validate=validate,
    export_columns=export_columns,
)
