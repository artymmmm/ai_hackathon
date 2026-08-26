"""Плагин кейса 2 (AI-guard / инъекция на входе). Экспортирует `PLUGIN: CasePlugin`
(конвенция — см. докстринг `core/pipeline.py`), подключаемый через `run.py --case 2`.

Стадии:
  load       — core.data.load_case2, добавляет doc_id.
  route      — быстрый офлайн-слой (модель (a) из baseline.py, персистится через model.py):
               уверенные тексты (`confidence >= guard_threshold`) получают вердикт сразу,
               остальные («серая зона») уходят в llm.
  llm        — cases.guard.grey_zone.classify_grey_zone; единственная точка сетевого вызова,
               и та — через ctx.llm (core/llm.py), который по умолчанию dry_run=True (без сети).
  export_columns — плоские колонки для xlsx/json.

prefilter/validate/calibrate не реализованы (нет входного отсева и постобработки сверх
already-калиброванного порога route) — не проставлять их не значит, что стадии сломаны:
`core.pipeline.run_pipeline` трактует отсутствующие стадии как passthrough (докстринг там же).
"""

from __future__ import annotations

from core.data import load_case2
from core.pipeline import CasePlugin, PipelineContext, Record
from core.schema import Verdict
from cases.guard.grey_zone import classify_grey_zone
from cases.guard.model import load_or_train, predict_proba
from cases.guard.taxonomy import tag_text

LABEL_NAMES = {0: "safe", 1: "masked", 2: "direct"}
DEFAULT_CONFIDENCE_THRESHOLD = 0.80  # см. report.md §6: coverage 95.2%, FPR 0.42% на test


def load(ctx: PipelineContext) -> list[Record]:
    split = ctx.config.get("split", "train")
    n = ctx.config.get("sample")
    seed = ctx.config.get("seed", 42)
    df = load_case2(split=split, n=n, seed=seed)
    return [
        {
            "doc_id": f"case2-{split}-{i}",
            "text": row["text"],
            "label": int(row["label"]),
            "gold_verdict": row["verdict_binary"],
        }
        for i, row in df.iterrows()
    ]


def _offline_verdict(rec: Record, pred_class: int, confidence: float, proba_row) -> Verdict:
    tags = tag_text(rec["text"])
    if pred_class == 0:
        verdict, action, subtype = "safe", "pass", None
    else:
        verdict, action = "injection_malicious", "block"
        subtype = "masked" if pred_class == 1 else "direct"
    return Verdict(
        doc_id=rec["doc_id"],
        verdict=verdict,
        confidence=confidence,
        action=action,
        evidence=tags,
        rationale=f"офлайн-слой (TF-IDF+LogReg): класс={LABEL_NAMES[pred_class]}, p={confidence:.3f}",
        artifacts={
            "subtype": subtype,
            "pred_class_3way": pred_class,
            "proba": [round(float(x), 4) for x in proba_row],
            "source": "offline",
        },
    )


def route(records: list[Record], ctx: PipelineContext) -> tuple[list[Verdict], list[Record]]:
    if not records:
        return [], []
    model = ctx.scratch.setdefault("guard_model", load_or_train())
    threshold = ctx.config.get("guard_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
    texts = [r["text"] for r in records]
    proba, classes = predict_proba(model, texts)

    auto_verdicts: list[Verdict] = []
    remaining: list[Record] = []
    for rec, p in zip(records, proba):
        idx = int(p.argmax())
        confidence = float(p[idx])
        pred_class = int(classes[idx])
        if confidence >= threshold:
            auto_verdicts.append(_offline_verdict(rec, pred_class, confidence, p))
        else:
            remaining.append(rec)
    return auto_verdicts, remaining


def export_columns(v: Verdict) -> dict:
    return {
        "doc_id": v.doc_id,
        "verdict": v.verdict,
        "subtype": v.artifacts.get("subtype"),
        "confidence": v.confidence,
        "action": v.action,
        "evidence": "; ".join(v.evidence),
        "rationale": v.rationale,
        "source": v.artifacts.get("source"),
    }


PLUGIN = CasePlugin(
    name="guard",
    load=load,
    route=route,
    llm=classify_grey_zone,
    export_columns=export_columns,
)
