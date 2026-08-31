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
from core.schema import ACTION_RU, Verdict
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
    records = [
        {
            "doc_id": f"case2-{split}-{i}",
            "text": row["text"],
            "label": int(row["label"]),
            "gold_verdict": row["verdict_binary"],
        }
        for i, row in df.iterrows()
    ]
    # Задание требует в выгрузке сам запрос, а Verdict текста не несёт (и не должен: контракт
    # общий на три кейса). Возвращаем текст в стадии `calibrate` через блокнот прогона —
    # ровно то, для чего `ctx.scratch` и заведён (см. докстринг core/pipeline.py).
    ctx.scratch["texts"] = {r["doc_id"]: r["text"] for r in records}
    return records


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


def calibrate(verdicts: list[Verdict], ctx: PipelineContext) -> list[Verdict]:
    """Эскалация пограничных запросов — прямое требование задания: «Если запрос пограничный
    (похож одновременно на безопасный и на инъекцию), участник выбирает наиболее вероятный класс
    и отмечает его как требующий эскалации».

    Пограничные — ровно те, что офлайн-слой не смог закрыть порогом и отправил в серую зону
    (`source == "llm_grey_zone"`, ~4.4% потока при threshold 0.80). Класс при этом остаётся
    тот, что выбрал LLM — меняется только рекомендуемое решение, поэтому измеренная F1 0.989
    (она считается по `verdict`) не затрагивается.
    """
    texts = ctx.scratch.get("texts", {})
    out = []
    for v in verdicts:
        update: dict = {"artifacts": {**v.artifacts, "text": texts.get(v.doc_id, "")}}
        if v.artifacts.get("source") == "llm_grey_zone" and v.action != "manual_review":
            update["action"] = "manual_review"
        out.append(v.model_copy(update=update))
    return out


# Формулировки классов ровно как в задании («safe / injection and malicious»).
VERDICT_RU = {"safe": "safe", "injection_malicious": "injection and malicious"}


def export_columns(v: Verdict) -> dict:
    return {
        "doc_id": v.doc_id,
        "запрос": v.artifacts.get("text", ""),
        "класс": VERDICT_RU.get(v.verdict, v.verdict),
        "решение": ACTION_RU.get(v.action, v.action),
        "обоснование": v.rationale,
        "признаки_атаки": "; ".join(v.evidence),
        "подтип": v.artifacts.get("subtype"),
        "уверенность": v.confidence,
        "verdict": v.verdict,
        "action": v.action,
        "source": v.artifacts.get("source"),
    }


PLUGIN = CasePlugin(
    name="guard",
    load=load,
    route=route,
    llm=classify_grey_zone,
    calibrate=calibrate,
    export_columns=export_columns,
)
