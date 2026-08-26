"""Кейс 1 — PII/PHI: обезличивание с сохранением смысла.

Подключение к `run.py --case 1` (см. `core/pipeline.py`): экспортируем `PLUGIN`.

Пайплайн — две стадии:
- `route`: детерминированный слой (`detectors.py`) находит форматные сущности, кладёт их в
  запись и передаёт ВСЁ дальше в `llm` (сама стадия `route` вердиктов не выносит).
- `llm`: второй проход поверх текста (`llm_layer.detect_pii_llm`) добирает свободнотекстовые
  сущности (имена, гео, демография — 20 из 55 типов, которые regex не берёт в принципе, см.
  `report.md`, раздел «где нужен LLM»), сливает их с детекторными через `llm_layer.merge_spans`
  и вызывает `aliaser.anonymize()` на объединённом списке — он не отличает источник спана.

Без ключей LLM (`LLMConfig.dry_run=True`, см. CLAUDE.md) `detect_pii_llm` не делает сетевых
вызовов и в норме не добавляет ничего поверх детекторов (стаб-JSON почти никогда не совпадает
дословно с текстом документа) — поэтому детерминированный путь даёт те же цифры, что и раньше,
если LLM-слой фактически выключен отсутствием ключа.
"""

from __future__ import annotations

from core.data import load_case1
from core.pipeline import CasePlugin, PipelineContext, Record
from core.schema import Verdict

from cases.pii.aliaser import anonymize, doc_salt
from cases.pii.detectors import detect_pii
from cases.pii.llm_layer import detect_pii_llm, merge_spans
from cases.pii.verify import audit


def load(ctx: PipelineContext) -> list[Record]:
    cfg = ctx.config
    df = load_case1(
        split=cfg.get("split", "test"),
        n=cfg.get("n", 200),
        seed=cfg.get("seed", 42),
    )
    records: list[Record] = []
    for _, row in df.iterrows():
        records.append(
            {
                "doc_id": row["uid"],
                "text": row["text"],
                "gold_spans": row["spans"],  # list[dict], уже распарсено load_case1
                "domain": row["domain"],
                "document_type": row.get("document_type"),
            }
        )
    return records


def route(records: list[Record], ctx: PipelineContext) -> tuple[list[Verdict], list[Record]]:
    """Только детекторный проход, вердиктов не выносит: результат кладётся в запись, а вся
    выборка целиком уходит в стадию `llm` (см. докстринг модуля)."""
    for rec in records:
        rec["detector_spans"] = detect_pii(rec["text"])
    return [], records


def llm(records: list[Record], ctx: PipelineContext) -> list[Verdict]:
    """Второй проход (`llm_layer.detect_pii_llm`) + слияние + подстановка. Вызывается всегда:
    в dry-run (по умолчанию, ключей нет) `ctx.llm` не делает сети и не находит ничего значимого
    поверх детекторов — эта стадия обязательная часть архитектуры, а не «включаемая опция»."""
    model = ctx.config.get("model")
    verdicts = []
    for rec in records:
        doc_id = str(rec["doc_id"])
        text = rec["text"]
        detector_spans = rec["detector_spans"]
        llm_spans = detect_pii_llm(
            text,
            ctx.llm,
            document_type=rec.get("document_type") or "",
            domain=rec.get("domain") or "",
            model=model,
        )
        spans = merge_spans(detector_spans, llm_spans)
        salt = doc_salt(doc_id)
        anon_text, vault = anonymize(text, spans, salt)

        verdicts.append(
            Verdict(
                doc_id=doc_id,
                verdict="anonymized",
                confidence=1.0,  # пересчитается в validate по факту self-audit
                action="pass",
                evidence=sorted({sp["label"] for sp in spans}),
                rationale=(
                    f"Детекторы нашли {len(detector_spans)} форматных PII-сущностей, "
                    f"LLM-слой добавил {len(llm_spans)}, свёрнуто в {len(vault)} уникальных "
                    f"сущностей."
                ),
                artifacts={
                    "original_text": text,
                    "anonymized_text": anon_text,
                    "pii_found": spans,
                    "detector_pii_found": detector_spans,
                    "llm_pii_found": llm_spans,
                    "vault": vault,
                    "domain": rec.get("domain"),
                },
            )
        )
    return verdicts


def validate(verdicts: list[Verdict], ctx: PipelineContext) -> list[Verdict]:
    """Self-audit: анонимизированный текст прогоняется детекторами повторно (см. verify.py).
    `true_leak_rate` > 0 на документе → `action=manual_review`, иначе документ проходит."""
    out = []
    for v in verdicts:
        a = v.artifacts
        report = audit(a["original_text"], a["anonymized_text"], a["pii_found"])
        a["self_audit"] = report
        leaked = report["n_true_leaks"] > 0 or report["n_literal_leftovers"] > 0
        out.append(
            v.model_copy(
                update={
                    "verdict": "leak_detected" if leaked else "clean",
                    "confidence": 0.5 if leaked else 1.0,
                    "action": "manual_review" if leaked else "pass",
                }
            )
        )
    return out


def export_columns(v: Verdict) -> dict:
    a = v.artifacts
    vault = a.get("vault", [])
    return {
        "doc_id": v.doc_id,
        "verdict": v.verdict,
        "action": v.action,
        "confidence": v.confidence,
        "pii_labels_found": sorted({e["label"] for e in a.get("pii_found", [])}),
        "n_entities_replaced": len(vault),
        "n_llm_entities": len(a.get("llm_pii_found", [])),
        "true_leak_rate": a.get("self_audit", {}).get("true_leak_rate"),
        "anonymized_text": a.get("anonymized_text", ""),
    }


PLUGIN = CasePlugin(
    name="pii",
    load=load,
    route=route,
    llm=llm,
    validate=validate,
    export_columns=export_columns,
)
