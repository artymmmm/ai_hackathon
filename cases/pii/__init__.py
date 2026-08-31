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

from concurrent.futures import ThreadPoolExecutor

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
    # `uid` в датасете НЕ идентифицирует документ: 100 000 разных текстов приходятся на 50 000
    # uid, каждый встречается ровно дважды с разными документами. Разводим их порядковым
    # номером вхождения. Это не косметика: соль подстановки берётся из doc_id, и на общей соли
    # один и тот же человек получал один псевдоним в обоих документах пары — то есть документы
    # можно было связать между собой по псевдониму. Исходный uid сохраняем для прослеживаемости.
    seen: dict[str, int] = {}
    records: list[Record] = []
    for _, row in df.iterrows():
        uid = str(row["uid"])
        seen[uid] = seen.get(uid, 0) + 1
        records.append(
            {
                "doc_id": f"{uid}#{seen[uid]}",
                "source_uid": uid,
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


def _anonymize_one(rec: Record, ctx: PipelineContext, model: str | None) -> Verdict:
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

    return Verdict(
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
            "source_uid": rec.get("source_uid"),
            "original_text": text,
            "anonymized_text": anon_text,
            "pii_found": spans,
            "detector_pii_found": detector_spans,
            "llm_pii_found": llm_spans,
            "vault": vault,
            "domain": rec.get("domain"),
        },
    )


def llm(records: list[Record], ctx: PipelineContext) -> list[Verdict]:
    """Второй проход (`llm_layer.detect_pii_llm`) + слияние + подстановка. Вызывается всегда:
    в dry-run (по умолчанию, ключей нет) `ctx.llm` не делает сети и не находит ничего значимого
    поверх детекторов — эта стадия обязательная часть архитектуры, а не «включаемая опция».

    Запросы к LLM независимы и блокирующие, поэтому распараллелены через ThreadPoolExecutor
    (лимит — LLMConfig.max_concurrency); ex.map сохраняет порядок результатов.
    """
    model = ctx.config.get("model")
    with ThreadPoolExecutor(max_workers=ctx.llm.config.max_concurrency) as ex:
        return list(ex.map(lambda rec: _anonymize_one(rec, ctx, model), records))


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


# Ячейка Excel не вмещает больше 32 767 символов — файл сохранится, но не откроется.
# Полный текст всегда остаётся в json-выгрузке (`core.export.to_json` берёт `model_dump()`,
# а не эти колонки), поэтому в xlsx безопасно подрезать.
_XLSX_CELL_LIMIT = 32000


def _clip(text: str) -> str:
    if len(text) <= _XLSX_CELL_LIMIT:
        return text
    return text[:_XLSX_CELL_LIMIT] + f"… [обрезано для xlsx, полностью — в json, всего {len(text)} символов]"


def export_columns(v: Verdict) -> dict:
    a = v.artifacts
    vault = a.get("vault", [])
    return {
        "doc_id": v.doc_id,
        "uid_датасета": a.get("source_uid"),
        "verdict": v.verdict,
        "action": v.action,
        "confidence": v.confidence,
        "pii_labels_found": sorted({e["label"] for e in a.get("pii_found", [])}),
        # Задание требует именно список найденных персональных данных и использованные замены,
        # а не только их типы и счётчик.
        "pii_found": _clip("; ".join(f'{e["label"]}: {e["text"]}' for e in a.get("pii_found", []))),
        "replacements": _clip("; ".join(
            f'{" / ".join(entry.get("original_values", []))} → {entry.get("alias", "")}'
            f' [{entry.get("label", "")}, {entry.get("occurrences", 0)} вхожд.]'
            for entry in vault
        )),
        "n_entities_replaced": len(vault),
        "n_llm_entities": len(a.get("llm_pii_found", [])),
        "true_leak_rate": a.get("self_audit", {}).get("true_leak_rate"),
        "anonymized_text": _clip(a.get("anonymized_text", "")),
    }


PLUGIN = CasePlugin(
    name="pii",
    load=load,
    route=route,
    llm=llm,
    validate=validate,
    export_columns=export_columns,
)
