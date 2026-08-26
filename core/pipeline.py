"""Стадии конвейера: load → prefilter → route → llm → validate → calibrate.

Кейс подключается регистрацией `CasePlugin` с реализациями нужных стадий.
Обязательны только `load` и `export_columns`; остальные опциональны (дефолт — passthrough:
всё уходит в LLM-стадию, без предрешённых вердиктов и без валидации/калибровки).

Конвенция подключения (см. CLAUDE.md § «границы владения файлами»): пакет `cases.<name>`
экспортирует модуль-уровневую константу `PLUGIN: CasePlugin` (обычно из `cases/<name>/__init__.py`,
либо оттуда реэкспортированную из подмодуля). `run.py` находит её по номеру кейса:

    1 → cases.pii.PLUGIN
    2 → cases.guard.PLUGIN
    3 → cases.codereview.PLUGIN

Семантика стадий:
- `load(ctx) -> list[Record]` — читает датасет (обычно через `core.data`), возвращает список
  записей (`dict`), у каждой обязателен ключ `doc_id` (строка).
- `prefilter(records, ctx) -> list[Record]` — детерминированная предобработка/отсев
  (например регекс-слой кейса 1, сигнатурный триаж кейса 3). Не порождает вердиктов.
- `route(records, ctx) -> (auto_verdicts, remaining)` — дешёвый слой, который уже способен
  вынести часть вердиктов без LLM (kNN/эвристики кейса 2); остальное идёт дальше.
- `llm(records, ctx) -> list[Verdict]` — вызовы `ctx.llm` (единственная точка контакта с LLM).
- `validate(verdicts, ctx) -> list[Verdict]` — самопроверка/консистентность (self-audit кейса 1,
  проверка патча кейса 3).
- `calibrate(verdicts, ctx) -> list[Verdict]` — порог эскалации → финальный `action`.

`ctx.scratch` — блокнот для передачи кейс-специфичного состояния между стадиями одного прогона
(например vault соответствий сущность→псевдоним в кейсе 1). Ядро в него не заглядывает.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from core.llm import LLMClient
from core.schema import Verdict

Record = dict[str, Any]


@dataclass
class PipelineContext:
    case: str
    config: dict[str, Any]
    llm: LLMClient
    scratch: dict[str, Any] = field(default_factory=dict)


@dataclass
class CasePlugin:
    name: str
    load: Callable[[PipelineContext], list[Record]]
    export_columns: Callable[[Verdict], dict[str, Any]]
    prefilter: Callable[[list[Record], PipelineContext], list[Record]] | None = None
    route: Callable[[list[Record], PipelineContext], tuple[list[Verdict], list[Record]]] | None = None
    llm: Callable[[list[Record], PipelineContext], list[Verdict]] | None = None
    validate: Callable[[list[Verdict], PipelineContext], list[Verdict]] | None = None
    calibrate: Callable[[list[Verdict], PipelineContext], list[Verdict]] | None = None


def run_pipeline(plugin: CasePlugin, ctx: PipelineContext) -> list[Verdict]:
    records = plugin.load(ctx)

    if plugin.prefilter:
        records = plugin.prefilter(records, ctx)

    if plugin.route:
        auto_verdicts, remaining = plugin.route(records, ctx)
    else:
        auto_verdicts, remaining = [], records

    llm_verdicts = plugin.llm(remaining, ctx) if plugin.llm and remaining else []

    verdicts = [*auto_verdicts, *llm_verdicts]

    if plugin.validate:
        verdicts = plugin.validate(verdicts, ctx)
    if plugin.calibrate:
        verdicts = plugin.calibrate(verdicts, ctx)

    return verdicts
