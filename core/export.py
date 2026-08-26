"""Выгрузка списка Verdict в xlsx и json. Колонки xlsx задаёт кейс через `columns_fn`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from openpyxl import Workbook

from core.schema import Verdict


def _cellify(value: object) -> object:
    """Excel-ячейка не умеет хранить произвольные python-объекты — сплющиваем в строку/число."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _default_columns(v: Verdict) -> dict:
    row = v.model_dump()
    row["artifacts"] = json.dumps(row["artifacts"], ensure_ascii=False)
    row["evidence"] = "; ".join(row["evidence"])
    return row


def to_xlsx(verdicts: list[Verdict], path: str,
            columns_fn: Callable[[Verdict], dict] | None = None,
            sheet_name: str = "verdicts") -> None:
    """columns_fn(verdict) -> dict колонок для одной строки. По умолчанию — плоский Verdict."""
    columns_fn = columns_fn or _default_columns
    rows = [columns_fn(v) for v in verdicts]

    header: list[str] = []
    for row in rows:
        for key in row:
            if key not in header:
                header.append(key)

    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name
    ws.append(header)
    for row in rows:
        ws.append([_cellify(row.get(h)) for h in header])

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def to_json(verdicts: list[Verdict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = [v.model_dump() for v in verdicts]
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
