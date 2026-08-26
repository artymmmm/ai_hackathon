"""Обёртка над flawfinder — текстовый статический анализатор C/C++ (regex по базе из 222
опасных паттернов/функций, https://dwheeler.com/flawfinder/), устанавливается через
`uv pip install flawfinder` (чистый Python-пакет, добавлен в .venv этой сессией).

ПРОВЕРЕНО (см. cases/codereview/improvements.md за деталями): flawfinder — построчный
regex-сканер исходного текста. Он не запускает препроцессор, не вызывает компилятор и не
исполняет ни строки анализируемого кода. Подтверждено чтением `--help` (никаких флагов
компиляции/выполнения) и практическим прогоном — сканирует файл и выдаёт хиты по паттернам,
как и `triage.py` в этом же кейсе, только с гораздо большей базой правил и подключённым
маппингом на CWE. Это укладывается в жёсткий запрет задания на компиляцию/исполнение
(CLAUDE.md, PLAN.md §5): анализируемый текст временно кладётся в файл на диске ТОЛЬКО чтобы
дать flawfinder путь для чтения (subprocess, никакого shell=True, никакой команды типа `gcc`/
`cc`/`python` над этим файлом).

Используется как дополнительный сигнал в промпте LLM (конфигурация A, cases/codereview/
improvements.md шаг 3) — независимый от размеченного корпуса источник, как и cert_rules.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

_FLAWFINDER_BIN = Path(sys.executable).parent / "flawfinder"


def run_flawfinder(code: str, timeout_s: float = 15.0) -> list[dict]:
    """Прогоняет flawfinder на фрагменте, возвращает список хитов
    [{line, level, cwe_ids, message}], отсортированных по убыванию серьёзности."""
    if not _FLAWFINDER_BIN.exists():
        return []
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp_path = f.name
    try:
        proc = subprocess.run(
            [str(_FLAWFINDER_BIN), "--sarif", tmp_path],
            capture_output=True, text=True, timeout=timeout_s, check=False,
        )
        if not proc.stdout.strip():
            return []
        sarif = json.loads(proc.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return []
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not sarif.get("runs"):
        return []
    run = sarif["runs"][0]
    rule_cwe: dict[str, list[str]] = {}
    for rule in run["tool"]["driver"].get("rules", []):
        cwes = [rel["target"]["id"] for rel in rule.get("relationships", [])
                if rel.get("target", {}).get("toolComponent", {}).get("name") == "CWE"]
        rule_cwe[rule["id"]] = cwes

    _LEVEL_RANK = {"error": 3, "warning": 2, "note": 1}
    hits = []
    for r in run.get("results", []):
        loc = (r.get("locations") or [{}])[0].get("physicalLocation", {}).get("region", {})
        hits.append({
            "line": loc.get("startLine"),
            "level": r.get("level", "note"),
            "cwe_ids": rule_cwe.get(r["ruleId"], []),
            "message": r.get("message", {}).get("text", ""),
        })
    hits.sort(key=lambda h: _LEVEL_RANK.get(h["level"], 0), reverse=True)
    return hits


def flawfinder_prompt_block(code: str, max_hits: int = 8) -> str:
    """Компактный текстовый блок для вставки в промпт LLM. Пусто, если хитов нет или
    flawfinder недоступен — это ДОПОЛНИТЕЛЬНЫЙ сигнал, отсутствие хита не значит "secure"."""
    hits = run_flawfinder(code)
    if not hits:
        return ""
    lines = [
        "Вывод текстового статического сканера flawfinder (regex по базе опасных функций/"
        "паттернов, НЕ анализ потока данных — может давать ложные срабатывания и пропуски, "
        "не финальный вердикт):"
    ]
    for h in hits[:max_hits]:
        cwe = f" ({', '.join(h['cwe_ids'])})" if h["cwe_ids"] else ""
        lines.append(f"- строка {h['line']} [{h['level']}]{cwe}: {h['message']}")
    if len(hits) > max_hits:
        lines.append(f"... и ещё {len(hits) - max_hits} хитов (не показаны, экономия места)")
    return "\n".join(lines)
