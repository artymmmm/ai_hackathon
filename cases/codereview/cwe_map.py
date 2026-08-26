"""Нормализация и валидация CWE-идентификаторов, которые выдаёт LLM.

Модель путает форматы: "CWE-119", "cwe119", "119", "CWE 119: Buffer Overflow",
"Improper Restriction of Operations within Bounds of a Memory Buffer (CWE-119)" и т.п.
`normalize_cwe` извлекает каноническую форму `CWE-<n>` из любого из них (или возвращает
`None`, если числа CWE в тексте нет вообще — модель могла ответить прозой).

Справочник имён `CWE_NAMES` построен по фактическому распределению `cwe_id` среди 774
восстановленных `vulnerable`-фрагментов (`research/case3_recovered_labels.csv`, топ виден в
`research/case3_label_matching.md` §5) плюс несколько CWE Top-25 для C/C++, которых нет в топе
восстановленных лейблов, но которые LLM реалистично может назвать (SQLi/командная инъекция и т.п.
на встраиваемых интерпретаторах, которых немного, но они есть в корпусе).
"""

from __future__ import annotations

import re

_CWE_DIGITS_RE = re.compile(r"cwe[\s_-]?(\d{1,4})", re.IGNORECASE)
_BARE_DIGITS_RE = re.compile(r"^\s*(\d{1,4})\s*$")

# Покрытие: все CWE, встретившиеся хотя бы дважды среди восстановленных vulnerable-лейблов
# (research/case3_label_matching.md §5), плюс типовые для C/C++ CWE Top-25.
CWE_NAMES: dict[str, str] = {
    "CWE-119": "Buffer Overflow — общее нарушение границ буфера",
    "CWE-20": "Improper Input Validation",
    "CWE-125": "Out-of-bounds Read",
    "CWE-399": "Resource Management Errors",
    "CWE-264": "Permissions, Privileges and Access Control",
    "CWE-200": "Information Exposure",
    "CWE-416": "Use After Free",
    "CWE-189": "Numeric Errors",
    "CWE-190": "Integer Overflow or Wraparound",
    "CWE-476": "NULL Pointer Dereference",
    "CWE-787": "Out-of-bounds Write",
    "CWE-362": "Race Condition (Concurrent Execution using Shared Resource)",
    "CWE-284": "Improper Access Control",
    "CWE-19": "Data Processing Errors",
    "CWE-310": "Cryptographic Issues",
    "CWE-59": "Improper Link Resolution Before File Access (Symlink Following)",
    "CWE-404": "Improper Resource Shutdown or Release",
    "CWE-400": "Uncontrolled Resource Consumption",
    "CWE-415": "Double Free",
    "CWE-269": "Improper Privilege Management",
    "CWE-79": "Improper Neutralization of Input During Web Page Generation (XSS)",
    "CWE-18": "Source Code — общая категория",
    "CWE-17": "Code — общая категория",
    "CWE-311": "Missing Encryption of Sensitive Data",
    "CWE-77": "Improper Neutralization of Special Elements used in a Command (Command Injection)",
    "CWE-22": "Improper Limitation of a Pathname to a Restricted Directory (Path Traversal)",
    "CWE-369": "Divide By Zero",
    "CWE-354": "Improper Validation of Integrity Check Value",
    "CWE-254": "Security Features — общая категория",
    "CWE-617": "Reachable Assertion",
    "CWE-770": "Allocation of Resources Without Limits or Throttling",
    "CWE-754": "Improper Check for Unusual or Exceptional Conditions",
    "CWE-388": "Error Handling — общая категория",
    "CWE-285": "Improper Authorization",
    "CWE-94": "Improper Control of Generation of Code (Code Injection)",
    "CWE-255": "Credentials Management Errors",
    "CWE-674": "Uncontrolled Recursion",
    "CWE-346": "Origin Validation Error",
    "CWE-320": "Key Management Errors",
    "CWE-120": "Buffer Copy without Checking Size of Input (Classic Buffer Overflow)",
    # Вне топа восстановленных лейблов, но типовые для C/C++ CWE Top-25 — держим в справочнике,
    # чтобы не проваливаться в UNKNOWN на разумных ответах модели.
    "CWE-78": "OS Command Injection",
    "CWE-89": "SQL Injection",
    "CWE-611": "Improper Restriction of XML External Entity Reference (XXE)",
    "CWE-434": "Unrestricted Upload of File with Dangerous Type",
    "CWE-798": "Use of Hard-coded Credentials",
    "CWE-863": "Incorrect Authorization",
    "CWE-352": "Cross-Site Request Forgery (CSRF)",
    "CWE-732": "Incorrect Permission Assignment for Critical Resource",
    "CWE-835": "Loop with Unreachable Exit Condition (Infinite Loop)",
    "CWE-131": "Incorrect Calculation of Buffer Size",
    "CWE-457": "Use of Uninitialized Variable",
    "CWE-843": "Type Confusion",
    "CWE-909": "Missing Initialization of Resource",
    "CWE-682": "Incorrect Calculation",
}

UNKNOWN_NAME = "неизвестный CWE или не найден в локальном справочнике"


def normalize_cwe(raw: object) -> str | None:
    """Приводит произвольный ответ модели к каноническому 'CWE-<n>'.

    Возвращает `None`, если в тексте нет распознаваемого номера CWE (пусто, "N/A",
    свободная проза без числа и т.п.) — это не ошибка, а честный сигнал «модель не назвала CWE».
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        n = int(raw)
        return f"CWE-{n}" if n > 0 else None
    text = str(raw).strip()
    if not text or text.lower() in {"none", "null", "n/a", "na", "-", "нет"}:
        return None
    m = _CWE_DIGITS_RE.search(text)
    if m:
        return f"CWE-{int(m.group(1))}"
    m = _BARE_DIGITS_RE.match(text)
    if m:
        return f"CWE-{int(m.group(1))}"
    return None


def cwe_name(cwe_id: str | None) -> str:
    """Человекочитаемое имя по канонической форме. Неизвестный/нераспознанный CWE — не ошибка."""
    if cwe_id is None:
        return UNKNOWN_NAME
    return CWE_NAMES.get(cwe_id, UNKNOWN_NAME)


def is_known(cwe_id: str | None) -> bool:
    return cwe_id is not None and cwe_id in CWE_NAMES
