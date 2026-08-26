"""Небольшая база правил SEI CERT C/C++ Coding Standard, подтягиваемых в промпт по конструкциям,
обнаруженным в конкретном фрагменте (через `cases.codereview.triage.score_fragment`).

Источник: публичный каталог SEI CERT C (https://wiki.sei.cmu.edu/confluence/display/c/,
актуальное зеркало https://cmu-sei.github.io/secure-coding-standards/). ID и заголовки правил
и краткое содержание сверены через WebFetch по MEM30-C и ARR38-C (см. cases/codereview/
improvements.md); остальные — устоявшиеся, широко публикуемые формулировки того же стандарта,
пересказаны своими словами, без копирования длинных фрагментов текста CERT.

Это доменное знание, не привязанное к конкретному датасету — переносится на любой C/C++ код,
в отличие от retrieval по размеченным примерам (см. `knn_baseline.py` — диагностика того, что
именно даёт retrieval по своему датасету).
"""

from __future__ import annotations

from cases.codereview.triage import score_fragment

# Правило -> (id, заголовок, короткая формулировка сути, категория триажа-триггер).
# `trigger_category` — то же имя категории, что и в `triage.CATEGORY_WEIGHTS`; правило
# подтягивается в промпт, если соответствующая категория сработала (`n_<category> > 0`).
RULES: list[dict] = [
    {
        "id": "STR31-C",
        "title": "Guarantee that storage for strings has sufficient space for character data and the null terminator",
        "summary": "Буфер под строку должен вмещать данные И завершающий нуль-терминатор; "
                    "strcpy/strcat/sprintf/gets не проверяют это сами — нужна явная проверка "
                    "длины или замена на safe-варианты (strncpy/strlcpy/snprintf) с корректным "
                    "учётом лишнего байта под '\\0'.",
        "trigger_category": "unsafe_func",
    },
    {
        "id": "ARR38-C",
        "title": "Guarantee that library functions do not form invalid pointers",
        "summary": "Размер (count/nbytes), переданный в memcpy/memmove/fread и аналоги, должен "
                    "быть проверен относительно реальной ёмкости целевого буфера ДО вызова — "
                    "иначе классический паттерн Heartbleed: атакующий контролирует длину и читает/"
                    "пишет за пределы буфера.",
        "trigger_category": "memcpy_unchecked",
    },
    {
        "id": "FIO30-C",
        "title": "Exclude user input from format strings",
        "summary": "Строка формата в printf-семействе не должна быть данными, пришедшими извне "
                    "(переменной, а не литералом) — иначе атакующий управляет спецификаторами "
                    "(%n и т.п.), что даёт запись произвольной памяти или утечку стека.",
        "trigger_category": "format_string",
    },
    {
        "id": "MEM32-C",
        "title": "Detect and handle memory allocation errors",
        "summary": "Результат malloc/calloc/realloc и системных аналогов (kmalloc и т.п.) может "
                    "быть NULL — использование указателя без проверки на NULL перед разыменованием "
                    "— падение или (реже) эксплуатируемая запись по нулевому смещению.",
        "trigger_category": "unchecked_alloc",
    },
    {
        "id": "MEM30-C",
        "title": "Do not access freed memory",
        "summary": "Разыменование указателя после free() того же указателя — undefined behavior: "
                    "память могла быть переиспользована под другие данные, атакующий, управляя "
                    "реаллокацией, может подменить содержимое (use-after-free -> RCE).",
        "trigger_category": "use_after_free",
    },
    {
        "id": "MEM31-C",
        "title": "Free dynamically allocated memory when no longer needed (double-free)",
        "summary": "free() одного и того же указателя дважды без переприсваивания между вызовами "
                    "повреждает метаданные аллокатора — эксплуатируемая порча кучи, часто "
                    "используется для получения примитива произвольной записи.",
        "trigger_category": "double_free",
    },
    {
        "id": "INT30-C / INT32-C",
        "title": "Ensure unsigned/signed integer operations do not wrap or overflow",
        "summary": "Арифметика (умножение/сложение) внутри аргумента размера для malloc/calloc и "
                    "аналогов может переполниться и дать маленькое или отрицательное значение — "
                    "аллокация получает меньше памяти, чем ожидает код, который затем пишет по "
                    "исходному (большому) размеру -> heap overflow через integer overflow.",
        "trigger_category": "size_int_overflow",
    },
]

# Fallback-правила — общего характера, добавляются, если ни одна триаж-категория не сработала
# (риск = none), чтобы промпт не оставался совсем без CERT-контекста на «тихих» фрагментах.
_FALLBACK_RULES = [
    {
        "id": "EXP34-C",
        "title": "Do not dereference null pointers",
        "summary": "Любое разыменование указателя (->, *p, p[i]) должно быть после проверки, что "
                    "указатель не NULL, если его значение может быть NULL по контракту функции "
                    "(результат поиска, необязательный параметр, результат предыдущего вызова).",
        "trigger_category": None,
    },
    {
        "id": "ARR30-C",
        "title": "Do not form or use out-of-bounds pointers or array subscripts",
        "summary": "Индекс массива/указательная арифметика должны оставаться в пределах "
                    "выделенного объекта на всех путях выполнения, включая граничные (i == size).",
        "trigger_category": None,
    },
]


def relevant_rules(code: str) -> list[dict]:
    """Правила, релевантные конкретному фрагменту — по категориям сигнатурного триажа
    (`triage.score_fragment`), не по лейблу и не по эталону. Пусто -> fallback-набор."""
    scores = score_fragment(code)
    hit_categories = {c for c in scores.get("categories", "").split(";") if c}
    selected = [r for r in RULES if r["trigger_category"] in hit_categories]
    if not selected:
        selected = _FALLBACK_RULES
    return selected


def rules_prompt_block(code: str) -> str:
    """Компактный текстовый блок для вставки в промпт LLM."""
    rules = relevant_rules(code)
    if not rules:
        return ""
    lines = ["Применимые правила безопасного кодирования (SEI CERT C/C++):"]
    for r in rules:
        lines.append(f"- {r['id']} ({r['title']}): {r['summary']}")
    return "\n".join(lines)
