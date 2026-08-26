"""
Сигнатурный триаж кейса 3.

Регулярочный статический анализ C/C++ фрагментов из "case 3/3 кейс_датасет.csv".
Код НЕ компилируется и НЕ исполняется — только текстовый разбор регулярными
выражениями и простыми эвристиками на строках.

Категории риска (см. PLAN.md, кейс 3):
  unsafe_func        strcpy/strcat/sprintf/gets
  memcpy_unchecked    memcpy/memmove без видимой проверки границ
  format_string      функции форматного вывода с не-литеральной строкой формата
  unchecked_alloc     malloc/calloc/... без проверки результата на NULL
  double_free        free(x) дважды подряд без переприсваивания x
  use_after_free      использование x после free(x) без переприсваивания
  size_int_overflow    арифметика (*, +) внутри аргумента размера alloc-вызова

Это слой приоритизации перед LLM-разбором, а не финальный вердикт: эвристики
дают ложные срабатывания и пропуски по конструкции (регулярки не понимают
поток управления). Ложные срабатывания — ожидаемая цена за скорость и за то,
что метод работает без исполнения кода.

Использование:
    .venv/bin/python cases/codereview/triage.py \
        --input "case 3/3 кейс_датасет.csv" \
        --output cases/codereview/out/triage_scores.csv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

# --- категория: небезопасные функции копирования/форматирования строк ---
UNSAFE_FUNC_RE = re.compile(r"\b(strcpy|strcat|sprintf|vsprintf|gets)\s*\(")

# --- категория: memcpy/memmove — сам факт вызова + эвристика на «проверено ли» ---
MEMCPY_CALL_RE = re.compile(
    r"\b(memcpy|memmove)\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^)]+?)\s*\)"
)
BOUND_KEYWORDS_RE = re.compile(r"\b(min|MIN|max|MAX)\b")
COMPARISON_RE = re.compile(r"[<>]=?")

# --- категория: format string — форматная функция вызвана без строкового литерала
# на позиции аргумента формата. Позиция формата зависит от функции: printf(fmt,...),
# fprintf(stream,fmt,...), snprintf(buf,size,fmt,...) — поэтому парсим аргументы
# по позициям, а не одной регуляркой (иначе snprintf(buf, ...) даёт систематический FP).
FORMAT_ARG_POS = {
    "printf": 1,
    "vprintf": 1,
    "fprintf": 2,
    "vfprintf": 2,
    "syslog": 2,
    "vsyslog": 2,
    "sprintf": 2,
    "vsprintf": 2,
    "snprintf": 3,
    "vsnprintf": 3,
}
FORMAT_CALL_START_RE = re.compile(r"\b(" + "|".join(FORMAT_ARG_POS) + r")\s*\(")


def _split_top_level_args(s: str, open_paren_pos: int):
    """s[open_paren_pos] == '('. Возвращает аргументы вызова, разбитые по
    запятым верхнего уровня (с учётом вложенных скобок вроде sizeof(x)).
    None, если вызов не закрылся в разумном окне (защита от патологий)."""
    depth = 0
    args = []
    current = []
    limit = min(len(s), open_paren_pos + 2000)
    for i in range(open_paren_pos, limit):
        ch = s[i]
        if ch == "(":
            depth += 1
            if depth > 1:
                current.append(ch)
        elif ch == ")":
            depth -= 1
            if depth == 0:
                args.append("".join(current).strip())
                return args
            current.append(ch)
        elif ch == "," and depth == 1:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    return None

# --- категория: unchecked malloc/alloc ---
ALLOC_FUNCS = r"(malloc|calloc|realloc|kmalloc|kzalloc|vmalloc|g_malloc|g_try_malloc|strdup|strndup)"
ALLOC_ASSIGN_RE = re.compile(
    r"\b(\w+)\s*=\s*(?:\([^)=]*\)\s*)?\b" + ALLOC_FUNCS + r"\s*\("
)
NULL_CHECK_WINDOW = 200  # символов после присваивания, где ищем проверку на NULL

# --- категория: double-free / use-after-free ---
FREE_CALL_RE = re.compile(r"\b(free|kfree|g_free|vfree)\s*\(\s*([A-Za-z_]\w*)\s*\)")
USE_AFTER_FREE_WINDOW = 400  # символов после free(), где ищем повторное использование

# --- категория: integer overflow в арифметике размера alloc-вызова ---
SIZE_ARITH_RE = re.compile(
    r"\b(malloc|calloc|realloc|kmalloc|kzalloc|vmalloc|alloca|g_malloc)\s*\(([^;]*?[*+][^;]*?)\)"
)

CATEGORY_WEIGHTS = {
    "unsafe_func": 3,
    "memcpy_unchecked": 3,
    "format_string": 4,
    "unchecked_alloc": 2,
    "double_free": 5,
    "use_after_free": 5,
    "size_int_overflow": 2,
}
CATEGORY_CAP = 3  # не более 3 засчитанных срабатываний одной категории на фрагмент


def _find_unsafe_func(code: str) -> int:
    return len(UNSAFE_FUNC_RE.findall(code))


def _find_memcpy_unchecked(code: str) -> int:
    hits = 0
    for m in MEMCPY_CALL_RE.finditer(code):
        size_expr = m.group(4)
        if "sizeof" in size_expr or BOUND_KEYWORDS_RE.search(size_expr):
            continue  # похоже на защищённый вызов
        idents = re.findall(r"[A-Za-z_]\w*", size_expr)
        window_start = max(0, m.start() - 400)
        preceding = code[window_start : m.start()]
        checked = False
        for ident in idents:
            for im in re.finditer(r"\bif\s*\([^)]*\)", preceding):
                if ident in im.group(0) and COMPARISON_RE.search(im.group(0)):
                    checked = True
                    break
            if checked:
                break
        if not checked:
            hits += 1
    return hits


def _find_format_string(code: str) -> int:
    hits = 0
    for m in FORMAT_CALL_START_RE.finditer(code):
        func = m.group(1)
        paren_pos = m.end() - 1
        args = _split_top_level_args(code, paren_pos)
        if not args:
            continue
        pos = FORMAT_ARG_POS[func]
        if pos > len(args):
            continue
        fmt_arg = args[pos - 1].strip()
        if not fmt_arg or fmt_arg.startswith('"') or fmt_arg.startswith('L"'):
            continue  # строковый литерал — не уязвимость
        if fmt_arg.upper() in {"NULL", "0"}:
            continue
        hits += 1
    return hits


def _find_unchecked_alloc(code: str) -> int:
    hits = 0
    for m in ALLOC_ASSIGN_RE.finditer(code):
        var = m.group(1)
        window = code[m.end() : m.end() + NULL_CHECK_WINDOW]
        null_check = re.search(
            r"\b(if|assert|ASSERT|BUG_ON)\s*\([^)]*\b" + re.escape(var) + r"\b[^)]*\)",
            window,
        )
        if not null_check:
            hits += 1
    return hits


def _find_free_misuse(code: str) -> tuple[int, int]:
    """Возвращает (double_free_hits, use_after_free_hits)."""
    double_free = 0
    use_after_free = 0
    for m in FREE_CALL_RE.finditer(code):
        var = m.group(2)
        window = code[m.end() : m.end() + USE_AFTER_FREE_WINDOW]
        reassign = re.search(r"\b" + re.escape(var) + r"\s*=[^=]", window)
        reassign_pos = reassign.start() if reassign else len(window)

        double_free_m = re.search(
            r"\bfree\s*\(\s*" + re.escape(var) + r"\s*\)", window
        )
        if double_free_m and double_free_m.start() < reassign_pos:
            double_free += 1

        use_m = re.search(
            r"\b" + re.escape(var) + r"\s*(->|\[)|[*]\s*" + re.escape(var) + r"\b",
            window,
        )
        if use_m and use_m.start() < reassign_pos:
            use_after_free += 1
    return double_free, use_after_free


def _find_size_int_overflow(code: str) -> int:
    hits = 0
    for m in SIZE_ARITH_RE.finditer(code):
        expr = m.group(2)
        # sizeof(x) * n — идиоматичный, но всё ещё формально CWE-190; считаем,
        # просто с меньшей уверенностью его не отделяем отдельной категорией.
        hits += 1
    return hits


def score_fragment(code: str) -> dict:
    if not isinstance(code, str) or not code:
        counts = {c: 0 for c in CATEGORY_WEIGHTS}
    else:
        double_free, use_after_free = _find_free_misuse(code)
        counts = {
            "unsafe_func": _find_unsafe_func(code),
            "memcpy_unchecked": _find_memcpy_unchecked(code),
            "format_string": _find_format_string(code),
            "unchecked_alloc": _find_unchecked_alloc(code),
            "double_free": double_free,
            "use_after_free": use_after_free,
            "size_int_overflow": _find_size_int_overflow(code),
        }

    risk_score = sum(
        min(counts[c], CATEGORY_CAP) * CATEGORY_WEIGHTS[c] for c in CATEGORY_WEIGHTS
    )
    categories_hit = [c for c, n in counts.items() if n > 0]

    if risk_score == 0:
        risk_level = "none"
    elif risk_score <= 3:
        risk_level = "low"
    elif risk_score <= 8:
        risk_level = "medium"
    else:
        risk_level = "high"

    result = {"risk_score": risk_score, "risk_level": risk_level}
    result.update({f"n_{c}": counts[c] for c in CATEGORY_WEIGHTS})
    result["categories"] = ";".join(categories_hit)
    return result


def run(input_path: Path, output_path: Path, summary_path: Path) -> None:
    df = pd.read_csv(input_path)
    scored = df["code"].apply(score_fragment).apply(pd.Series)
    out = pd.concat([df[["unique_id"]], scored], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    total = len(out)
    summary = {
        "total_fragments": total,
        "risk_level_distribution": out["risk_level"].value_counts().to_dict(),
        "category_hit_counts": {
            c: int((out[f"n_{c}"] > 0).sum()) for c in CATEGORY_WEIGHTS
        },
        "category_hit_share": {
            c: round(float((out[f"n_{c}"] > 0).mean()), 4) for c in CATEGORY_WEIGHTS
        },
        "mean_risk_score": round(float(out["risk_score"].mean()), 3),
        "median_risk_score": float(out["risk_score"].median()),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"Всего фрагментов: {total}")
    print("Распределение по risk_level:")
    for level in ["none", "low", "medium", "high"]:
        n = int((out["risk_level"] == level).sum())
        print(f"  {level:8s} {n:6d}  ({n / total:.1%})")
    print("\nПопадание по категориям (фрагментов с хотя бы 1 срабатыванием):")
    for c in CATEGORY_WEIGHTS:
        n = int((out[f"n_{c}"] > 0).sum())
        print(f"  {c:20s} {n:6d}  ({n / total:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("case 3/3 кейс_датасет.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("cases/codereview/out/triage_scores.csv")
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("cases/codereview/out/triage_summary.json"),
    )
    args = parser.parse_args()
    run(args.input, args.output, args.summary)


if __name__ == "__main__":
    main()
