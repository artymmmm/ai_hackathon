"""Прогон cppcheck по выложенным фрагментам корпуса кейса 3.

cppcheck — статический анализатор с собственным парсером: компилятор он не вызывает и код
не исполняет, поэтому запрет задания не нарушается (см. CLAUDE.md, «жёсткие запреты»).

Результат — csv по одной строке на фрагмент: список сработавших id и severity, чтобы порог
подбирался потом по измеренному вкладу, а не угадывался заранее.
"""

from __future__ import annotations

import collections
import csv
import subprocess
import sys
from pathlib import Path


def main(frag_dir: str, out_csv: str, jobs: str = "8") -> None:
    frags = sorted(Path(frag_dir).glob("*.c"))
    cmd = [
        "cppcheck",
        "--enable=warning,style,portability",
        "--inline-suppr",
        "--suppress=missingInclude",
        "--suppress=missingIncludeSystem",
        "--suppress=unknownMacro",
        "--suppress=syntaxError",
        "--suppress=preprocessorErrorDirective",
        "--template={file}\t{severity}\t{id}",
        "--quiet",
        f"-j{jobs}",
        frag_dir,
    ]
    print("запуск:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")

    hits: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for line in proc.stderr.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        path, severity, check_id = parts
        uid = Path(path).stem
        if uid.isdigit():
            hits[uid].append((severity, check_id))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["unique_id", "n_hits", "any_hit", "has_error", "severities", "check_ids"])
        for frag in frags:
            uid = frag.stem
            h = hits.get(uid, [])
            w.writerow([uid, len(h), bool(h), any(s == "error" for s, _ in h),
                        ";".join(sorted({s for s, _ in h})),
                        ";".join(sorted({c for _, c in h}))])
    n_hit = sum(1 for frag in frags if hits.get(frag.stem))
    print(f"фрагментов {len(frags)}, со срабатыванием {n_hit} ({n_hit/len(frags):.1%})")
    top = collections.Counter(c for v in hits.values() for _, c in v)
    print("частые проверки:", top.most_common(12))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "8")
