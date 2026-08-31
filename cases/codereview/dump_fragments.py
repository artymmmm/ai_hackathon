"""Выкладывает фрагменты корпуса кейса 3 в отдельные файлы для сторонних статических анализаторов.

Только запись на диск. Ни компиляции, ни исполнения — прямой запрет задания и CLAUDE.md.
Анализаторы (cppcheck, semgrep) разбирают текст собственными парсерами и компилятор не зовут.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.data import load_case3  # noqa: E402


def main(out_dir: str, ext: str = ".c") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_case3()
    for uid, code in zip(df["unique_id"].astype(int), df["code"].astype(str)):
        (out / f"{uid}{ext}").write_text(code, encoding="utf-8", errors="replace")
    print(f"записано {len(df)} фрагментов в {out} с расширением {ext}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ".c")
