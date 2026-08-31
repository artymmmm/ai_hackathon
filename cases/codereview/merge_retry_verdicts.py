"""Вливает добор обрезанных фрагментов в основную выгрузку кейса 3.

130 из 137 заглушек боевого прогона — обрыв ответа на max_tokens=2048 ровно там, где модель
писала патч. Поднимать лимит для всего корпуса нельзя (сменился бы ключ кеша и весь прогон
пришлось бы оплачивать заново), поэтому те же фрагменты пройдены отдельно с max_tokens=4096
и их вердикты подменяются здесь. Пять фрагментов не уложились и в 4096 — остаются uncertain
с action=manual_review, что для них честный исход.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cases.codereview import export_columns  # noqa: E402
from core.export import to_json, to_xlsx  # noqa: E402
from core.schema import Verdict  # noqa: E402


def main(base_path: str, retry_path: str, out_dir: str) -> None:
    base = json.loads(Path(base_path).read_text(encoding="utf-8"))
    retry = {v["doc_id"]: v for v in json.loads(Path(retry_path).read_text(encoding="utf-8"))}
    merged = [retry.get(v["doc_id"], v) for v in base]
    replaced = sum(1 for v in base if v["doc_id"] in retry)

    verdicts = [Verdict.model_validate(v) for v in merged]
    out = Path(out_dir)
    to_xlsx(verdicts, str(out / "case3_verdicts.xlsx"), columns_fn=export_columns)
    to_json(verdicts, str(out / "case3_verdicts.json"))
    print(f"вердиктов {len(verdicts)}, подменено {replaced}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
