"""ШАГ 4: сравнение наших сгенерированных патчей с реальными апстримными фиксами BigVul.

Идея (PLAN.md/задание): в BigVul для `vul==1` строк есть пара (func_before, func_after) —
func_before это ровно то, что попало к нам как "уязвимый фрагмент" (см. `research/
case3_label_matching.md`), func_after — реальный фикс, принятый мейнтейнерами апстрима.
Мы НЕ используем это на инференсе (это была бы утечка) — только здесь, постфактум, для оценки
качества уже сгенерированных нами патчей против настоящего эталона.

Источник наших патчей: `out/bench/case3_deepseek-chat.json` — уже посчитанный прогон на eval-150
(та же дисциплина, что и все остальные шаги — используем существующие артефакты, не тратим
новый LLM-бюджет). Берём фрагменты, где gold_label == vulnerable И verdict == "vulnerable" И
patched_code непустой (то есть: модель верно нашла уязвимость и предложила патч) — только для
них вообще есть с чем сравнивать реальный фикс. Дополнительно (если посчитан заранее) подмешивает
патчи из `cases/codereview/out/config_experiment_verdicts_config_A.json`/`config_B.json` (шаг 3)
как второй, отдельно помеченный источник.

Матчинг с BigVul: тот же метод, что в `research/case3_label_matching.md` §2 — SHA-256 от
normalize(code) = re.sub(r'\\s+', '', code), точное совпадение с normalize(func_before) среди
строк BigVul с vul==1.

Метрики (намеренно грубые — сам факт валидации против эталона важнее точности метрики,
как и указано в задании):
  1. touched_decile_overlap  — функция делится на 10 "децилей" по номеру строки; сравниваются
     МНОЖЕСТВА децилей, которые задело наше изменение (original vs ours) и апстримное
     (func_before vs func_after) — Jaccard. Не требует точного совпадения нумерации строк
     (разное форматирование между источниками), достаточно грубого "где по функции менялось".
  2. text_similarity_to_upstream — difflib.SequenceMatcher.ratio() между нашим patched_code
     и апстримным func_after (после нормализации пробелов) — насколько ИТОГОВЫЙ текст похож.
  3. new_token_overlap — Jaccard множества НОВЫХ токенов (появившихся в патче, которых не было
     в оригинале) между нашим патчем и апстримным — грубая проверка "чинили тем же способом"
     (например оба добавили вызов snprintf/strncpy или сравнение с размером буфера).
  4. same_triage_categories_resolved — сигнатурные категории triage.py, сработавшие на
     оригинале: сравниваем, исчезли ли они и в нашем патче, и в апстримном (переиспользует
     `patch_check.check_vulnerable_pattern_gone`, независимую проверку кейса).

Код НЕ компилируется и НЕ исполняется — сравнение чисто текстовое (diff/токены), тот же принцип,
что и во всём остальном кейсе 3.
"""

from __future__ import annotations

import difflib
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cases.codereview.patch_check import check_vulnerable_pattern_gone  # noqa: E402
from cases.codereview.features import tokenize_code  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_OUT_DIR = Path(__file__).resolve().parent / "out"
_BASELINE_VERDICTS = _ROOT / "out" / "bench" / "case3_deepseek-chat.json"
_GOLD_CSV = _ROOT / "research" / "case3_recovered_labels.csv"


def _normalize(code: str) -> str:
    return re.sub(r"\s+", "", code)


def _hash(code: str) -> str:
    return hashlib.sha256(_normalize(code).encode("utf-8", errors="ignore")).hexdigest()


def build_bigvul_fix_map() -> dict[str, dict]:
    """hash(normalize(func_before)) -> {func_after, cwe, project} только для vul==1 строк
    БЕЗ конфликтов (один и тот же нормализованный func_before с разными func_after — отбрасываем,
    как и в research/case3_label_matching.md — не гадаем, какой фикс правильный)."""
    paths = glob.glob(str(
        Path.home() / ".cache" / "huggingface" / "hub" / "datasets--bstee615--bigvul"
        / "snapshots" / "*" / "data" / "*.parquet"
    ))
    if not paths:
        raise FileNotFoundError(
            "BigVul parquet не найден в кеше HF — скачать: "
            "hf download bstee615/bigvul --repo-type dataset --include '*.parquet'"
        )
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    vul = df[df["vul"] == 1].dropna(subset=["func_before", "func_after"])

    fix_map: dict[str, dict] = {}
    conflicts: set[str] = set()
    for _, row in vul.iterrows():
        h = _hash(row["func_before"])
        entry = {"func_after": row["func_after"], "cwe": row.get("CWE ID"), "project": row.get("project")}
        if h in fix_map and fix_map[h]["func_after"] != row["func_after"]:
            conflicts.add(h)
            continue
        fix_map[h] = entry
    for h in conflicts:
        fix_map.pop(h, None)
    return fix_map


def touched_decile_overlap(original: str, variant_a: str, variant_b: str) -> float:
    """Jaccard децилей строк, задетых (original->variant_a) и (original->variant_b) отдельно."""
    def touched_deciles(orig_lines: list[str], new_lines: list[str]) -> set[int]:
        sm = difflib.SequenceMatcher(None, orig_lines, new_lines, autojunk=False)
        n = max(1, len(orig_lines))
        deciles = set()
        for tag, i1, i2, _, _ in sm.get_opcodes():
            if tag == "equal":
                continue
            for i in range(i1, max(i2, i1 + 1)):
                deciles.add(min(9, int(i / n * 10)))
        return deciles

    orig_lines = original.splitlines()
    da = touched_deciles(orig_lines, variant_a.splitlines())
    db = touched_deciles(orig_lines, variant_b.splitlines())
    if not da and not db:
        return None  # обе версии текстуально идентичны оригиналу — не о чем судить
    union = da | db
    return len(da & db) / len(union) if union else 0.0


def new_token_overlap(original: str, variant_a: str, variant_b: str) -> float | None:
    orig_tokens = set(tokenize_code(original))
    new_a = set(tokenize_code(variant_a)) - orig_tokens
    new_b = set(tokenize_code(variant_b)) - orig_tokens
    if not new_a and not new_b:
        return None
    union = new_a | new_b
    return len(new_a & new_b) / len(union) if union else 0.0


def text_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, re.sub(r"\s+", " ", a), re.sub(r"\s+", " ", b)).ratio()


def evaluate_one(original: str, our_patch: str, upstream_fix: str) -> dict:
    our_gone = check_vulnerable_pattern_gone(original, our_patch)
    upstream_gone = check_vulnerable_pattern_gone(original, upstream_fix)
    same_categories_resolved = (
        our_gone["status"] == upstream_gone["status"] == "gone"
        or our_gone["status"] == upstream_gone["status"] == "not_applicable"
    )
    return {
        "touched_decile_overlap": touched_decile_overlap(original, our_patch, upstream_fix),
        "text_similarity_to_upstream": round(text_similarity(our_patch, upstream_fix), 4),
        "new_token_overlap": new_token_overlap(original, our_patch, upstream_fix),
        "our_patch_status_vs_original": our_gone["status"],
        "upstream_fix_status_vs_original": upstream_gone["status"],
        "same_triage_resolution_outcome": same_categories_resolved,
    }


def collect_candidates(verdicts_path: Path, gold: dict[str, str], source_label: str) -> list[dict]:
    data = json.loads(verdicts_path.read_text(encoding="utf-8"))
    out = []
    for d in data:
        doc_id = d["doc_id"]
        if gold.get(doc_id) != "1":
            continue  # только истинно vulnerable — иначе нечего сравнивать с "реальным фиксом"
        if d["verdict"] != "vulnerable":
            continue
        a = d.get("artifacts", {})
        patched = a.get("patched_code", "")
        code = a.get("code", "")
        if not patched.strip() or not code.strip():
            continue
        out.append({"doc_id": doc_id, "code": code, "patched_code": patched, "source": source_label})
    return out


def main() -> None:
    print("Строим карту апстримных фиксов BigVul (hash(func_before) -> func_after)...")
    fix_map = build_bigvul_fix_map()
    print(f"{len(fix_map)} однозначных пар (func_before, func_after) без конфликтов.")

    gold_df = pd.read_csv(_GOLD_CSV)
    gold_df["unique_id"] = gold_df["unique_id"].astype(int)
    gold = {str(int(r.unique_id)): str(r.recovered_label) for r in gold_df.itertuples()
            if pd.notna(r.recovered_label)}

    candidates = collect_candidates(_BASELINE_VERDICTS, gold, "bare_baseline_eval150")
    for extra_name in ("config_A", "config_B"):
        p = _OUT_DIR / f"config_experiment_verdicts_{extra_name}.json"
        if p.exists():
            candidates += collect_candidates(p, gold, extra_name)
    print(f"Кандидатов (verdict=vulnerable, gold=vulnerable, патч непустой): {len(candidates)}")

    results = []
    for c in candidates:
        h = _hash(c["code"])
        fix = fix_map.get(h)
        if fix is None:
            results.append({**c, "has_upstream_fix": False})
            continue
        metrics = evaluate_one(c["code"], c["patched_code"], fix["func_after"])
        results.append({**c, "has_upstream_fix": True, "upstream_cwe": fix.get("cwe"),
                         "upstream_project": fix.get("project"), **metrics})

    n_with_fix = sum(1 for r in results if r["has_upstream_fix"])
    print(f"\nНайден апстримный фикс для {n_with_fix} / {len(results)} кандидатов.")

    if n_with_fix:
        with_fix = [r for r in results if r["has_upstream_fix"]]
        dec = [r["touched_decile_overlap"] for r in with_fix if r["touched_decile_overlap"] is not None]
        sim = [r["text_similarity_to_upstream"] for r in with_fix]
        tok = [r["new_token_overlap"] for r in with_fix if r["new_token_overlap"] is not None]
        same_outcome = sum(1 for r in with_fix if r["same_triage_resolution_outcome"])
        summary = {
            "n_candidates": len(results),
            "n_with_upstream_fix": n_with_fix,
            "mean_touched_decile_overlap": round(sum(dec) / len(dec), 4) if dec else None,
            "mean_text_similarity_to_upstream": round(sum(sim) / len(sim), 4) if sim else None,
            "mean_new_token_overlap": round(sum(tok) / len(tok), 4) if tok else None,
            "same_triage_resolution_outcome_share": round(same_outcome / len(with_fix), 4),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        summary = {"n_candidates": len(results), "n_with_upstream_fix": 0}
        print("Ни для одного кандидата не нашлось апстримного фикса в BigVul — "
              "см. detailed results для причин (возможно все vulnerable-TP пришли из DiverseVul-only "
              "части, где func_after недоступен).")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    (_OUT_DIR / "upstream_patch_eval.json").write_text(
        json.dumps({"summary": summary, "details": results}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n-> {_OUT_DIR / 'upstream_patch_eval.json'}")


if __name__ == "__main__":
    main()
