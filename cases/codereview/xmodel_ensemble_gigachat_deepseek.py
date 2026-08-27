"""Кросс-модельный ансамбль cert_only: deepseek-chat x GigaChat-2-Max, n=150 (STATE.md, задача агента xmodel).

Вход: два файла вердиктов на одной конфигурации cert_only (одинаковые 150 doc_id):
  - out/bench/case3_deepseek-chat_cert_only.json
  - out/bench/case3_gc-max_cert_only.json

Правила ансамбля (см. постановку задачи координатора):
  - union:            vulnerable, если хотя бы одна модель сказала vulnerable
  - intersection:     vulnerable, только если обе сказали vulnerable
  - union_uncertain:  vulnerable, если одна сказала vulnerable, а другая vulnerable ИЛИ uncertain
  - any_signal:       vulnerable, если хотя бы одна сказала vulnerable ИЛИ uncertain (верхняя
                       граница recall, заведомо шумная)

Для каждого правила пишется файл вердиктов out/bench/case3_ensemble_<rule>.json (совместимый
с evaluate.py: valid Verdict, verdict в {"secure","vulnerable"}), метрики считаются ТОЛЬКО через
evaluate.py (не пересчитываются вручную здесь) — вызывается как подпроцесс на каждый файл.

Плюс: матрица согласованности 3x3 (secure/vulnerable/uncertain x secure/vulnerable/uncertain) и
разбор, кто из моделей поймал какие ИСТИННЫЕ vulnerable в одиночку (по gold из evaluate.load_gold).

Запуск:
    .venv/bin/python cases/codereview/xmodel_ensemble_gigachat_deepseek.py
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cases.codereview.evaluate import load_gold, load_verdicts  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_BENCH = _ROOT / "out" / "bench"
_PY = _ROOT / ".venv" / "bin" / "python"
_EVALUATE = _ROOT / "cases" / "codereview" / "evaluate.py"

_DS_PATH = _BENCH / "case3_deepseek-chat_cert_only.json"
_GC_PATH = _BENCH / "case3_gc-max_cert_only.json"

_LABELS3 = ["secure", "vulnerable", "uncertain"]


def _combine(ds_v: str, gc_v: str, rule: str) -> str:
    vs = {ds_v, gc_v}
    if rule == "union":
        return "vulnerable" if "vulnerable" in vs else "secure"
    if rule == "intersection":
        return "vulnerable" if ds_v == "vulnerable" and gc_v == "vulnerable" else "secure"
    if rule == "union_uncertain":
        # vulnerable, если одна сказала vulnerable, а вторая vulnerable или uncertain
        if ds_v == "vulnerable" and gc_v in ("vulnerable", "uncertain"):
            return "vulnerable"
        if gc_v == "vulnerable" and ds_v in ("vulnerable", "uncertain"):
            return "vulnerable"
        return "secure"
    if rule == "any_signal":
        return "vulnerable" if ("vulnerable" in vs or "uncertain" in vs) else "secure"
    raise ValueError(rule)


def main():
    ds_verdicts = {v.doc_id: v for v in load_verdicts(_DS_PATH)}
    gc_verdicts = {v.doc_id: v for v in load_verdicts(_GC_PATH)}
    ds_ids, gc_ids = set(ds_verdicts), set(gc_verdicts)
    common = ds_ids & gc_ids
    print(f"deepseek cert_only: {len(ds_ids)} вердиктов; GigaChat cert_only: {len(gc_ids)} вердиктов; "
          f"общих doc_id: {len(common)}")
    if ds_ids != gc_ids:
        print(f"  ! разные множества id: только в deepseek={len(ds_ids - gc_ids)}, "
              f"только в GigaChat={len(gc_ids - ds_ids)}")

    common_sorted = sorted(common, key=lambda x: int(x) if x.isdigit() else x)

    # --- 3x3 матрица согласованности (сырые вердикты, включая uncertain) ---
    matrix = {a: {b: 0 for b in _LABELS3} for a in _LABELS3}
    for doc_id in common_sorted:
        a = ds_verdicts[doc_id].verdict
        b = gc_verdicts[doc_id].verdict
        a = a if a in _LABELS3 else "secure"
        b = b if b in _LABELS3 else "secure"
        matrix[a][b] += 1
    n_agree_exact = sum(matrix[k][k] for k in _LABELS3)
    print(f"\nматрица согласованности 3x3 (строки=deepseek, столбцы=GigaChat), n={len(common_sorted)}:")
    header = "              " + "  ".join(f"{b:>11}" for b in _LABELS3)
    print(header)
    for a in _LABELS3:
        print(f"{a:>12}  " + "  ".join(f"{matrix[a][b]:>11}" for b in _LABELS3))
    print(f"точное совпадение вердикта (secure/vulnerable/uncertain): {n_agree_exact}/{len(common_sorted)} "
          f"({n_agree_exact/len(common_sorted):.3f})")

    # --- gold-based: кто поймал какие ИСТИННЫЕ vulnerable в одиночку ---
    gold = load_gold()
    scoreable = [doc_id for doc_id in common_sorted
                 if doc_id in gold and gold[doc_id]["label"] in ("secure", "vulnerable")]
    true_vuln = [doc_id for doc_id in scoreable if gold[doc_id]["label"] == "vulnerable"]
    print(f"\nиз {len(common_sorted)} общих: {len(scoreable)} однозначно размечены (secure/vulnerable), "
          f"из них {len(true_vuln)} истинно vulnerable")

    ds_hit = {doc_id for doc_id in true_vuln if ds_verdicts[doc_id].verdict == "vulnerable"}
    gc_hit = {doc_id for doc_id in true_vuln if gc_verdicts[doc_id].verdict == "vulnerable"}
    both_hit = ds_hit & gc_hit
    only_ds = ds_hit - gc_hit
    only_gc = gc_hit - ds_hit
    neither = set(true_vuln) - ds_hit - gc_hit
    print(f"поймали (verdict=='vulnerable') из {len(true_vuln)} истинных vulnerable:")
    print(f"  только deepseek: {len(only_ds)}  {sorted(only_ds, key=int)}")
    print(f"  только GigaChat: {len(only_gc)}  {sorted(only_gc, key=int)}")
    print(f"  обе модели:      {len(both_hit)}  {sorted(both_hit, key=int)}")
    print(f"  ни одна:         {len(neither)}  {sorted(neither, key=int)}")
    union_hit = ds_hit | gc_hit
    print(f"  объединение (union recall на true positives): {len(union_hit)}/{len(true_vuln)}")
    if len(ds_hit) or len(gc_hit):
        overlap_frac = len(both_hit) / len(union_hit) if union_hit else 0.0
        print(f"  доля пересечения находок (both / union): {overlap_frac:.3f} "
              f"(близко к 1.0 => модели находят одно и то же, ансамбль бессмысленен)")

    # --- построить и оценить файлы вердиктов по каждому правилу ---
    rules = ["union", "intersection", "union_uncertain", "any_signal"]
    summary = {}
    for rule in rules:
        out_verdicts = []
        for doc_id in common_sorted:
            ds_v = ds_verdicts[doc_id]
            combined = _combine(ds_v.verdict, gc_verdicts[doc_id].verdict, rule)
            d = ds_v.model_dump()
            d["verdict"] = combined
            d["rationale"] = f"[ensemble:{rule}] ds={ds_v.verdict} gc={gc_verdicts[doc_id].verdict}"
            out_verdicts.append(d)
        out_path = _BENCH / f"case3_ensemble_{rule}.json"
        out_path.write_text(json.dumps(out_verdicts, ensure_ascii=False, indent=2), encoding="utf-8")

        metrics_out = _ROOT / "cases" / "codereview" / "out" / f"xmodel_eval_metrics_ensemble_{rule}.json"
        proc = subprocess.run(
            [str(_PY), str(_EVALUATE), "--verdicts", str(out_path), "--output", str(metrics_out)],
            capture_output=True, text=True, cwd=str(_ROOT),
        )
        print(f"\n=== правило: {rule} -> {out_path.name} ===")
        print(proc.stdout)
        if proc.returncode != 0:
            print("STDERR:", proc.stderr, file=sys.stderr)
            continue
        m = json.loads(metrics_out.read_text(encoding="utf-8"))
        cm = m["confusion_matrix"]
        tp = cm["pred_vulnerable"]["true_vulnerable"]
        fp = cm["pred_vulnerable"]["true_secure"]
        fn = cm["pred_secure"]["true_vulnerable"]
        tn = cm["pred_secure"]["true_secure"]
        summary[rule] = {
            "precision": m["precision_vulnerable"], "recall": m["recall_vulnerable"],
            "f1": m["f1_vulnerable"], "fpr": m["fpr_vulnerable"],
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_scoreable": m["n_scoreable"],
        }

    print("\n=== сводная таблица правил ансамбля, n=150 ===")
    print(f"{'правило':<18}{'P':>7}{'R':>7}{'F1':>7}{'FPR':>7}{'tp':>5}{'fp':>5}{'fn':>5}{'tn':>5}")
    for rule in rules:
        s = summary.get(rule)
        if not s:
            continue
        print(f"{rule:<18}{s['precision']:>7.3f}{s['recall']:>7.3f}{s['f1']:>7.3f}{s['fpr']:>7.3f}"
              f"{s['tp']:>5}{s['fp']:>5}{s['fn']:>5}{s['tn']:>5}")

    summary_path = _BENCH / "case3_ensemble_gigachat_deepseek_summary.json"
    summary_path.write_text(json.dumps({
        "n_common": len(common_sorted), "n_scoreable": len(scoreable), "n_true_vulnerable": len(true_vuln),
        "agreement_matrix_3x3": matrix, "exact_agreement": n_agree_exact,
        "only_deepseek_hits": sorted(only_ds, key=int), "only_gigachat_hits": sorted(only_gc, key=int),
        "both_hits": sorted(both_hit, key=int), "neither_hits": sorted(neither, key=int),
        "rules": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсводка -> {summary_path}")


if __name__ == "__main__":
    main()
