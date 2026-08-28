"""Прогон дообученного чекпойнта `mahdin70/CodeBERT-PrimeVul-BigVul` на эталонных 600
фрагментах кейса 3 (offline, CPU, без обращений к платным API).

Модель — MultiTaskCodeBERT (custom класс, `trust_remote_code`) поверх microsoft/codebert-base,
дообучена на mahdin70/balanced_merged_bigvul_primevul. См. проверку утечки:
cases/codereview/check_finetuned_leakage.py -> out/bench/case3_finetuned_leakage.json
(37.5% из 600 совпадают по нормализованному MD5 с обучающим сплитом).

Фрагменты берутся так же, как в run_knowledge_variants_full.py: core.data.load_case3(),
фильтр по unique_id из out/bench/case3_eval600_ids.txt.

Сохраняет сырые вероятности в out/bench/case3_finetuned_600_raw.json (doc_id -> prob_vulnerable),
и вердикты при пороге 0.5 в out/bench/case3_finetuned_600.json (формат Verdict, тот же, что
понимает cases/codereview/evaluate.py). Порог для равного FPR подбирается отдельным скриптом
(threshold_sweep.py), т.к. требует уже посчитанных сырых вероятностей + gold-меток.

Запуск:
    .venv/bin/python cases/codereview/run_finetuned_600.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.data import load_case3  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _ROOT / "cases" / "codereview" / "out" / "hf_model_codebert_bigvul_primevul"
_EVAL600_IDS = _ROOT / "out" / "bench" / "case3_eval600_ids.txt"
_RAW_OUT = _ROOT / "out" / "bench" / "case3_finetuned_600_raw.json"
_VERDICTS_OUT_05 = _ROOT / "out" / "bench" / "case3_finetuned_600.json"


def main() -> None:
    t0 = time.time()
    eval_ids = {x.strip() for x in _EVAL600_IDS.read_text().split() if x.strip()}
    corpus = load_case3()
    corpus["unique_id"] = corpus["unique_id"].astype(str)
    sub = corpus[corpus["unique_id"].isin(eval_ids)].reset_index(drop=True)
    print(f"эталон 600: id в файле {len(eval_ids)}, найдено в корпусе {len(sub)}")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    config = AutoConfig.from_pretrained(str(_MODEL_DIR), trust_remote_code=True)
    model = AutoModel.from_pretrained(str(_MODEL_DIR), config=config, trust_remote_code=True)
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    print(f"модель загружена ({time.time() - t0:.1f}s)")

    raw: dict[str, dict] = {}
    verdicts_05: list[dict] = []

    with torch.no_grad():
        for i, row in sub.iterrows():
            doc_id = str(row["unique_id"])
            code = row["code"]
            inputs = tokenizer(code, return_tensors="pt", padding="max_length",
                                truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs)
            probs = torch.softmax(out["vul_logits"], dim=1)[0]
            prob_vulnerable = float(probs[1].item())
            truncated = len(tokenizer(code)["input_ids"]) > 512

            raw[doc_id] = {"prob_vulnerable": prob_vulnerable, "truncated": truncated,
                            "original_length_chars": len(code)}

            verdict_label = "vulnerable" if prob_vulnerable >= 0.5 else "secure"
            verdicts_05.append({
                "doc_id": doc_id,
                "verdict": verdict_label,
                "confidence": prob_vulnerable if verdict_label == "vulnerable" else 1.0 - prob_vulnerable,
                "action": "block" if verdict_label == "vulnerable" else "pass",
                "evidence": [],
                "rationale": f"CodeBERT-PrimeVul-BigVul: p(vulnerable)={prob_vulnerable:.4f}, порог 0.5",
                "artifacts": {
                    "source": "finetuned_codebert_bigvul_primevul",
                    "prob_vulnerable": prob_vulnerable,
                    "threshold": 0.5,
                    "truncated_to_512_tokens": truncated,
                },
            })

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(sub)} ({time.time() - t0:.0f}s)")

    _RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    _RAW_OUT.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    _VERDICTS_OUT_05.write_text(json.dumps(verdicts_05, ensure_ascii=False, indent=2), encoding="utf-8")

    n_truncated = sum(1 for r in raw.values() if r["truncated"])
    dt = time.time() - t0
    print(f"готово: {len(raw)} фрагментов, обрезано по 512 токенов: {n_truncated}, время: {dt:.0f}s")
    print(f"-> {_RAW_OUT}")
    print(f"-> {_VERDICTS_OUT_05}")


if __name__ == "__main__":
    main()
