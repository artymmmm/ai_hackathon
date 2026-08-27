#!/usr/bin/env bash
# Прогон контрольного набора (614 вызовов) для одной модели.
#
#   ./bench.sh <тег> <модель> [доп. флаги run.py ...]
#
# Примеры:
#   ./bench.sh gigachat-max GigaChat-2-Max --backend gigachat --api-key-env GIGACHAT_AUTH_KEY --max-concurrency 1
#   ./bench.sh opus5 anthropic/claude-opus-5 --provider Anthropic
#   ./bench.sh nemotron nvidia/nemotron-3-ultra-550b-a55b:free --max-concurrency 2
#
# Результаты: out/bench/case{1,2,3}_<тег>.json + out/bench/<тег>.log
set -uo pipefail
cd "$(dirname "$0")"

TAG="${1:?нужен тег прогона}"; MODEL="${2:?нужна модель}"; shift 2
PY=.venv/bin/python
LOG="out/bench/${TAG}.log"
mkdir -p out/bench

# Значения по умолчанию — OpenRouter; переопределяются флагами после <модель>.
# cases/pii/evaluate.py не принимает --max-concurrency, поэтому флаг вынесен отдельно
# и передаётся только в run.py.
DEFAULTS=(--base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY)
CONC=(--max-concurrency ${BENCH_CONCURRENCY:-4})
# reasoning-моделям дефолтных 1024 не хватает: весь бюджет уходит в рассуждения.
MAXTOK=(${BENCH_MAX_TOKENS:+--max-tokens $BENCH_MAX_TOKENS})

set -a; [ -f .env ] && . ./.env; set +a

{
echo "=== $TAG | $MODEL | $(date '+%F %T') ==="

echo; echo "--- кейс 3 (150 фрагментов, фиксированный набор) ---"
$PY run.py --case 3 --ids-file out/bench/case3_eval_ids.txt --model "$MODEL" "${DEFAULTS[@]}" "${CONC[@]}" "${MAXTOK[@]}" "$@" \
  && cp out/case3_verdicts.json "out/bench/case3_${TAG}.json" \
  && $PY cases/codereview/evaluate.py --verdicts "out/bench/case3_${TAG}.json"

echo; echo "--- кейс 1 (ablation, n=200) ---"
$PY cases/pii/evaluate.py --n 200 --split test --ablation --model "$MODEL" "${DEFAULTS[@]}" "${CONC[@]}" "${MAXTOK[@]}" "$@"

echo; echo "--- кейс 2 (n=1000) ---"
$PY run.py --case 2 --sample 1000 --split test --model "$MODEL" "${DEFAULTS[@]}" "${CONC[@]}" "${MAXTOK[@]}" "$@" \
  && cp out/case2_verdicts.json "out/bench/case2_${TAG}.json"

echo; echo "=== готово: $TAG ==="
} 2>&1 | tee "$LOG"
