#!/bin/bash
# Боевой прогон трёх кейсов на прямом DeepSeek в off-peak.
# Порядок: кейс 2 -> кейс 3 -> кейс 1. Кейс 1 последний: его объём — единственная ручка,
# которую можно ужать, если деньги кончатся, не портя два других кейса.
cd /Users/artemmartynov/claude/ai_hackathon
set -a && . ./.env && set +a
PY=.venv/bin/python
COMMON="--model deepseek-chat --base-url https://api.deepseek.com/v1 --api-key-env DEEPSEEK_API_KEY --max-concurrency 64"

bal() { curl -s -H "Authorization: Bearer $DEEPSEEK_API_KEY" https://api.deepseek.com/user/balance \
        | $PY -c "import json,sys;print('БАЛАНС: \$'+json.load(sys.stdin)['balance_infos'][0]['total_balance'])"; }

echo "=== СТАРТ $(date -u '+%F %T UTC') ==="; bal

echo; echo "=== КЕЙС 2 (10 000, test) ==="
$PY run.py --case 2 --split test $COMMON --max-tokens 1024 --cache-path out/prod_case2_cache.sqlite3
bal

echo; echo "=== КЕЙС 3 (18 864, весь корпус) ==="
$PY run.py --case 3 $COMMON --max-tokens 2048 --cache-path out/prod_case3_cache.sqlite3
bal

echo; echo "=== КЕЙС 1 (100 000, test) ==="
$PY run.py --case 1 --sample 100000 --split test $COMMON --max-tokens 1024 --cache-path out/prod_case1_cache.sqlite3
bal

echo; echo "=== ФИНИШ $(date -u '+%F %T UTC') ==="
