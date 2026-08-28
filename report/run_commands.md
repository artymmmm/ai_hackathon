# Команды для полного прогона

Всё запускается из корня репозитория. Ключи подхватываются из `.env` автоматически (`run.py`
делает это сам). Питон — только `.venv/bin/python`.

---

## СНАЧАЛА ПРОЧИТАТЬ: кейс 3 сейчас прогонится не той конфигурацией

`run.py --case 3` идёт через `cases/codereview/reviewer.py::review_fragments`, а это **базовый
промпт, F1 0.265**. Наша поставочная конфигурация — `cert_only` (промпт
`SYSTEM_PROMPT_SENSITIVE` + блок `cert_rules_block`), **F1 0.386**, и она живёт в
`cases/codereview/reviewer_configs.py`, то есть в экспериментальном модуле, а не в
production-пути.

**Полный прогон кейса 3 до устранения этого расхождения даст результат на 0.121 F1 хуже
измеренного.** Нужно либо перевести `reviewer.py` на конфигурацию `cert_only`, либо прокинуть
её выбор флагом. Работа небольшая, но обязательная.

Второе, что надо знать про кейс 3: стадия `validate` делает **дополнительный вызов LLM**
(`patch_check.second_opinion`) на каждый вердикт `vulnerable` с непустым патчем. Это примерно
18.5% фрагментов, то есть около 3 500 лишних вызовов на полном корпусе — плюс ~15% к смете.
Отключать не нужно, это часть контроля качества патчей, но в бюджет закладывать надо.

---

## Проверка перед тратой денег

Всегда сначала вхолостую — сеть не трогается, заглушки детерминированные:

```bash
.venv/bin/python run.py --case 1 --sample 20 --split test --dry-run
.venv/bin/python run.py --case 2 --sample 20 --split test --dry-run
.venv/bin/python run.py --case 3 --sample 20 --dry-run
```

Затем маленький живой прогон на 20 документах — убедиться, что ключ и провайдер отвечают,
и посмотреть на реальный расход в строке `llm usage`.

---

## Кейс 1 — анонимизация

Полный тестовый сплит — 100 000 документов. Задание объёма не требует, поэтому `--sample`
это законный способ уложиться в бюджет; размер выборки указывается в отчёте.

```bash
# выборка 50 000 (рекомендуется, ~$21)
.venv/bin/python run.py --case 1 --sample 50000 --split test \
  --model deepseek/deepseek-chat \
  --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY \
  --max-concurrency 64 --max-tokens 1024 \
  --cache-path out/llm_cache_final_case1.sqlite3 \
  --out-dir out/final

# весь сплит 100 000 (~$42)
.venv/bin/python run.py --case 1 --split test \
  --model deepseek/deepseek-chat \
  --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY \
  --max-concurrency 64 --max-tokens 1024 \
  --cache-path out/llm_cache_final_case1.sqlite3 \
  --out-dir out/final
```

Результат: `out/final/case1_verdicts.xlsx` и `.json`. Время при conc 64 — порядка двух часов
на полный сплит.

Метрики по выборке считаются отдельно, это не стоит денег:
```bash
.venv/bin/python cases/pii/evaluate.py --n 2000 --split test
```

---

## Кейс 2 — страж инъекций

Дёшево при любом раскладе: офлайн-слой закрывает ~95% потока, в LLM уходит серая зона.
Полный тест — 10 000, примерно $0.06.

```bash
.venv/bin/python run.py --case 2 --split test \
  --model deepseek/deepseek-chat \
  --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY \
  --max-concurrency 64 \
  --cache-path out/llm_cache_final_case2.sqlite3 \
  --out-dir out/final
```

Здесь модель почти не важна (разброс F1 между девятью моделями — 0.0115). Бесплатный
вариант без вывоза данных за периметр:

```bash
.venv/bin/python run.py --case 2 --split test \
  --model GigaChat-2-Max --backend gigachat --api-key-env GIGACHAT_AUTH_KEY \
  --max-concurrency 1 --out-dir out/final
```

---

## Кейс 3 — ревизор кода

**Только после устранения расхождения, описанного вверху.** Корпус — 18 864 фрагмента целиком,
выборка не нужна.

```bash
.venv/bin/python run.py --case 3 \
  --model deepseek/deepseek-chat \
  --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY \
  --max-concurrency 64 --max-tokens 2048 \
  --cache-path out/llm_cache_final_case3.sqlite3 \
  --out-dir out/final
```

Ожидание: около 45 минут при conc 64, порядка $27 с учётом второго мнения по патчам.

Метрики против восстановленной разметки — бесплатно, по готовым вердиктам:
```bash
.venv/bin/python cases/codereview/evaluate.py --verdicts out/final/case3_verdicts.json
```

---

## Если прогон прервался

Кеш ответов в SQLite сохраняет всё по хешу (промпт, модель, параметры). Повторный запуск той
же команды с тем же `--cache-path` **не платит второй раз** за уже полученные ответы — он
добирает недостающее. Поэтому прерывать безопасно, а `--cache-path` для финальных прогонов
задан отдельным файлом на кейс.

## Контроль расхода

Остаток на OpenRouter:
```bash
curl -s https://openrouter.ai/api/v1/credits -H "Authorization: Bearer $OPENROUTER_API_KEY"
```

Реальное списание оказывается примерно на 11% выше расчётного по прайсу (замер 28.08:
расчёт $1.44, списано $1.593) — закладывать этот запас.

## Про off-peak у прямого DeepSeek

У прямого API (`https://api.deepseek.com/v1`, ключ `DEEPSEEK_API_KEY`) есть скидка 50% вне
пика. Пик — 01:00–04:00 и 06:00–10:00 UTC в будни, то есть по Москве 04:00–07:00 и 09:00–13:00;
всё остальное время и выходные целиком идут со скидкой.

Внимание: **в пик прямой API дороже OpenRouter в полтора раза** ($0.44/M вход против $0.2574/M).
Уходить туда имеет смысл только вне пика. И это смена модели: наши метрики кейса 1 измерены
на OpenRouter, на прямом API их надо перемерить.
