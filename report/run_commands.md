# Команды для полного прогона

Всё запускается из корня репозитория. Ключи подхватываются из `.env` автоматически (`run.py`
делает это сам). Питон — только `.venv/bin/python`.

---

## Маршрут: прямой DeepSeek в off-peak

Все три кейса идут на `deepseek-chat` через прямой API `https://api.deepseek.com/v1`
(ключ `DEEPSEEK_API_KEY`) **вне пикового окна**. Пик — 01:00–04:00 и 06:00–10:00 UTC в будни
(по Москве 04:00–07:00 и 09:00–13:00); всё остальное время и выходные целиком идут со скидкой
50%. Скидка подтверждена счётом 28.08 (`out/bench/deepseek_billing_check.json`).

Смета по замеренным токенам с учётом кеша префикса — `out/bench/full_run_cost_offpeak.json`:
кейс 1 (100 000) $12.25, кейс 3 (18 864) $9.55 + $0.70 на второе мнение по патчам,
кейс 2 (10 000) $0.12. **Итого $22.62.**

Обе конфигурации, на которых измерены метрики, уже стоят в production-пути:
`run.py --case 3` идёт через `reviewer_configs.review_fragments_cert_only` (F1 0.386),
промпт кейса 1 — с переставленным вниз блоком «Контекст документа» (F1 0.919, кеш 0.758).

Что надо знать про кейс 3: стадия `validate` делает **дополнительный вызов LLM**
(`patch_check.second_opinion`) на каждый вердикт `vulnerable` с непустым патчем. Это 18.5%
фрагментов, около 3 500 лишних вызовов на полном корпусе — учтено в смете отдельной строкой.
Отключать не нужно, это часть контроля качества патчей.

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

Полный тестовый сплит — 100 000 документов, гоним весь. Метрики поставочного промпта
(«Контекст документа» внизу, кеш префикса 0.758): гибрид F1 **0.919** на n=200 seed 42 и
**0.899** на независимых n=500 seed 7 — оба замера на этом же прямом DeepSeek.

```bash
# весь сплит 100 000 — $12.25 в off-peak с кешем префикса
.venv/bin/python run.py --case 1 --split test \
  --model deepseek-chat \
  --base-url https://api.deepseek.com/v1 --api-key-env DEEPSEEK_API_KEY \
  --max-concurrency 64 --max-tokens 1024 \
  --cache-path out/llm_cache_final_case1.sqlite3 \
  --out-dir out/final
```

Результат: `out/final/case1_verdicts.xlsx` и `.json`. Время при conc 64 — порядка двух часов
на полный сплит. Гоним ЦЕЛИКОМ: с кешем префикса выборка больше не окупается.

Метрики по выборке считаются отдельно, это не стоит денег:
```bash
.venv/bin/python cases/pii/evaluate.py --n 2000 --split test
```

---

## Кейс 2 — страж инъекций

Дёшево при любом раскладе: офлайн-слой закрывает ~95% потока, в LLM уходит серая зона.
Полный тест — 10 000, примерно $0.12.

```bash
.venv/bin/python run.py --case 2 --split test \
  --model deepseek-chat \
  --base-url https://api.deepseek.com/v1 --api-key-env DEEPSEEK_API_KEY \
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

Корпус — 18 864 фрагмента целиком, выборка не нужна. Плагин уже идёт поставочной
конфигурацией `cert_only` (F1 0.386). **`--max-tokens 2048` обязателен**: на 1024 ответы
обрезаются, а на 2048 обрезается 2.4% и уходит в JSON-ретрай.

```bash
.venv/bin/python run.py --case 3 \
  --model deepseek-chat \
  --base-url https://api.deepseek.com/v1 --api-key-env DEEPSEEK_API_KEY \
  --max-concurrency 64 --max-tokens 2048 \
  --cache-path out/llm_cache_final_case3.sqlite3 \
  --out-dir out/final
```

Ожидание: около 45 минут при conc 64, $10.25 в off-peak с учётом второго мнения по патчам.

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

Остаток на прямом DeepSeek (основной счёт маршрута) и на OpenRouter (запасной):
```bash
curl -s https://api.deepseek.com/user/balance -H "Authorization: Bearer $DEEPSEEK_API_KEY"
curl -s https://openrouter.ai/api/v1/credits -H "Authorization: Bearer $OPENROUTER_API_KEY"
```

Реальное списание выше расчётного по прайсу: два замера дали +11% и +6.6%. В сметах заложен
консервативный ×1.11. Биллинг DeepSeek приходит с задержкой — баланс сразу после прогона
не меняется, это нормально.

## Про off-peak у прямого DeepSeek

У прямого API (`https://api.deepseek.com/v1`, ключ `DEEPSEEK_API_KEY`) есть скидка 50% вне
пика. Пик — 01:00–04:00 и 06:00–10:00 UTC в будни, то есть по Москве 04:00–07:00 и 09:00–13:00;
всё остальное время и выходные целиком идут со скидкой.

Скидка подтверждена счётом: за прогон 28.08 в 10:10 UTC списано $0.026 при расчёте по пиковому
прайсу $0.049 — ровно 53% (`out/bench/deepseek_billing_check.json`).

Внимание: **в пик прямой API дороже OpenRouter в полтора раза** ($0.44/M вход против $0.2574/M).
Запускать только вне пика. Метрики кейса 1 (0.919 / 0.899) и кейса 3 (0.386) измерены на этом
же прямом API — перемерять не нужно.
