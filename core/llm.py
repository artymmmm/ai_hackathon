"""Единственная точка контакта с LLM во всём проекте.

Никакой другой модуль не импортирует ни `anthropic`, ни `openai`, ни делает HTTP-запросы
к LLM-провайдеру напрямую — только через `LLMClient` отсюда.

- Базовый транспорт — OpenAI-совместимый `/v1/chat/completions` (цель — OpenRouter).
  Anthropic SDK и GigaChat — опциональные бэкенды за тем же интерфейсом
  (`backend="anthropic"`, `backend="gigachat"`; последний — `core/llm_gigachat.py`,
  см. `research/gigachat_setup.md`).
- Модель — строка из конфига (`LLMConfig.model`), не хардкод.
- Кеш ответов в SQLite по хешу (промпт, модель, параметры): повторный прогон — бесплатно и мгновенно.
- `LLMConfig.dry_run=True` — детерминированные заглушки, ноль сетевых вызовов.
- Учёт токенов и стоимости накапливается в `LLMClient.usage` по всему прогону.
- Ретраи с экспоненциальной задержкой (+ уважение `Retry-After` на 429), ограничение
  параллелизма через `complete_many`.
- `complete_json`: просим JSON в промпте, строго парсим, повторяем при провале.
  Нативный structured output намеренно не используется — он есть не у всех провайдеров.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal


class LLMError(RuntimeError):
    """Неустранимая ошибка вызова LLM (после исчерпания ретраев или явный отказ)."""


class LLMJSONError(LLMError):
    """LLM так и не вернула валидный JSON после всех повторов."""


class _TransientError(RuntimeError):
    """Внутренний сигнал ретраябельной ошибки (429, 5xx, таймаут, сеть). Наружу не выходит."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


# Грубая таблица цен (USD за 1M токенов: input, output) для оценки стоимости прогона.
# Не источник истины — прикидка порядка величины для отчёта. Актуализировать по мере надобности,
# либо переопределить через LLMConfig.price_per_1m_input/output.
_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4.1": (2.00, 8.00),
    "openai/gpt-4.1-mini": (0.40, 1.60),
    "anthropic/claude-opus-4.5": (5.00, 25.00),
    "anthropic/claude-sonnet-4.5": (3.00, 15.00),
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
}


@dataclass
class LLMConfig:
    model: str = "openai/gpt-4o-mini"
    backend: Literal["openai_compat", "anthropic", "gigachat"] = "openai_compat"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_s: float = 60.0
    max_retries: int = 5
    max_concurrency: int = 4
    dry_run: bool = True
    cache_path: str = "out/llm_cache.sqlite3"
    price_per_1m_input: float | None = None
    price_per_1m_output: float | None = None
    provider_order: tuple[str, ...] | None = None
    """OpenRouter: жёсткий порядок провайдеров-исполнителей.

    Без фиксации OpenRouter волен отдать запросы одной и той же «модели» разным провайдерам,
    у которых отличаются квантизация и настройки — тогда сравнение моделей между собой
    измеряет не модель, а маршрутизацию. Для воспроизводимого бенчмарка задавать обязательно.
    Игнорируется всеми бэкендами кроме `openai_compat`.
    """
    allow_fallbacks: bool = True
    """False вместе с `provider_order` — запрет уходить к другому провайдеру при недоступности."""


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    cached: bool
    latency_ms: float
    raw: dict = field(default_factory=dict)


@dataclass
class UsageTotals:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    prompt_cache_hit_tokens: int = 0
    """Токены промпта, попавшие в кеш префикса ПРОВАЙДЕРА (напр. DeepSeek). Не путать с
    `cache_hits` — это наш собственный SQLite-кеш ответов, совсем другая сущность."""
    prompt_cache_miss_tokens: int = 0

    def as_dict(self) -> dict:
        cache_total = self.prompt_cache_hit_tokens + self.prompt_cache_miss_tokens
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "prompt_cache_hit_tokens": self.prompt_cache_hit_tokens,
            "prompt_cache_miss_tokens": self.prompt_cache_miss_tokens,
            "prompt_cache_hit_rate": (
                round(self.prompt_cache_hit_tokens / cache_total, 4) if cache_total else 0.0
            ),
        }


def _estimate_tokens(text: str) -> int:
    """Грубая оценка (~4 символа/токен), когда провайдер не вернул usage."""
    return max(1, len(text) // 4)


def _cache_key(model: str, system: str | None, prompt: str, temperature: float,
               max_tokens: int, extra: dict) -> str:
    payload = json.dumps(
        {"model": model, "system": system, "prompt": prompt, "temperature": temperature,
         "max_tokens": max_tokens, "extra": extra},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _ResponseCache:
    """SQLite-кеш ответов по хешу (промпт, модель, параметры)."""

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "key TEXT PRIMARY KEY, response TEXT NOT NULL, created_at REAL NOT NULL)"
        )
        self._conn.commit()
        self._lock = threading.Lock()

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, response: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, response, created_at) VALUES (?, ?, ?)",
                (key, json.dumps(response, ensure_ascii=False), time.time()),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def _openai_compat_call(*, base_url: str, api_key_env: str, model: str, system: str | None,
                         prompt: str, temperature: float, max_tokens: int, timeout_s: float,
                         extra_params: dict) -> tuple[str, int, int, int, int]:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise LLMError(
            f"API-ключ не найден в переменной окружения {api_key_env}. "
            "Либо задать ключ, либо запускать с --dry-run."
        )
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, **extra_params}
    req = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        if e.code == 429 or e.code >= 500:
            retry_after = e.headers.get("Retry-After")
            raise _TransientError(
                f"HTTP {e.code}: {body_text}",
                retry_after=float(retry_after) if retry_after else None,
            )
        raise LLMError(f"HTTP {e.code} от {base_url}: {body_text}")
    except (urllib.error.URLError, TimeoutError) as e:
        raise _TransientError(f"сетевая ошибка: {e}")

    choice = payload["choices"][0]
    text = choice["message"].get("content")
    usage = payload.get("usage") or {}
    if not text:
        # Reasoning-модели (gpt-5-nano и т.п.) при finish_reason == "length" могут потратить
        # весь max_tokens на внутренние рассуждения и вернуть content: null — это не сетевая
        # ошибка и не брак модели по существу, повтор с теми же max_tokens ничего не изменит.
        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            reasoning_note = (
                f", из них на рассуждения ушло {reasoning_tokens}" if reasoning_tokens is not None else ""
            )
            raise LLMError(
                f"пустой content у модели {model}: finish_reason=length, "
                f"весь max_tokens={max_tokens} израсходован до генерации ответа"
                f"{reasoning_note}. Повтор с тем же max_tokens не поможет — увеличить max_tokens."
            )
        # Пустой content при любом другом finish_reason похож на сбой провайдера — ретраим.
        raise _TransientError(f"пустой content у модели {model}, finish_reason={finish_reason!r}")
    prompt_tokens = usage.get("prompt_tokens") or _estimate_tokens(prompt + (system or ""))
    completion_tokens = usage.get("completion_tokens") or _estimate_tokens(text)
    # DeepSeek-специфичные поля учёта попаданий в кеш префикса. У провайдеров, которые их не
    # присылают (OpenRouter, большинство прочих), остаются нулями — ничего не ломаем.
    cache_hit_tokens = usage.get("prompt_cache_hit_tokens") or 0
    cache_miss_tokens = usage.get("prompt_cache_miss_tokens") or 0
    return text, prompt_tokens, completion_tokens, cache_hit_tokens, cache_miss_tokens


def _anthropic_call(*, api_key_env: str, model: str, system: str | None, prompt: str,
                     temperature: float, max_tokens: int, timeout_s: float,
                     extra_params: dict) -> tuple[str, int, int]:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise LLMError(f"API-ключ не найден в переменной окружения {api_key_env}.")
    import anthropic  # опциональный бэкенд — импортируется только при реальном использовании

    client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s)
    try:
        resp = client.messages.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            system=system or "", messages=[{"role": "user", "content": prompt}],
            **extra_params,
        )
    except anthropic.RateLimitError as e:
        raise _TransientError(f"rate limit: {e}")
    except anthropic.APIStatusError as e:
        if e.status_code >= 500:
            raise _TransientError(f"HTTP {e.status_code}: {e}")
        raise LLMError(f"HTTP {e.status_code}: {e}")
    except anthropic.APIConnectionError as e:
        raise _TransientError(f"сетевая ошибка: {e}")

    text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
    return text, resp.usage.input_tokens, resp.usage.output_tokens


def _call_with_retries(fn: Callable[[], tuple], max_retries: int) -> tuple:
    attempt = 0
    while True:
        try:
            return fn()
        except _TransientError as e:
            attempt += 1
            if attempt > max_retries:
                raise LLMError(f"исчерпаны ретраи ({max_retries}): {e}") from e
            delay = e.retry_after if e.retry_after else min(60.0, (2 ** attempt) + random.random())
            time.sleep(delay)


def _dry_run_text(model: str, system: str | None, prompt: str, extra: dict) -> str:
    h = hashlib.sha256(
        f"{model}|{system}|{prompt}|{json.dumps(extra, sort_keys=True)}".encode("utf-8")
    ).hexdigest()
    return f"[dry-run stub {h[:12]}]"


def _seed_from(*parts: str) -> int:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _fill_example(value: Any, rng: random.Random, key: str = "") -> Any:
    """Заполняет пример детерминированной заглушкой той же формы (для dry-run JSON)."""
    if isinstance(value, bool):
        return rng.choice([True, False])
    if isinstance(value, int):
        return rng.randint(0, 100)
    if isinstance(value, float):
        return round(rng.uniform(0.0, 1.0), 3)
    if isinstance(value, str):
        return rng.choice([value, f"dryrun_{key or 'value'}"])
    if isinstance(value, list):
        return [_fill_example(v, rng, key) for v in value]
    if isinstance(value, dict):
        return {k: _fill_example(v, rng, k) for k, v in value.items()}
    return value


def _extract_json(text: str) -> dict:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("в ответе не найден JSON-объект")
    candidate = stripped[start:end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"невалидный JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise ValueError("верхний уровень JSON должен быть объектом")
    return parsed


class LLMClient:
    """Провайдер-агностичный клиент. См. докстринг модуля."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._cache: _ResponseCache | None = None if config.dry_run else _ResponseCache(config.cache_path)
        self.usage = UsageTotals()

    def _price(self, model: str) -> tuple[float, float]:
        if self.config.price_per_1m_input is not None and self.config.price_per_1m_output is not None:
            return self.config.price_per_1m_input, self.config.price_per_1m_output
        return _PRICING_PER_1M.get(model, (0.0, 0.0))

    def complete(self, prompt: str, *, system: str | None = None, model: str | None = None,
                 temperature: float | None = None, max_tokens: int | None = None,
                 use_cache: bool = True, **extra_params: Any) -> LLMResponse:
        model = model or self.config.model
        temperature = self.config.temperature if temperature is None else temperature
        max_tokens = self.config.max_tokens if max_tokens is None else max_tokens

        if self.config.dry_run:
            t0 = time.monotonic()
            text = _dry_run_text(model, system, prompt, extra_params)
            latency_ms = (time.monotonic() - t0) * 1000
            prompt_tokens = _estimate_tokens(prompt + (system or ""))
            completion_tokens = _estimate_tokens(text)
            self.usage.calls += 1
            self.usage.prompt_tokens += prompt_tokens
            self.usage.completion_tokens += completion_tokens
            self.usage.total_tokens += prompt_tokens + completion_tokens
            return LLMResponse(
                text=text, model=model, prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens,
                cost_usd=0.0, cached=False, latency_ms=latency_ms, raw={"dry_run": True},
            )

        # Спецификация провайдера идёт в тело запроса И в ключ кеша: ответы разных
        # провайдеров одной модели нельзя считать взаимозаменяемыми.
        if self.config.backend == "openai_compat" and self.config.provider_order:
            extra_params = {
                **extra_params,
                "provider": {
                    "order": list(self.config.provider_order),
                    "allow_fallbacks": self.config.allow_fallbacks,
                },
            }

        assert self._cache is not None
        key = _cache_key(model, system, prompt, temperature, max_tokens, extra_params)
        if use_cache:
            cached = self._cache.get(key)
            if cached is not None:
                self.usage.calls += 1
                self.usage.cache_hits += 1
                self.usage.prompt_tokens += cached["prompt_tokens"]
                self.usage.completion_tokens += cached["completion_tokens"]
                self.usage.total_tokens += cached["prompt_tokens"] + cached["completion_tokens"]
                return LLMResponse(
                    text=cached["text"], model=model, prompt_tokens=cached["prompt_tokens"],
                    completion_tokens=cached["completion_tokens"],
                    total_tokens=cached["prompt_tokens"] + cached["completion_tokens"],
                    cost_usd=0.0, cached=True, latency_ms=0.0, raw={"cached": True},
                )

        t0 = time.monotonic()
        if self.config.backend == "openai_compat":
            def call() -> tuple[str, int, int, int, int]:
                return _openai_compat_call(
                    base_url=self.config.base_url, api_key_env=self.config.api_key_env,
                    model=model, system=system, prompt=prompt, temperature=temperature,
                    max_tokens=max_tokens, timeout_s=self.config.timeout_s, extra_params=extra_params,
                )
        elif self.config.backend == "anthropic":
            def call() -> tuple[str, int, int]:
                return _anthropic_call(
                    api_key_env=self.config.api_key_env, model=model, system=system, prompt=prompt,
                    temperature=temperature, max_tokens=max_tokens, timeout_s=self.config.timeout_s,
                    extra_params=extra_params,
                )
        elif self.config.backend == "gigachat":
            def call() -> tuple[str, int, int]:
                from core.llm_gigachat import gigachat_call  # опциональный бэкенд, лениво
                return gigachat_call(
                    api_key_env=self.config.api_key_env, model=model, system=system, prompt=prompt,
                    temperature=temperature, max_tokens=max_tokens, timeout_s=self.config.timeout_s,
                    extra_params=extra_params,
                )
        else:
            raise LLMError(f"неизвестный backend: {self.config.backend}")

        result = _call_with_retries(call, self.config.max_retries)
        # openai_compat возвращает 5-элементный кортеж (с полями кеша провайдера), остальные
        # бэкенды (anthropic, gigachat) — исходный 3-элементный. Нормализуем без их правки.
        if len(result) == 5:
            text, prompt_tokens, completion_tokens, cache_hit_tokens, cache_miss_tokens = result
        else:
            text, prompt_tokens, completion_tokens = result
            cache_hit_tokens, cache_miss_tokens = 0, 0
        latency_ms = (time.monotonic() - t0) * 1000

        price_in, price_out = self._price(model)
        cost = (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000

        if use_cache:
            self._cache.put(key, {"text": text, "prompt_tokens": prompt_tokens,
                                   "completion_tokens": completion_tokens})

        self.usage.calls += 1
        self.usage.prompt_tokens += prompt_tokens
        self.usage.completion_tokens += completion_tokens
        self.usage.total_tokens += prompt_tokens + completion_tokens
        self.usage.cost_usd += cost
        self.usage.prompt_cache_hit_tokens += cache_hit_tokens
        self.usage.prompt_cache_miss_tokens += cache_miss_tokens

        return LLMResponse(
            text=text, model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens, cost_usd=cost, cached=False,
            latency_ms=latency_ms, raw={},
        )

    def complete_many(self, requests: list[dict], *, max_workers: int | None = None) -> list[LLMResponse]:
        """requests — список kwargs для `complete()`. Параллелизм ограничен max_concurrency."""
        workers = max(1, max_workers or self.config.max_concurrency)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(lambda kw: self.complete(**kw), requests))

    def complete_json(self, prompt: str, *, example: dict, system: str | None = None,
                       model: str | None = None, temperature: float | None = None,
                       max_tokens: int | None = None, max_json_retries: int = 2,
                       use_cache: bool = True, **extra_params: Any) -> dict:
        """Просит JSON по форме `example`, строго парсит, повторяет запрос при провале.

        `example` — JSON-пример желаемой структуры (ключи и типы, значения не важны).
        В dry-run возвращает детерминированную заглушку той же формы, без сети.
        """
        if self.config.dry_run:
            seed = _seed_from(model or self.config.model, system or "", prompt)
            result = _fill_example(example, random.Random(seed))
            result_text = json.dumps(result, ensure_ascii=False)
            prompt_tokens = _estimate_tokens(prompt + (system or ""))
            completion_tokens = _estimate_tokens(result_text)
            self.usage.calls += 1
            self.usage.prompt_tokens += prompt_tokens
            self.usage.completion_tokens += completion_tokens
            self.usage.total_tokens += prompt_tokens + completion_tokens
            return result

        schema_hint = json.dumps(example, ensure_ascii=False, indent=2)
        instruction = (
            "Ответ верни СТРОГО в виде одного JSON-объекта, без markdown-обёртки и пояснений вне JSON. "
            f"Пример структуры и типов (значения не копировать буквально):\n{schema_hint}"
        )
        full_system = f"{system}\n\n{instruction}" if system else instruction

        current_prompt = prompt
        last_err: Exception | None = None
        for attempt in range(max_json_retries + 1):
            resp = self.complete(
                current_prompt, system=full_system, model=model, temperature=temperature,
                max_tokens=max_tokens, use_cache=use_cache and attempt == 0, **extra_params,
            )
            try:
                return _extract_json(resp.text)
            except ValueError as e:
                last_err = e
                current_prompt = (
                    f"{prompt}\n\n[Предыдущий ответ не был валидным JSON: {e}. "
                    "Верни ТОЛЬКО валидный JSON-объект по формату выше.]"
                )
        raise LLMJSONError(
            f"не удалось получить валидный JSON за {max_json_retries + 1} попыток: {last_err}"
        )

    def usage_summary(self) -> dict:
        return self.usage.as_dict()

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()
