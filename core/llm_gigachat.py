"""GigaChat-бэкенд для `core/llm.py` (`LLMConfig.backend = "gigachat"`).

Опциональный бэкенд — импортируется только при реальном вызове `backend="gigachat"`,
как и `anthropic` в `core/llm.py`. Никакой другой модуль проекта не должен импортировать
этот файл напрямую.

Отличие GigaChat от остальных бэкендов: вместо статичного API-ключа в заголовке нужен
предварительный OAuth-обмен Authorization key → access token с TTL 30 минут (см.
`research/gigachat_setup.md`). Поэтому это отдельный модуль, а не ещё один вариант
`_openai_compat_call` с другим `base_url`.

Переменные окружения (см. `research/gigachat_setup.md`, §7):
- `<api_key_env>`   — Authorization key (`base64(Client ID:Client Secret)`), имя переменной
                      задаётся через `LLMConfig.api_key_env` (по умолчанию для проекта — как
                      и у остальных бэкендов, конкретное имя выбирает вызывающий код).
- `GIGACHAT_SCOPE`      — необязательно, по умолчанию `GIGACHAT_API_PERS` (физлицо).
- `GIGACHAT_BASE_URL`   — необязательно, по умолчанию продовый эндпоинт модели.
- `GIGACHAT_TOKEN_URL`  — необязательно, по умолчанию продовый OAuth-эндпоинт.
- `GIGACHAT_CA_BUNDLE`  — необязательно, путь к сертификату НУЦ Минцифры, если он не
                          установлен в системное/venv-хранилище доверенных сертификатов.
- `GIGACHAT_VERIFY_SSL` — необязательно, `false`/`0`/`no` отключает проверку TLS-сертификата
                          (не рекомендуется, только для разовых экспериментов).
"""

from __future__ import annotations

import json
import os
import ssl
import threading
import time
import urllib.error
import urllib.request
import uuid

from core.llm import LLMError, _TransientError, _estimate_tokens

DEFAULT_TOKEN_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
DEFAULT_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"
DEFAULT_SCOPE = "GIGACHAT_API_PERS"
_TOKEN_REFRESH_MARGIN_S = 60.0  # обновлять токен за минуту до истечения (живёт 30 минут)

_token_lock = threading.Lock()
# authorization_key -> (access_token, время_истечения в time.monotonic())
_token_cache: dict[str, tuple[str, float]] = {}


def _ssl_context() -> ssl.SSLContext:
    """См. `research/gigachat_setup.md`, §3 — проблема сертификата НУЦ Минцифры."""
    ca_bundle = os.environ.get("GIGACHAT_CA_BUNDLE")
    if ca_bundle:
        return ssl.create_default_context(cafile=ca_bundle)
    if os.environ.get("GIGACHAT_VERIFY_SSL", "true").strip().lower() in ("0", "false", "no"):
        return ssl._create_unverified_context()  # осознанно небезопасно, см. research-файл
    return ssl.create_default_context()


def _fetch_access_token(auth_key: str, timeout_s: float) -> tuple[str, float]:
    """Один OAuth-обмен Authorization key -> access token. Не кеширует — вызывать через
    `_get_access_token`."""
    token_url = os.environ.get("GIGACHAT_TOKEN_URL", DEFAULT_TOKEN_URL)
    scope = os.environ.get("GIGACHAT_SCOPE", DEFAULT_SCOPE)
    req = urllib.request.Request(
        url=token_url,
        data=f"scope={scope}".encode("utf-8"),
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": f"Basic {auth_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s, context=_ssl_context()) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        if e.code == 429 or e.code >= 500:
            raise _TransientError(f"GigaChat OAuth HTTP {e.code}: {body_text}")
        raise LLMError(f"GigaChat OAuth HTTP {e.code}: {body_text}")
    except (urllib.error.URLError, TimeoutError) as e:
        raise _TransientError(f"GigaChat OAuth сетевая ошибка: {e}")

    access_token = payload.get("access_token")
    if not access_token:
        raise LLMError(f"GigaChat OAuth не вернул access_token: {payload}")
    # expires_at у GigaChat — unix-миллисекунды; переводим в дедлайн по monotonic-часам,
    # чтобы не зависеть от расхождения системных часов с сервером.
    expires_at_ms = payload.get("expires_at")
    ttl_s = 30 * 60 - _TOKEN_REFRESH_MARGIN_S
    if isinstance(expires_at_ms, (int, float)):
        ttl_s = max(1.0, expires_at_ms / 1000 - time.time() - _TOKEN_REFRESH_MARGIN_S)
    return access_token, time.monotonic() + ttl_s


def _get_access_token(auth_key: str, timeout_s: float) -> str:
    """Возвращает кешированный access token или обменивает Authorization key на новый,
    если кеша нет или он истёк (с запасом `_TOKEN_REFRESH_MARGIN_S`)."""
    with _token_lock:
        cached = _token_cache.get(auth_key)
        if cached and cached[1] > time.monotonic():
            return cached[0]

    token, expires_at = _fetch_access_token(auth_key, timeout_s)

    with _token_lock:
        _token_cache[auth_key] = (token, expires_at)
    return token


def gigachat_call(*, api_key_env: str, model: str, system: str | None, prompt: str,
                   temperature: float, max_tokens: int, timeout_s: float,
                   extra_params: dict) -> tuple[str, int, int]:
    """Совместим по сигнатуре с `_openai_compat_call`/`_anthropic_call` из `core/llm.py`:
    возвращает `(text, prompt_tokens, completion_tokens)`."""
    auth_key = os.environ.get(api_key_env)
    if not auth_key:
        raise LLMError(
            f"GigaChat Authorization key не найден в переменной окружения {api_key_env}. "
            "Получить: GigaChat Studio -> проект GigaChat API -> Настройки API. "
            "Подробности — research/gigachat_setup.md. Либо запускать с --dry-run."
        )

    access_token = _get_access_token(auth_key, timeout_s)

    base_url = os.environ.get("GIGACHAT_BASE_URL", DEFAULT_BASE_URL)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, **extra_params}
    req = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s, context=_ssl_context()) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        if e.code == 401:
            # Токен мог протухнуть между кешем и запросом (гонка/рассинхрон часов) —
            # сбросить кеш, чтобы следующий вызов получил новый токен.
            with _token_lock:
                _token_cache.pop(auth_key, None)
        if e.code == 429 or e.code >= 500:
            retry_after = e.headers.get("Retry-After")
            raise _TransientError(
                f"GigaChat HTTP {e.code}: {body_text}",
                retry_after=float(retry_after) if retry_after else None,
            )
        raise LLMError(f"GigaChat HTTP {e.code} от {base_url}: {body_text}")
    except (urllib.error.URLError, TimeoutError) as e:
        raise _TransientError(f"GigaChat сетевая ошибка: {e}")

    text = payload["choices"][0]["message"]["content"]
    usage = payload.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens") or _estimate_tokens(prompt + (system or ""))
    completion_tokens = usage.get("completion_tokens") or _estimate_tokens(text)
    return text, prompt_tokens, completion_tokens
