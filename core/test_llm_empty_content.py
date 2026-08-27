"""Проверка разбора ответа openai_compat-бэкенда на пустой/отсутствующий content.

Без сети: подменяем urllib.request.urlopen заготовленным HTTP-ответом и зовём
`_openai_compat_call` напрямую. Запуск:
    .venv/bin/python -m pytest core/test_llm_empty_content.py -q
"""

from __future__ import annotations

import io
import json
from contextlib import contextmanager
from unittest import mock

import pytest

from core.llm import LLMError, _TransientError, _openai_compat_call


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@contextmanager
def _mock_payload(payload: dict):
    fake = _FakeResponse(json.dumps(payload).encode("utf-8"))
    with mock.patch("urllib.request.urlopen", return_value=fake):
        yield


def _call():
    return _openai_compat_call(
        base_url="https://openrouter.ai/api/v1", api_key_env="FAKE_KEY_ENV",
        model="openai/gpt-5-nano", system=None, prompt="длинный промпт с кодом",
        temperature=0.0, max_tokens=1024, timeout_s=10.0, extra_params={},
    )


def test_content_none_length_raises_llmerror_with_details(monkeypatch):
    monkeypatch.setenv("FAKE_KEY_ENV", "x")
    payload = {
        "choices": [{"message": {"content": None}, "finish_reason": "length"}],
        "usage": {"completion_tokens": 1024,
                  "completion_tokens_details": {"reasoning_tokens": 1024}},
    }
    with _mock_payload(payload):
        with pytest.raises(LLMError) as exc_info:
            _call()
    msg = str(exc_info.value)
    assert "finish_reason" in msg
    assert "max_tokens" in msg
    assert "1024" in msg  # reasoning_tokens упомянуты


def test_content_none_stop_raises_transient(monkeypatch):
    monkeypatch.setenv("FAKE_KEY_ENV", "x")
    payload = {
        "choices": [{"message": {"content": None}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 0},
    }
    with _mock_payload(payload):
        with pytest.raises(_TransientError):
            _call()


def test_empty_string_content_length_same_as_none(monkeypatch):
    monkeypatch.setenv("FAKE_KEY_ENV", "x")
    payload = {
        "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
        "usage": {"completion_tokens": 1024},
    }
    with _mock_payload(payload):
        with pytest.raises(LLMError) as exc_info:
            _call()
    msg = str(exc_info.value)
    assert "finish_reason" in msg
    assert "max_tokens" in msg


def test_normal_response_parses_as_before(monkeypatch):
    monkeypatch.setenv("FAKE_KEY_ENV", "x")
    payload = {
        "choices": [{"message": {"content": "обычный ответ"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 42, "completion_tokens": 7},
    }
    with _mock_payload(payload):
        text, prompt_tokens, completion_tokens = _call()
    assert text == "обычный ответ"
    assert prompt_tokens == 42
    assert completion_tokens == 7
