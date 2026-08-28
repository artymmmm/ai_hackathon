"""Разовый замер попадания в кеш префикса провайдера (DeepSeek `prompt_cache_hit_tokens`) для
промпта кейса 1, в двух вариантах порядка блоков — текущем и с переставленным вниз переменным
блоком «Контекст документа». Не абстракция про запас — одноразовый измерительный скрипт.

`build_prompt` (текущий, рабочий, не трогаем) берём из `cases.pii.llm_layer` как есть.
Вариант `reordered` строим здесь же перестановкой того же `_PROMPT_TEMPLATE` (читаем модуль,
не копируем текст правил руками — меньше риск разъехаться с оригиналом), `llm_layer.py` не
меняем ни строкой.

Запуск: `.venv/bin/python cases/pii/measure_prefix_cache.py`
Результат: `cases/pii/out/prefix_cache_measure.json`
Кеш LLM-ответов — отдельные свежие SQLite-файлы `out/pii/cache_prefix_{current,reordered}.sqlite3`,
удаляются перед прогоном, чтобы наш локальный кеш не съел вызовы и всё реально ушло в сеть.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from run import load_dotenv  # noqa: E402

load_dotenv()

from core.llm import LLMClient, LLMConfig, LLMJSONError  # noqa: E402
from cases.pii import llm_layer as ll  # noqa: E402

N = 60
SEED = 777
MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com/v1"
DATA_PATH = ROOT / "case 1" / "data" / "test-00000-of-00001.parquet"
OUT_JSON = Path(__file__).resolve().parent / "out" / "prefix_cache_measure.json"

# Блок, перестановку которого измеряем — байт-в-байт совпадает с тем, что стоит в
# `llm_layer._PROMPT_TEMPLATE` между списком лейблов и «Правила:» (см. строки 85-88 файла).
_CONTEXT_BLOCK = "Контекст документа:\n- Тип документа: {document_type}\n- Домен: {domain}\n\n"
_TEXT_MARKER = "Текст документа:"

assert ll._PROMPT_TEMPLATE.count(_CONTEXT_BLOCK) == 1, "шаблон в llm_layer.py разъехался с ожидаемым блоком"
_REORDERED_TEMPLATE = ll._PROMPT_TEMPLATE.replace(_CONTEXT_BLOCK, "", 1).replace(
    _TEXT_MARKER, _CONTEXT_BLOCK + _TEXT_MARKER, 1
)


def build_prompt_reordered(text: str, document_type: str = "", domain: str = "") -> str:
    return _REORDERED_TEMPLATE.format(
        labels=ll._labels_with_examples(),
        document_type=document_type or "unknown",
        domain=domain or "unknown",
        text=text,
    )


def _make_client(cache_path: Path) -> LLMClient:
    if cache_path.exists():
        cache_path.unlink()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = LLMConfig(
        model=MODEL,
        backend="openai_compat",
        base_url=BASE_URL,
        api_key_env="DEEPSEEK_API_KEY",
        max_tokens=1024,
        max_concurrency=4,
        dry_run=False,
        cache_path=str(cache_path),
    )
    return LLMClient(cfg)


def _safe_call(client: LLMClient, prompt: str) -> None:
    """Качество ответа не меряем — важен только usage. JSON-ошибки глушим (не влияют на учёт
    токенов), остальное логируем, но не роняем прогон из-за одного документа."""
    try:
        client.complete_json(prompt, example=ll.EXAMPLE_JSON, system=ll.SYSTEM_PROMPT, model=MODEL)
    except LLMJSONError:
        pass
    except Exception as e:  # noqa: BLE001
        print(f"[warn] {e}", file=sys.stderr)


def run_variant(name: str, prompt_fn, cache_path: Path, prompts: list[str]) -> dict:
    client = _make_client(cache_path)
    started = datetime.now(timezone.utc).isoformat()

    # Прогрев: один вызов последовательно, чтобы отделить неизбежный холодный промах.
    _safe_call(client, prompts[0])
    usage_warmup = client.usage_summary()

    # Остальные 59 — в 4 потока.
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(lambda p: _safe_call(client, p), prompts[1:]))

    usage_full = client.usage_summary()
    finished = datetime.now(timezone.utc).isoformat()
    client.close()

    hit_total = usage_full["prompt_cache_hit_tokens"]
    miss_total = usage_full["prompt_cache_miss_tokens"]
    hit_warmup = usage_warmup["prompt_cache_hit_tokens"]
    miss_warmup = usage_warmup["prompt_cache_miss_tokens"]
    steady_hit_tokens = hit_total - hit_warmup
    steady_total_tokens = (hit_total + miss_total) - (hit_warmup + miss_warmup)
    steady_hit_rate = steady_hit_tokens / steady_total_tokens if steady_total_tokens else 0.0

    # Оценка стоимости своим счётом (в usage_summary().cost_usd — 0, т.к. для deepseek-chat нет
    # цены в core/llm.py и мы её туда не подставляли — это трогало бы общий прайсинг клиента,
    # а он бы всё равно не различил hit/miss). Прайс — пиковый DeepSeek: вход-промах $0.44/M,
    # вход-попадание $0.014/M, выход $1.10/M (допущение — эта цена не зашита в core/llm.py).
    cost_estimate_usd = (
        miss_total * 0.44 + hit_total * 0.014 + usage_full["completion_tokens"] * 1.10
    ) / 1_000_000

    return {
        "variant": name,
        "n_docs": len(prompts),
        "model": MODEL,
        "base_url": BASE_URL,
        "cache_path": str(cache_path),
        "started_utc": started,
        "finished_utc": finished,
        "usage_after_warmup": usage_warmup,
        "usage_after_full_run": usage_full,
        "steady_hit_tokens": steady_hit_tokens,
        "steady_total_cache_tokens": steady_total_tokens,
        "steady_hit_rate": round(steady_hit_rate, 4),
        "cost_estimate_usd": round(cost_estimate_usd, 6),
        "cost_estimate_note": "пиковый прайс DeepSeek $0.44/M miss, $0.014/M hit, $1.10/M output — допущение, не зашито в core/llm.py",
    }


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    sample = df.sample(n=N, random_state=SEED).reset_index(drop=True)

    prompts_current = [
        ll.build_prompt(row["text"], row["document_type"], row["domain"])
        for _, row in sample.iterrows()
    ]
    prompts_reordered = [
        build_prompt_reordered(row["text"], row["document_type"], row["domain"])
        for _, row in sample.iterrows()
    ]

    results = {
        "current": run_variant(
            "current", ll.build_prompt, ROOT / "out" / "pii" / "cache_prefix_current.sqlite3", prompts_current
        ),
        "reordered": run_variant(
            "reordered", build_prompt_reordered, ROOT / "out" / "pii" / "cache_prefix_reordered.sqlite3",
            prompts_reordered,
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
