"""CLI: python run.py --case 1|2|3 [--sample N] [--dry-run] [--model ...]

Находит плагин кейса (`cases.pii`/`cases.guard`/`cases.codereview`, атрибут `PLUGIN`),
прогоняет `core.pipeline.run_pipeline`, выгружает результат в out/case{N}_verdicts.{xlsx,json}.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

from core.export import to_json, to_xlsx
from core.llm import LLMClient, LLMConfig
from core.pipeline import CasePlugin, PipelineContext, run_pipeline

def load_dotenv(path: str = ".env") -> None:
    """Подхватывает ключи из `.env`, не перетирая уже заданные переменные окружения.

    Нужен потому, что ключ удобнее держать в файле (он в .gitignore), чем экспортировать
    в каждой сессии шелла. Формат простой: KEY=value, строки с # игнорируются.
    """
    f = Path(path)
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


CASE_MODULES = {1: "cases.pii", 2: "cases.guard", 3: "cases.codereview"}
CASE_NAMES = {1: "pii", 2: "guard", 3: "codereview"}


def load_plugin(case: int) -> CasePlugin:
    module_name = CASE_MODULES[case]
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise SystemExit(f"не удалось импортировать {module_name}: {e}")
    plugin = getattr(module, "PLUGIN", None)
    if plugin is None:
        raise SystemExit(
            f"{module_name} не экспортирует PLUGIN: CasePlugin — плагин кейса ещё не подключён."
        )
    return plugin


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Прогон конвейера по одному из трёх кейсов ДКБ.")
    p.add_argument("--case", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--sample", type=int, default=None, help="размер выборки (по умолчанию — весь датасет)")
    p.add_argument("--split", default="train", choices=["train", "test"], help="игнорируется кейсом 3")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="без сети, детерминированные заглушки")
    p.add_argument("--model", default="openai/gpt-4o-mini", help="строка модели для LLM-стадии")
    p.add_argument("--backend", default="openai_compat", choices=["openai_compat", "anthropic", "gigachat"])
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-concurrency", type=int, default=4)
    p.add_argument("--cache-path", default="out/llm_cache.sqlite3")
    p.add_argument("--out-dir", default="out")
    p.add_argument("--ids-file", default=None, help="файл со списком doc_id: фиксированный набор для сравнения моделей")
    p.add_argument("--price-in", type=float, default=None, help="цена за 1M входных токенов, USD")
    p.add_argument("--price-out", type=float, default=None, help="цена за 1M выходных токенов, USD")
    p.add_argument("--provider", default=None,
                   help="OpenRouter: фиксация провайдера-исполнителя, через запятую "
                        "(обязательно для воспроизводимого сравнения моделей)")
    p.add_argument("--allow-fallbacks", action="store_true",
                   help="разрешить уход к другому провайдеру при недоступности заданного")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)

    llm_config = LLMConfig(
        model=args.model, backend=args.backend, base_url=args.base_url,
        api_key_env=args.api_key_env, temperature=args.temperature, max_tokens=args.max_tokens,
        max_concurrency=args.max_concurrency, dry_run=args.dry_run, cache_path=args.cache_path,
        price_per_1m_input=args.price_in, price_per_1m_output=args.price_out,
        provider_order=tuple(x.strip() for x in args.provider.split(",")) if args.provider else None,
        allow_fallbacks=args.allow_fallbacks,
    )
    llm = LLMClient(llm_config)
    plugin = load_plugin(args.case)
    config = vars(args)
    # Плагины исторически читают размер выборки под разными именами (`sample` / `n`).
    # Держим оба ключа синхронными, чтобы --sample действовал на любой кейс.
    if args.sample is not None:
        config["n"] = args.sample
    ctx = PipelineContext(case=CASE_NAMES[args.case], config=config, llm=llm)

    verdicts = run_pipeline(plugin, ctx)

    out_dir = Path(args.out_dir)
    xlsx_path = out_dir / f"case{args.case}_verdicts.xlsx"
    json_path = out_dir / f"case{args.case}_verdicts.json"
    to_xlsx(verdicts, str(xlsx_path), columns_fn=plugin.export_columns)
    to_json(verdicts, str(json_path))

    llm.close()

    print(f"case {args.case} ({plugin.name}): {len(verdicts)} вердиктов")
    print(f"  xlsx -> {xlsx_path}")
    print(f"  json -> {json_path}")
    print(f"  llm usage: {json.dumps(llm.usage_summary(), ensure_ascii=False)}")


if __name__ == "__main__":
    main()
