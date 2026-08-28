"""Доказательство без единого сетевого вызова: production-путь плагина кейса 3 после правки
строит ПОБАЙТОВО те же (system, prompt, параметры), что измеренная конфигурация cert_only.
Критерий — ключ кеша `core.llm._cache_key` совпадает с уже лежащим в кеше прогонов на 600.
Код из датасета не исполняется, только подставляется в промпт (CLAUDE.md)."""
import sys, json, sqlite3, collections
sys.path.insert(0, '.')
from core.llm import _cache_key, _extract_json
from core.pipeline import PipelineContext
from cases.codereview.reviewer_configs import (
    SYSTEM_PROMPT_SENSITIVE, build_prompt, _JSON_EXAMPLE_SENSITIVE, _to_verdict_ext)
import cases.codereview as plugin_mod

class _FakeLLM:
    """Подменяет ctx.llm: ничего не шлёт, только фиксирует (system, prompt) production-пути."""
    class config: max_concurrency = 1
    def __init__(self): self.seen = {}
    def complete_json(self, prompt, *, example, system=None, **kw):
        self.seen[len(self.seen)] = (system, prompt, example)
        raise RuntimeError("no-network-probe")

def full_system(system, example):
    schema = json.dumps(example, ensure_ascii=False, indent=2)
    instr = ("Ответ верни СТРОГО в виде одного JSON-объекта, без markdown-обёртки и пояснений вне JSON. "
             f"Пример структуры и типов (значения не копировать буквально):\n{schema}")
    return f"{system}\n\n{instr}"

ctx = PipelineContext(case="codereview", config={
    "ids_file": "out/bench/case3_eval600_ids.txt", "sample": None, "seed": 42}, llm=_FakeLLM())
records = plugin_mod.PLUGIN.load(ctx)
print("записей через production load():", len(records))

# 1) прогоняем production-стадию llm через подменённого клиента — собираем её промпты
plugin_mod.PLUGIN.llm(records, ctx)
seen = list(ctx.llm.seen.values())
print("промптов, построенных production-путём:", len(seen))

# 2) сверяем каждый с кешем измеренного прогона cert_only
con = sqlite3.connect("out/llm_cache_case3_ablation.sqlite3")
hit = 0
unparsed = 0
verdict_by_id = {}
for rec, (system, prompt, example) in zip(records, seen):
    key = _cache_key("deepseek-chat", full_system(system, example), prompt, 0.0, 2048, {})
    row = con.execute("select response from cache where key=?", (key,)).fetchone()
    if row:
        hit += 1
        try:
            parsed = _extract_json(json.loads(row[0])["text"])
        except ValueError:
            unparsed += 1
            continue
        v = _to_verdict_ext(rec["doc_id"], parsed, full_code=rec["code"],
                            truncated=False, original_length=len(rec["code"]))
        verdict_by_id[rec["doc_id"]] = v.verdict
print(f"ключ кеша совпал (промпт побайтово тот же): {hit} / {len(seen)}")
print(f"из них ответ обрезан по max_tokens и не парсится: {unparsed}")

ref = {d["doc_id"]: d["verdict"] for d in json.load(open("out/bench/case3_cert_only_600.json"))}
common = [k for k in verdict_by_id if k in ref]
agree = sum(1 for k in common if verdict_by_id[k] == ref[k])
print(f"вердикты сошлись с эталоном cert_only: {agree} / {len(common)}")
print("эталон  :", collections.Counter(ref[k] for k in common))
print("из кеша :", collections.Counter(verdict_by_id[k] for k in common))
