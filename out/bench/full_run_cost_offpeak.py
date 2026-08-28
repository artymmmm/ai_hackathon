"""Смета полного прогона: прямой DeepSeek, off-peak, с кешем префикса.
Все входные числа — из измеренных артефактов, источники указаны рядом."""
import json

# --- прайс. Пиковый — допущение из cases/pii/measure_prefix_cache.py; off-peak = -50%.
PEAK = {"miss": 0.44, "hit": 0.014, "out": 1.10}          # USD / 1M токенов
OFF  = {k: v / 2 for k, v in PEAK.items()}
# Калибровка по единственному живому счёту: списано $0.026003424 при расчёте $0.024389
# (out/bench/deepseek_billing_check.json).
CAL = 1.0662

def cost(hit, miss, out, price=OFF, cal=CAL):
    return (hit * price["hit"] + miss * price["miss"] + out * price["out"]) / 1e6 * cal

rows = []

# ---------------- КЕЙС 1: 100 000 документов, полный test ----------------
# prompt 1351 / completion 65 на документ — out/pii/ablation_metrics_after_fixes.json
# (400 вызовов / 200 док) и ablation_metrics_n500_seed7_after_fixes.json (1002 / 500 док).
# Кешируемый префикс ИЗМЕРЕН: cases/pii/out/prefix_cache_measure.json.
N1, P1, C1 = 100_000, 1351, 65
for name, hit_tok in (("кейс 1, 100k, текущий порядок блоков", 640),
                      ("кейс 1, 100k, «Контекст документа» вниз", 1024)):
    rows.append((name, N1, hit_tok * N1, (P1 - hit_tok) * N1, C1 * N1))
rows.append(("кейс 1, 100k, БЕЗ кеша (для сравнения)", N1, 0, P1 * N1, C1 * N1))

# ---------------- КЕЙС 3: 18 864 фрагмента ----------------
# prompt 2388 / completion 577 на фрагмент — STATE.md, сверено с
# out/bench/case3_binary_cert_only_600_usage.json (2401 / фрагмент).
# Кешируемый префикс НЕ измерен, ОЦЕНКА: system-сообщение 5112 символов при 2.94 симв/токен
# (7014 симв. промпта на 2388 токенов, выборка 300 фрагментов корпуса) -> ~1739 ток,
# округление вниз до кратности 64 -> 1728.
N3, P3, C3 = 18_864, 2388, 577
rows.append(("кейс 3, 18 864, cert_only, кеш как есть", N3, 1728 * N3, (P3 - 1728) * N3, C3 * N3))
rows.append(("кейс 3, 18 864, cert_only, doc_id вниз", N3, 1984 * N3, (P3 - 1984) * N3, C3 * N3))
rows.append(("кейс 3, 18 864, cert_only, БЕЗ кеша", N3, 0, P3 * N3, C3 * N3))
# второе мнение по патчам: 18.5% фрагментов (111/600 в out/bench/case3_cert_only_600.json),
# промпт 2125 симв. -> ~723 ток (system 816 симв. -> 256 ток кеша), ответ ~150 ток.
N3B = round(N3 * 0.185)
rows.append(("кейс 3, второе мнение по патчам (18.5%)", N3B, 256 * N3B, (723 - 256) * N3B, 150 * N3B))

# ---------------- КЕЙС 2: 10 000 документов ----------------
# 44 вызова и 45 052 токена на 1000 документов (STATE.md); доля completion 12% взята из
# out/guard/case2_llm_on_errors.json. Кеш не учитываем — величина под порогом значимости.
N2 = 10_000
t2 = 45_052 / 1000 * N2
rows.append(("кейс 2, 10 000, серая зона", N2, 0, round(t2 * 0.88), round(t2 * 0.12)))

out = []
for name, n, hit, miss, comp in rows:
    out.append({"конфигурация": name, "n": n,
                "hit_ток": hit, "miss_ток": miss, "out_ток": comp,
                "off_peak_usd": round(cost(hit, miss, comp), 2),
                "peak_usd": round(cost(hit, miss, comp, PEAK), 2)})
    print(f"{name:<44} off-peak ${out[-1]['off_peak_usd']:>6.2f}   пик ${out[-1]['peak_usd']:>6.2f}")

json.dump({"прайс_пик_usd_за_1m": PEAK, "прайс_offpeak_usd_за_1m": OFF,
           "калибровка_по_счёту": CAL, "строки": out},
          open("out/bench/full_run_cost_offpeak.json", "w"), ensure_ascii=False, indent=1)
