"""Поиск ближайших размеченных примеров (retrieval) поверх TF-IDF признаков из `features.py`.

Используется двояко:
  1. Диагностика near-duplicate / кросс-проектного переноса (`retrieval_diagnostics.py`) —
     ЧИСТО офлайн, без LLM: насколько высокое сходство top-1/top-5 соседей и насколько
     результат держится, если пул поиска и оценка взяты из РАЗНЫХ open-source проектов.
  2. Построение few-shot контекста для конфигурации B (retrieval-augmented промпт,
     `reviewer_configs.py`) — k ближайших размеченных примеров с их вердиктом/CWE кладутся
     в промпт.

Дисциплина утечки (одна и та же во всех использованиях):
  - индекс строится ТОЛЬКО по обучающему пулу (никогда не включает сам анализируемый/eval
    фрагмент — гарантируется на уровне вызывающего кода: eval исключается из пула так же,
    как в `knn_baseline.py`, и здесь же добавлена явная проверка по unique_id, чтобы
    фрагмент не мог найти сам себя в соседях, даже если случайно оказался в пуле).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from cases.codereview.features import CodeFeaturizer, truncate

NEIGHBOR_CODE_CHARS = 1200  # обрезка кода соседа в few-shot промпте — компактность важнее полноты


class NearestExampleRetriever:
    def __init__(self, pool_df: pd.DataFrame):
        """pool_df: колонки unique_id, code, label (0/1), опционально cwe_id, source_project."""
        self.pool_df = pool_df.reset_index(drop=True)
        self.featurizer = CodeFeaturizer()
        self.X = self.featurizer.fit_transform(self.pool_df["code"].tolist())
        self._nn = NearestNeighbors(metric="cosine", algorithm="brute")
        self._nn.fit(self.X)
        self._id_to_pos = {uid: i for i, uid in enumerate(self.pool_df["unique_id"])}

    def query_vector(self, code: str) -> "np.ndarray":
        return self.featurizer.transform([code])

    def query(self, code: str, k: int = 5, exclude_unique_id: int | None = None) -> list[dict]:
        """Топ-k ближайших соседей по косинусному сходству. `exclude_unique_id` — страховка:
        если сам анализируемый фрагмент случайно оказался бы в пуле (не должно случаться при
        правильном исключении eval из пула), он не попадёт в свои же соседи."""
        vec = self.query_vector(code)
        k_query = k + (1 if exclude_unique_id is not None else 0)
        k_query = min(k_query, len(self.pool_df))
        dist, idx = self._nn.kneighbors(vec, n_neighbors=k_query)
        results = []
        for d, i in zip(dist[0], idx[0]):
            row = self.pool_df.iloc[i]
            if exclude_unique_id is not None and int(row["unique_id"]) == int(exclude_unique_id):
                continue
            results.append({
                "unique_id": int(row["unique_id"]),
                "similarity": round(1.0 - float(d), 4),
                "label": int(row["label"]) if pd.notna(row.get("label")) else None,
                "cwe_id": row.get("cwe_id") if pd.notna(row.get("cwe_id")) else None,
                "source_project": row.get("source_project") if pd.notna(row.get("source_project")) else None,
                "code": row["code"],
            })
            if len(results) >= k:
                break
        return results

    def query_many_similarities(self, eval_df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
        """Для диагностики: top-1 и средняя top-k схожесть для каждого eval-фрагмента.
        Явная проверка нулевого пересечения id пула и eval (see assert)."""
        overlap = set(self.pool_df["unique_id"]) & set(eval_df["unique_id"])
        assert not overlap, f"УТЕЧКА в retrieval-диагностике: {len(overlap)} id пересекаются"

        rows = []
        for _, r in eval_df.iterrows():
            vec = self.query_vector(r["code"])
            k_eff = min(k, len(self.pool_df))
            dist, idx = self._nn.kneighbors(vec, n_neighbors=k_eff)
            sims = 1.0 - dist[0]
            nearest_pos = idx[0][0]
            nearest_row = self.pool_df.iloc[nearest_pos]
            rows.append({
                "unique_id": int(r["unique_id"]),
                "top1_similarity": round(float(sims[0]), 4),
                "topk_mean_similarity": round(float(sims.mean()), 4),
                "top1_neighbor_id": int(nearest_row["unique_id"]),
                "top1_neighbor_label": int(nearest_row["label"]) if pd.notna(nearest_row.get("label")) else None,
                "top1_neighbor_project": nearest_row.get("source_project"),
                "own_project": r.get("source_project"),
            })
        return pd.DataFrame(rows)


def format_neighbors_block(neighbors: list[dict]) -> str:
    """Компактный текстовый блок few-shot примеров для промпта конфигурации B."""
    if not neighbors:
        return ""
    lines = [
        f"Похожие размеченные примеры из обучающего корпуса (top-{len(neighbors)} по "
        "сходству кода, НЕ гарантия — используй как справочный контекст, не копируй вердикт "
        "механически, если сам код перед тобой отличается по сути):"
    ]
    for i, n in enumerate(neighbors, 1):
        verdict = "vulnerable" if n["label"] == 1 else "secure"
        cwe = f", {n['cwe_id']}" if n.get("cwe_id") else ""
        lines.append(
            f"\n[Пример {i}, сходство={n['similarity']:.2f}, вердикт={verdict}{cwe}]\n"
            f"{truncate(n['code'], NEIGHBOR_CODE_CHARS)}"
        )
    return "\n".join(lines)
