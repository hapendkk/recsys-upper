from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class TwoTowerHybridRecommender:
    """Two-Tower retrieval scores reranked with classical content/CF/popularity signals."""

    name = "two_tower_hybrid"

    def __init__(
        self,
        two_tower: Any,
        recommenders: dict[str, Any],
        two_tower_weight: float = 0.30,
        content_weight: float = 0.35,
        item_weight: float = 0.25,
        popularity_weight: float = 0.10,
    ):
        self.two_tower = two_tower
        self.content = recommenders["content_tfidf"]
        self.item_knn = recommenders["item_knn"]
        self.popularity = recommenders["popularity"]
        self.two_tower_weight = two_tower_weight
        self.content_weight = content_weight
        self.item_weight = item_weight
        self.popularity_weight = popularity_weight
        self.n_items = int(self.popularity.n_items)

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float32)
        max_score = float(scores.max()) if scores.size else 0.0
        if max_score > 0:
            scores = scores / max_score
        return scores

    def _two_tower_scores(self, user_ids: np.ndarray, histories: list[list[int]]) -> np.ndarray:
        import torch

        with torch.no_grad():
            user_emb = self.two_tower.encode_users(user_ids, histories)
            return (user_emb @ self.two_tower.item_embeddings.T).astype(np.float32)

    def _combined_scores(self, tt_row: np.ndarray, history_items: list[int], weights: list[float]) -> np.ndarray:
        content = self._normalize(self.content.score_items(history_items, weights))
        item = self._normalize(self.item_knn.score_items(history_items, weights))
        pop = self._normalize(self.popularity.score_items(history_items, weights))
        tt = self._normalize(tt_row)
        total = self.two_tower_weight + self.content_weight + self.item_weight + self.popularity_weight
        return (
            self.two_tower_weight * tt
            + self.content_weight * content
            + self.item_weight * item
            + self.popularity_weight * pop
        ) / max(total, 1e-9)

    def recommend_many(self, requests: list[tuple[int, list[tuple[int, float]], int]], k: int = 10) -> list[np.ndarray]:
        user_ids = np.asarray([req[0] for req in requests], dtype=np.int64)
        histories = [[int(item) for item, _ in req[1]] for req in requests]
        weight_lists = [[float(weight) for _, weight in req[1]] for req in requests]

        tt_scores = np.zeros((len(requests), self.n_items), dtype=np.float32)
        chunk = 256
        for start in range(0, len(requests), chunk):
            end = min(start + chunk, len(requests))
            tt_scores[start:end] = self._two_tower_scores(user_ids[start:end], histories[start:end])

        result: list[np.ndarray] = []
        for idx, (_, history_pairs, _) in enumerate(requests):
            history_items = [int(item) for item, _ in history_pairs]
            weights = [float(weight) for _, weight in history_pairs]
            scores = self._combined_scores(tt_scores[idx], history_items, weights)
            if history_items:
                scores[history_items] = -np.inf
            k_eff = min(k, self.n_items)
            part = np.argpartition(-scores, kth=k_eff - 1)[:k_eff]
            part = part[np.argsort(-scores[part], kind="mergesort")]
            result.append(part.astype(np.int32))
        return result

    def recommend(self, user_idx: int, history: Sequence[int], k: int = 10, exclude: set[int] | None = None, weights: Sequence[float] | None = None) -> np.ndarray:
        hist = [int(i) for i in history]
        w = [float(x) for x in (weights if weights is not None else [1.0] * len(hist))]
        requests = [(int(user_idx), [(i, wi) for i, wi in zip(hist, w)], -1)]
        return self.recommend_many(requests, k=k)[0]
