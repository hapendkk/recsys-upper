from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import TOP_K_SIMILAR_ITEMS


def _top_indices(scores: np.ndarray, k: int, exclude: set[int] | None = None) -> np.ndarray:
    scores = scores.astype(np.float32, copy=True)
    if exclude:
        scores[list(exclude)] = -np.inf
    scores[~np.isfinite(scores)] = -np.inf
    finite = int(np.isfinite(scores).sum())
    k = min(k, finite, len(scores))
    if k <= 0:
        return np.array([], dtype=np.int32)
    part = np.argpartition(-scores, kth=k - 1)[:k]
    part = part[np.argsort(-scores[part], kind="mergesort")]
    return part.astype(np.int32)


def _safe_weights(weights: Sequence[float] | None, n: int) -> np.ndarray:
    if weights is None:
        return np.ones(n, dtype=np.float32)
    arr = np.asarray(list(weights), dtype=np.float32)
    if len(arr) != n:
        raise ValueError("weights length must match history length")
    return np.maximum(arr, 0.0)


class PopularityRecommender:
    name = "popularity"

    def fit(self, interactions: pd.DataFrame, items: pd.DataFrame) -> "PopularityRecommender":
        self.n_items = int(items["item_idx"].max()) + 1 if len(items) else 0
        counts = interactions.groupby("item_idx").size().reindex(range(self.n_items), fill_value=0)
        self.scores = counts.to_numpy(dtype=np.float32)
        self.rank = np.argsort(-self.scores, kind="mergesort")
        return self

    def score_items(self, history: Sequence[int], weights: Sequence[float] | None = None) -> np.ndarray:
        return self.scores.copy()

    def recommend(self, user_idx: int, history: Sequence[int], k: int = 10, exclude: set[int] | None = None, weights: Sequence[float] | None = None) -> np.ndarray:
        return _top_indices(self.scores, k=k, exclude=exclude)


class RecentRecommender:
    name = "recent"

    def fit(self, interactions: pd.DataFrame, items: pd.DataFrame) -> "RecentRecommender":
        self.n_items = int(items["item_idx"].max()) + 1 if len(items) else 0
        latest = interactions.groupby("item_idx")["timestamp"].max().reindex(range(self.n_items), fill_value=0)
        self.scores = latest.to_numpy(dtype=np.float32)
        return self

    def score_items(self, history: Sequence[int], weights: Sequence[float] | None = None) -> np.ndarray:
        return self.scores.copy()

    def recommend(self, user_idx: int, history: Sequence[int], k: int = 10, exclude: set[int] | None = None, weights: Sequence[float] | None = None) -> np.ndarray:
        return _top_indices(self.scores, k=k, exclude=exclude)


class AttributeRecommender:
    def __init__(self, attribute: str):
        self.attribute = attribute
        self.name = f"same_{attribute}"

    @staticmethod
    def _pseudo_category(text: str) -> str:
        t = text.lower()
        rules = [
            ("Skin Care", ["serum", "moisturizer", "moisturiser", "cream", "lotion", "face", "skincare", "skin care", "hyaluronic", "retinol", "sunscreen", "spf", "cleanser", "toner", "facial", "anti-aging", "anti ageing"]),
            ("Hair Care", ["shampoo", "conditioner", "hair", "keratin", "hairdryer", "hair dryer", "leave-in", "scalp"]),
            ("Fragrance", ["perfume", "fragrance", "cologne", "eau de", "body spray", "cologne", "parfum"]),
            ("Makeup", ["lipstick", "mascara", "foundation", "eyeshadow", "blush", "concealer", "makeup", "lip gloss", "lipstick", "bb cream", "cc cream"]),
            ("Nails", ["nail", "manicure", "polish", "acrylic", "cuticle", "nail file", "nail clipper"]),
            ("Oral Care", ["toothbrush", "toothpaste", "whitening", "oral", "mouthwash", "floss", "dental"]),
            ("Shaving & Grooming", ["razor", "shaving", "trimmer", "shaver", "clipper", "epilator", "waxing", "beard", "grooming"]),
            ("Bath & Body", ["soap", "body wash", "bubble bath", "bath", "hand wash", "bar soap", "body lotion"]),
        ]
        for label, keywords in rules:
            if any(k in t for k in keywords):
                return label
        return "Beauty Other"

    @classmethod
    def _category_attr(cls, row: pd.Series) -> str:
        generic = {"Unknown", "All Beauty", "Premium Beauty", "Beauty & Personal Care", "Personal Care", "Health & Household"}
        primary = str(row.get("primary_category", "Unknown"))
        if primary not in generic:
            return primary
        parts = [p.strip() for p in str(row.get("category_path", "")).split(">") if p.strip()]
        parts = [p for p in parts if p not in generic]
        if parts:
            return parts[-1]
        text = " ".join(str(row.get(c, "")) for c in ["title", "description", "features", "text_blob"])
        return cls._pseudo_category(text)

    def fit(self, interactions: pd.DataFrame, items: pd.DataFrame) -> "AttributeRecommender":
        item_frame = items.sort_values("item_idx").reset_index(drop=True)
        self.n_items = len(item_frame)
        if self.attribute == "primary_category":
            self.item_attr = item_frame.apply(self._category_attr, axis=1).to_numpy()
        else:
            self.item_attr = item_frame[self.attribute].fillna("Unknown").astype(str).to_numpy()
        self.attr_to_items: dict[str, np.ndarray] = {}
        for attr, idx in pd.Series(np.arange(self.n_items)).groupby(self.item_attr):
            self.attr_to_items[str(attr)] = idx.to_numpy(dtype=np.int32)
        counts = interactions.groupby("item_idx").size().reindex(range(self.n_items), fill_value=0)
        self.popularity = counts.to_numpy(dtype=np.float32)
        return self

    def score_items(self, history: Sequence[int], weights: Sequence[float] | None = None) -> np.ndarray:
        hist = [int(i) for i in history if 0 <= int(i) < self.n_items]
        w = _safe_weights(weights, len(hist)) if hist else np.array([], dtype=np.float32)
        scores = np.zeros(self.n_items, dtype=np.float32)
        for item_idx, weight in zip(hist, w):
            attr = self.item_attr[item_idx]
            candidates = self.attr_to_items.get(attr)
            if candidates is not None and len(candidates):
                scores[candidates] += float(weight)
        if self.popularity.max() > 0:
            scores += 1e-3 * self.popularity / self.popularity.max()
        return scores

    def recommend(self, user_idx: int, history: Sequence[int], k: int = 10, exclude: set[int] | None = None, weights: Sequence[float] | None = None) -> np.ndarray:
        return _top_indices(self.score_items(history, weights), k=k, exclude=exclude)


class ContentBasedRecommender:
    name = "content_tfidf"

    def __init__(self, max_features: int = 50_000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            stop_words="english",
            sublinear_tf=True,
            norm="l2",
        )

    def fit(self, interactions: pd.DataFrame, items: pd.DataFrame) -> "ContentBasedRecommender":
        item_frame = items.sort_values("item_idx").reset_index(drop=True)
        self.n_items = len(item_frame)
        corpus = item_frame["text_blob"].fillna("").astype(str).tolist()
        self.tfidf = self.vectorizer.fit_transform(corpus).tocsr()
        counts = interactions.groupby("item_idx").size().reindex(range(self.n_items), fill_value=0)
        self.popularity = counts.to_numpy(dtype=np.float32)
        return self

    def score_items(self, history: Sequence[int], weights: Sequence[float] | None = None) -> np.ndarray:
        hist = [int(i) for i in history if 0 <= int(i) < self.n_items]
        if not hist:
            return self.popularity / max(self.popularity.max(), 1.0)
        w = _safe_weights(weights, len(hist))
        profile = np.asarray(self.tfidf[hist].T @ w).ravel()
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile = profile / norm
        scores = np.asarray(self.tfidf @ profile, dtype=np.float32).ravel()
        return scores

    def recommend(self, user_idx: int, history: Sequence[int], k: int = 10, exclude: set[int] | None = None, weights: Sequence[float] | None = None) -> np.ndarray:
        if not history:
            return _top_indices(self.popularity, k=k, exclude=exclude)
        return _top_indices(self.score_items(history, weights), k=k, exclude=exclude)


class ItemKNNRecommender:
    name = "item_knn"

    def __init__(self, top_k: int = TOP_K_SIMILAR_ITEMS, batch_size: int = 256):
        self.top_k = top_k
        self.batch_size = batch_size

    def fit(self, interactions: pd.DataFrame, items: pd.DataFrame) -> "ItemKNNRecommender":
        self.n_users = int(interactions["user_idx"].max()) + 1 if len(interactions) else 0
        self.n_items = int(items["item_idx"].max()) + 1 if len(items) else 0
        rows = interactions["user_idx"].to_numpy(dtype=np.int32)
        cols = interactions["item_idx"].to_numpy(dtype=np.int32)
        data = interactions["rating"].to_numpy(dtype=np.float32)
        x = sparse.coo_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items)).tocsr()
        item_user = normalize(x.T.tocsr(), norm="l2", axis=1)

        top_k = min(self.top_k, self.n_items - 1)
        sim_indices = np.zeros((self.n_items, top_k), dtype=np.int32)
        sim_scores = np.zeros((self.n_items, top_k), dtype=np.float32)
        for start in range(0, self.n_items, self.batch_size):
            end = min(start + self.batch_size, self.n_items)
            block = item_user[start:end]
            sims = (block @ item_user.T).toarray().astype(np.float32)
            for local_row, global_row in enumerate(range(start, end)):
                sims[local_row, global_row] = -np.inf
                k = min(top_k, self.n_items - 1)
                part = np.argpartition(-sims[local_row], kth=k - 1)[:k]
                part = part[np.argsort(-sims[local_row, part], kind="mergesort")]
                sim_indices[global_row, : len(part)] = part.astype(np.int32)
                sim_scores[global_row, : len(part)] = sims[local_row, part]
                if len(part) < top_k:
                    sim_indices[global_row, len(part) :] = part[:1]
                    sim_scores[global_row, len(part) :] = sims[local_row, part[:1]]

        rows_rep = np.repeat(np.arange(self.n_items, dtype=np.int32), top_k)
        self.sim_matrix = sparse.csr_matrix(
            (sim_scores.ravel(), (rows_rep, sim_indices.ravel())),
            shape=(self.n_items, self.n_items),
        )
        counts = interactions.groupby("item_idx").size().reindex(range(self.n_items), fill_value=0)
        self.popularity = counts.to_numpy(dtype=np.float32)
        return self

    def score_items(self, history: Sequence[int], weights: Sequence[float] | None = None) -> np.ndarray:
        hist = [int(i) for i in history if 0 <= int(i) < self.n_items]
        if not hist:
            return self.popularity / max(self.popularity.max(), 1.0)
        w = _safe_weights(weights, len(hist))
        h = np.zeros(self.n_items, dtype=np.float32)
        h[hist] = w
        scores = np.asarray(self.sim_matrix.T @ h, dtype=np.float32).ravel()
        return scores

    def recommend(self, user_idx: int, history: Sequence[int], k: int = 10, exclude: set[int] | None = None, weights: Sequence[float] | None = None) -> np.ndarray:
        if not history:
            return _top_indices(self.popularity, k=k, exclude=exclude)
        return _top_indices(self.score_items(history, weights), k=k, exclude=exclude)


class HybridRecommender:
    name = "hybrid"

    def __init__(self, content_weight: float = 0.65, item_weight: float = 0.25, popularity_weight: float = 0.10):
        self.content_weight = content_weight
        self.item_weight = item_weight
        self.popularity_weight = popularity_weight

    def fit(self, interactions: pd.DataFrame, items: pd.DataFrame) -> "HybridRecommender":
        self.content = ContentBasedRecommender().fit(interactions, items)
        self.item_knn = ItemKNNRecommender().fit(interactions, items)
        self.popularity = PopularityRecommender().fit(interactions, items)
        self.n_items = self.popularity.n_items
        return self

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float32)
        max_score = float(scores.max()) if scores.size else 0.0
        if max_score > 0:
            scores = scores / max_score
        return scores

    def score_items(self, history: Sequence[int], weights: Sequence[float] | None = None) -> np.ndarray:
        content = self._normalize(self.content.score_items(history, weights))
        item = self._normalize(self.item_knn.score_items(history, weights))
        pop = self._normalize(self.popularity.score_items(history, weights))
        total = self.content_weight + self.item_weight + self.popularity_weight
        if total <= 0:
            return pop
        return (self.content_weight * content + self.item_weight * item + self.popularity_weight * pop) / total

    def recommend(self, user_idx: int, history: Sequence[int], k: int = 10, exclude: set[int] | None = None, weights: Sequence[float] | None = None) -> np.ndarray:
        if not history:
            return self.popularity.recommend(user_idx, history, k=k, exclude=exclude, weights=weights)
        return _top_indices(self.score_items(history, weights), k=k, exclude=exclude)


def build_baseline_recommenders(items: pd.DataFrame, train_interactions: pd.DataFrame) -> dict[str, Any]:
    return {
        "popularity": PopularityRecommender().fit(train_interactions, items),
        "recent": RecentRecommender().fit(train_interactions, items),
        "same_brand": AttributeRecommender("brand").fit(train_interactions, items),
        "same_category": AttributeRecommender("primary_category").fit(train_interactions, items),
        "content_tfidf": ContentBasedRecommender().fit(train_interactions, items),
        "item_knn": ItemKNNRecommender().fit(train_interactions, items),
        "hybrid": HybridRecommender().fit(train_interactions, items),
    }
