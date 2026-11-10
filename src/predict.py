from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_DIR
from .data_utils import load_processed
from .two_tower_hybrid import TwoTowerHybridRecommender

BASELINES_PATH = MODEL_DIR / "baselines.joblib"
TWO_TOWER_PATH = MODEL_DIR / "two_tower.pt"


def load_baseline_bundle(path: Path = BASELINES_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Baseline bundle not found: {path}. Run `python -m src.train_baselines` first.")
    return joblib.load(path)


def load_two_tower_recommender():
    if not TWO_TOWER_PATH.exists():
        return None
    from .two_tower import TwoTowerRecommender

    return TwoTowerRecommender(TWO_TOWER_PATH)


def user_history(
    interactions: pd.DataFrame,
    user_idx: int,
    splits: Sequence[str] | None = None,
) -> tuple[list[int], list[float]]:
    subset = interactions[interactions["user_idx"] == user_idx]
    if splits is not None:
        subset = subset[subset["split"].isin(list(splits))]
    subset = subset.sort_values(["timestamp", "item_idx"], kind="mergesort")
    return subset["item_idx"].astype(int).tolist(), subset["rating"].astype(float).tolist()


def _baseline_scores(recommender: Any, history_items: list[int], weights: list[float], rec_items: np.ndarray) -> np.ndarray:
    if hasattr(recommender, "score_items"):
        try:
            scores = recommender.score_items(history_items, weights)
            return np.asarray(scores[rec_items], dtype=np.float32)
        except Exception:
            pass
    return -np.arange(len(rec_items), dtype=np.float32)


def recommend_for_user(
    method: str,
    user_idx: int,
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    users: pd.DataFrame,
    k: int = 10,
    history_splits: Sequence[str] | None = None,
    baseline_bundle: dict[str, Any] | None = None,
    two_tower: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    history_items, weights = user_history(interactions, user_idx, splits=history_splits)
    exclude = set(history_items)

    if method in {"two_tower", "two_tower_hybrid"}:
        if two_tower is None:
            raise RuntimeError("Two-Tower model is not trained yet")
        requests = [(int(user_idx), [(int(i), float(w)) for i, w in zip(history_items, weights)], -1)]
        if method == "two_tower_hybrid":
            if baseline_bundle is None:
                baseline_bundle = load_baseline_bundle()
            hybrid = TwoTowerHybridRecommender(two_tower, baseline_bundle["recommenders"])
            rec_items = hybrid.recommend_many(requests, k=k)[0]
            return rec_items, -np.arange(len(rec_items), dtype=np.float32)
        rec_items = two_tower.recommend_many(requests, k=k)[0]
        user_emb = two_tower.encode_users(np.array([user_idx], dtype=np.int64), [history_items])
        scores_all = (user_emb @ two_tower.item_embeddings.T).astype(np.float32)[0]
        return rec_items, np.asarray(scores_all[rec_items], dtype=np.float32)

    if baseline_bundle is None:
        baseline_bundle = load_baseline_bundle()
    recommender = baseline_bundle["recommenders"][method]
    rec_items = recommender.recommend(
        user_idx,
        history_items,
        k=k,
        exclude=exclude,
        weights=weights,
    )
    scores = _baseline_scores(recommender, history_items, weights, rec_items)
    return rec_items, scores


def recommendations_frame(
    rec_items: np.ndarray,
    scores: np.ndarray,
    items: pd.DataFrame,
) -> pd.DataFrame:
    item_lookup = items.set_index("item_idx")
    rows = []
    for rank, (item_idx, score) in enumerate(zip(rec_items, scores), start=1):
        item_idx = int(item_idx)
        if item_idx not in item_lookup.index:
            continue
        row = item_lookup.loc[item_idx]
        rows.append(
            {
                "rank": rank,
                "item_idx": item_idx,
                "title": row.get("title", ""),
                "brand": row.get("brand", ""),
                "category": row.get("primary_category", ""),
                "price": row.get("price", np.nan),
                "avg_rating": row.get("average_rating", np.nan),
                "score": float(score),
                "image_url": row.get("image_url", ""),
            }
        )
    return pd.DataFrame(rows)


def item_similar_frame(
    seed_item_idx: int,
    method: str,
    items: pd.DataFrame,
    baseline_bundle: dict[str, Any] | None = None,
    k: int = 10,
) -> pd.DataFrame:
    if baseline_bundle is None:
        baseline_bundle = load_baseline_bundle()
    recommender = baseline_bundle["recommenders"][method]
    rec_items = recommender.recommend(-1, [int(seed_item_idx)], k=k, exclude={int(seed_item_idx)}, weights=[1.0])
    scores = _baseline_scores(recommender, [int(seed_item_idx)], [1.0], rec_items)
    return recommendations_frame(rec_items, scores, items)
