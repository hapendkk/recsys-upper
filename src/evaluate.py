from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .metrics import aggregate_metrics, evaluate_recommendation_list


def histories_by_user(interactions: pd.DataFrame, splits: Sequence[str]) -> dict[int, list[tuple[int, float]]]:
    subset = interactions[interactions["split"].isin(list(splits))]
    histories: dict[int, list[tuple[int, float]]] = {}
    for row in subset[["user_idx", "item_idx", "rating"]].itertuples(index=False):
        histories.setdefault(int(row.user_idx), []).append((int(row.item_idx), float(row.rating)))
    return histories


def _recommend_for_user(recommender: Any, user_idx: int, history_pairs: list[tuple[int, float]], k: int, exclude_splits_items: set[int] | None) -> np.ndarray:
    history_items = [item for item, _ in history_pairs]
    weights = [weight for _, weight in history_pairs]
    exclude = set(history_items)
    if exclude_splits_items:
        exclude |= exclude_splits_items
    return recommender.recommend(user_idx, history_items, k=k, exclude=exclude, weights=weights)


def evaluate_method(
    name: str,
    recommender: Any,
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    target_split: str = "test",
    history_splits: Sequence[str] = ("train", "valid"),
    k: int = 10,
    max_users: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    targets = interactions[interactions["split"] == target_split].copy()
    if max_users is not None and len(targets) > max_users:
        targets = targets.sample(n=max_users, random_state=seed).sort_values(["user_idx", "timestamp"])
    histories = histories_by_user(interactions, history_splits)
    counts = (
        interactions[interactions["split"].isin(list(history_splits))]
        .groupby("item_idx")
        .size()
        .reindex(range(len(items)), fill_value=0)
        .astype(float)
    )
    total = float(counts.sum())
    popularity = (counts + 1.0) / (total + len(items))

    records: list[dict[str, float]] = []
    all_recs: list[int] = []
    target_items: list[int] = []

    requests = []
    for row in targets[["user_idx", "item_idx"]].itertuples(index=False):
        user_idx = int(row.user_idx)
        target_item = int(row.item_idx)
        history_pairs = histories.get(user_idx, [])
        requests.append((user_idx, history_pairs, target_item))

    if hasattr(recommender, "recommend_many"):
        rec_lists = recommender.recommend_many(requests, k=k)
    else:
        rec_lists = [
            _recommend_for_user(recommender, user_idx, history_pairs, k=k, exclude_splits_items=None)
            for user_idx, history_pairs, _ in requests
        ]

    for (_, history_pairs, target_item), recs in zip(requests, rec_lists):
        rec_array = np.asarray(recs, dtype=np.int32)
        metrics = evaluate_recommendation_list(rec_array, target_item, ks=(k,))
        records.append(metrics)
        all_recs.extend(rec_array.tolist())
        target_items.append(target_item)

    user_ids = [user_idx for user_idx, _, _ in requests]
    for record, user_idx in zip(records, user_ids):
        record["user_idx"] = float(user_idx)

    metric_df = pd.DataFrame(records)
    result: dict[str, Any] = {
        "method": name,
        "target_split": target_split,
        "history_splits": "+".join(history_splits),
        "k": k,
        "users": int(len(records)),
    }
    for col in metric_df.columns:
        if col not in {"user_idx", "rank"}:
            result[col] = float(metric_df[col].mean())

    rec_array = np.asarray(all_recs, dtype=np.int32)
    if len(rec_array):
        result["coverage"] = float(len(set(rec_array.tolist())) / max(len(items), 1))
        pop_scores = popularity.to_numpy()[rec_array]
        result["novelty"] = float((-np.log(pop_scores)).mean())
    else:
        result["coverage"] = 0.0
        result["novelty"] = 0.0
    return result


def evaluate_methods(
    recommenders: dict[str, Any],
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    target_split: str = "test",
    history_splits: Sequence[str] = ("train", "valid"),
    k: int = 10,
    max_users: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    rows = []
    for name, recommender in recommenders.items():
        rows.append(
            evaluate_method(
                name=name,
                recommender=recommender,
                interactions=interactions,
                items=items,
                target_split=target_split,
                history_splits=history_splits,
                k=k,
                max_users=max_users,
                seed=seed,
            )
        )
    return pd.DataFrame(rows)
