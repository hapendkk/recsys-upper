from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def rank_of_target(recommendations: Sequence[int], target: int) -> int | None:
    try:
        return int(np.where(np.asarray(recommendations) == target)[0][0]) + 1
    except (IndexError, ValueError):
        return None


def ndcg_for_single_target(rank: int | None) -> float:
    if rank is None:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def evaluate_recommendation_list(recommendations: Sequence[int], target: int, ks: Sequence[int] = (5, 10, 20)) -> dict[str, float]:
    rank = rank_of_target(recommendations, target)
    out = {"hit": float(rank is not None), "rank": float(rank) if rank is not None else np.nan, "mrr": 0.0, "ndcg": 0.0}
    rec_array = np.asarray(recommendations)
    for k in ks:
        top = rec_array[:k]
        hit = target in set(top.tolist())
        out[f"recall@{k}"] = float(hit)
        out[f"hit@{k}"] = float(hit)
        out[f"ndcg@{k}"] = ndcg_for_single_target(rank_of_target(top, target))
        out[f"mrr@{k}"] = (1.0 / rank) if (rank is not None and rank <= k) else 0.0
    if rank is not None:
        out["mrr"] = 1.0 / rank
        out["ndcg"] = ndcg_for_single_target(rank)
    return out


def aggregate_metrics(records: Iterable[dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(list(records))
    if df.empty:
        return df
    numeric = df.select_dtypes(include=[np.number]).copy()
    if "rank" in numeric.columns:
        numeric["rank"] = numeric["rank"].replace([np.inf, -np.inf], np.nan)
    agg = numeric.mean(numeric_only=True).to_frame("mean").T
    agg["users"] = len(df)
    return agg
