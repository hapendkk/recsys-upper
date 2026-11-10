from __future__ import annotations

import argparse
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_DIR, REPORT_DIR
from .data_utils import load_processed
from .evaluate import histories_by_user
from .metrics import evaluate_recommendation_list
from .two_tower import TwoTowerRecommender


def _recommend_all(recommender: Any, requests: list[tuple[int, list[tuple[int, float]], int]], k: int) -> list[np.ndarray]:
    if hasattr(recommender, "recommend_many"):
        return recommender.recommend_many(requests, k=k)
    out = []
    for user_idx, history_pairs, _ in requests:
        hist_items = [int(i) for i, _ in history_pairs]
        weights = [float(w) for _, w in history_pairs]
        out.append(
            recommender.recommend(
                user_idx,
                hist_items,
                k=k,
                exclude=set(hist_items),
                weights=weights,
            )
        )
    return out


def _bucket(values: pd.Series, bins: list[int], labels: list[str]) -> pd.Series:
    return pd.cut(values, bins=bins, labels=labels, include_lowest=True, right=False).astype(str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cold-start / popularity analysis")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--methods", nargs="+", default=["content_tfidf", "item_knn", "hybrid", "two_tower"])
    args = parser.parse_args()

    interactions, items, users = load_processed()
    bundle = joblib.load(MODEL_DIR / "baselines.joblib")
    recommenders = dict(bundle["recommenders"])
    if "two_tower" in args.methods and (MODEL_DIR / "two_tower.pt").exists():
        recommenders["two_tower"] = TwoTowerRecommender(MODEL_DIR / "two_tower.pt")

    targets = interactions[interactions["split"] == "test"].sort_values(["user_idx", "timestamp"])
    histories = histories_by_user(interactions, ("train", "valid"))
    history_counts = interactions[interactions["split"].isin(["train", "valid"])].groupby("item_idx").size()

    rows = []
    for method in args.methods:
        if method not in recommenders:
            continue
        recommender = recommenders[method]
        requests = []
        for row in targets[["user_idx", "item_idx"]].itertuples(index=False):
            pairs = histories.get(int(row.user_idx), [])
            requests.append((int(row.user_idx), pairs, int(row.item_idx)))
        rec_lists = _recommend_all(recommender, requests, k=args.k)

        records = []
        for (_, history_pairs, target_item), recs in zip(requests, rec_lists):
            metrics = evaluate_recommendation_list(np.asarray(recs, dtype=np.int32), target_item, ks=(args.k,))
            records.append(
                {
                    "user_idx": float(_),
                    "target_item": float(target_item),
                    "history_size": float(len(history_pairs)),
                    "item_popularity": float(history_counts.get(target_item, 0)),
                    f"recall@{args.k}": metrics[f"recall@{args.k}"],
                    f"ndcg@{args.k}": metrics[f"ndcg@{args.k}"],
                }
            )
        df = pd.DataFrame(records)
        df["history_bucket"] = _bucket(df["history_size"], [0, 3, 6, 11, 10_000], ["<=2", "3-5", "6-10", "11+"])
        df["popularity_bucket"] = _bucket(df["item_popularity"], [0, 3, 6, 11, 10_000], ["<=2", "3-5", "6-10", "11+"])
        for bucket_col in ["history_bucket", "popularity_bucket"]:
            agg = (
                df.groupby(bucket_col, observed=False)[[f"recall@{args.k}", f"ndcg@{args.k}"]]
                .mean()
                .reset_index()
            )
            agg["users"] = df.groupby(bucket_col, observed=False).size().to_numpy()
            for _, r in agg.iterrows():
                rows.append(
                    {
                        "method": method,
                        "bucket_type": "user_history" if bucket_col == "history_bucket" else "target_item_popularity",
                        "bucket": r[bucket_col],
                        "users": int(r["users"]),
                        f"recall@{args.k}": float(r[f"recall@{args.k}"]),
                        f"ndcg@{args.k}": float(r[f"ndcg@{args.k}"]),
                    }
                )

    result = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(REPORT_DIR / "cold_start.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
