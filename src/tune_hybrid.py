from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPORT_DIR
from .data_utils import load_processed
from .evaluate import evaluate_methods
from .recommenders import HybridRecommender, build_baseline_recommenders


def _weight_combos(step: float, include_current: bool = True) -> list[tuple[float, float, float]]:
    values = np.round(np.arange(0.0, 1.0 + step / 2, step), 6)
    combos = [
        (float(c), float(i), float(p))
        for c in values
        for i in values
        for p in values
        if abs(c + i + p - 1.0) < 1e-6
    ]
    if include_current:
        current = (0.65, 0.25, 0.10)
        if not any(abs(c - current[0]) < 1e-6 and abs(i - current[1]) < 1e-6 and abs(p - current[2]) < 1e-6 for c, i, p in combos):
            combos.append(current)
    return combos


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-search hybrid component weights on validation")
    parser.add_argument("--max-users", type=int, default=1000)
    parser.add_argument("--step", type=float, default=0.2)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    interactions, items, _users = load_processed()
    train_interactions = interactions[interactions["split"] == "train"]
    components = build_baseline_recommenders(items, train_interactions)
    content = components["content_tfidf"]
    item_knn = components["item_knn"]
    popularity = components["popularity"]

    rows = []
    for content_w, item_w, pop_w in _weight_combos(args.step):
        hybrid = HybridRecommender(content_weight=content_w, item_weight=item_w, popularity_weight=pop_w)
        hybrid.content = content
        hybrid.item_knn = item_knn
        hybrid.popularity = popularity
        hybrid.n_items = popularity.n_items
        name = f"hybrid_c{content_w:.2f}_i{item_w:.2f}_p{pop_w:.2f}"
        metrics = evaluate_methods(
            {name: hybrid},
            interactions,
            items,
            target_split="valid",
            history_splits=("train",),
            k=args.k,
            max_users=args.max_users,
            seed=args.seed,
        )
        rows.append(metrics.iloc[0].to_dict())
        print(name, "recall@10=", round(float(metrics.iloc[0].get("recall@10", 0.0)), 4), "ndcg@10=", round(float(metrics.iloc[0].get("ndcg@10", 0.0)), 4))

    df = pd.DataFrame(rows)
    df["content_w"] = [name.split("_c")[1].split("_")[0] for name in df["method"]]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT_DIR / "hybrid_weight_search.csv", index=False)
    df.to_json(REPORT_DIR / "hybrid_weight_search.json", orient="records", indent=2, force_ascii=False)

    top = df.sort_values(["recall@10", "ndcg@10"], ascending=False).head(10)
    print("\nTop hybrid weights by Recall@10/NDCG@10 on validation sample:")
    print(top[["method", "recall@10", "ndcg@10", "hit@10", "mrr@10", "coverage", "novelty"]].to_string(index=False))


if __name__ == "__main__":
    main()
