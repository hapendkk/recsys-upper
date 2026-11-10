from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .config import REPORT_DIR
from .data_utils import load_processed
from .evaluate import histories_by_user
from .metrics import evaluate_recommendation_list
from .recommenders import build_baseline_recommenders


def _hit_at_k(recommender, requests: list[tuple[int, list[tuple[int, float]], int]], k: int) -> np.ndarray:
    hits = []
    for user_idx, history_pairs, target_item in requests:
        history_items = [item for item, _ in history_pairs]
        weights = [weight for _, weight in history_pairs]
        recs = recommender.recommend(user_idx, history_items, k=k, exclude=set(history_items), weights=weights)
        hits.append(evaluate_recommendation_list(recs, target_item, ks=(k,))[f"hit@{k}"])
    return np.asarray(hits, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap/McNemar significance for hybrid vs content_tfidf")
    parser.add_argument("--max-users", type=int, default=2000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    interactions, items, _users = load_processed()
    train_interactions = interactions[interactions["split"] == "train"]
    recommenders = build_baseline_recommenders(items, train_interactions)
    content = recommenders["content_tfidf"]
    hybrid = recommenders["hybrid"]

    targets = interactions[interactions["split"] == "test"].copy()
    if args.max_users is not None and len(targets) > args.max_users:
        targets = targets.sample(n=args.max_users, random_state=args.seed).sort_values(["user_idx", "timestamp"])
    histories = histories_by_user(interactions, ("train", "valid"))
    requests = [
        (int(row.user_idx), histories.get(int(row.user_idx), []), int(row.item_idx))
        for row in targets[["user_idx", "item_idx"]].itertuples(index=False)
    ]

    content_hits = _hit_at_k(content, requests, args.k)
    hybrid_hits = _hit_at_k(hybrid, requests, args.k)
    diff = hybrid_hits - content_hits
    n = len(diff)
    rng = np.random.default_rng(args.seed)
    boot_means = np.empty(args.bootstraps, dtype=np.float64)
    for b in range(args.bootstraps):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = float(diff[idx].mean())
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    boot_p_two_sided = float(min(1.0, 2.0 * min(np.mean(boot_means <= 0), np.mean(boot_means >= 0))))

    b = int(np.sum((hybrid_hits == 1) & (content_hits == 0)))
    c = int(np.sum((hybrid_hits == 0) & (content_hits == 1)))
    if b + c > 0:
        mcnemar_stat = (abs(b - c) - 1.0) ** 2 / (b + c)
        mcnemar_p = float(chi2.sf(mcnemar_stat, 1))
    else:
        mcnemar_stat = 0.0
        mcnemar_p = 1.0

    result = {
        "method_a": "content_tfidf",
        "method_b": "hybrid",
        "k": args.k,
        "users": int(n),
        "content_hit_rate": float(content_hits.mean()),
        "hybrid_hit_rate": float(hybrid_hits.mean()),
        "hit_rate_diff": float(diff.mean()),
        "bootstrap_ci_low": float(ci_low),
        "bootstrap_ci_high": float(ci_high),
        "bootstrap_p_two_sided": boot_p_two_sided,
        "mcnemar_b_hybrid_only": b,
        "mcnemar_c_content_only": c,
        "mcnemar_stat": float(mcnemar_stat),
        "mcnemar_p": mcnemar_p,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(REPORT_DIR / "significance_hybrid_vs_content.csv", index=False)
    with open(REPORT_DIR / "significance_hybrid_vs_content.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
