from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .config import MODEL_DIR, REPORT_DIR
from .data_utils import load_processed
from .evaluate import evaluate_methods
from .recommenders import build_baseline_recommenders


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate classical RecSys baselines")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=None, help="Optional cap per split for fast experiments")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()

    interactions, items, _ = load_processed()
    train = interactions[interactions["split"] == "train"].copy()
    recommenders = build_baseline_recommenders(items, train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"recommenders": recommenders, "items": items}, MODEL_DIR / "baselines.joblib", compress=3)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    valid_metrics = evaluate_methods(
        recommenders,
        interactions,
        items,
        target_split="valid",
        history_splits=("train",),
        k=args.k,
        max_users=args.max_users,
    )
    valid_metrics["eval_split"] = "valid"
    all_metrics.append(valid_metrics)
    print(valid_metrics.to_string(index=False))

    if not args.skip_test:
        test_metrics = evaluate_methods(
            recommenders,
            interactions,
            items,
            target_split="test",
            history_splits=("train", "valid"),
            k=args.k,
            max_users=args.max_users,
        )
        test_metrics["eval_split"] = "test"
        all_metrics.append(test_metrics)
        print(test_metrics.to_string(index=False))

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(REPORT_DIR / "baseline_metrics.csv", index=False)
    metrics.to_json(REPORT_DIR / "baseline_metrics.json", orient="records", indent=2, force_ascii=False)


if __name__ == "__main__":
    main()
