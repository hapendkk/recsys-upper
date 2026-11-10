from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MODEL_DIR, REPORT_DIR
from .data_utils import load_processed
from .evaluate import evaluate_methods
from .two_tower import TwoTowerRecommender, save_item_embeddings, save_two_tower, train_two_tower


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Two-Tower neural recommender")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--user-emb-dim", type=int, default=32)
    parser.add_argument("--item-emb-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--out-dim", type=int, default=64)
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--n-random-negatives", type=int, default=256)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--eval-max-users", type=int, default=2000)
    parser.add_argument("--output", default="two_tower.pt")
    parser.add_argument("--train-splits", default="train", help="comma-separated split names used for training")
    parser.add_argument("--no-user-id-emb", action="store_true")
    parser.add_argument("--normalize-embeddings", action="store_true")
    parser.add_argument("--embedding-init-std", type=float, default=0.02)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    interactions, items, users = load_processed()
    model, features, checkpoint = train_two_tower(
        interactions=interactions,
        items=items,
        users=users,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        user_emb_dim=args.user_emb_dim,
        item_emb_dim=args.item_emb_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        max_history=args.max_history,
        n_random_negatives=args.n_random_negatives,
        max_examples=args.max_examples,
        seed=args.seed,
        train_splits=tuple(s.strip() for s in args.train_splits.split(",") if s.strip()),
        use_user_id_emb=not args.no_user_id_emb,
        normalize_embeddings=args.normalize_embeddings,
        embedding_init_std=args.embedding_init_std,
        dropout=args.dropout,
    )

    ckpt_path = save_two_tower(checkpoint, MODEL_DIR / args.output)
    emb_path = ckpt_path.with_name(ckpt_path.stem + "_item_emb.npy")
    if emb_path.exists():
        emb_path.unlink()
    recommender = TwoTowerRecommender(ckpt_path)
    save_item_embeddings(recommender, emb_path)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    valid_metrics = evaluate_methods(
        {"two_tower": recommender},
        interactions,
        items,
        target_split="valid",
        history_splits=("train",),
        k=args.k,
        max_users=args.eval_max_users,
        seed=args.seed,
    )
    valid_metrics["eval_split"] = "valid"
    rows.append(valid_metrics)
    print(valid_metrics.to_string(index=False))

    test_metrics = evaluate_methods(
        {"two_tower": recommender},
        interactions,
        items,
        target_split="test",
        history_splits=("train", "valid"),
        k=args.k,
        max_users=args.eval_max_users,
        seed=args.seed,
    )
    test_metrics["eval_split"] = "test"
    rows.append(test_metrics)
    print(test_metrics.to_string(index=False))

    metrics = pd.concat(rows, ignore_index=True)
    metrics.to_csv(REPORT_DIR / "two_tower_metrics.csv", index=False)
    metrics.to_json(REPORT_DIR / "two_tower_metrics.json", orient="records", indent=2, force_ascii=False)


if __name__ == "__main__":
    main()
