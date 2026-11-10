from __future__ import annotations

import pandas as pd

from .config import PROCESSED_DIR


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    interactions = pd.read_parquet(PROCESSED_DIR / "interactions.parquet")
    items = pd.read_parquet(PROCESSED_DIR / "items.parquet")
    users = pd.read_parquet(PROCESSED_DIR / "users.parquet")
    interactions = interactions.sort_values(["user_idx", "timestamp", "item_idx"], kind="mergesort").reset_index(drop=True)
    items = items.sort_values("item_idx").reset_index(drop=True)
    users = users.sort_values("user_idx").reset_index(drop=True)
    return interactions, items, users
