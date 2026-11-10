from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    DATASET_REPO,
    DEFAULT_MIN_ITEM_INTERACTIONS,
    DEFAULT_MIN_USER_INTERACTIONS,
    EXPECTED_SIZES,
    META_FILENAME,
    META_REPO_PATH,
    MIN_POSITIVE_RATING,
    PROCESSED_DIR,
    RATINGS_FILENAME,
    RATINGS_REPO_PATH,
    RAW_DIR,
    REPORT_DIR,
)


def ensure_raw_file(filename: str, repo_path: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / filename
    expected = EXPECTED_SIZES.get(filename)
    if target.exists() and (expected is None or target.stat().st_size == expected):
        return target

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("huggingface_hub is required to download the dataset") from exc

    cached = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=repo_path,
        repo_type="dataset",
    )
    shutil.copyfile(cached, target)
    return target


def load_ratings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"user_id", "parent_asin", "rating", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Ratings file misses columns: {sorted(missing)}")
    df = df.rename(columns={"parent_asin": "item_id"})
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(np.int64)
    return df[["user_id", "item_id", "rating", "timestamp"]]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _flatten_categories(value: Any) -> list[str]:
    out: list[str] = []
    for item in _as_list(value):
        if isinstance(item, list):
            out.extend(str(x) for x in item if x is not None)
        elif item is not None:
            out.append(str(item))
    return out


def _parse_price(value: Any) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).replace("$", "").replace(",", "").strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def _first_image(images: Any) -> str:
    for image in _as_list(images):
        if isinstance(image, dict):
            url = image.get("large") or image.get("hi_res") or image.get("thumb")
            if url:
                return str(url)
    return ""


def _details_dict(details: Any) -> dict[str, Any]:
    if isinstance(details, dict):
        return details
    if isinstance(details, str):
        try:
            parsed = json.loads(details)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def metadata_record(raw: dict[str, Any]) -> dict[str, Any] | None:
    item_id = raw.get("parent_asin") or raw.get("asin")
    if not item_id:
        return None

    details = _details_dict(raw.get("details"))
    brand = details.get("Brand") or raw.get("store") or "Unknown"
    categories = _flatten_categories(raw.get("categories"))
    if not categories:
        categories = _flatten_categories(raw.get("main_category"))
    category_path = " > ".join(categories) if categories else "Unknown"
    primary_category = categories[-1] if categories else "Unknown"

    description = " ".join(str(x) for x in _as_list(raw.get("description")) if x is not None)
    features = " ".join(str(x) for x in _as_list(raw.get("features")) if x is not None)
    title = str(raw.get("title") or item_id)
    text_parts = [title, category_path, str(brand), description, features]

    return {
        "item_id": str(item_id),
        "title": title,
        "brand": str(brand),
        "primary_category": primary_category,
        "category_path": category_path,
        "description": description,
        "features": features,
        "price": _parse_price(raw.get("price")),
        "average_rating": pd.to_numeric(raw.get("average_rating"), errors="coerce"),
        "rating_number": pd.to_numeric(raw.get("rating_number"), errors="coerce"),
        "image_url": _first_image(raw.get("images")),
        "text_blob": " ".join(text_parts),
    }


def load_metadata(path: Path, needed_item_ids: set[str] | None = None) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            record = metadata_record(raw)
            if record is None:
                continue
            if needed_item_ids is not None and record["item_id"] not in needed_item_ids:
                continue
            records.append(record)
    return pd.DataFrame(records)


def support_filter(
    df: pd.DataFrame,
    min_user_interactions: int,
    min_item_interactions: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    user_counts = df.groupby("user_idx").size()
    item_counts = df.groupby("item_idx").size()
    keep_users = set(user_counts[user_counts >= min_user_interactions].index)
    keep_items = set(item_counts[item_counts >= min_item_interactions].index)
    filtered = df[df["user_idx"].isin(keep_users) & df["item_idx"].isin(keep_items)].copy()
    stats = {
        "initial_rows": int(len(df)),
        "final_rows": int(len(filtered)),
        "users": int(filtered["user_idx"].nunique()),
        "items": int(filtered["item_idx"].nunique()),
        "min_user_interactions": int(min_user_interactions),
        "min_item_interactions": int(min_item_interactions),
    }
    return filtered, stats


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["split"] = "train"
    order_cols = ["user_idx", "timestamp", "item_idx"]
    out = out.sort_values(order_cols, kind="mergesort").reset_index(drop=True)

    test_positions: list[int] = []
    valid_positions: list[int] = []
    for _, group in out.groupby("user_idx", sort=False):
        indices = group.index.to_numpy()
        if len(indices) >= 3:
            valid_positions.append(int(indices[-2]))
            test_positions.append(int(indices[-1]))
    out.loc[valid_positions, "split"] = "valid"
    out.loc[test_positions, "split"] = "test"
    return out.sort_values(["user_idx", "timestamp", "item_idx"], kind="mergesort").reset_index(drop=True)


def prepare(
    min_user_interactions: int = DEFAULT_MIN_USER_INTERACTIONS,
    min_item_interactions: int = DEFAULT_MIN_ITEM_INTERACTIONS,
    min_rating: float = MIN_POSITIVE_RATING,
) -> dict[str, Any]:
    ratings_path = ensure_raw_file(RATINGS_FILENAME, RATINGS_REPO_PATH)
    meta_path = ensure_raw_file(META_FILENAME, META_REPO_PATH)

    ratings = load_ratings(ratings_path)
    report: dict[str, Any] = {
        "raw_ratings_rows": int(len(ratings)),
        "raw_unique_users": int(ratings["user_id"].nunique()),
        "raw_unique_items": int(ratings["item_id"].nunique()),
        "min_positive_rating": min_rating,
        "min_user_interactions": min_user_interactions,
        "min_item_interactions": min_item_interactions,
    }
    all_user_counts = ratings["user_id"].value_counts()
    report["all_ratings_users"] = int(all_user_counts.size)
    report["all_ratings_single_review_users"] = int((all_user_counts == 1).sum())
    report["all_ratings_single_review_share"] = float((all_user_counts == 1).mean())

    positive = ratings[ratings["rating"] >= min_rating].copy()
    positive = positive.sort_values(["user_id", "item_id", "timestamp"], kind="mergesort")
    positive = positive.drop_duplicates(subset=["user_id", "item_id"], keep="last")
    report["positive_rows"] = int(len(positive))
    positive_user_counts = positive["user_id"].value_counts()
    report["positive_users"] = int(positive_user_counts.size)
    report["positive_single_review_users"] = int((positive_user_counts == 1).sum())
    report["positive_single_review_share"] = float((positive_user_counts == 1).mean())

    user_codes, user_uniques = pd.factorize(positive["user_id"], sort=True)
    item_codes, item_uniques = pd.factorize(positive["item_id"], sort=True)
    coded = pd.DataFrame(
        {
            "user_id": positive["user_id"].to_numpy(),
            "item_id": positive["item_id"].to_numpy(),
            "user_idx": user_codes.astype(np.int32),
            "item_idx": item_codes.astype(np.int32),
            "rating": positive["rating"].astype(np.float32),
            "timestamp": positive["timestamp"].astype(np.int64),
        }
    )

    core, core_stats = support_filter(
        coded,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
    )
    report["support_filter"] = core_stats

    needed_item_ids = set(core["item_id"].unique())
    meta = load_metadata(meta_path, needed_item_ids=needed_item_ids)
    items = pd.DataFrame({"item_id": item_uniques})
    items = items.merge(meta, on="item_id", how="left")
    for col in ["title", "brand", "primary_category", "category_path", "description", "features", "image_url", "text_blob"]:
        items[col] = items[col].fillna("Unknown" if col in {"title", "brand", "primary_category", "category_path", "text_blob"} else "")
    items["average_rating"] = items["average_rating"].fillna(0.0)
    items["rating_number"] = items["rating_number"].fillna(0.0)
    items["price"] = items["price"].astype(float)

    kept_item_ids = set(core["item_id"].unique())
    items = items[items["item_id"].isin(kept_item_ids)].copy()
    item_map = {item_id: idx for idx, item_id in enumerate(sorted(items["item_id"].tolist()))}
    items["item_idx"] = items["item_id"].map(item_map).astype(np.int32)
    items = items.sort_values("item_idx").reset_index(drop=True)

    user_map = {user_id: idx for idx, user_id in enumerate(sorted(core["user_id"].unique()))}
    core["user_idx"] = core["user_id"].map(user_map).astype(np.int32)
    core["item_idx"] = core["item_id"].map(item_map).astype(np.int32)

    core = assign_splits(core)
    users = pd.DataFrame({"user_idx": np.arange(len(user_map), dtype=np.int32), "user_id": sorted(user_map, key=user_map.get)})

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    core.to_parquet(PROCESSED_DIR / "interactions.parquet", index=False)
    items.to_parquet(PROCESSED_DIR / "items.parquet", index=False)
    users.to_parquet(PROCESSED_DIR / "users.parquet", index=False)

    report.update(
        {
            "processed_rows": int(len(core)),
            "processed_users": int(core["user_idx"].nunique()),
            "processed_items": int(items.shape[0]),
            "split_counts": core["split"].value_counts().to_dict(),
            "timestamp_min": int(core["timestamp"].min()),
            "timestamp_max": int(core["timestamp"].max()),
            "metadata_matched_items": int(items["title"].ne("Unknown").sum()),
        }
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "data_prep_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Amazon Beauty RecSys dataset")
    parser.add_argument("--min-user-interactions", type=int, default=DEFAULT_MIN_USER_INTERACTIONS)
    parser.add_argument("--min-item-interactions", type=int, default=DEFAULT_MIN_ITEM_INTERACTIONS)
    parser.add_argument("--min-rating", type=float, default=MIN_POSITIVE_RATING)
    args = parser.parse_args()
    report = prepare(
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
        min_rating=args.min_rating,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
