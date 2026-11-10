import numpy as np
import pandas as pd

from src.data_prep import _parse_price, assign_splits, metadata_record, support_filter


def test_parse_price_handles_common_formats():
    assert _parse_price("$12.99") == 12.99
    assert _parse_price("1,299.00") == 1299.0
    assert _parse_price(None) != _parse_price(None)  # NaN
    assert _parse_price("unknown") != _parse_price("unknown")


def test_metadata_record_extracts_core_fields():
    raw = {
        "parent_asin": "B000TEST",
        "title": "Test Serum",
        "store": "BrandX",
        "categories": [["All Beauty", "Skin Care", "Serums"]],
        "description": ["Hydrating", "serum"],
        "features": ["Vitamin C"],
        "price": "$19.99",
        "average_rating": 4.5,
        "rating_number": 120,
        "images": [{"large": "https://example.com/img.jpg"}],
    }
    rec = metadata_record(raw)
    assert rec["item_id"] == "B000TEST"
    assert rec["brand"] == "BrandX"
    assert rec["primary_category"] == "Serums"
    assert "Test Serum" in rec["text_blob"]
    assert rec["price"] == 19.99


def test_support_filter_keeps_only_supported_users_and_items():
    df = pd.DataFrame(
        {
            "user_idx": [0, 0, 0, 1, 1, 2, 2],
            "item_idx": [0, 1, 2, 0, 1, 0, 3],
            "rating": [5, 4, 3, 5, 4, 5, 4],
            "timestamp": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    filtered, stats = support_filter(df, min_user_interactions=3, min_item_interactions=2)
    assert stats["final_rows"] == 2
    assert set(filtered["user_idx"]) == {0}
    assert set(filtered["item_idx"]) == {0, 1}


def test_assign_splits_is_per_user_leave_last_two_out():
    df = pd.DataFrame(
        {
            "user_idx": [0, 0, 0, 0, 1, 1, 1],
            "item_idx": [0, 1, 2, 3, 4, 5, 6],
            "rating": [5, 4, 3, 5, 4, 3, 5],
            "timestamp": [10, 20, 30, 40, 1, 2, 3],
        }
    )
    out = assign_splits(df)
    u0 = out[out["user_idx"] == 0].sort_values("timestamp")
    assert u0.iloc[-1]["split"] == "test"
    assert u0.iloc[-2]["split"] == "valid"
    assert set(u0.iloc[:-2]["split"]) == {"train"}
    # No target item appears in train for the same user.
    train_items = set(u0[u0["split"] == "train"]["item_idx"])
    target_items = set(u0[u0["split"].isin(["valid", "test"])]["item_idx"])
    assert train_items.isdisjoint(target_items)
