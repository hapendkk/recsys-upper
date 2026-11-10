import numpy as np
import pandas as pd

from src.predict import recommendations_frame, user_history


def make_frames():
    interactions = pd.DataFrame(
        {
            "user_idx": [0, 0, 0],
            "item_idx": [0, 1, 2],
            "rating": [5.0, 4.0, 3.0],
            "timestamp": [3, 1, 2],
            "split": ["train", "valid", "test"],
        }
    )
    items = pd.DataFrame(
        {
            "item_idx": [0, 1, 2],
            "item_id": ["i0", "i1", "i2"],
            "title": ["Item 0", "Item 1", "Item 2"],
            "brand": ["A", "B", "C"],
            "primary_category": ["Cat A", "Cat B", "Cat C"],
            "price": [1.0, 2.0, np.nan],
            "average_rating": [4.0, 4.1, 4.2],
            "image_url": ["", "", ""],
        }
    )
    return interactions, items


def test_user_history_filters_splits_and_sorts_by_time():
    interactions, _ = make_frames()
    hist_items, weights = user_history(interactions, 0, splits=("train", "valid"))
    assert hist_items == [1, 0]  # timestamps 1 then 3
    assert weights == [4.0, 5.0]


def test_recommendations_frame_maps_items_and_scores():
    _, items = make_frames()
    df = recommendations_frame(np.array([2, 0]), np.array([0.9, 0.1]), items)
    assert df["rank"].tolist() == [1, 2]
    assert df["item_idx"].tolist() == [2, 0]
    assert df.loc[0, "title"] == "Item 2"
    assert np.isnan(df.loc[0, "price"])
