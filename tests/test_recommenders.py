import numpy as np
import pandas as pd

from src.recommenders import AttributeRecommender, ContentBasedRecommender, HybridRecommender, ItemKNNRecommender, PopularityRecommender, RecentRecommender


def make_items():
    return pd.DataFrame(
        {
            "item_idx": [0, 1, 2, 3],
            "item_id": ["i0", "i1", "i2", "i3"],
            "title": ["Red lipstick", "Red lipstick matte", "Blue mascara", "Hair dryer"],
            "brand": ["A", "A", "B", "C"],
            "primary_category": ["Makeup", "Makeup", "Eyes", "Hair"],
            "category_path": ["Beauty > Makeup", "Beauty > Makeup", "Beauty > Eyes", "Beauty > Hair"],
            "description": ["red lip", "red lip matte", "blue eye", "dry hair"],
            "features": ["lipstick", "matte lipstick", "mascara", "dryer"],
            "text_blob": ["red lipstick makeup", "red lipstick matte makeup", "blue mascara eyes", "hair dryer hair"],
            "price": [10.0, 12.0, 8.0, 20.0],
            "average_rating": [4.0, 4.2, 4.1, 4.3],
            "rating_number": [100, 120, 90, 80],
            "image_url": ["", "", "", ""],
        }
    )


def make_interactions():
    return pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1, 2, 3],
            "item_idx": [0, 1, 0, 2, 1, 3],
            "rating": [5.0, 4.0, 5.0, 4.0, 3.0, 5.0],
            "timestamp": [1, 2, 3, 4, 5, 6],
        }
    )


def test_popularity_and_recent_exclude_history():
    items = make_items()
    inter = make_interactions()
    pop = PopularityRecommender().fit(inter, items)
    recent = RecentRecommender().fit(inter, items)
    assert set(pop.recommend(0, [0, 1], k=2, exclude={0, 1}).tolist()) == {2, 3}
    assert recent.recommend(0, [0, 1], k=2, exclude={0, 1}).tolist() == [3, 2]


def test_attribute_recommender_uses_brand():
    items = make_items()
    inter = make_interactions()
    rec = AttributeRecommender("brand").fit(inter, items)
    out = rec.recommend(0, [0], k=3, exclude={0})
    assert 1 in out


def test_content_based_prefers_similar_text_and_excludes_history():
    items = make_items()
    inter = make_interactions()
    rec = ContentBasedRecommender(max_features=100).fit(inter, items)
    out = rec.recommend(0, [0], k=3, exclude={0})
    assert 0 not in out
    assert 1 in out


def test_item_knn_returns_items_and_excludes_history():
    items = make_items()
    inter = make_interactions()
    rec = ItemKNNRecommender(top_k=2).fit(inter, items)
    out = rec.recommend(0, [0, 1], k=2, exclude={0, 1})
    assert len(out) == 2
    assert set(out).isdisjoint({0, 1})


def test_hybrid_recommend_is_deterministic_and_excludes_history():
    items = make_items()
    inter = make_interactions()
    rec = HybridRecommender(content_weight=0.6, item_weight=0.3, popularity_weight=0.1).fit(inter, items)
    out1 = rec.recommend(0, [0], k=3, exclude={0})
    out2 = rec.recommend(0, [0], k=3, exclude={0})
    assert out1.tolist() == out2.tolist()
    assert 0 not in out1
