import math

import pytest

from src.metrics import aggregate_metrics, evaluate_recommendation_list, rank_of_target


def test_rank_of_target_first_and_missing():
    assert rank_of_target([10, 20, 30], 20) == 2
    assert rank_of_target([10, 20, 30], 99) is None


def test_evaluate_recommendation_list_rank_one():
    metrics = evaluate_recommendation_list([5, 1, 2, 3, 4], target=5, ks=(10,))
    assert metrics["hit"] == 1.0
    assert metrics["recall@10"] == 1.0
    assert metrics["hit@10"] == 1.0
    assert metrics["mrr@10"] == 1.0
    assert metrics["ndcg@10"] == pytest.approx(1.0)


def test_evaluate_recommendation_list_rank_three():
    metrics = evaluate_recommendation_list([1, 2, 7, 4, 5], target=7, ks=(10,))
    assert metrics["recall@10"] == 1.0
    assert metrics["mrr@10"] == pytest.approx(1.0 / 3)
    assert metrics["ndcg@10"] == pytest.approx(1.0 / math.log2(4))


def test_evaluate_recommendation_list_miss_and_k_cutoff():
    metrics = evaluate_recommendation_list([1, 2, 3], target=7, ks=(2, 10))
    assert metrics["recall@2"] == 0.0
    assert metrics["hit@2"] == 0.0
    assert metrics["mrr@2"] == 0.0
    assert metrics["ndcg@2"] == 0.0


def test_aggregate_metrics_empty_and_mean():
    assert aggregate_metrics([]).empty
    agg = aggregate_metrics([{"recall@10": 0.0, "users": 1}, {"recall@10": 1.0, "users": 1}])
    assert agg.loc["mean", "recall@10"] == pytest.approx(0.5)
    assert agg.loc["mean", "users"] == 2
