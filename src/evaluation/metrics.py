"""
metrics.py — Evaluation metrics for the H&M Recommendation System.

Centralises all metric computation.  The primary metric is MAP@12
as defined by the Kaggle H&M competition.

Functions
---------
average_precision_at_k
    AP@K for a single user.
calculate_map_at_12
    Mean AP@12 across all test users.
hit_rate_at_k
    Fraction of users with at least one hit in top-K.
ndcg_at_k
    Normalised Discounted Cumulative Gain @ K.
"""

from __future__ import annotations

import math
from typing import Dict, List, Set

import polars as pl

from src.config import TOP_K_RECOMMENDATIONS


# ---------------------------------------------------------------------------
# Single-user helpers
# ---------------------------------------------------------------------------

def average_precision_at_k(
    predicted_item_ids: List[int],
    actual_item_ids: Set[int],
    k: int = TOP_K_RECOMMENDATIONS,
) -> float:
    """Compute Average Precision at K for a single user.

    Parameters
    ----------
    predicted_item_ids : list of int
        Ordered list of predicted item IDs (most relevant first).
    actual_item_ids : set of int
        Ground-truth item IDs the user actually interacted with.
    k : int
        Cutoff rank.

    Returns
    -------
    float
        AP@K score in [0, 1].
    """
    if not actual_item_ids:
        return 0.0

    hit_count = 0
    cumulative_precision_sum = 0.0

    for rank, predicted_id in enumerate(predicted_item_ids[:k]):
        if predicted_id in actual_item_ids:
            hit_count += 1
            cumulative_precision_sum += hit_count / (rank + 1.0)

    denominator = min(len(actual_item_ids), k)
    return cumulative_precision_sum / denominator if denominator > 0 else 0.0


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def _build_user_ground_truth(test_df: pl.DataFrame) -> Dict[int, Set[int]]:
    """Group test transactions into a {user_id: set(item_ids)} dict."""
    grouped = test_df.group_by("user_id").agg(
        pl.col("item_id").alias("actual_items")
    )
    return {
        int(row["user_id"]): set(row["actual_items"])
        for row in grouped.iter_rows(named=True)
    }


def calculate_map_at_12(
    predictions: Dict[int, List[int]],
    test_df: pl.DataFrame,
) -> float:
    """Mean Average Precision @ 12 across all users in *test_df*.

    Parameters
    ----------
    predictions : dict[int, list[int]]
        Mapping from ``user_id`` to an ordered list of up to 12 predicted
        item IDs.
    test_df : pl.DataFrame
        Test transactions with columns ``user_id`` and ``item_id``.

    Returns
    -------
    float
        MAP@12 score in [0, 1].
    """
    ground_truth = _build_user_ground_truth(test_df)
    if not ground_truth:
        return 0.0

    total_ap = 0.0
    for user_id, actual_set in ground_truth.items():
        user_predictions = predictions.get(user_id, [])
        total_ap += average_precision_at_k(
            user_predictions, actual_set, k=TOP_K_RECOMMENDATIONS
        )

    return total_ap / len(ground_truth)


def hit_rate_at_k(
    predictions: Dict[int, List[int]],
    test_df: pl.DataFrame,
    k: int = TOP_K_RECOMMENDATIONS,
) -> float:
    """Fraction of users with at least one hit in their top-K recommendations."""
    ground_truth = _build_user_ground_truth(test_df)
    if not ground_truth:
        return 0.0

    hit_count = 0
    for user_id, actual_set in ground_truth.items():
        user_predictions = predictions.get(user_id, [])
        if any(item_id in actual_set for item_id in user_predictions[:k]):
            hit_count += 1

    return hit_count / len(ground_truth)


def ndcg_at_k(
    predictions: Dict[int, List[int]],
    test_df: pl.DataFrame,
    k: int = TOP_K_RECOMMENDATIONS,
) -> float:
    """Normalised Discounted Cumulative Gain @ K averaged over test users."""
    ground_truth = _build_user_ground_truth(test_df)
    if not ground_truth:
        return 0.0

    total_ndcg = 0.0
    for user_id, actual_set in ground_truth.items():
        user_predictions = predictions.get(user_id, [])

        # DCG
        dcg = 0.0
        for rank, predicted_id in enumerate(user_predictions[:k]):
            if predicted_id in actual_set:
                dcg += 1.0 / math.log2(rank + 2)

        # Ideal DCG
        ideal_hit_count = min(len(actual_set), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hit_count))

        total_ndcg += (dcg / idcg) if idcg > 0 else 0.0

    return total_ndcg / len(ground_truth)
