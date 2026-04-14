"""
popularity_baseline.py — Global popularity baseline recommender.

Recommends the K most-purchased products across all users.  Serves as the
simplest baseline to compare against collaborative and hybrid models.
"""

from __future__ import annotations

from typing import List

import polars as pl

from src.config import TOP_K_RECOMMENDATIONS


class PopularityBaseline:
    """Recommend the globally most popular items regardless of user."""

    def __init__(self, top_k: int = TOP_K_RECOMMENDATIONS) -> None:
        self._top_k = top_k
        self._most_popular_item_ids: List[int] = []

    # ------------------------------------------------------------------
    def fit(self, train_df: pl.DataFrame) -> None:
        """Compute the top-K most frequent items in *train_df*.

        Parameters
        ----------
        train_df : pl.DataFrame
            Must contain an ``item_id`` column.
        """
        top_items = (
            train_df.group_by("item_id")
            .agg(pl.len().alias("purchase_count"))
            .sort("purchase_count", descending=True)
            .head(self._top_k)
        )
        self._most_popular_item_ids = top_items["item_id"].to_list()

    # ------------------------------------------------------------------
    def predict(self, user_id: int | None = None) -> List[int]:
        """Return the most popular item IDs (user-independent).

        Parameters
        ----------
        user_id : int or None
            Ignored — popularity baseline is not personalised.

        Returns
        -------
        list of int
        """
        return self._most_popular_item_ids
