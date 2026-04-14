"""
data_loader.py — Load Parquet datasets using Polars.

Responsible *only* for reading parquet files and joining customer/article
metadata onto transaction DataFrames.  All model logic, baselines, and
evaluation metrics live in their own dedicated modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import polars as pl

from src.config import (
    ARTICLES_PARQUET_PATH,
    CUSTOMERS_PARQUET_PATH,
    TEST_PARQUET_PATH,
    TRAIN_PARQUET_PATH,
)


class DataLoaderPolars:
    """Read H&M Parquet files and join customer / article metadata."""

    def __init__(
        self,
        articles_path: Path = ARTICLES_PARQUET_PATH,
        customers_path: Path = CUSTOMERS_PARQUET_PATH,
        train_path: Path = TRAIN_PARQUET_PATH,
        test_path: Path = TEST_PARQUET_PATH,
    ) -> None:
        self._articles_path = Path(articles_path)
        self._customers_path = Path(customers_path)
        self._train_path = Path(train_path)
        self._test_path = Path(test_path)

        # Populated by load_all_dataframes()
        self.articles: pl.DataFrame | None = None
        self.customers: pl.DataFrame | None = None
        self.train: pl.DataFrame | None = None
        self.test: pl.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_all_dataframes(
        self,
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Load train, test, customers, and articles DataFrames.

        Returns
        -------
        tuple of (train_df, test_df, customers_df, articles_df)
        """
        self.articles = pl.read_parquet(self._articles_path)
        self.customers = pl.read_parquet(self._customers_path)
        self.train = pl.read_parquet(self._train_path)
        self.test = pl.read_parquet(self._test_path)
        return self.train, self.test, self.customers, self.articles

    def join_features_to_transactions(
        self, transactions_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Left-join customer and article information onto a transactions DF.

        Parameters
        ----------
        transactions_df : pl.DataFrame
            Must contain ``user_id`` and ``item_id`` columns.

        Returns
        -------
        pl.DataFrame
            Transactions enriched with customer demographics and article
            metadata.
        """
        if self.customers is None or self.articles is None:
            raise RuntimeError(
                "Call load_all_dataframes() before join_features_to_transactions()."
            )

        # Join customers on user_id
        enriched_df = transactions_df.join(self.customers, on="user_id", how="left")

        # Join articles — handle 'item_id' vs 'article_id' schema variants
        if "item_id" in self.articles.columns:
            enriched_df = enriched_df.join(self.articles, on="item_id", how="left")
        else:
            enriched_df = enriched_df.join(
                self.articles, left_on="item_id", right_on="article_id", how="left"
            )

        return enriched_df
