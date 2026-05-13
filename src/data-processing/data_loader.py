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

class DataLoaderPolar :
    def __init__(
        self,
        article_data_path: Path = ARTICLES_PARQUET_PATH ,
        customer_data_path: Path = CUSTOMERS_PARQUET_PATH,
        train_data_path: Path = TRAIN_PARQUET_PATH,
        test_data_path: Path = TEST_PARQUET_PATH,
    ) -> None:
        self._article_data_path = Path(article_data_path)
        self._customer_data_path = Path(customer_data_path)
        self._train_data_path = Path(train_data_path)
        self._test_data_path = Path(test_data_path)
        
        self.articles : pl.DataFrame | None = None
        self.customers : pl.DataFrame | None = None
        self.train : pl.DataFrame | None = None
        self.test : pl.DataFrame | None = None
        
    
    def load_all_dataframes(
        self ,
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame] :
        #load article_df , customer_df , train_df , test_df -> return tuple of them
        self.articles = pl.read_parquet(self._article_data_path)
        self.customers = pl.read_parquet(self._customer_data_path)
        self.train = pl.read_parquet(self._train_data_path)
        self.test = pl.read_parquet(self._test_data_path)
        return self.train , self.test , self.customers , self.articles
    
    def join_feature_customers_article_to_transaction_df(
        self ,
        transaction_df : pl.DataFrame ,
    ) -> pl.DataFrame :
        # Left join customers and article information onto a transaction dataframe
        if self.customers is None or self.articles is None:
            raise RuntimeError(
                "Call load_all_dataframes() before join_features_to_transactions()."
            )
        transactions_with_features = transaction_df.join(self.customers, on="user_id", how="left")

        # Join articles — handle 'item_id' vs 'article_id' schema variants
        if "item_id" in self.articles.columns:
            transactions_with_features = transactions_with_features.join(self.articles, on="item_id", how="left")
        else:
            transactions_with_features = transactions_with_features.join(
                self.articles, left_on="item_id", right_on="article_id", how="left"
            )

        return transactions_with_features