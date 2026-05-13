import sys
from pathlib import Path
import polars as pl
import unittest

# Ensure output encoding is utf-8
sys.stdout.reconfigure(encoding='utf-8')

# Ensure we can import from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.data_loader import DataLoaderPolar

class TestParquetDataSanity(unittest.TestCase):
    """
    Test suite to check Parquet data sanity and
    mapping between transactions, users, and articles.
    """

    @classmethod
    def setUpClass(cls):
        """Load data once for all tests in this class."""
        print("\n--- LOADING PARQUET DATA FOR SANITY CHECK ---")
        cls.loader = DataLoaderPolar()
        cls.train_df, cls.test_df, cls.customers_df, cls.articles_df = cls.loader.load_all_dataframes()
        
        # Print general overview
        print(f"[*] Train Data : {cls.train_df.shape[0]} rows, {cls.train_df.shape[1]} cols")
        print(f"[*] Test Data  : {cls.test_df.shape[0]} rows, {cls.test_df.shape[1]} cols")
        print(f"[*] Customers  : {cls.customers_df.shape[0]} rows, {cls.customers_df.shape[1]} cols")
        print(f"[*] Articles   : {cls.articles_df.shape[0]} rows, {cls.articles_df.shape[1]} cols")
        print("-" * 45)

    def test_customer_mapping_in_train(self):
        """Check if user_ids in train exist in customers."""
        train_users = self.train_df.select("user_id").unique()
        customer_users = self.customers_df.select("user_id").unique()
        
        missing_users = train_users.join(customer_users, on="user_id", how="anti")
        missing_count = missing_users.height
        
        print(f"[?] Customer Mapping: {missing_count} users in train lack profile info.")
        
    def test_article_mapping_in_train(self):
        """Check if item_ids in train exist in articles."""
        train_items = self.train_df.select("item_id").unique()
        
        article_key = "item_id" if "item_id" in self.articles_df.columns else "article_id"
        article_items = self.articles_df.select(pl.col(article_key).alias("item_id")).unique()
        
        missing_items = train_items.join(article_items, on="item_id", how="anti")
        missing_count = missing_items.height
        
        print(f"[?] Article Mapping : {missing_count} items in train lack article info.")
        
    def test_transaction_dataframe_nulls(self):
        """Check for nulls in key columns of train data."""
        null_users = self.train_df.filter(pl.col("user_id").is_null()).height
        null_items = self.train_df.filter(pl.col("item_id").is_null()).height
        
        print(f"[?] Null Check      : Train data has {null_users} null user_ids and {null_items} null item_ids.")
        
        self.assertEqual(null_users, 0, "Null user_id found in train data!")
        self.assertEqual(null_items, 0, "Null item_id found in train data!")

if __name__ == '__main__':
    unittest.main(verbosity=2)

