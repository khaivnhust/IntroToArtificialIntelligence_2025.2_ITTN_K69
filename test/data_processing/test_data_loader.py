import unittest
import sys
from pathlib import Path
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.data_loader import DataLoaderPolar

class TestDataLoaderPolar(unittest.TestCase):
    
    def setUp(self):
        self.loader = DataLoaderPolar()
        
    def test_load_all_dataframes(self):
        train_df, test_df, customers_df, articles_df = self.loader.load_all_dataframes()
        
        self.assertIsInstance(train_df, pl.DataFrame)
        self.assertIsInstance(test_df, pl.DataFrame)
        self.assertIsInstance(customers_df, pl.DataFrame)
        self.assertIsInstance(articles_df, pl.DataFrame)
        
        self.assertIsNotNone(self.loader.train)
        self.assertIsNotNone(self.loader.test)
        self.assertIsNotNone(self.loader.customers)
        self.assertIsNotNone(self.loader.articles)
        
    def test_join_features_to_transaction_df(self):
        
        train_df, _, _, _ = self.loader.load_all_dataframes()        
        sample_transactions_df = train_df.head(10)
        joined_df = self.loader.join_feature_customers_article_to_transaction_df(sample_transactions_df)
        
        self.assertEqual(joined_df.height, 10)
        
        expected_columns = ['user_id', 'item_id', 'price']
        for col in expected_columns:
            self.assertIn(col, joined_df.columns, f"Column {col} is missing after join.")
            
        if 'age' in self.loader.customers.columns:
             self.assertIn('age', joined_df.columns, "Column 'age' from customers was not joined.")

if __name__ == '__main__':
    unittest.main()
