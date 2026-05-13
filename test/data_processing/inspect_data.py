import sys
from pathlib import Path
import polars as pl

# Ensure output encoding is utf-8
sys.stdout.reconfigure(encoding='utf-8')

# Ensure we can import from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.data_loader import DataLoaderPolar

def inspect_datasets():
    print("=" * 60)
    print("               DATASET INSPECTION SCRIPT")
    print("=" * 60)
    
    loader = DataLoaderPolar()
    print("\nLoading datasets... please wait.\n")
    train_df, test_df, customers_df, articles_df = loader.load_all_dataframes()
    
    # Set polars formatting to show all columns
    pl.Config.set_tbl_cols(-1)
    
    # Print sample of Customers
    print("-" * 60)
    print("1. CUSTOMERS DATA (First 3 rows)")
    print(f"Shape: {customers_df.shape}")
    print("-" * 60)
    print(customers_df.head(3))
    print("\n")
    
    # Print sample of Articles
    print("-" * 60)
    print("2. ARTICLES DATA (First 3 rows)")
    print(f"Shape: {articles_df.shape}")
    print("-" * 60)
    print(articles_df.head(3))
    print("\n")
    
    # Print sample of Train Transactions
    print("-" * 60)
    print("3. TRAIN TRANSACTIONS (First 3 rows)")
    print(f"Shape: {train_df.shape}")
    print("-" * 60)
    print(train_df.head(3))
    print("\n")
    
    # Print sample of Joined Transactions (Train + Customers + Articles)
    print("-" * 60)
    print("4. JOINED TRANSACTIONS WITH FEATURES (First 3 rows)")
    print("-" * 60)
    sample_train = train_df.head(3)
    joined_df = loader.join_feature_customers_article_to_transaction_df(sample_train)
    print(f"Shape of Joined Data: {joined_df.shape}")
    print("Selecting key columns to avoid messy output...")
    print("-" * 60)
    
    # Select a few key columns to display neatly
    key_cols = [col for col in ['user_id', 'item_id', 'price', 'age', 'club_member_status', 'prod_name', 'product_type_name', 'detail_desc'] if col in joined_df.columns]
    
    print(joined_df.select(key_cols))
    print("\n")

if __name__ == "__main__":
    inspect_datasets()
