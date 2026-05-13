#!/usr/bin/env python3
"""Debug script to check why articles join returns None."""

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

# Initialize loader and load data
loader = DataLoaderPolar()
train_df, test_df, customers_df, articles_df = loader.load_all_dataframes()

print("=" * 90)
print("SCHEMA OF ARTICLES")
print("=" * 90)
print(f"Columns: {articles_df.columns}")
print(f"Shape: {articles_df.shape}\n")

# Check if articles uses item_id or article_id
if "item_id" in articles_df.columns:
    print("Articles has column: item_id")
    id_col = "item_id"
elif "article_id" in articles_df.columns:
    print("Articles has column: article_id")
    id_col = "article_id"
else:
    print("NOT FOUND item_id or article_id!")
    id_col = None

print("\n" + "=" * 90)
print("FIRST FEW ROWS OF ARTICLES")
print("=" * 90)
# Instead of pandas, we can just print polars head
print(articles_df.head(5))

print("\n" + "=" * 90)
print("CHECKING ITEM_IDs FROM SAMPLE TRANSACTIONS")
print("=" * 90)

# Get sample transactions
sample_transactions = train_df.head(3)
sample_ids = sample_transactions["item_id"].to_list()
print(f"Sample item_ids: {sample_ids}\n")

# Check if these item_ids exist in articles
for item_id in sample_ids:
    if id_col:
        match = articles_df.filter(pl.col(id_col) == item_id)
        print(f"item_id {item_id}: Found in articles? {match.shape[0] > 0}")
        if match.shape[0] > 0:
            print(f"  -> prod_name: {match['prod_name'].to_list()[0]}")

print("\n" + "=" * 90)
print("STATISTICS")
print("=" * 90)
print(f"Total unique item_ids in train_df: {train_df['item_id'].n_unique()}")
print(f"Total unique {id_col} in articles_df: {articles_df[id_col].n_unique()}")

# Find how many item_ids from train match with articles
if id_col:
    matching_ids = train_df.join(
        articles_df.select(pl.col(id_col)), 
        left_on="item_id", 
        right_on=id_col, 
        how="inner"
    )
    print(f"Item_ids from train that match articles: {matching_ids.shape[0]}")
    print(f"Match rate: {matching_ids.shape[0] / train_df.shape[0] * 100:.2f}%")

