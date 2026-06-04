"""
preprocess_hm.py - Build processed parquet files from the raw H&M Kaggle CSVs.

Expected raw files:
  data/raw/articles.csv
  data/raw/customers.csv
  data/raw/transactions_train.csv

Outputs:
  data/processed/articles_cleaned.parquet
  data/processed/customers_fixed.parquet
  data/processed/hm_train.parquet
  data/processed/hm_test.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def encode_ids(transactions: pl.DataFrame, customers: pl.DataFrame, articles: pl.DataFrame):
    customer_mapping = (
        customers.select("customer_id")
        .unique()
        .sort("customer_id")
        .with_row_index("user_id")
    )
    article_mapping = (
        articles.select("article_id")
        .unique()
        .sort("article_id")
        .with_row_index("item_id")
    )

    customers = customers.join(customer_mapping, on="customer_id", how="left").drop("customer_id")
    articles = articles.join(article_mapping, on="article_id", how="left")
    transactions = (
        transactions.join(customer_mapping, on="customer_id", how="inner")
        .join(article_mapping, on="article_id", how="inner")
        .drop(["customer_id", "article_id"])
    )
    return transactions, customers, articles


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw H&M Kaggle CSVs.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--months", type=int, default=12, help="Keep the latest N months of transactions.")
    parser.add_argument("--test-days", type=int, default=7, help="Use the last N days as test set.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    articles = pl.read_csv(args.raw_dir / "articles.csv")
    customers = pl.read_csv(args.raw_dir / "customers.csv")
    transactions = pl.read_csv(
        args.raw_dir / "transactions_train.csv",
        try_parse_dates=True,
    )

    min_date = transactions.select(
        pl.col("t_dat").max().dt.offset_by(f"-{args.months}mo")
    ).item()
    transactions = transactions.filter(pl.col("t_dat") >= min_date)

    transactions, customers, articles = encode_ids(transactions, customers, articles)
    transactions = transactions.sort("t_dat")

    split_date = transactions.select(
        pl.col("t_dat").max().dt.offset_by(f"-{args.test_days}d")
    ).item()
    train = transactions.filter(pl.col("t_dat") < split_date)
    test = transactions.filter(pl.col("t_dat") >= split_date)

    articles.write_parquet(args.out_dir / "articles_cleaned.parquet")
    customers.write_parquet(args.out_dir / "customers_fixed.parquet")
    train.write_parquet(args.out_dir / "hm_train.parquet")
    test.write_parquet(args.out_dir / "hm_test.parquet")

    print(f"Wrote articles: {articles.shape}")
    print(f"Wrote customers: {customers.shape}")
    print(f"Wrote train: {train.shape}")
    print(f"Wrote test: {test.shape}")


if __name__ == "__main__":
    main()
