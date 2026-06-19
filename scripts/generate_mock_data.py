import os
import numpy as np
import polars as pl
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ARTICLES_PARQUET_PATH,
    CUSTOMERS_PARQUET_PATH,
    TRAIN_PARQUET_PATH,
    TEST_PARQUET_PATH,
    VISUAL_FEATURES_NPZ_PATH,
    BEST_CHECKPOINT_PATH,
    MF_EMBEDDING_DIM,
    MLP_LAYER_SIZES,
    VISUAL_FEATURE_DIM
)
from src.models.hybrid_model import HybridRecommendationModel

def main():
    print("Generating mock dataset and checkpoint...")
    
    # Create directories
    ARTICLES_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    num_users = 50
    num_items = 100
    
    # 1. Articles Cleaned
    articles_data = {
        "item_id": list(range(num_items)),
        "article_id": [100000 + i for i in range(num_items)],
        "prod_name": [f"Fashion Item {i}" for i in range(num_items)],
        "product_type_name": [f"Type {i % 5}" for i in range(num_items)],
        "colour_group_name": [f"Colour {i % 8}" for i in range(num_items)],
        "department_name": [f"Dept {i % 3}" for i in range(num_items)],
        "product_group_name": [f"Group {i % 4}" for i in range(num_items)],
        "section_name": [f"Section {i % 2}" for i in range(num_items)],
        "detail_desc": [f"Detailed description for fashion item {i}. Very comfortable and fashionable." for i in range(num_items)]
    }
    articles_df = pl.DataFrame(articles_data)
    articles_df.write_parquet(ARTICLES_PARQUET_PATH)
    print(f"Saved mock articles to {ARTICLES_PARQUET_PATH}")
    
    # 2. Customers Cleaned
    customers_data = {
        "user_id": list(range(num_users)),
        "customer_id": [f"cust_{i}" for i in range(num_users)],
        "age": [20 + (i % 45) for i in range(num_users)],
        "club_member_status": ["ACTIVE" if i % 10 != 0 else "PRE-CREATE" for i in range(num_users)],
        "FN": [1.0 if i % 2 == 0 else 0.0 for i in range(num_users)],
        "Active": [1.0 if i % 3 != 0 else 0.0 for i in range(num_users)],
        "fashion_news_frequency": ["Regularly" if i % 4 == 0 else "NONE" for i in range(num_users)]
    }
    customers_df = pl.DataFrame(customers_data)
    customers_df.write_parquet(CUSTOMERS_PARQUET_PATH)
    print(f"Saved mock customers to {CUSTOMERS_PARQUET_PATH}")
    
    # 3. Train Transactions
    np.random.seed(42)
    train_rows = 1000
    train_users = np.random.randint(0, num_users, size=train_rows)
    train_items = np.random.randint(0, num_items, size=train_rows)
    
    # ensure max user_id and item_id are in dataset to define ID space size correctly
    train_users[0] = num_users - 1
    train_items[0] = num_items - 1
    
    train_data = {
        "user_id": train_users.tolist(),
        "item_id": train_items.tolist(),
        "price": np.random.uniform(0.01, 0.1, size=train_rows).tolist(),
        "t_dat": ["2020-09-01"] * train_rows
    }
    train_df = pl.DataFrame(train_data)
    train_df.write_parquet(TRAIN_PARQUET_PATH)
    print(f"Saved mock train data to {TRAIN_PARQUET_PATH}")
    
    # 4. Test Transactions
    test_rows = 100
    test_users = np.random.randint(0, num_users, size=test_rows)
    test_items = np.random.randint(0, num_items, size=test_rows)
    test_data = {
        "user_id": test_users.tolist(),
        "item_id": test_items.tolist(),
        "price": np.random.uniform(0.01, 0.1, size=test_rows).tolist(),
        "t_dat": ["2020-09-23"] * test_rows
    }
    test_df = pl.DataFrame(test_data)
    test_df.write_parquet(TEST_PARQUET_PATH)
    print(f"Saved mock test data to {TEST_PARQUET_PATH}")
    
    # 5. Visual Features (.npz)
    visual_features = {}
    for i in range(num_items):
        vector = np.random.normal(0.0, 1.0, size=(VISUAL_FEATURE_DIM,)).astype(np.float32)
        visual_features[str(i)] = vector
        visual_features[str(100000 + i)] = vector
    np.savez(VISUAL_FEATURES_NPZ_PATH, **visual_features)
    print(f"Saved mock visual features to {VISUAL_FEATURES_NPZ_PATH}")
    
    # 6. Checkpoint Model (.pt)
    model = HybridRecommendationModel(
        num_users=num_users,
        num_items=num_items,
        visual_feature_dim=VISUAL_FEATURE_DIM,
        mf_embedding_dim=MF_EMBEDDING_DIM,
        mlp_layer_sizes=list(MLP_LAYER_SIZES)
    )
    torch.save(model.state_dict(), BEST_CHECKPOINT_PATH)
    print(f"Saved mock model checkpoint to {BEST_CHECKPOINT_PATH}")
    print("All mock files generated successfully.")

if __name__ == "__main__":
    main()
