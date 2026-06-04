"""
evaluate_recommenders.py - Compare recommender methods with MAP@12.

This script evaluates the global popularity baseline and, when a checkpoint is
available, the Hybrid NCF + visual + metadata model. The candidate set contains
each user's test items plus sampled negatives, matching the training script's
leakage-free evaluation path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import polars as pl
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_hybrid import (  # noqa: E402
    build_item_to_article_id_mapping,
    evaluate_map_at_12_on_test,
)
from src.config import BEST_CHECKPOINT_PATH, VISUAL_FEATURES_NPZ_PATH  # noqa: E402
from src.evaluation.metrics import calculate_map_at_12  # noqa: E402
from src.features.metadata_feature_encoder import MetadataFeatureEncoder  # noqa: E402
from src.features.visual_feature_extract import VisualFeatureExtractor  # noqa: E402
from src.models.inference_pipeline import InferencePipeline  # noqa: E402
from src.models.popularity_baseline import PopularityBaseline  # noqa: E402
from src.data_processing.data_loader import DataLoaderPolars  # noqa: E402


def build_user_to_seen_items(train_df: pl.DataFrame) -> Dict[int, set[int]]:
    grouped = train_df.group_by("user_id").agg(pl.col("item_id").alias("items"))
    return {
        int(row["user_id"]): set(int(item_id) for item_id in row["items"])
        for row in grouped.iter_rows(named=True)
    }


def evaluate_popularity(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    top_k: int = 12,
) -> float:
    baseline = PopularityBaseline(top_k=top_k)
    baseline.fit(train_df)
    top_items = baseline.predict()

    predictions: Dict[int, List[int]] = {}
    for row in test_df.select("user_id").unique().iter_rows(named=True):
        predictions[int(row["user_id"])] = top_items

    return calculate_map_at_12(predictions, test_df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate popularity and hybrid recommenders.")
    parser.add_argument("--checkpoint", type=Path, default=BEST_CHECKPOINT_PATH)
    parser.add_argument("--npz-path", type=Path, default=VISUAL_FEATURES_NPZ_PATH)
    parser.add_argument("--negative-candidates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_df, test_df, _, articles_df = DataLoaderPolars().load_all_dataframes()

    popularity_map = evaluate_popularity(train_df, test_df)
    print(f"Popularity MAP@12: {popularity_map:.6f}")

    if not args.checkpoint.exists():
        print(f"Hybrid MAP@12: skipped, checkpoint not found at {args.checkpoint}")
        return

    pipeline = InferencePipeline(
        train_df=train_df,
        articles_df=articles_df,
        checkpoint_path=args.checkpoint,
        visual_features_path=args.npz_path,
    )
    if not pipeline.model_is_loaded:
        print(f"Hybrid MAP@12: skipped, checkpoint could not be loaded from {args.checkpoint}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = VisualFeatureExtractor(
        args.npz_path,
        item_id_to_article_id=build_item_to_article_id_mapping(articles_df),
    )
    metadata_encoder = MetadataFeatureEncoder(articles_df)

    hybrid_map = evaluate_map_at_12_on_test(
        model=pipeline.model,
        test_df=test_df,
        feature_extractor=feature_extractor,
        metadata_encoder=metadata_encoder,
        device=device,
        all_item_id_pool=train_df["item_id"].unique().to_numpy(),
        user_to_purchased_items=build_user_to_seen_items(train_df),
        max_negative_candidates_per_user=args.negative_candidates,
        random_seed=args.seed,
    )
    print(f"Hybrid MAP@12: {hybrid_map:.6f}")


if __name__ == "__main__":
    main()
