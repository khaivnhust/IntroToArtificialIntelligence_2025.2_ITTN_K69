"""
Evaluate saved recommender checkpoints without retraining.

Use this after training MF/NCF with ``train_compare_recommenders.py`` and
Hybrid with ``train_hybrid.py``. The script writes a metrics CSV and comparison
plot under ``reports/test_results`` by default.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_compare_recommenders import (  # noqa: E402
    build_evaluation_data,
    build_user_to_seen_items,
    evaluate_hybrid_checkpoint,
    evaluate_popularity,
    evaluate_torch_cf_model,
    load_data,
    log_runtime_environment,
    parse_model_list,
    plot_results,
    positive_int_or_none,
    write_metrics_csv,
)
from src.config import (  # noqa: E402
    BEST_CHECKPOINT_PATH,
    CHECKPOINT_DIR,
    DATA_DIR,
    MF_EMBEDDING_DIM,
    MLP_LAYER_SIZES,
    VISUAL_FEATURES_NPZ_PATH,
)
from src.models.matrix_factorization import MatrixFactorization  # noqa: E402
from src.models.ncf import NeuralCollaborativeFiltering  # noqa: E402


logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_state_dict_if_exists(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> bool:
    if not checkpoint_path.exists():
        logger.warning("Checkpoint not found: %s", checkpoint_path)
        return False

    logger.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    logger.info("Checkpoint loaded: %s", checkpoint_path)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved recommender checkpoints.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "test_results")
    parser.add_argument("--models", type=parse_model_list, default=parse_model_list("popularity,mf,ncf,hybrid"))
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--mf-checkpoint", type=Path, default=CHECKPOINT_DIR / "mf_best.pt")
    parser.add_argument("--ncf-checkpoint", type=Path, default=CHECKPOINT_DIR / "ncf_best.pt")
    parser.add_argument("--hybrid-checkpoint", type=Path, default=BEST_CHECKPOINT_PATH)
    parser.add_argument("--visual-features", type=Path, default=VISUAL_FEATURES_NPZ_PATH)
    parser.add_argument("--max-train-rows", type=positive_int_or_none, default=None)
    parser.add_argument("--max-eval-users", type=positive_int_or_none, default=1000)
    parser.add_argument("--negative-candidates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--mf-dim", type=int, default=MF_EMBEDDING_DIM)
    parser.add_argument("--mlp-layers", type=int, nargs="+", default=list(MLP_LAYER_SIZES))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_runtime_environment(device)

    train_df, test_df, articles_df = load_data(args.data_dir, args.max_train_rows)
    user_to_seen_items = build_user_to_seen_items(train_df)
    all_item_id_pool = train_df["item_id"].unique().to_numpy()
    max_user_id = max(int(train_df["user_id"].max()), int(test_df["user_id"].max()))
    max_item_id = max(int(train_df["item_id"].max()), int(test_df["item_id"].max()))
    num_users = max_user_id + 1
    num_items = max_item_id + 1

    evaluation_data = build_evaluation_data(
        test_df=test_df,
        all_item_id_pool=all_item_id_pool,
        user_to_seen_items=user_to_seen_items,
        max_negative_candidates_per_user=args.negative_candidates,
        max_eval_users=args.max_eval_users,
        seed=args.seed,
    )
    logger.info("Evaluation users: %s", f"{len(evaluation_data.users):,}")

    rows: list[dict[str, str | float]] = []

    if "popularity" in args.models:
        logger.info("Testing Popularity baseline ...")
        rows.append({"model": "Popularity", **evaluate_popularity(train_df, evaluation_data)})

    if "mf" in args.models:
        logger.info("Testing MF checkpoint ...")
        model = MatrixFactorization(num_users=num_users, num_items=num_items, embedding_dim=args.mf_dim).to(device)
        if load_state_dict_if_exists(model, args.mf_checkpoint, device):
            rows.append({"model": "MF", **evaluate_torch_cf_model(model, evaluation_data, device, args.batch_size)})

    if "ncf" in args.models:
        logger.info("Testing NCF checkpoint ...")
        model = NeuralCollaborativeFiltering(
            num_users=num_users,
            num_items=num_items,
            mf_embedding_dim=args.mf_dim,
            mlp_layer_sizes=args.mlp_layers,
        ).to(device)
        if load_state_dict_if_exists(model, args.ncf_checkpoint, device):
            rows.append({"model": "NCF", **evaluate_torch_cf_model(model, evaluation_data, device, args.batch_size)})

    if "hybrid" in args.models:
        logger.info("Testing Hybrid checkpoint ...")
        metrics = evaluate_hybrid_checkpoint(
            checkpoint_path=args.hybrid_checkpoint,
            articles_df=articles_df,
            evaluation_data=evaluation_data,
            num_users=num_users,
            num_items=num_items,
            mf_embedding_dim=args.mf_dim,
            mlp_layer_sizes=args.mlp_layers,
            visual_features_path=args.visual_features,
            device=device,
            batch_size=args.batch_size,
        )
        if metrics is not None:
            rows.append({"model": "Hybrid", **metrics})

    metrics_path = args.output_dir / "metrics_comparison.csv"
    write_metrics_csv(rows, metrics_path)
    plot_results(rows, histories={}, output_dir=args.output_dir)

    logger.info("Wrote metrics: %s", metrics_path)
    for row in rows:
        logger.info(
            "%s | MAP@12=%.6f | HitRate@12=%.6f | NDCG@12=%.6f",
            row["model"],
            float(row["map_at_12"]),
            float(row["hit_rate_at_12"]),
            float(row["ndcg_at_12"]),
        )


if __name__ == "__main__":
    main()
