"""
train_hybrid.py — Training pipeline for the Hybrid NCF + Visual Features model.

Key techniques
--------------
- Negative Sampling  : for each positive (user, item) pair, sample N negatives.
- Mixed Precision    : torch.amp.autocast + GradScaler for faster GPU training.
- Early Stopping     : halt training when validation loss stops improving.
- Checkpoint Saving  : best model weights saved to checkpoints/.
- Logging            : all training progress is written via Python logging.

Usage
-----
    python scripts/train_hybrid.py
    python scripts/train_hybrid.py --batch-size 2048 --num-epochs 50 --no-amp
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so ``from src.…`` imports work when
# this script is invoked directly (e.g. ``python scripts/train_hybrid.py``).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainingConfig, BEST_CHECKPOINT_PATH  # noqa: E402
from src.evaluation.metrics import calculate_map_at_12  # noqa: E402
from src.features.visual_feature_extractor import VisualFeatureExtractor  # noqa: E402
from src.models.hybrid_model import HybridRecommendationModel  # noqa: E402
from src.preprocessing.data_loader import DataLoaderPolars  # noqa: E402
from src.utils.early_stopping import EarlyStopping  # noqa: E402

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Negative-Sampling Dataset
# ═══════════════════════════════════════════════════════════════════════════
class NegativeSamplingDataset(Dataset):
    """Creates (user, item, visual_feature, label) samples.

    For every positive interaction a configurable number of negative
    (user, random_unseen_item) pairs are generated.  Call ``resample()``
    at the start of each training epoch to get fresh negatives.
    """

    def __init__(
        self,
        positive_user_ids: np.ndarray,
        positive_item_ids: np.ndarray,
        all_item_id_pool: np.ndarray,
        user_to_purchased_items: Dict[int, set],
        num_negatives_per_positive: int,
        feature_extractor: VisualFeatureExtractor,
    ) -> None:
        self._positive_user_ids = positive_user_ids
        self._positive_item_ids = positive_item_ids
        self._all_item_id_pool = all_item_id_pool
        self._user_to_purchased_items = user_to_purchased_items
        self._num_negatives = num_negatives_per_positive
        self._feature_extractor = feature_extractor

        # Populated by resample()
        self._sampled_users: List[int] = []
        self._sampled_items: List[int] = []
        self._sampled_labels: List[float] = []

        self.resample()

    # ------------------------------------------------------------------
    def resample(self) -> None:
        """Re-generate positive + negative samples (call once per epoch)."""
        users: List[int] = []
        items: List[int] = []
        labels: List[float] = []

        for user_id, item_id in zip(self._positive_user_ids, self._positive_item_ids):
            user_id_int = int(user_id)
            item_id_int = int(item_id)

            # Positive sample
            users.append(user_id_int)
            items.append(item_id_int)
            labels.append(1.0)

            # Negative samples
            purchased_items = self._user_to_purchased_items.get(user_id_int, set())
            negatives_found = 0
            max_attempts = self._num_negatives * 10

            for _ in range(max_attempts):
                if negatives_found >= self._num_negatives:
                    break
                candidate_item = int(random.choice(self._all_item_id_pool))
                if candidate_item not in purchased_items:
                    users.append(user_id_int)
                    items.append(candidate_item)
                    labels.append(0.0)
                    negatives_found += 1

        self._sampled_users = users
        self._sampled_items = items
        self._sampled_labels = labels

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._sampled_labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        user_tensor = torch.tensor(self._sampled_users[index], dtype=torch.long)
        item_tensor = torch.tensor(self._sampled_items[index], dtype=torch.long)
        label_tensor = torch.tensor(self._sampled_labels[index], dtype=torch.float32)
        visual_tensor = self._feature_extractor.get_feature_vectors(
            [self._sampled_items[index]]
        ).squeeze(0)
        return user_tensor, item_tensor, visual_tensor, label_tensor


# ═══════════════════════════════════════════════════════════════════════════
# Data Preparation
# ═══════════════════════════════════════════════════════════════════════════
def prepare_training_data(config: TrainingConfig):
    """Load datasets and build interaction mappings for training.

    Returns
    -------
    tuple containing:
        train_user_ids, train_item_ids,
        validation_user_ids, validation_item_ids,
        test_df,
        all_item_id_pool,
        user_to_purchased_items,
        num_users, num_items
    """
    logger.info("Loading Parquet files from %s ...", config.data_dir)

    data_loader = DataLoaderPolars(
        articles_path=config.data_dir / "articles_cleaned.parquet",
        customers_path=config.data_dir / "customers_fixed.parquet",
        train_path=config.data_dir / "hm_train.parquet",
        test_path=config.data_dir / "hm_test.parquet",
    )
    train_df, test_df, _, _ = data_loader.load_all_dataframes()

    logger.info("  Train rows : %s", f"{len(train_df):,}")
    logger.info("  Test rows  : %s", f"{len(test_df):,}")

    # ID space
    num_users = int(train_df["user_id"].max()) + 1
    num_items = int(train_df["item_id"].max()) + 1
    logger.info("  num_users  : %s  |  num_items : %s", f"{num_users:,}", f"{num_items:,}")

    # Build user -> set(purchased items) for negative sampling
    logger.info("Building user-to-purchased-items mapping ...")
    grouped = train_df.group_by("user_id").agg(pl.col("item_id").alias("items"))
    user_to_purchased_items: Dict[int, set] = {
        int(row["user_id"]): set(row["items"])
        for row in grouped.iter_rows(named=True)
    }

    all_item_id_pool = train_df["item_id"].unique().to_numpy()

    # Train / validation split (last ~5% as quick validation)
    total_rows = len(train_df)
    split_index = int(total_rows * 0.95)

    train_slice = train_df.slice(0, split_index)
    validation_slice = train_df.slice(split_index, total_rows - split_index)

    train_user_ids = train_slice["user_id"].to_numpy()
    train_item_ids = train_slice["item_id"].to_numpy()
    validation_user_ids = validation_slice["user_id"].to_numpy()
    validation_item_ids = validation_slice["item_id"].to_numpy()

    return (
        train_user_ids,
        train_item_ids,
        validation_user_ids,
        validation_item_ids,
        test_df,
        all_item_id_pool,
        user_to_purchased_items,
        num_users,
        num_items,
    )


# ═══════════════════════════════════════════════════════════════════════════
# MAP@12 Evaluation
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_map_at_12_on_test(
    model: HybridRecommendationModel,
    test_df: pl.DataFrame,
    feature_extractor: VisualFeatureExtractor,
    device: torch.device,
) -> float:
    """Score candidate items per test user and compute MAP@12.

    For efficiency, only items that appear in the test set for each user
    are scored (rather than the full item catalogue).
    """
    model.eval()

    test_grouped = test_df.group_by("user_id").agg(
        pl.col("item_id").alias("items")
    )

    predictions: Dict[int, List[int]] = {}

    with torch.no_grad():
        for row in test_grouped.iter_rows(named=True):
            user_id = int(row["user_id"])
            candidate_item_ids = list(row["items"])

            if not candidate_item_ids:
                continue

            user_tensor = torch.tensor(
                [user_id] * len(candidate_item_ids), dtype=torch.long, device=device
            )
            item_tensor = torch.tensor(
                candidate_item_ids, dtype=torch.long, device=device
            )
            visual_tensor = feature_extractor.get_feature_vectors(
                candidate_item_ids
            ).to(device)

            scores = model(user_tensor, item_tensor, visual_tensor).cpu().numpy()
            top_12_indices = np.argsort(scores)[::-1][:12]
            predictions[user_id] = [candidate_item_ids[i] for i in top_12_indices]

    return calculate_map_at_12(predictions, test_df)


# ═══════════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════════
def train(config: TrainingConfig) -> None:
    """Full training pipeline: data → model → train → evaluate → save."""

    # -- Reproducibility ------------------------------------------------------
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    # -- Device ---------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_mixed_precision = config.use_mixed_precision and device.type == "cuda"
    logger.info("Device: %s  |  Mixed Precision: %s", device, use_mixed_precision)

    # -- Checkpoint directories -----------------------------------------------
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "hybrid_best.pt"
    last_checkpoint_path = checkpoint_dir / "hybrid_last.pt"

    # -- Data -----------------------------------------------------------------
    (
        train_user_ids,
        train_item_ids,
        validation_user_ids,
        validation_item_ids,
        test_df,
        all_item_id_pool,
        user_to_purchased_items,
        num_users,
        num_items,
    ) = prepare_training_data(config)

    # -- Visual feature extractor ---------------------------------------------
    logger.info("Loading visual features ...")
    feature_extractor = VisualFeatureExtractor(config.visual_features_path)

    # -- Model ----------------------------------------------------------------
    logger.info("Building HybridRecommendationModel ...")
    model = HybridRecommendationModel(
        num_users=num_users,
        num_items=num_items,
        visual_feature_dim=feature_extractor.feature_dim,
        mf_embedding_dim=config.mf_embedding_dim,
        mlp_layer_sizes=config.mlp_layer_sizes,
    ).to(device)

    trainable_parameter_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info("Trainable parameters: %s", f"{trainable_parameter_count:,}")

    # -- Optimiser & loss -----------------------------------------------------
    loss_function = nn.BCELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5, min_lr=1e-6
    )
    gradient_scaler = GradScaler(enabled=use_mixed_precision)

    # -- Early stopping -------------------------------------------------------
    early_stopper = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
        checkpoint_path=best_checkpoint_path,
    )

    # -- Validation dataset (fewer negatives for speed) -----------------------
    validation_dataset = NegativeSamplingDataset(
        positive_user_ids=validation_user_ids,
        positive_item_ids=validation_item_ids,
        all_item_id_pool=all_item_id_pool,
        user_to_purchased_items=user_to_purchased_items,
        num_negatives_per_positive=1,
        feature_extractor=feature_extractor,
    )
    validation_data_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # -- Training loop --------------------------------------------------------
    logger.info("=" * 70)
    logger.info("Starting training ...")
    logger.info("=" * 70)

    for epoch in range(1, config.num_epochs + 1):
        epoch_start_time = time.time()

        # 1. Resample negatives for this epoch
        logger.info("Epoch %d/%d — resampling negatives ...", epoch, config.num_epochs)
        training_dataset = NegativeSamplingDataset(
            positive_user_ids=train_user_ids,
            positive_item_ids=train_item_ids,
            all_item_id_pool=all_item_id_pool,
            user_to_purchased_items=user_to_purchased_items,
            num_negatives_per_positive=config.num_negatives_per_positive,
            feature_extractor=feature_extractor,
        )
        training_data_loader = DataLoader(
            training_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.dataloader_num_workers,
            pin_memory=(device.type == "cuda"),
        )

        # 2. Train one epoch
        model.train()
        epoch_loss_sum = 0.0
        total_batches = len(training_data_loader)

        for batch_index, (users, items, visuals, labels) in enumerate(training_data_loader, 1):
            users = users.to(device, non_blocking=True)
            items = items.to(device, non_blocking=True)
            visuals = visuals.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_mixed_precision):
                predictions = model(users, items, visuals)
                batch_loss = loss_function(predictions, labels)

            gradient_scaler.scale(batch_loss).backward()
            gradient_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            gradient_scaler.step(optimizer)
            gradient_scaler.update()

            epoch_loss_sum += batch_loss.item()

            # Log progress ~5 times per epoch
            if batch_index % max(1, total_batches // 5) == 0:
                logger.info(
                    "  [%5d/%d]  batch_loss=%.5f",
                    batch_index,
                    total_batches,
                    batch_loss.item(),
                )

        average_train_loss = epoch_loss_sum / total_batches

        # 3. Validation
        if epoch % config.evaluate_every_n_epochs == 0:
            model.eval()
            validation_loss_sum = 0.0

            with torch.no_grad():
                for users, items, visuals, labels in validation_data_loader:
                    users = users.to(device, non_blocking=True)
                    items = items.to(device, non_blocking=True)
                    visuals = visuals.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    with torch.autocast(device_type=device.type, enabled=use_mixed_precision):
                        predictions = model(users, items, visuals)
                        batch_loss = loss_function(predictions, labels)

                    validation_loss_sum += batch_loss.item()

            average_validation_loss = validation_loss_sum / len(validation_data_loader)
            epoch_elapsed_seconds = time.time() - epoch_start_time

            logger.info(
                "Epoch %3d | train_loss=%.5f | val_loss=%.5f | %.1fs",
                epoch,
                average_train_loss,
                average_validation_loss,
                epoch_elapsed_seconds,
            )

            lr_scheduler.step(average_validation_loss)

            if early_stopper.step(average_validation_loss, model, epoch):
                logger.info("Early stopping triggered after epoch %d.", epoch)
                break

        # 4. Save "last" checkpoint every epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
                "scaler_state_dict": gradient_scaler.state_dict(),
            },
            last_checkpoint_path,
        )

    # -- Final MAP@12 evaluation on test set ----------------------------------
    logger.info("=" * 70)
    logger.info("Loading best weights for final evaluation ...")
    if best_checkpoint_path.exists():
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))

    map_at_12_score = evaluate_map_at_12_on_test(
        model, test_df, feature_extractor, device
    )
    logger.info("Final MAP@12 on test set: %.6f", map_at_12_score)
    logger.info("=" * 70)
    logger.info("Best checkpoint : %s", best_checkpoint_path)
    logger.info("Last checkpoint : %s", last_checkpoint_path)


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════
def parse_command_line_arguments() -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description="Train the Hybrid NCF + Visual Features recommendation model."
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--npz-path", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--mf-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-negatives", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true", help="Disable Mixed Precision")
    args = parser.parse_args()

    config = TrainingConfig(
        mf_embedding_dim=args.mf_dim,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        num_negatives_per_positive=args.num_negatives,
        early_stopping_patience=args.patience,
        random_seed=args.seed,
        use_mixed_precision=not args.no_amp,
    )

    if args.data_dir:
        config.data_dir = Path(args.data_dir)
    if args.npz_path:
        config.visual_features_path = Path(args.npz_path)
    if args.checkpoint_dir:
        config.checkpoint_dir = Path(args.checkpoint_dir)

    return config


if __name__ == "__main__":
    training_config = parse_command_line_arguments()
    train(training_config)
