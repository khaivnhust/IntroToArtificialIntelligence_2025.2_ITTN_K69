"""
Train and compare recommender baselines.

This script covers the report workflow that is missing from the project:

- Popularity baseline evaluation.
- Matrix Factorization (MF) training and evaluation.
- Neural Collaborative Filtering (NCF) training and evaluation.
- Optional Hybrid checkpoint evaluation, if ``checkpoints/hybrid_best.pt`` exists.
- CSV and PNG plots for metric comparison and training loss curves.

The evaluation ranks each user's true test items mixed with sampled negatives.
That keeps comparisons tractable while using the same candidate set for all
models.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler

try:
    from torch.amp import GradScaler
except ImportError:  # pragma: no cover - compatibility with older PyTorch
    from torch.cuda.amp import GradScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_hybrid import build_item_to_article_id_mapping  # noqa: E402
from src.config import (  # noqa: E402
    BEST_CHECKPOINT_PATH,
    CHECKPOINT_DIR,
    DATA_DIR,
    MF_EMBEDDING_DIM,
    MLP_LAYER_SIZES,
    TOP_K_RECOMMENDATIONS,
    VISUAL_FEATURES_NPZ_PATH,
)
from src.data_processing.data_loader import DataLoaderPolars  # noqa: E402
from src.evaluation.metrics import average_precision_at_k  # noqa: E402
from src.features.metadata_feature_encoder import MetadataFeatureEncoder  # noqa: E402
from src.features.visual_feature_extract import VisualFeatureExtractor  # noqa: E402
from src.models.hybrid_model import HybridRecommendationModel  # noqa: E402
from src.models.matrix_factorization import MatrixFactorization  # noqa: E402
from src.models.ncf import NeuralCollaborativeFiltering  # noqa: E402
from src.models.popularity_baseline import PopularityBaseline  # noqa: E402


logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def create_gradient_scaler(enabled: bool) -> GradScaler:
    try:
        return GradScaler("cuda", enabled=enabled)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        return GradScaler(enabled=enabled)


def log_runtime_environment(device: torch.device) -> None:
    logger.info("Python executable: %s", sys.executable)
    logger.info("Torch version: %s", torch.__version__)
    logger.info("Selected device: %s", device)
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info("CUDA version: %s", torch.version.cuda)
        logger.info("GPU: %s | VRAM: %.1f GB", gpu_name, gpu_memory_gb)
    else:
        logger.warning("CUDA is not available; training/evaluation will run on CPU.")


def should_log_progress(index: int, total: int, step: int | None = None) -> bool:
    if total <= 0:
        return False
    if index == 1 or index == total:
        return True
    if step is not None and step > 0:
        return index % step == 0
    return index % max(1, total // 10) == 0


@dataclass
class EvaluationData:
    users: list[int]
    actual_by_user: dict[int, set[int]]
    candidates_by_user: dict[int, list[int]]


class ImplicitFeedbackDataset(Dataset):
    """Positive interactions plus lazily sampled user-specific negatives."""

    def __init__(
        self,
        positive_user_ids: np.ndarray,
        positive_item_ids: np.ndarray,
        all_item_id_pool: np.ndarray,
        user_to_seen_items: dict[int, set[int]],
        num_negatives_per_positive: int,
        seed: int,
    ) -> None:
        self._positive_user_ids = positive_user_ids
        self._positive_item_ids = positive_item_ids
        self._all_item_id_pool = all_item_id_pool
        self._user_to_seen_items = user_to_seen_items
        self._num_negatives = num_negatives_per_positive
        self._seed = int(seed)
        self._pool_size = len(all_item_id_pool)
        if self._pool_size <= 0:
            raise ValueError("all_item_id_pool must not be empty.")

    def resample(self) -> None:
        """No-op kept for compatibility; sampling happens in __getitem__."""
        pass

    def __len__(self) -> int:
        return len(self._positive_user_ids) * (1 + self._num_negatives)

    def __getitem__(self, index: int):
        group_size = 1 + self._num_negatives
        positive_index = index // group_size
        is_positive = index % group_size == 0

        user_id = int(self._positive_user_ids[positive_index])
        positive_item_id = int(self._positive_item_ids[positive_index])
        if is_positive:
            return user_id, positive_item_id, 1.0

        seen_items = self._user_to_seen_items.get(user_id, set())
        item_id = positive_item_id
        max_attempts = max(20, self._num_negatives * 10)
        state = (self._seed ^ ((index + 1) * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF
        for attempt in range(max_attempts):
            state = (
                state * 6364136223846793005
                + 1442695040888963407
                + attempt
            ) & 0xFFFFFFFFFFFFFFFF
            candidate = int(self._all_item_id_pool[state % self._pool_size])
            if candidate not in seen_items:
                item_id = candidate
                break

        return user_id, item_id, 0.0


def implicit_feedback_collate(batch):
    user_ids, item_ids, labels = zip(*batch)
    return (
        torch.tensor(user_ids, dtype=torch.long),
        torch.tensor(item_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )


def parse_model_list(value: str) -> list[str]:
    allowed = {"popularity", "mf", "ncf", "hybrid"}
    models = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(models) - allowed)
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown model(s): {', '.join(unknown)}")
    return models


def load_data(data_dir: Path, max_train_rows: int | None) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    logger.info("Loading parquet data from %s ...", data_dir)
    loader = DataLoaderPolars(
        articles_path=data_dir / "articles_cleaned.parquet",
        customers_path=data_dir / "customers_cleaned.parquet",
        train_path=data_dir / "hm_train.parquet",
        test_path=data_dir / "hm_test.parquet",
    )
    train_df, test_df, _, articles_df = loader.load_all_dataframes()
    logger.info(
        "Loaded data | train=%s rows | test=%s rows | articles=%s rows",
        f"{len(train_df):,}",
        f"{len(test_df):,}",
        f"{len(articles_df):,}",
    )
    if max_train_rows is not None and max_train_rows > 0:
        train_df = train_df.tail(max_train_rows)
        logger.info("Using last %s train rows for this run.", f"{len(train_df):,}")
    return train_df, test_df, articles_df


def build_user_to_seen_items(train_df: pl.DataFrame) -> dict[int, set[int]]:
    logger.info("Building user -> seen items map ...")
    grouped = train_df.group_by("user_id").agg(pl.col("item_id").alias("items"))
    mapping = {
        int(row["user_id"]): set(int(item_id) for item_id in row["items"])
        for row in grouped.iter_rows(named=True)
    }
    logger.info("Built seen-item map for %s users.", f"{len(mapping):,}")
    return mapping


def split_train_validation(
    train_df: pl.DataFrame,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    split_index = max(1, int(len(train_df) * (1.0 - validation_fraction)))
    train_slice = train_df.slice(0, split_index)
    validation_slice = train_df.slice(split_index, len(train_df) - split_index)
    return (
        train_slice["user_id"].to_numpy(),
        train_slice["item_id"].to_numpy(),
        validation_slice["user_id"].to_numpy(),
        validation_slice["item_id"].to_numpy(),
    )


def build_evaluation_data(
    test_df: pl.DataFrame,
    all_item_id_pool: np.ndarray,
    user_to_seen_items: dict[int, set[int]],
    max_negative_candidates_per_user: int,
    max_eval_users: int | None,
    seed: int,
) -> EvaluationData:
    rng = np.random.default_rng(seed)
    all_items = np.asarray(all_item_id_pool, dtype=np.int64)
    grouped = test_df.group_by("user_id").agg(pl.col("item_id").alias("items")).sort("user_id")
    total_groups = grouped.height
    target_users = min(total_groups, max_eval_users) if max_eval_users is not None else total_groups
    logger.info(
        "Building evaluation candidates for %s users with up to %s negatives/user ...",
        f"{target_users:,}",
        f"{max_negative_candidates_per_user:,}",
    )

    users: list[int] = []
    actual_by_user: dict[int, set[int]] = {}
    candidates_by_user: dict[int, list[int]] = {}

    for row_index, row in enumerate(grouped.iter_rows(named=True), start=1):
        if max_eval_users is not None and len(users) >= max_eval_users:
            break

        user_id = int(row["user_id"])
        actual_items = set(int(item_id) for item_id in row["items"])
        seen_items = user_to_seen_items.get(user_id, set())
        negative_pool = [
            int(item_id)
            for item_id in all_items
            if int(item_id) not in actual_items and int(item_id) not in seen_items
        ]
        if max_negative_candidates_per_user > 0 and len(negative_pool) > max_negative_candidates_per_user:
            negative_pool = rng.choice(
                negative_pool,
                size=max_negative_candidates_per_user,
                replace=False,
            ).tolist()

        candidates = list(actual_items) + [int(item_id) for item_id in negative_pool]
        if not candidates:
            continue

        users.append(user_id)
        actual_by_user[user_id] = actual_items
        candidates_by_user[user_id] = candidates
        if should_log_progress(len(users), target_users, step=500):
            logger.info(
                "Evaluation candidates: %s/%s users ready.",
                f"{len(users):,}",
                f"{target_users:,}",
            )

    return EvaluationData(users, actual_by_user, candidates_by_user)


def compute_metrics(
    predictions: dict[int, list[int]],
    actual_by_user: dict[int, set[int]],
    k: int = TOP_K_RECOMMENDATIONS,
) -> dict[str, float]:
    if not actual_by_user:
        return {"map_at_12": 0.0, "hit_rate_at_12": 0.0, "ndcg_at_12": 0.0}

    total_ap = 0.0
    hit_count = 0
    total_ndcg = 0.0

    for user_id, actual_items in actual_by_user.items():
        ranked_items = predictions.get(user_id, [])
        total_ap += average_precision_at_k(ranked_items, actual_items, k=k)
        if any(item_id in actual_items for item_id in ranked_items[:k]):
            hit_count += 1

        dcg = 0.0
        for rank, item_id in enumerate(ranked_items[:k]):
            if item_id in actual_items:
                dcg += 1.0 / np.log2(rank + 2)
        ideal_hits = min(len(actual_items), k)
        idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
        total_ndcg += (dcg / idcg) if idcg > 0 else 0.0

    user_count = len(actual_by_user)
    return {
        "map_at_12": total_ap / user_count,
        "hit_rate_at_12": hit_count / user_count,
        "ndcg_at_12": total_ndcg / user_count,
    }


def evaluate_popularity(
    train_df: pl.DataFrame,
    evaluation_data: EvaluationData,
) -> dict[str, float]:
    baseline = PopularityBaseline(top_k=TOP_K_RECOMMENDATIONS)
    baseline.fit(train_df)

    popularity_counts = (
        train_df.group_by("item_id")
        .agg(pl.len().alias("purchase_count"))
        .to_dict(as_series=False)
    )
    item_to_count = {
        int(item_id): int(count)
        for item_id, count in zip(popularity_counts["item_id"], popularity_counts["purchase_count"])
    }

    predictions: dict[int, list[int]] = {}
    total_users = len(evaluation_data.users)
    logger.info("Popularity evaluation: ranking %s users ...", f"{total_users:,}")
    for index, user_id in enumerate(evaluation_data.users, start=1):
        candidates = evaluation_data.candidates_by_user[user_id]
        ranked = sorted(candidates, key=lambda item_id: item_to_count.get(int(item_id), 0), reverse=True)
        predictions[user_id] = ranked[:TOP_K_RECOMMENDATIONS]
        if should_log_progress(index, total_users, step=1000):
            logger.info("Popularity evaluation: %s/%s users ranked.", f"{index:,}", f"{total_users:,}")

    return compute_metrics(predictions, evaluation_data.actual_by_user)


def train_collaborative_model(
    model_name: str,
    model: nn.Module,
    train_user_ids: np.ndarray,
    train_item_ids: np.ndarray,
    validation_user_ids: np.ndarray,
    validation_item_ids: np.ndarray,
    all_item_id_pool: np.ndarray,
    user_to_seen_items: dict[int, set[int]],
    num_negatives: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    checkpoint_path: Path,
    seed: int,
    use_mixed_precision: bool,
) -> list[dict[str, float]]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    gradient_scaler = create_gradient_scaler(use_mixed_precision)
    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    data_loader_workers = 4

    validation_dataset = ImplicitFeedbackDataset(
        validation_user_ids,
        validation_item_ids,
        all_item_id_pool,
        user_to_seen_items,
        num_negatives_per_positive=1,
        seed=seed + 1000,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=data_loader_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=implicit_feedback_collate,
    )

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_dataset = ImplicitFeedbackDataset(
            train_user_ids,
            train_item_ids,
            all_item_id_pool,
            user_to_seen_items,
            num_negatives_per_positive=num_negatives,
            seed=seed + epoch,
        )
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(seed + epoch)
        training_sampler = RandomSampler(
            train_dataset,
            replacement=True,
            num_samples=len(train_dataset),
            generator=sampler_generator,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=training_sampler,
            num_workers=data_loader_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=implicit_feedback_collate,
        )
        logger.info(
            "%s epoch %d/%d | training samples=%s | batches=%s | sampler=replacement | amp=%s",
            model_name,
            epoch,
            num_epochs,
            f"{len(train_dataset):,}",
            f"{len(train_loader):,}",
            use_mixed_precision,
        )

        model.train()
        train_loss_sum = 0.0
        total_batches = len(train_loader)
        for batch_index, (users, items, labels) in enumerate(train_loader, start=1):
            users = users.to(device, non_blocking=True)
            items = items.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_mixed_precision):
                logits = model(users, items)
                loss = loss_fn(logits, labels)

            gradient_scaler.scale(loss).backward()
            gradient_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            gradient_scaler.step(optimizer)
            gradient_scaler.update()
            train_loss_sum += loss.item()
            if should_log_progress(batch_index, total_batches, step=500):
                logger.info(
                    "%s epoch %d/%d | batch %s/%s | batch_loss=%.5f",
                    model_name,
                    epoch,
                    num_epochs,
                    f"{batch_index:,}",
                    f"{total_batches:,}",
                    loss.item(),
                )

        train_loss = train_loss_sum / max(len(train_loader), 1)
        validation_loss = evaluate_bce_loss(
            model,
            validation_loader,
            loss_fn,
            device,
            use_mixed_precision,
        )
        elapsed = time.time() - start_time

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": validation_loss,
                "seconds": elapsed,
            }
        )
        logger.info(
            "%s epoch %d/%d | train_loss=%.5f | val_loss=%.5f | %.1fs",
            model_name,
            epoch,
            num_epochs,
            train_loss,
            validation_loss,
            elapsed,
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return history


def evaluate_bce_loss(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_mixed_precision: bool = False,
) -> float:
    model.eval()
    loss_sum = 0.0
    with torch.no_grad():
        for users, items, labels in data_loader:
            users = users.to(device, non_blocking=True)
            items = items.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_mixed_precision):
                loss_sum += loss_fn(model(users, items), labels).item()
    return loss_sum / max(len(data_loader), 1)


def evaluate_torch_cf_model(
    model: nn.Module,
    evaluation_data: EvaluationData,
    device: torch.device,
    batch_size: int,
    use_mixed_precision: bool = False,
) -> dict[str, float]:
    model.eval()
    predictions: dict[int, list[int]] = {}

    with torch.no_grad():
        total_users = len(evaluation_data.users)
        logger.info("Torch model evaluation: scoring %s users ...", f"{total_users:,}")
        for index, user_id in enumerate(evaluation_data.users, start=1):
            candidates = evaluation_data.candidates_by_user[user_id]
            scores: list[float] = []
            for start in range(0, len(candidates), batch_size):
                batch_items = candidates[start : start + batch_size]
                users = torch.full((len(batch_items),), user_id, dtype=torch.long, device=device)
                items = torch.tensor(batch_items, dtype=torch.long, device=device)
                with torch.autocast(device_type=device.type, enabled=use_mixed_precision):
                    batch_scores = model(users, items).detach().cpu().numpy().tolist()
                scores.extend(float(score) for score in batch_scores)

            top_indices = np.argsort(scores)[::-1][:TOP_K_RECOMMENDATIONS]
            predictions[user_id] = [candidates[index] for index in top_indices]
            if should_log_progress(index, total_users, step=500):
                logger.info(
                    "Torch model evaluation: %s/%s users scored.",
                    f"{index:,}",
                    f"{total_users:,}",
                )

    return compute_metrics(predictions, evaluation_data.actual_by_user)


def evaluate_hybrid_checkpoint(
    checkpoint_path: Path,
    articles_df: pl.DataFrame,
    evaluation_data: EvaluationData,
    num_users: int,
    num_items: int,
    mf_embedding_dim: int,
    mlp_layer_sizes: Sequence[int],
    visual_features_path: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, float] | None:
    if not checkpoint_path.exists():
        logger.warning("Hybrid checkpoint not found: %s", checkpoint_path)
        return None

    item_to_article_id = build_item_to_article_id_mapping(articles_df)
    feature_extractor = VisualFeatureExtractor(
        visual_features_path,
        item_id_to_article_id=item_to_article_id,
    )
    metadata_encoder = MetadataFeatureEncoder(articles_df)
    model = HybridRecommendationModel(
        num_users=num_users,
        num_items=num_items,
        visual_feature_dim=feature_extractor.feature_dim,
        metadata_feature_dim=metadata_encoder.feature_dim,
        mf_embedding_dim=mf_embedding_dim,
        mlp_layer_sizes=list(mlp_layer_sizes),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    predictions: dict[int, list[int]] = {}
    with torch.no_grad():
        total_users = len(evaluation_data.users)
        logger.info("Hybrid evaluation: scoring %s users ...", f"{total_users:,}")
        for index, user_id in enumerate(evaluation_data.users, start=1):
            candidates = evaluation_data.candidates_by_user[user_id]
            scores: list[float] = []
            for start in range(0, len(candidates), batch_size):
                batch_items = candidates[start : start + batch_size]
                users = torch.full((len(batch_items),), user_id, dtype=torch.long, device=device)
                items = torch.tensor(batch_items, dtype=torch.long, device=device)
                visual = feature_extractor.get_feature_vectors(batch_items).to(device)
                metadata = metadata_encoder.get_feature_vectors(batch_items).to(device)
                batch_scores = model(users, items, visual, metadata).detach().cpu().numpy().tolist()
                scores.extend(float(score) for score in batch_scores)

            top_indices = np.argsort(scores)[::-1][:TOP_K_RECOMMENDATIONS]
            predictions[user_id] = [candidates[index] for index in top_indices]
            if should_log_progress(index, total_users, step=500):
                logger.info("Hybrid evaluation: %s/%s users scored.", f"{index:,}", f"{total_users:,}")

    return compute_metrics(predictions, evaluation_data.actual_by_user)


def write_metrics_csv(rows: list[dict[str, str | float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["model", "map_at_12", "hit_rate_at_12", "ndcg_at_12"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_history_csv(histories: dict[str, list[dict[str, float]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["model", "epoch", "train_loss", "val_loss", "seconds"],
        )
        writer.writeheader()
        for model_name, history in histories.items():
            for row in history:
                writer.writerow({"model": model_name, **row})


def plot_results(
    metrics_rows: list[dict[str, str | float]],
    histories: dict[str, list[dict[str, float]]],
    output_dir: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed; CSV files were written but plots were skipped.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = [str(row["model"]) for row in metrics_rows]
    metric_names = ["map_at_12", "hit_rate_at_12", "ndcg_at_12"]

    fig, axes = plt.subplots(1, len(metric_names), figsize=(14, 4))
    for axis, metric_name in zip(axes, metric_names):
        values = [float(row[metric_name]) for row in metrics_rows]
        axis.bar(model_names, values, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"][: len(model_names)])
        axis.set_title(metric_name)
        axis.set_ylim(bottom=0)
        axis.tick_params(axis="x", rotation=25)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "model_metric_comparison.png", dpi=160)
    plt.close(fig)

    if histories:
        fig, axis = plt.subplots(figsize=(8, 5))
        for model_name, history in histories.items():
            epochs = [row["epoch"] for row in history]
            train_losses = [row["train_loss"] for row in history]
            validation_losses = [row["val_loss"] for row in history]
            axis.plot(epochs, train_losses, marker="o", label=f"{model_name} train")
            axis.plot(epochs, validation_losses, marker="s", linestyle="--", label=f"{model_name} val")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("BCE loss")
        axis.set_title("Training Curves")
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "training_loss_curves.png", dpi=160)
        plt.close(fig)


def positive_int_or_none(value: str) -> int | None:
    if value.lower() in {"none", "all", "0"}:
        return None
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare recommender models.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "model_comparison")
    parser.add_argument("--models", type=parse_model_list, default=parse_model_list("popularity,mf,ncf,hybrid"))
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--hybrid-checkpoint", type=Path, default=BEST_CHECKPOINT_PATH)
    parser.add_argument("--visual-features", type=Path, default=VISUAL_FEATURES_NPZ_PATH)
    parser.add_argument("--max-train-rows", type=positive_int_or_none, default=None)
    parser.add_argument("--max-eval-users", type=positive_int_or_none, default=1000)
    parser.add_argument("--negative-candidates", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-negatives", type=int, default=4)
    parser.add_argument("--mf-dim", type=int, default=MF_EMBEDDING_DIM)
    parser.add_argument("--mlp-layers", type=int, nargs="+", default=list(MLP_LAYER_SIZES))
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP for MF/NCF training and evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_runtime_environment(device)
    use_mixed_precision = device.type == "cuda" and not args.no_amp
    logger.info("MF/NCF Mixed Precision: %s", use_mixed_precision)

    train_df, test_df, articles_df = load_data(args.data_dir, args.max_train_rows)
    user_to_seen_items = build_user_to_seen_items(train_df)
    all_item_id_pool = train_df["item_id"].unique().to_numpy()
    max_user_id = max(int(train_df["user_id"].max()), int(test_df["user_id"].max()))
    max_item_id = max(int(train_df["item_id"].max()), int(test_df["item_id"].max()))
    num_users = max_user_id + 1
    num_items = max_item_id + 1

    logger.info("Train rows: %s | Test rows: %s", f"{len(train_df):,}", f"{len(test_df):,}")
    logger.info("Users: %s | Items: %s", f"{num_users:,}", f"{num_items:,}")

    evaluation_data = build_evaluation_data(
        test_df=test_df,
        all_item_id_pool=all_item_id_pool,
        user_to_seen_items=user_to_seen_items,
        max_negative_candidates_per_user=args.negative_candidates,
        max_eval_users=args.max_eval_users,
        seed=args.seed,
    )
    logger.info("Evaluation users: %s", f"{len(evaluation_data.users):,}")

    metrics_rows: list[dict[str, str | float]] = []
    histories: dict[str, list[dict[str, float]]] = {}

    train_user_ids, train_item_ids, val_user_ids, val_item_ids = split_train_validation(
        train_df, args.validation_fraction
    )

    if "mf" in args.models:
        logger.info("Training Matrix Factorization ...")
        mf_model = MatrixFactorization(num_users=num_users, num_items=num_items, embedding_dim=args.mf_dim)
        history = train_collaborative_model(
            model_name="MF",
            model=mf_model,
            train_user_ids=train_user_ids,
            train_item_ids=train_item_ids,
            validation_user_ids=val_user_ids,
            validation_item_ids=val_item_ids,
            all_item_id_pool=all_item_id_pool,
            user_to_seen_items=user_to_seen_items,
            num_negatives=args.num_negatives,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            checkpoint_path=args.checkpoint_dir / "mf_best.pt",
            seed=args.seed,
            use_mixed_precision=use_mixed_precision,
        )
        histories["MF"] = history
        metrics = evaluate_torch_cf_model(
            mf_model,
            evaluation_data,
            device,
            args.batch_size,
            use_mixed_precision=use_mixed_precision,
        )
        metrics_rows.append({"model": "MF", **metrics})

    if "ncf" in args.models:
        logger.info("Training NCF ...")
        ncf_model = NeuralCollaborativeFiltering(
            num_users=num_users,
            num_items=num_items,
            mf_embedding_dim=args.mf_dim,
            mlp_layer_sizes=args.mlp_layers,
        )
        history = train_collaborative_model(
            model_name="NCF",
            model=ncf_model,
            train_user_ids=train_user_ids,
            train_item_ids=train_item_ids,
            validation_user_ids=val_user_ids,
            validation_item_ids=val_item_ids,
            all_item_id_pool=all_item_id_pool,
            user_to_seen_items=user_to_seen_items,
            num_negatives=args.num_negatives,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            checkpoint_path=args.checkpoint_dir / "ncf_best.pt",
            seed=args.seed + 10,
            use_mixed_precision=use_mixed_precision,
        )
        histories["NCF"] = history
        metrics = evaluate_torch_cf_model(
            ncf_model,
            evaluation_data,
            device,
            args.batch_size,
            use_mixed_precision=use_mixed_precision,
        )
        metrics_rows.append({"model": "NCF", **metrics})

    if "popularity" in args.models:
        logger.info("Evaluating Popularity baseline ...")
        metrics = evaluate_popularity(train_df, evaluation_data)
        metrics_rows.append({"model": "Popularity", **metrics})

    if "hybrid" in args.models:
        logger.info("Evaluating Hybrid checkpoint ...")
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
            metrics_rows.append({"model": "Hybrid", **metrics})

    metrics_path = args.output_dir / "metrics_comparison.csv"
    history_path = args.output_dir / "training_history.csv"
    write_metrics_csv(metrics_rows, metrics_path)
    write_history_csv(histories, history_path)
    plot_results(metrics_rows, histories, args.output_dir)

    logger.info("Wrote metrics: %s", metrics_path)
    logger.info("Wrote history: %s", history_path)
    for row in metrics_rows:
        logger.info(
            "%s | MAP@12=%.6f | HitRate@12=%.6f | NDCG@12=%.6f",
            row["model"],
            float(row["map_at_12"]),
            float(row["hit_rate_at_12"]),
            float(row["ndcg_at_12"]),
        )


if __name__ == "__main__":
    main()
