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
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler

try:
    from torch.amp import GradScaler

    _GRAD_SCALER_SUPPORTS_DEVICE_TYPE = True
except ImportError:
    from torch.cuda.amp import GradScaler

    _GRAD_SCALER_SUPPORTS_DEVICE_TYPE = False

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so ``from src.…`` imports work when
# this script is invoked directly (e.g. ``python scripts/train_hybrid.py``).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainingConfig, BEST_CHECKPOINT_PATH  # noqa: E402
from src.evaluation.metrics import calculate_map_at_12  # noqa: E402
from src.features.metadata_feature_encoder import MetadataFeatureEncoder  # noqa: E402
from src.features.visual_feature_extract import VisualFeatureExtractor  # noqa: E402
from src.models.hybrid_model import HybridRecommendationModel  # noqa: E402
from src.data_processing.data_loader import DataLoaderPolars  # noqa: E402
from src.utils.early_stopping import EarlyStopping  # noqa: E402

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def should_log_progress(index: int, total: int, step: int | None = None) -> bool:
    if total <= 0:
        return False
    if index == 1 or index == total:
        return True
    if step is not None and step > 0:
        return index % step == 0
    return index % max(1, total // 10) == 0


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(int(seconds), 60)
    hours, remaining_minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{remaining_minutes:02d}m{remaining_seconds:02d}s"
    return f"{remaining_minutes}m{remaining_seconds:02d}s"


def create_gradient_scaler(enabled: bool) -> GradScaler:
    if _GRAD_SCALER_SUPPORTS_DEVICE_TYPE:
        return GradScaler("cuda", enabled=enabled)
    return GradScaler(enabled=enabled)


class ProgressHeartbeat:
    """Periodically log progress while a long blocking operation is running."""

    def __init__(
        self,
        message_factory: Callable[[], str],
        interval_seconds: int,
    ) -> None:
        self._message_factory = message_factory
        self._interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "ProgressHeartbeat":
        if self._interval_seconds > 0:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            logger.info("Still running: %s", self._message_factory())


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


# ═══════════════════════════════════════════════════════════════════════════
# Negative-Sampling Dataset
# ═══════════════════════════════════════════════════════════════════════════
class NegativeSamplingDataset(Dataset):
    """Creates (user, item, label) samples.

    For every positive interaction a configurable number of negative
    (user, random_unseen_item) pairs are generated lazily.
    """

    def __init__(
        self,
        positive_user_ids: np.ndarray,
        positive_item_ids: np.ndarray,
        all_item_id_pool: np.ndarray,
        user_to_purchased_items: Dict[int, set],
        num_negatives_per_positive: int,
        feature_extractor: VisualFeatureExtractor,
        metadata_encoder: MetadataFeatureEncoder | None = None,
        seed: int | None = None,
    ) -> None:
        self._positive_user_ids = positive_user_ids
        self._positive_item_ids = positive_item_ids
        self._all_item_id_pool = all_item_id_pool
        self._user_to_purchased_items = user_to_purchased_items
        self._num_negatives = num_negatives_per_positive
        self._rng = random.Random(seed)

        # These arguments are accepted for backward compatibility. Feature
        # tensors are fetched by HybridBatchCollator once per batch.
        _ = feature_extractor, metadata_encoder

    # ------------------------------------------------------------------
    def resample(self) -> None:
        """No-op. Negative sampling is now performed lazily in __getitem__."""
        pass

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._positive_user_ids) * (1 + self._num_negatives)

    def __getitem__(self, index: int) -> Tuple[int, int, float]:
        group_size = 1 + self._num_negatives
        pos_index = index // group_size
        is_positive = (index % group_size == 0)

        user_id_int = int(self._positive_user_ids[pos_index])

        if is_positive:
            item_id_int = int(self._positive_item_ids[pos_index])
            label = 1.0
        else:
            purchased_items = self._user_to_purchased_items.get(user_id_int, set())
            item_id_int = int(self._positive_item_ids[pos_index])  # fallback
            label = 0.0
            max_attempts = max(20, self._num_negatives * 10)
            for _ in range(max_attempts):
                candidate_item = int(self._rng.choice(self._all_item_id_pool))
                if candidate_item not in purchased_items:
                    item_id_int = candidate_item
                    break

        return user_id_int, item_id_int, label


class HybridBatchCollator:
    """Build tensors and fetch content features once per batch."""

    def __init__(
        self,
        feature_extractor: VisualFeatureExtractor,
        metadata_encoder: MetadataFeatureEncoder | None = None,
    ) -> None:
        self._feature_extractor = feature_extractor
        self._metadata_encoder = metadata_encoder

    def __call__(
        self,
        batch: List[Tuple[int, int, float]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        user_ids, item_ids, labels = zip(*batch)
        item_id_list = [int(item_id) for item_id in item_ids]

        users = torch.tensor(user_ids, dtype=torch.long)
        items = torch.tensor(item_id_list, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.float32)
        visual_tensor = self._feature_extractor.get_feature_vectors(item_id_list)
        if self._metadata_encoder is None:
            metadata_tensor = torch.empty((len(item_id_list), 0), dtype=torch.float32)
        else:
            metadata_tensor = self._metadata_encoder.get_feature_vectors(item_id_list)

        return users, items, visual_tensor, metadata_tensor, label_tensor


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
        articles_df,
        all_item_id_pool,
        user_to_purchased_items,
        num_users, num_items
    """
    logger.info("Loading Parquet files from %s ...", config.data_dir)

    data_loader = DataLoaderPolars(
        articles_path=config.data_dir / "articles_cleaned.parquet",
        customers_path=config.data_dir / "customers_cleaned.parquet",
        train_path=config.data_dir / "hm_train.parquet",
        test_path=config.data_dir / "hm_test.parquet",
    )
    train_df, test_df, _, articles_df = data_loader.load_all_dataframes()

    if config.max_train_rows is not None and config.max_train_rows > 0:
        original_train_rows = len(train_df)
        train_df = train_df.tail(config.max_train_rows)
        logger.info(
            "Using last %s/%s train rows for this run.",
            f"{len(train_df):,}",
            f"{original_train_rows:,}",
        )

    logger.info("  Train rows : %s", f"{len(train_df):,}")
    logger.info("  Test rows  : %s", f"{len(test_df):,}")

    # ID space
    max_user_id = max(int(train_df["user_id"].max()), int(test_df["user_id"].max()))
    max_item_id = max(int(train_df["item_id"].max()), int(test_df["item_id"].max()))
    num_users = max_user_id + 1
    num_items = max_item_id + 1
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
        articles_df,
        all_item_id_pool,
        user_to_purchased_items,
        num_users,
        num_items,
    )


# ═══════════════════════════════════════════════════════════════════════════
# MAP@12 Evaluation
# ═══════════════════════════════════════════════════════════════════════════
def build_item_to_article_id_mapping(articles_df: pl.DataFrame) -> Dict[int, int]:
    """Build mapping from encoded item_id to raw H&M article_id."""
    if "item_id" not in articles_df.columns or "article_id" not in articles_df.columns:
        return {}
    return {
        int(row["item_id"]): int(row["article_id"])
        for row in articles_df.select(["item_id", "article_id"]).iter_rows(named=True)
    }


def sample_negative_candidates(
    all_items: np.ndarray,
    excluded_items: set[int],
    max_candidates: int,
    rng: np.random.Generator,
) -> list[int]:
    """Sample negatives without scanning the whole catalogue for every user."""
    if max_candidates <= 0:
        return [
            int(item_id)
            for item_id in all_items
            if int(item_id) not in excluded_items
        ]

    target_count = min(max_candidates, len(all_items))
    sampled: list[int] = []
    sampled_set: set[int] = set()
    max_attempts = max(target_count * 20, 1000)

    for _ in range(max_attempts):
        if len(sampled) >= target_count:
            break
        candidate = int(rng.choice(all_items))
        if candidate in excluded_items or candidate in sampled_set:
            continue
        sampled.append(candidate)
        sampled_set.add(candidate)

    if len(sampled) < target_count:
        for item_id in all_items:
            candidate = int(item_id)
            if candidate in excluded_items or candidate in sampled_set:
                continue
            sampled.append(candidate)
            sampled_set.add(candidate)
            if len(sampled) >= target_count:
                break

    return sampled


def evaluate_map_at_12_on_test(
    model: HybridRecommendationModel,
    test_df: pl.DataFrame,
    feature_extractor: VisualFeatureExtractor,
    metadata_encoder: MetadataFeatureEncoder | None,
    device: torch.device,
    all_item_id_pool: np.ndarray,
    user_to_purchased_items: Dict[int, set],
    max_negative_candidates_per_user: int = 1000,
    random_seed: int = 42,
    batch_size: int = 4096,
    max_eval_users: int | None = None,
    heartbeat_interval_seconds: int = 30,
) -> float:
    """Score candidate items per test user and compute MAP@12.

    Ground-truth test items are mixed with sampled negatives from the item
    catalogue. This avoids leakage from ranking only the user's actual test
    items while keeping evaluation tractable on local hardware.
    """
    model.eval()
    rng = np.random.default_rng(random_seed)
    all_items = np.asarray(all_item_id_pool, dtype=np.int64)

    test_grouped = test_df.group_by("user_id").agg(
        pl.col("item_id").alias("items")
    )

    total_users = test_grouped.height
    target_users = min(total_users, max_eval_users) if max_eval_users is not None else total_users
    predictions: Dict[int, List[int]] = {}
    progress_state = {
        "scored_users": 0,
        "target_users": target_users,
        "last_candidates": 0,
        "elapsed_start": time.time(),
    }

    logger.info(
        "Final MAP@12 evaluation: scoring %s/%s users with up to %s negatives/user ...",
        f"{target_users:,}",
        f"{total_users:,}",
        f"{max_negative_candidates_per_user:,}",
    )

    with torch.no_grad():
        with ProgressHeartbeat(
            lambda: (
                "MAP@12 evaluation "
                f"{progress_state['scored_users']:,}/{progress_state['target_users']:,} users scored; "
                f"last_candidates={progress_state['last_candidates']:,}; "
                f"elapsed={format_seconds(time.time() - progress_state['elapsed_start'])}"
            ),
            heartbeat_interval_seconds,
        ):
            for user_index, row in enumerate(test_grouped.iter_rows(named=True), start=1):
                if max_eval_users is not None and user_index > max_eval_users:
                    break

                user_id = int(row["user_id"])
                actual_items = set(int(item_id) for item_id in row["items"])
                seen_items = set(user_to_purchased_items.get(user_id, set()))
                excluded_items = actual_items | seen_items
                sampled_negatives = sample_negative_candidates(
                    all_items=all_items,
                    excluded_items=excluded_items,
                    max_candidates=max_negative_candidates_per_user,
                    rng=rng,
                )

                candidate_item_ids = list(actual_items) + [int(item_id) for item_id in sampled_negatives]
                progress_state["last_candidates"] = len(candidate_item_ids)

                if not candidate_item_ids:
                    continue

                score_chunks: list[np.ndarray] = []
                for start_index in range(0, len(candidate_item_ids), batch_size):
                    batch_item_ids = candidate_item_ids[start_index : start_index + batch_size]
                    user_tensor = torch.full(
                        (len(batch_item_ids),), user_id, dtype=torch.long, device=device
                    )
                    item_tensor = torch.tensor(
                        batch_item_ids, dtype=torch.long, device=device
                    )
                    visual_tensor = feature_extractor.get_feature_vectors(
                        batch_item_ids
                    ).to(device)
                    metadata_tensor = (
                        metadata_encoder.get_feature_vectors(batch_item_ids).to(device)
                        if metadata_encoder is not None
                        else None
                    )
                    score_chunks.append(
                        model(user_tensor, item_tensor, visual_tensor, metadata_tensor).cpu().numpy()
                    )

                scores = np.concatenate(score_chunks)
                top_12_indices = np.argsort(scores)[::-1][:12]
                predictions[user_id] = [candidate_item_ids[i] for i in top_12_indices]
                progress_state["scored_users"] = len(predictions)

                if should_log_progress(len(predictions), target_users, step=max(1, target_users // 10)):
                    logger.info(
                        "Final MAP@12 evaluation: %s/%s users scored.",
                        f"{len(predictions):,}",
                        f"{target_users:,}",
                    )

    if max_eval_users is not None:
        evaluated_user_ids = list(predictions.keys())
        test_df = test_df.filter(pl.col("user_id").is_in(evaluated_user_ids))

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
    log_runtime_environment(device)
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
        articles_df,
        all_item_id_pool,
        user_to_purchased_items,
        num_users,
        num_items,
    ) = prepare_training_data(config)

    # -- Content feature extractors -------------------------------------------
    logger.info("Loading visual and metadata features ...")
    item_to_article_id = build_item_to_article_id_mapping(articles_df)
    feature_extractor = VisualFeatureExtractor(
        config.visual_features_path,
        item_id_to_article_id=item_to_article_id,
    )
    metadata_encoder = MetadataFeatureEncoder(articles_df)

    # -- Model ----------------------------------------------------------------
    logger.info("Building HybridRecommendationModel ...")
    model = HybridRecommendationModel(
        num_users=num_users,
        num_items=num_items,
        visual_feature_dim=feature_extractor.feature_dim,
        metadata_feature_dim=metadata_encoder.feature_dim,
        mf_embedding_dim=config.mf_embedding_dim,
        mlp_layer_sizes=config.mlp_layer_sizes,
    ).to(device)

    trainable_parameter_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info("Trainable parameters: %s", f"{trainable_parameter_count:,}")

    # -- Optimiser & loss -----------------------------------------------------
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5, min_lr=1e-6
    )
    gradient_scaler = create_gradient_scaler(use_mixed_precision)

    # -- Early stopping -------------------------------------------------------
    early_stopper = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
        checkpoint_path=best_checkpoint_path,
    )
    hybrid_collator = HybridBatchCollator(feature_extractor, metadata_encoder)

    # -- Validation dataset (fewer negatives for speed) -----------------------
    validation_dataset = NegativeSamplingDataset(
        positive_user_ids=validation_user_ids,
        positive_item_ids=validation_item_ids,
        all_item_id_pool=all_item_id_pool,
        user_to_purchased_items=user_to_purchased_items,
        num_negatives_per_positive=1,
        feature_extractor=feature_extractor,
        metadata_encoder=metadata_encoder,
        seed=config.random_seed + 1000,
    )
    validation_data_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=hybrid_collator,
    )

    # -- Training loop --------------------------------------------------------
    logger.info("=" * 70)
    logger.info("Starting training ...")
    logger.info("=" * 70)

    for epoch in range(1, config.num_epochs + 1):
        epoch_start_time = time.time()

        # 1. Build lazy negative-sampling dataset for this epoch
        logger.info("Epoch %d/%d | preparing lazy negative sampler ...", epoch, config.num_epochs)
        training_dataset = NegativeSamplingDataset(
            positive_user_ids=train_user_ids,
            positive_item_ids=train_item_ids,
            all_item_id_pool=all_item_id_pool,
            user_to_purchased_items=user_to_purchased_items,
            num_negatives_per_positive=config.num_negatives_per_positive,
            feature_extractor=feature_extractor,
            metadata_encoder=metadata_encoder,
            seed=config.random_seed + epoch,
        )
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(config.random_seed + epoch)
        training_sampler = RandomSampler(
            training_dataset,
            replacement=True,
            num_samples=len(training_dataset),
            generator=sampler_generator,
        )
        training_data_loader = DataLoader(
            training_dataset,
            batch_size=config.batch_size,
            sampler=training_sampler,
            num_workers=config.dataloader_num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=hybrid_collator,
        )

        # 2. Train one epoch
        model.train()
        epoch_loss_sum = 0.0
        total_batches = len(training_data_loader)
        progress_step = (
            config.log_every_n_batches
            if config.log_every_n_batches > 0
            else max(1, total_batches // 10)
        )
        completed_batches = 0
        progress_state = {
            "phase": "starting",
            "current_batch": 0,
            "completed_batches": 0,
            "total_batches": total_batches,
            "last_loss": None,
            "elapsed_start": epoch_start_time,
        }
        logger.info(
            "Epoch %d/%d | training samples=%s | batches=%s | batch_size=%s | sampler=replacement",
            epoch,
            config.num_epochs,
            f"{len(training_dataset):,}",
            f"{total_batches:,}",
            f"{config.batch_size:,}",
        )
        logger.info(
            "Epoch %d/%d | replacement sampler avoids allocating a huge shuffle permutation.",
            epoch,
            config.num_epochs,
        )

        training_iterator = iter(training_data_loader)
        with ProgressHeartbeat(
            lambda: (
                f"epoch {epoch}/{config.num_epochs} {progress_state['phase']}; "
                f"batch={progress_state['current_batch']:,}/{progress_state['total_batches']:,}; "
                f"completed={progress_state['completed_batches']:,}; "
                f"last_loss={progress_state['last_loss']}; "
                f"elapsed={format_seconds(time.time() - progress_state['elapsed_start'])}"
            ),
            config.heartbeat_interval_seconds,
        ):
            for batch_index in range(1, total_batches + 1):
                progress_state["phase"] = "loading batch"
                progress_state["current_batch"] = batch_index
                batch_load_started_at = time.time()
                try:
                    users, items, visuals, metadata, labels = next(training_iterator)
                except StopIteration:
                    break
                batch_load_seconds = time.time() - batch_load_started_at

                progress_state["phase"] = "training batch"
                batch_train_started_at = time.time()
                users = users.to(device, non_blocking=True)
                items = items.to(device, non_blocking=True)
                visuals = visuals.to(device, non_blocking=True)
                metadata = metadata.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=device.type, enabled=use_mixed_precision):
                    predictions = model(users, items, visuals, metadata)
                    batch_loss = loss_function(predictions, labels)

                gradient_scaler.scale(batch_loss).backward()
                gradient_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                gradient_scaler.step(optimizer)
                gradient_scaler.update()

                batch_loss_value = batch_loss.item()
                batch_train_seconds = time.time() - batch_train_started_at
                epoch_loss_sum += batch_loss_value
                completed_batches += 1
                progress_state["phase"] = "completed batch"
                progress_state["completed_batches"] = completed_batches
                progress_state["last_loss"] = f"{batch_loss_value:.5f}"

                if should_log_progress(completed_batches, total_batches, step=progress_step):
                    logger.info(
                        "Epoch %d/%d | batch %s/%s | loss=%.5f | load=%s | train=%s",
                        epoch,
                        config.num_epochs,
                        f"{completed_batches:,}",
                        f"{total_batches:,}",
                        batch_loss_value,
                        format_seconds(batch_load_seconds),
                        format_seconds(batch_train_seconds),
                    )

        average_train_loss = epoch_loss_sum / max(completed_batches, 1)

        # 3. Validation
        if epoch % config.evaluate_every_n_epochs == 0:
            model.eval()
            validation_loss_sum = 0.0
            validation_batches = len(validation_data_loader)
            completed_validation_batches = 0
            validation_progress_state = {
                "phase": "starting validation",
                "current_batch": 0,
                "completed_batches": 0,
                "total_batches": validation_batches,
                "elapsed_start": time.time(),
            }
            logger.info(
                "Epoch %d/%d | validation samples=%s | batches=%s",
                epoch,
                config.num_epochs,
                f"{len(validation_dataset):,}",
                f"{validation_batches:,}",
            )

            with torch.no_grad():
                validation_iterator = iter(validation_data_loader)
                with ProgressHeartbeat(
                    lambda: (
                        f"epoch {epoch}/{config.num_epochs} validation "
                        f"{validation_progress_state['phase']}; "
                        f"batch={validation_progress_state['current_batch']:,}/{validation_progress_state['total_batches']:,}; "
                        f"completed={validation_progress_state['completed_batches']:,}; "
                        f"elapsed={format_seconds(time.time() - validation_progress_state['elapsed_start'])}"
                    ),
                    config.heartbeat_interval_seconds,
                ):
                    for validation_batch_index in range(1, validation_batches + 1):
                        validation_progress_state["phase"] = "loading batch"
                        validation_progress_state["current_batch"] = validation_batch_index
                        try:
                            users, items, visuals, metadata, labels = next(validation_iterator)
                        except StopIteration:
                            break

                        validation_progress_state["phase"] = "running batch"
                        users = users.to(device, non_blocking=True)
                        items = items.to(device, non_blocking=True)
                        visuals = visuals.to(device, non_blocking=True)
                        metadata = metadata.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)

                        with torch.autocast(device_type=device.type, enabled=use_mixed_precision):
                            predictions = model(users, items, visuals, metadata)
                            batch_loss = loss_function(predictions, labels)

                        validation_loss_sum += batch_loss.item()
                        completed_validation_batches += 1
                        validation_progress_state["phase"] = "completed batch"
                        validation_progress_state["completed_batches"] = completed_validation_batches

                        if should_log_progress(
                            completed_validation_batches,
                            validation_batches,
                            step=max(1, validation_batches // 5),
                        ):
                            logger.info(
                                "Epoch %d/%d | validation batch %s/%s",
                                epoch,
                                config.num_epochs,
                                f"{completed_validation_batches:,}",
                                f"{validation_batches:,}",
                            )

            average_validation_loss = validation_loss_sum / max(completed_validation_batches, 1)
            epoch_elapsed_seconds = time.time() - epoch_start_time

            logger.info(
                "Epoch %3d | train_loss=%.5f | val_loss=%.5f | %s",
                epoch,
                average_train_loss,
                average_validation_loss,
                format_seconds(epoch_elapsed_seconds),
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
                "metadata_feature_dim": metadata_encoder.feature_dim,
                "visual_feature_dim": feature_extractor.feature_dim,
            },
            last_checkpoint_path,
        )

    # -- Final MAP@12 evaluation on test set ----------------------------------
    logger.info("=" * 70)
    logger.info("Loading best weights for final evaluation ...")
    if best_checkpoint_path.exists():
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
        logger.info("Best checkpoint loaded from %s", best_checkpoint_path)
    else:
        logger.warning("Best checkpoint not found at %s; evaluating current model weights.", best_checkpoint_path)

    logger.info("Starting final MAP@12 evaluation on test set ...")
    map_at_12_score = evaluate_map_at_12_on_test(
        model=model,
        test_df=test_df,
        feature_extractor=feature_extractor,
        metadata_encoder=metadata_encoder,
        device=device,
        all_item_id_pool=all_item_id_pool,
        user_to_purchased_items=user_to_purchased_items,
        random_seed=config.random_seed,
        batch_size=config.eval_batch_size,
        max_eval_users=config.max_eval_users,
        heartbeat_interval_seconds=config.heartbeat_interval_seconds,
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
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--log-every-batches", type=int, default=0)
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    parser.add_argument("--max-eval-users", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
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
        max_train_rows=args.max_train_rows,
        log_every_n_batches=args.log_every_batches,
        heartbeat_interval_seconds=args.heartbeat_seconds,
        max_eval_users=args.max_eval_users,
        eval_batch_size=args.eval_batch_size,
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
