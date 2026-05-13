from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTICLES_PARQUET_PATH = DATA_DIR / "articles_cleaned.parquet"
CUSTOMERS_PARQUET_PATH = DATA_DIR / "customers_cleaned.parquet"
TRAIN_PARQUET_PATH = DATA_DIR / "hm_train.parquet"
TEST_PARQUET_PATH = DATA_DIR / "hm_test.parquet"
VISUAL_FEATURES_NPZ_PATH = DATA_DIR / "visual_features_sample.npz"


CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "hybrid_best.pt"
LAST_CHECKPOINT_PATH = CHECKPOINT_DIR / "hybrid_last.pt"

SAVED_MODELS_DIR = PROJECT_ROOT / "models"

TOP_K_RECOMMEND = 12

#Visual Feature dimension (ResNet-50)
VISUAL_FEATURE_DIM = 2048

#Model hyper parameters 

MF_EMBEDDING_DIM = 16
MLP_LAYER_SIZES : List[int] = [128 , 64, 32, 16]
INFERENCE_BATCH_SIZE = 4096

@dataclass 
class TrainingConfig:
    """All tuneable knobs for the training pipeline."""

    # Paths
    data_dir: Path = DATA_DIR
    visual_features_path: Path = VISUAL_FEATURES_NPZ_PATH
    checkpoint_dir: Path = CHECKPOINT_DIR

    # Model architecture
    mf_embedding_dim: int = MF_EMBEDDING_DIM
    mlp_layer_sizes: List[int] = field(default_factory=lambda: list(MLP_LAYER_SIZES))
    visual_feature_dim: int = VISUAL_FEATURE_DIM

    # Training hyper-parameters
    num_negatives_per_positive: int = 4
    batch_size: int = 4096
    num_epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4

    # Miscellaneous
    random_seed: int = 42
    dataloader_num_workers: int = 0   # keep 0 on Windows to avoid issues
    use_mixed_precision: bool = True  # requires CUDA
    evaluate_every_n_epochs: int = 1
