"""
inference_pipeline.py — High-level inference API for the recommendation system.

``InferencePipeline`` encapsulates model loading, feature extraction, and
scoring so that the Streamlit UI only needs to call simple methods without
dealing with tensors, devices, or batch slicing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import polars as pl
import torch

from src.config import (
    BEST_CHECKPOINT_PATH,
    INFERENCE_BATCH_SIZE,
    MF_EMBEDDING_DIM,
    MLP_LAYER_SIZES,
    TOP_K_RECOMMENDATIONS,
    VISUAL_FEATURES_NPZ_PATH,
)
from src.features.visual_feature_extractor import VisualFeatureExtractor
from src.models.hybrid_model import HybridRecommendationModel
from src.models.popularity_baseline import PopularityBaseline

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Facade that handles model loading and recommendation generation.

    Parameters
    ----------
    train_df : pl.DataFrame
        Training transactions (needed to discover num_users/num_items and
        to fit the popularity baseline).
    checkpoint_path : Path, optional
        Path to a saved ``HybridRecommendationModel`` state dict.
    """

    def __init__(
        self,
        train_df: pl.DataFrame,
        checkpoint_path: Path = BEST_CHECKPOINT_PATH,
        visual_features_path: Path = VISUAL_FEATURES_NPZ_PATH,
    ) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Determine ID space
        self.num_users = int(train_df["user_id"].max()) + 1
        self.num_items = int(train_df["item_id"].max()) + 1
        self.all_item_ids = train_df["item_id"].unique().sort().to_numpy()

        # Visual feature extractor
        self._feature_extractor = VisualFeatureExtractor(visual_features_path)

        # Hybrid model
        self._model = HybridRecommendationModel(
            num_users=self.num_users,
            num_items=self.num_items,
            visual_feature_dim=self._feature_extractor.feature_dim,
            mf_embedding_dim=MF_EMBEDDING_DIM,
            mlp_layer_sizes=list(MLP_LAYER_SIZES),
        )
        self._model_is_loaded = self._try_load_checkpoint(checkpoint_path)
        self._model = self._model.to(self._device)

        # Popularity baseline (always available as fallback)
        self._popularity_baseline = PopularityBaseline()
        self._popularity_baseline.fit(train_df)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _try_load_checkpoint(self, checkpoint_path: Path) -> bool:
        """Attempt to load model weights.  Returns True on success."""
        path = Path(checkpoint_path)
        if not path.exists():
            logger.warning("Checkpoint not found: %s", path)
            return False
        try:
            state_dict = torch.load(path, map_location="cpu")
            self._model.load_state_dict(state_dict)
            self._model.eval()
            logger.info("Loaded checkpoint: %s", path)
            return True
        except Exception as exc:
            logger.warning("Failed to load checkpoint %s: %s", path, exc)
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def model_is_loaded(self) -> bool:
        return self._model_is_loaded

    def recommend_hybrid(
        self,
        user_id: int,
        top_k: int = TOP_K_RECOMMENDATIONS,
    ) -> List[Tuple[int, float]]:
        """Score all known items for *user_id* and return top-K with scores.

        Returns
        -------
        list of (item_id, score) sorted by descending score.
        """
        self._model.eval()
        all_scores: list[Tuple[int, float]] = []

        with torch.no_grad():
            for batch_start in range(0, len(self.all_item_ids), INFERENCE_BATCH_SIZE):
                batch_item_ids = self.all_item_ids[batch_start : batch_start + INFERENCE_BATCH_SIZE]
                batch_size = len(batch_item_ids)

                user_tensor = torch.full(
                    (batch_size,), user_id, dtype=torch.long, device=self._device
                )
                item_tensor = torch.tensor(
                    batch_item_ids, dtype=torch.long, device=self._device
                )
                visual_tensor = self._feature_extractor.get_feature_vectors(
                    batch_item_ids.tolist()
                ).to(self._device)

                scores = self._model(user_tensor, item_tensor, visual_tensor).cpu().numpy()
                all_scores.extend(zip(batch_item_ids.tolist(), scores.tolist()))

        all_scores.sort(key=lambda pair: pair[1], reverse=True)
        return all_scores[:top_k]

    def recommend_popular(self) -> List[Tuple[int, float]]:
        """Return most popular items (score 0.0 since no personalisation)."""
        return [(item_id, 0.0) for item_id in self._popularity_baseline.predict()]
