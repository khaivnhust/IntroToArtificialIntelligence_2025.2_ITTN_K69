"""
visual_feature_extractor.py — Load pre-computed ResNet-50 visual embeddings.

Maps each article_id to its 2048-dimensional feature vector stored inside
an ``.npz`` archive.  Articles without a matching vector receive a zero
vector so downstream models always get a fixed-size input.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from src.config import VISUAL_FEATURES_NPZ_PATH, VISUAL_FEATURE_DIM

logger = logging.getLogger(__name__)


class VisualFeatureExtractor:
    """Load and serve pre-computed visual feature vectors from an NPZ file."""

    def __init__(
        self,
        npz_path: Path = VISUAL_FEATURES_NPZ_PATH,
        feature_dim: int = VISUAL_FEATURE_DIM,
    ) -> None:
        self.npz_path = Path(npz_path)
        self.feature_dim = feature_dim
        self._features: Dict[str, np.ndarray] = {}
        self._load_features_from_npz()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _load_features_from_npz(self) -> None:
        """Read every array from the NPZ archive into an in-memory dict."""
        try:
            data = np.load(self.npz_path)
            for key in data.files:
                self._features[key] = data[key]
            logger.info(
                "Loaded %d visual feature vectors from %s",
                len(self._features),
                self.npz_path.name,
            )
        except Exception as exc:
            logger.warning(
                "Failed to load visual features from %s: %s", self.npz_path, exc
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_feature_vectors(self, article_ids: List) -> torch.Tensor:
        """Return a ``(batch_size, feature_dim)`` float32 tensor for *article_ids*.

        Parameters
        ----------
        article_ids : list of int or str
            Article identifiers.  Strings and zero-padded variants are both
            tried when looking up the NPZ keys.

        Returns
        -------
        torch.Tensor
            Shape ``(len(article_ids), self.feature_dim)``.  Missing articles
            are represented by zero vectors.
        """
        batch_size = len(article_ids)
        feature_matrix = np.zeros((batch_size, self.feature_dim), dtype=np.float32)

        for index, article_id in enumerate(article_ids):
            string_key = str(article_id)
            padded_key = string_key.zfill(9)

            if string_key in self._features:
                feature_matrix[index] = self._features[string_key]
            elif padded_key in self._features:
                feature_matrix[index] = self._features[padded_key]

        return torch.tensor(feature_matrix, dtype=torch.float32)
