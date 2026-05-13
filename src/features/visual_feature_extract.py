from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from src.config import VISUAL_FEATURES_NPZ_PATH, VISUAL_FEATURE_DIM
logger = logging.getLogger(__name__)

class VisualFeatureExtractor:
    def __init__(
        self,
        npz_path: Path = VISUAL_FEATURES_NPZ_PATH,
        feature_dimension: int = VISUAL_FEATURE_DIM,
    ) -> None:
        self._npz_path = Path(npz_path)
        self._feature_dimension = feature_dimension
        self._features: Dict[str, np.ndarray] = {}
        self._load_features_from_npz()
    
    def _load_features_from_npz(self) -> None:
        try:
            data_features = np.load(self._npz_path)
            for key in data_features.files:
                self._features[key] = data_features[key]
            logger.info(
                "Loaded %d visual feature vectors from %s",
                len(self._features),
                self._npz_path.name,
            )
        except Exception as exc:
            logger.warning(
                "Failed to load visual features from %s: %s", self._npz_path, exc
            )
    def get_feature_vectors(self , article_ids : List) -> torch.Tensor:
        """Returns torch.Tensor
            Shape ``(len(article_ids), self.feature_dim)``.  Missing articles
            are represented by zero vectors.
        """
        batch_size = len(article_ids)
        feature_matrix = np.zeros((batch_size , self._feature_dimension) , dtype=np.float32)
        for index, article_id in enumerate(article_ids):
            string_key = str(article_id)
            padded_key = string_key.zfill(9)

            if string_key in self._features:
                feature_matrix[index] = self._features[string_key]
            elif padded_key in self._features:
                feature_matrix[index] = self._features[padded_key]

        return torch.tensor(feature_matrix, dtype=torch.float32)
    