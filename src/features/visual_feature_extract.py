from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Mapping
import numpy as np
import torch
from src.config import VISUAL_FEATURES_NPZ_PATH, VISUAL_FEATURE_DIM
logger = logging.getLogger(__name__)

class VisualFeatureExtractor:
    def __init__(
        self,
        npz_path: Path = VISUAL_FEATURES_NPZ_PATH,
        feature_dimension: int = VISUAL_FEATURE_DIM,
        item_id_to_article_id : Mapping[int , int] | None = None ,
    ) -> None:
        self._npz_path = Path(npz_path)
        self._feature_dimension = feature_dimension
        self._item_id_to_article_id = {
            int(item_id) : int(article_id)
            for item_id  , article_id in (item_id_to_article_id or {}).items()
        }
        self._features: Dict[str, np.ndarray] = {}
        self._load_features_from_npz()

    @property
    def feature_dim(self) -> int:
        return self._feature_dimension
    
    def _load_features_from_npz(self) -> None:
        try:
            with np.load(self._npz_path) as data_features:
                for key in data_features.files:
                    self._features[key] = data_features[key].astype(np.float32)
            logger.info(
                "Loaded %d visual feature vectors from %s",
                len(self._features),
                self._npz_path.name,
            )
        except Exception as exc:
            logger.warning(
                "Failed to load visual features from %s: %s", self._npz_path, exc
            )
    def get_feature_vectors(self, item_or_article_ids: List) -> torch.Tensor:
        """Returns torch.Tensor
            Shape ``(len(item_or_article_ids), self.feature_dim)``. Missing articles
            are represented by zero vectors.
        """
        batch_size = len(item_or_article_ids)
        feature_matrix = np.zeros((batch_size , self._feature_dimension) , dtype=np.float32)
        for index, item_or_article_id in enumerate(item_or_article_ids):
            resolved_id = self._resolve_article_id(item_or_article_id)
            string_key = str(resolved_id)
            padded_9_key = string_key.zfill(9)
            padded_10_key = string_key.zfill(10)

            if string_key in self._features:
                feature_matrix[index] = self._features[string_key]
            elif padded_9_key in self._features:
                feature_matrix[index] = self._features[padded_9_key]
            elif padded_10_key in self._features:
                feature_matrix[index] = self._features[padded_10_key]

        return torch.tensor(feature_matrix, dtype=torch.float32)
    
    def _resolve_article_id(self , item_or_article_id) -> int | str :
        try :
            numeric_id = int(item_or_article_id)
        except (TypeError , ValueError):
            return item_or_article_id
        
        return self._item_id_to_article_id.get(numeric_id , numeric_id)
