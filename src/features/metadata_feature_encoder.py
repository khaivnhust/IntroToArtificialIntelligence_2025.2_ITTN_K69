"""
Transform metadata to vector -> model hybrid
"""

from __future__ import annotations

from typing import List

import numpy as np
import polars as pl
import torch


DEFAULT_METADATA_COLUMNS = [
    "product_type_no",
    "graphical_appearance_no",
    "colour_group_code",
    "perceived_colour_value_id",
    "index_code",
    "index_group_no",
    "section_no",
    "garment_group_no",
]

class MetadataFeatureEncoder:
    """Encode per-article metadata into fixed-size float32 vectors.
       Ex :  item_id 45793 -> [0.34, 0.12, 0.88, ...]
    """
    def __init__(
        self,
        articles_df: pl.DataFrame,
        item_id_column: str = "item_id",
        metadata_columns: list[str] | None = None,
    ) -> None:
        if item_id_column not in articles_df.columns:
            raise ValueError(f"articles_df must contain {item_id_column!r}")

        self.item_id_column = item_id_column
        self.metadata_columns = [
            column
            for column in (metadata_columns or DEFAULT_METADATA_COLUMNS)
            if column in articles_df.columns
        ]
        self.feature_dim = len(self.metadata_columns)
        self._features: dict[int, np.ndarray] = {}

        self._fit_transform(articles_df)

    def _fit_transform(self, articles_df : pl.DataFrame):
        """ transform metadata -> vector"""
        if not self.metadata_columns:
            return

        encoded_columns: list[np.ndarray] = []
        for column in self.metadata_columns:
            series = articles_df[column]
            if series.dtype.is_numeric():
                values = series.fill_null(0).cast(pl.Float32).to_numpy()
                max_abs = float(np.nanmax(np.abs(values))) if len(values) else 0.0
                if max_abs > 0:
                    values = values / max_abs
            else:
                values = self._encode_categorical(series)
            encoded_columns.append(values.astype(np.float32))

        matrix = np.column_stack(encoded_columns).astype(np.float32)
        item_ids = articles_df[self.item_id_column].to_numpy()

        for item_id, vector in zip(item_ids, matrix):
            self._features[int(item_id)] = vector

    @staticmethod
    def _encode_categorical(series: pl.Series) -> np.ndarray:
        """mapping string/category -> number then normalize"""
        values = series.fill_null("__missing__").cast(pl.Utf8).to_list()
        categories = {value: index + 1 for index, value in enumerate(sorted(set(values)))}
        denominator = max(len(categories), 1)
        return np.array([categories[value] / denominator for value in values], dtype=np.float32)

    def get_feature_vectors(self, item_ids: List[int]) -> torch.Tensor:
        """Return a ``(batch_size, feature_dim)`` metadata tensor."""
        matrix = np.zeros((len(item_ids), self.feature_dim), dtype=np.float32)
        for index, item_id in enumerate(item_ids):
            vector = self._features.get(int(item_id))
            if vector is not None:
                matrix[index] = vector
        return torch.tensor(matrix, dtype=torch.float32)


metadata_feature_encoder = MetadataFeatureEncoder
