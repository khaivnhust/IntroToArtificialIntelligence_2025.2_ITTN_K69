"""
hybrid_model.py — Hybrid Recommendation Model (NCF + Visual Features).

Concatenates the latent representation produced by the NCF sub-network
with a pre-computed 2048-dim visual feature vector, then routes the
combined vector through dense fusion layers to predict the probability
of a user–item interaction.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from src.models.ncf import NeuralCollaborativeFiltering
from src.config import MF_EMBEDDING_DIM, MLP_LAYER_SIZES, VISUAL_FEATURE_DIM


class HybridRecommendationModel(nn.Module):
    """NCF + visual feature fusion model."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        visual_feature_dim: int = VISUAL_FEATURE_DIM,
        mf_embedding_dim: int = MF_EMBEDDING_DIM,
        mlp_layer_sizes: List[int] | None = None,
    ) -> None:
        super().__init__()

        if mlp_layer_sizes is None:
            mlp_layer_sizes = list(MLP_LAYER_SIZES)

        # -- NCF sub-network --------------------------------------------------
        self.ncf = NeuralCollaborativeFiltering(
            num_users=num_users,
            num_items=num_items,
            mf_embedding_dim=mf_embedding_dim,
            mlp_layer_sizes=mlp_layer_sizes,
        )

        # NCF latent dimension = GMF dim + last MLP layer dim
        ncf_latent_dim = mf_embedding_dim + mlp_layer_sizes[-1]

        # -- Fusion dense layers (NCF latent + visual) -------------------------
        fusion_input_dim = ncf_latent_dim + visual_feature_dim

        self.fusion_dense_layers = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> torch.Tensor:
        """Predict interaction probability using CF + visual signals.

        Parameters
        ----------
        user_indices : torch.Tensor
            Shape ``(batch_size,)``.
        item_indices : torch.Tensor
            Shape ``(batch_size,)``.
        visual_features : torch.Tensor
            Shape ``(batch_size, visual_feature_dim)``.

        Returns
        -------
        torch.Tensor
            Predicted probabilities, shape ``(batch_size,)``.
        """
        # --- 1. NCF latent representation (skip standalone predict layer) -----
        gmf_user = self.ncf.user_embedding_gmf(user_indices)
        gmf_item = self.ncf.item_embedding_gmf(item_indices)
        gmf_vector = torch.mul(gmf_user, gmf_item)

        mlp_user = self.ncf.user_embedding_mlp(user_indices)
        mlp_item = self.ncf.item_embedding_mlp(item_indices)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_vector = self.ncf.mlp_hidden_layers(mlp_input)

        ncf_latent = torch.cat([gmf_vector, mlp_vector], dim=-1)

        # --- 2. Fuse NCF latent + visual features ----------------------------
        hybrid_vector = torch.cat([ncf_latent, visual_features], dim=-1)

        prediction = self.fusion_dense_layers(hybrid_vector)
        return prediction.squeeze(-1)
