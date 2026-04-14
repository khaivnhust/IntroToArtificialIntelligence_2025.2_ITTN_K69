"""
matrix_factorization.py — Matrix Factorization model for collaborative filtering.

Learns per-user and per-item latent embeddings plus bias terms.  The
predicted interaction score is the dot product of the latent vectors
plus additive biases.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MatrixFactorization(nn.Module):
    """Latent-factor model with user/item embeddings and biases."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        self._initialize_weights()

    # ------------------------------------------------------------------
    def _initialize_weights(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    # ------------------------------------------------------------------
    def forward(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute predicted scores for (user, item) pairs.

        Parameters
        ----------
        user_indices, item_indices : torch.Tensor
            1-D tensors of shape ``(batch_size,)``.

        Returns
        -------
        torch.Tensor
            Predicted scores of shape ``(batch_size,)``.
        """
        user_latent = self.user_embedding(user_indices)
        item_latent = self.item_embedding(item_indices)

        user_bias_value = self.user_bias(user_indices).squeeze(-1)
        item_bias_value = self.item_bias(item_indices).squeeze(-1)

        dot_product = (user_latent * item_latent).sum(dim=1)
        return dot_product + user_bias_value + item_bias_value
