"""
ncf.py — Neural Collaborative Filtering (NCF) model.

Combines Generalised Matrix Factorisation (GMF) and a Multi-Layer
Perceptron (MLP) path, following the architecture proposed by He et al.
(2017).  Each path maintains independent user/item embedding tables so
that the GMF and MLP branches can learn complementary representations.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class NeuralCollaborativeFiltering(nn.Module):
    """NCF = GMF branch + MLP branch, fused via a final linear layer."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        mf_embedding_dim: int = 8,
        mlp_layer_sizes: List[int] | None = None,
    ) -> None:
        super().__init__()

        if mlp_layer_sizes is None:
            mlp_layer_sizes = [64, 32, 16, 8]

        # -- GMF embeddings ---------------------------------------------------
        self.user_embedding_gmf = nn.Embedding(num_users, mf_embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, mf_embedding_dim)

        # -- MLP embeddings (half of the first MLP layer width each) ----------
        mlp_input_half_dim = mlp_layer_sizes[0] // 2
        self.user_embedding_mlp = nn.Embedding(num_users, mlp_input_half_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, mlp_input_half_dim)

        # -- MLP hidden layers ------------------------------------------------
        mlp_modules: list[nn.Module] = []
        for layer_index in range(len(mlp_layer_sizes) - 1):
            mlp_modules.append(
                nn.Linear(mlp_layer_sizes[layer_index], mlp_layer_sizes[layer_index + 1])
            )
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(0.1))
        self.mlp_hidden_layers = nn.Sequential(*mlp_modules)

        # -- Final prediction layer (standalone NCF usage) --------------------
        combined_latent_dim = mf_embedding_dim + mlp_layer_sizes[-1]
        self.prediction_layer = nn.Linear(combined_latent_dim, 1)
        self.output_activation = nn.Sigmoid()

        self._initialize_weights()

    # ------------------------------------------------------------------
    def _initialize_weights(self) -> None:
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)

        for module in self.mlp_hidden_layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

        nn.init.kaiming_uniform_(
            self.prediction_layer.weight, a=1, nonlinearity="sigmoid"
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Predict interaction probability for (user, item) pairs.

        Parameters
        ----------
        user_indices, item_indices : torch.Tensor
            1-D tensors of shape ``(batch_size,)``.

        Returns
        -------
        torch.Tensor
            Predicted probabilities of shape ``(batch_size,)``.
        """
        # GMF branch
        gmf_user = self.user_embedding_gmf(user_indices)
        gmf_item = self.item_embedding_gmf(item_indices)
        gmf_vector = torch.mul(gmf_user, gmf_item)

        # MLP branch
        mlp_user = self.user_embedding_mlp(user_indices)
        mlp_item = self.item_embedding_mlp(item_indices)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_vector = self.mlp_hidden_layers(mlp_input)

        # Fuse GMF + MLP
        ncf_latent = torch.cat([gmf_vector, mlp_vector], dim=-1)

        prediction = self.output_activation(self.prediction_layer(ncf_latent))
        return prediction.squeeze(-1)
