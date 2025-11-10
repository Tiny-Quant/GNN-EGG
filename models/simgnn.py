"""Implementation of the SimGNN architecture for GED regression."""
from __future__ import annotations

from typing import Iterable, List, Optional

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import softmax
from torch_scatter import scatter_add


class AttentionPooling(nn.Module):
    """Attention-based pooling used by SimGNN."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        if x.numel() == 0:
            num_graphs = int(batch.max().item() + 1) if batch.numel() > 0 else 1
            return x.new_zeros(num_graphs, x.size(-1))

        scores = self.attention(x).squeeze(-1)
        weights = softmax(scores, batch)
        pooled = scatter_add(weights.unsqueeze(-1) * x, batch, dim=0)
        return pooled


class SimGNN(nn.Module):
    """Full SimGNN model with tensor network and regression head."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        tensor_channels: int = 16,
        mlp_hidden_dims: Optional[Iterable[int]] = None,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("SimGNN requires at least one GCN layer")

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(1, num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.activation = nn.ReLU()
        self.att_pool = AttentionPooling(hidden_channels)

        self.tensor_channels = tensor_channels
        if tensor_channels <= 0:
            raise ValueError("tensor_channels must be positive")

        self.tensor_weight = nn.Parameter(
            torch.empty(tensor_channels, hidden_channels, hidden_channels)
        )
        self.tensor_bias = nn.Parameter(torch.zeros(tensor_channels))
        nn.init.xavier_uniform_(self.tensor_weight)

        mlp_hidden_dims = list(mlp_hidden_dims or [128, 64])
        feature_dim = hidden_channels * 2 + tensor_channels
        mlp_layers: List[nn.Module] = []
        prev_dim = feature_dim
        for dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, dim))
            mlp_layers.append(nn.ReLU())
            prev_dim = dim
        mlp_layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def encode_graph(self, data: Data) -> Tensor:
        x = data.x
        if x is None:
            raise ValueError("Input graphs must contain node features in `x`")
        edge_index = data.edge_index
        edge_weight = getattr(data, "edge_weight", None)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = self.activation(x)

        graph_emb = self.att_pool(x, batch)
        return graph_emb

    def tensor_projector(self, h1: Tensor, h2: Tensor) -> Tensor:
        projected = torch.einsum("bi,oij,bj->bo", h1, self.tensor_weight, h2)
        projected = torch.tanh(projected + self.tensor_bias)
        return projected

    def forward(self, data1: Data | Batch, data2: Data | Batch) -> Tensor:
        h1 = self.encode_graph(data1)
        h2 = self.encode_graph(data2)

        tensor_scores = self.tensor_projector(h1, h2)
        combined = torch.cat([torch.abs(h1 - h2), h1 * h2, tensor_scores], dim=-1)
        out = self.mlp(combined).view(-1)
        return out
