"""Implementation of the SimGNN architecture for GED regression."""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import softmax
from torch_scatter import scatter_add

from tqdm.auto import tqdm

from .utils import convert_hard_to_soft_edges


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


def _to_device(data: Batch | Data | Tensor, device: torch.device) -> Batch | Data | Tensor:
    if isinstance(data, (Batch, Data, Tensor)):
        return data.to(device)
    raise TypeError(f"Unsupported data type {type(data)!r} for device transfer")


def save_checkpoint(
    path: str, model: SimGNN, optimizer: torch.optim.Optimizer, epoch: int
) -> None:
    """Persist training state so notebook sessions can resume later."""

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def train_simgnn(
    model: SimGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    device: torch.device | str = "cpu",
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    checkpoint_path: str | None = None,
    checkpoint_interval: int = 10,
) -> List[Dict[str, float]]:
    """Train ``model`` using ``loader`` and report detailed metrics via tqdm.

    Parameters
    ----------
    model:
        The ``SimGNN`` instance to optimise.
    loader:
        A ``DataLoader`` producing ``(graph_1, graph_2, norm, raw, factor)`` tuples.
    optimizer:
        Optimiser configured for ``model`` parameters.
    epochs:
        Number of full passes over ``loader``.
    device:
        Device identifier or instance where computations will be executed.
    scheduler:
        Optional learning-rate scheduler stepped once per epoch.
    checkpoint_path:
        If provided, checkpoints are written periodically.
    checkpoint_interval:
        Number of epochs between checkpoints.

    Returns
    -------
    list of dict
        Per-epoch metrics containing the loss, average GED values, and correlations.
    """

    torch_device = torch.device(device)
    model = model.to(torch_device)
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        total_examples = 0
        sum_norm = 0.0
        sum_raw = 0.0
        preds: List[Tensor] = []
        norm_targets: List[Tensor] = []
        raw_targets: List[Tensor] = []
        factors: List[Tensor] = []

        batch_iter = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in batch_iter:
            if len(batch) == 3:
                data1, data2, norm_target = batch
                raw_target = norm_target
                factor = torch.ones_like(norm_target)
            elif len(batch) == 5:
                data1, data2, norm_target, raw_target, factor = batch
            else:
                raise ValueError(
                    "Expected batches of length 3 or 5 (data1, data2, norm[, raw, factor])"
                )

            data1 = convert_hard_to_soft_edges(data1)
            data2 = convert_hard_to_soft_edges(data2)

            data1 = _to_device(data1, torch_device)
            data2 = _to_device(data2, torch_device)
            norm_target = _to_device(norm_target, torch_device)
            raw_target = _to_device(raw_target, torch_device)
            factor = _to_device(factor, torch_device)

            optimizer.zero_grad()
            pred_norm = model(data1, data2)
            loss = F.mse_loss(pred_norm, norm_target)
            loss.backward()
            optimizer.step()

            batch_size = norm_target.size(0)
            epoch_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            sum_norm += float(norm_target.sum().item())
            sum_raw += float(raw_target.sum().item())

            preds.append(pred_norm.detach().cpu())
            norm_targets.append(norm_target.detach().cpu())
            raw_targets.append(raw_target.detach().cpu())
            factors.append(factor.detach().cpu())

            avg_loss = epoch_loss / max(total_examples, 1)
            avg_norm = sum_norm / max(total_examples, 1)
            avg_raw = sum_raw / max(total_examples, 1)
            batch_iter.set_postfix(loss=f"{avg_loss:.4f}", norm_ged=f"{avg_norm:.4f}", raw_ged=f"{avg_raw:.4f}")

        batch_iter.close()

        if scheduler is not None:
            scheduler.step()

        preds_tensor = torch.cat(preds) if preds else torch.empty(0)
        norm_tensor = torch.cat(norm_targets) if norm_targets else torch.empty(0)
        raw_tensor = torch.cat(raw_targets) if raw_targets else torch.empty(0)
        factor_tensor = torch.cat(factors) if factors else torch.empty(0)

        if preds_tensor.numel() >= 2 and norm_tensor.numel() >= 2:
            corr_norm = torch.corrcoef(torch.stack([preds_tensor, norm_tensor]))[0, 1].item()
        else:
            corr_norm = float("nan")

        if preds_tensor.numel() >= 1 and factor_tensor.numel() == preds_tensor.numel():
            raw_pred = preds_tensor * factor_tensor
        else:
            raw_pred = torch.empty(0)

        if raw_pred.numel() >= 2 and raw_tensor.numel() >= 2:
            corr_raw = torch.corrcoef(torch.stack([raw_pred, raw_tensor]))[0, 1].item()
        else:
            corr_raw = float("nan")

        avg_loss = epoch_loss / max(total_examples, 1)
        avg_norm = sum_norm / max(total_examples, 1)
        avg_raw = sum_raw / max(total_examples, 1)

        epoch_metrics: Dict[str, float] = {
            "epoch": float(epoch),
            "loss": avg_loss,
            "avg_norm_ged": avg_norm,
            "avg_raw_ged": avg_raw,
            "corr_norm": corr_norm,
            "corr_raw": corr_raw,
        }
        history.append(epoch_metrics)

        tqdm.write(
            "Epoch {epoch:03d} | Loss: {loss:.4f} | Norm GED: {norm:.4f} | Raw GED: {raw:.4f} | "
            "Corr(norm): {corr_norm:.4f} | Corr(raw): {corr_raw:.4f}".format(
                epoch=epoch,
                loss=avg_loss,
                norm=avg_norm,
                raw=avg_raw,
                corr_norm=corr_norm,
                corr_raw=corr_raw,
            )
        )

        if (
            checkpoint_path is not None
            and (epoch % max(checkpoint_interval, 1) == 0 or epoch == epochs)
        ):
            save_checkpoint(checkpoint_path, model, optimizer, epoch)

    return history
