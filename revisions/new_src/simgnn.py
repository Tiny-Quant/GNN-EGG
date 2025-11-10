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
from torch_geometric.nn import global_mean_pool, GINEConv
from torch_geometric.data import Batch

from tqdm.auto import tqdm

from .utils import convert_hard_to_soft_edges
from .ged_dataset import GEDDataset, collate_pairs

class GINEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()

        def make_mlp(in_f, out_f):
            return nn.Sequential(
                nn.Linear(in_f, out_f),
                nn.ReLU(),
                nn.Linear(out_f, out_f)
            )

        # Declare edge_dim=1 so GINE accepts edge_attr with shape (E, 1)
        self.conv1 = GINEConv(make_mlp(in_dim, hidden_dim), edge_dim=1)
        self.conv2 = GINEConv(make_mlp(hidden_dim, hidden_dim), edge_dim=1)

    def forward(self, x, edge_index, batch, edge_weight=None):
        # Convert scalar weights → (E, 1) edge_attr
        if edge_weight is None:
            edge_attr = torch.ones(edge_index.size(1), 1, device=x.device)
        else:
            edge_attr = edge_weight.view(-1, 1)

        x = F.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr=edge_attr))
        return x

class TensorNetworkModule(nn.Module):
    def __init__(self, dim, channels=8):
        super().__init__()
        self.dim = dim
        self.channels = channels

        self.W = nn.Parameter(
            torch.randn(channels, dim, dim) * (1.0 / (dim ** 0.5))
        )

        self.fc = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.ReLU()
        )

    def forward(self, g1, g2):
        B = g1.size(0)
        sims = []
        for c in range(self.channels):
            inter = g1 @ self.W[c]
            s = (inter * g2).sum(dim=1, keepdim=True)
            sims.append(s)
        S = torch.cat(sims, dim=1)
        return self.fc(S)


class SimGNN(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim=64,
        hist_bins=16,
        dropout=0.2,
        use_tensor=False,
        tensor_channels=8
    ):
        super().__init__()
        self.encoder = GINEncoder(in_dim, hidden_dim)
        self.pool = global_mean_pool

        self.hist_bins = hist_bins
        self.use_tensor = use_tensor

        if use_tensor:
            self.tensor = TensorNetworkModule(hidden_dim, channels=tensor_channels)
            sim_feat_dim = tensor_channels
        else:
            sim_feat_dim = hist_bins + 3

        mlp_in = hidden_dim * 2 + sim_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def _compute_similarity_hist_stats(self, node_embs1, node_embs2, b1, b2):
        device = node_embs1.device
        B = int(max(b1.max().item(), b2.max().item()) + 1)

        sim_feats = []
        for i in range(B):
            m1 = torch.where(b1 == i)[0]
            m2 = torch.where(b2 == i)[0]

            if m1.numel() == 0 or m2.numel() == 0:
                hist = torch.zeros(self.hist_bins, device=device)
                stats = torch.zeros(3, device=device)
                sim_feats.append(torch.cat([hist, stats], dim=0))
                continue

            E1 = node_embs1[m1]
            E2 = node_embs2[m2]

            E1n = F.normalize(E1, p=2, dim=1)
            E2n = F.normalize(E2, p=2, dim=1)

            sim_mat = E1n @ E2n.T
            flat = sim_mat.reshape(-1)

            hist = torch.histc(flat, bins=self.hist_bins, min=-1.0, max=1.0)
            hist = hist / (hist.sum() + 1e-8)

            mean_sim = flat.mean()
            max_sim = flat.max()
            std_sim = flat.std(unbiased=False) if flat.numel() > 1 else torch.tensor(0., device=device)

            stats = torch.stack([mean_sim, max_sim, std_sim])
            sim_feats.append(torch.cat([hist, stats], dim=0))

        return torch.stack(sim_feats, dim=0)

    def forward(self, data1: Batch, data2: Batch):
        # Extract dense soft edges from convert_hard_to_soft_edges
        x1, e1, b1 = data1.x, data1.edge_index, data1.batch
        w1 = getattr(data1, "edge_weight", None)

        x2, e2, b2 = data2.x, data2.edge_index, data2.batch
        w2 = getattr(data2, "edge_weight", None)

        node_embs1 = self.encoder(x1, e1, b1, edge_weight=w1)
        node_embs2 = self.encoder(x2, e2, b2, edge_weight=w2)

        g_emb1 = self.pool(node_embs1, b1)
        g_emb2 = self.pool(node_embs2, b2)

        if self.use_tensor:
            sim_feats = self.tensor(g_emb1, g_emb2)
        else:
            sim_feats = self._compute_similarity_hist_stats(
                node_embs1, node_embs2, b1, b2
            )

        x = torch.cat([g_emb1, g_emb2, sim_feats], dim=1)
        return self.mlp(x).squeeze(-1)


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
    data: GEDDataset, 
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int, 
    epochs: int,
    device: torch.device | str = "cpu",
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    checkpoint_path: str | None = None,
    checkpoint_interval: int = 10,
) -> List[Dict[str, float]]:

    torch_device = torch.device(device)
    model = model.to(torch_device)
    history: List[Dict[str, float]] = []

    loader = DataLoader(
        data, batch_size, shuffle=True,
        collate_fn=collate_pairs, drop_last=True
    )

    # One calm progress bar for epochs only
    epoch_bar = tqdm(
        range(1, epochs + 1),
        desc="Training SimGNN",
        leave=True
    )

    for epoch in epoch_bar:
        model.train()

        # Reset epoch stats
        epoch_loss = 0.0
        total_examples = 0
        sum_norm = 0.0
        sum_raw = 0.0

        preds = []
        norm_targets = []
        raw_targets = []
        factors = []

        # No tqdm inside. No flicker.
        for batch in loader:
            if len(batch) == 3:
                data1, data2, norm_target = batch
                raw_target = norm_target
                factor = torch.ones_like(norm_target)
            else:
                data1, data2, norm_target, raw_target, factor = batch

            # Convert and move to device
            data1 = _to_device(convert_hard_to_soft_edges(data1), torch_device)
            data2 = _to_device(convert_hard_to_soft_edges(data2), torch_device)
            norm_target = _to_device(norm_target, torch_device)
            raw_target = _to_device(raw_target, torch_device)
            factor = _to_device(factor, torch_device)

            # Forward + backward
            optimizer.zero_grad()
            pred_norm = model(data1, data2)
            loss = F.mse_loss(pred_norm, norm_target)
            loss.backward()
            optimizer.step()

            # Update aggregates
            bsz = norm_target.size(0)
            epoch_loss += float(loss.item()) * bsz
            total_examples += bsz
            sum_norm += float(norm_target.sum().item())
            sum_raw += float(raw_target.sum().item())

            preds.append(pred_norm.detach().cpu())
            norm_targets.append(norm_target.detach().cpu())
            raw_targets.append(raw_target.detach().cpu())
            factors.append(factor.detach().cpu())

        # Scheduler per epoch
        if scheduler is not None:
            scheduler.step()

        # Prepare concatenated tensors
        preds_tensor = torch.cat(preds) if preds else torch.empty(0)
        norm_tensor = torch.cat(norm_targets) if norm_targets else torch.empty(0)
        raw_tensor = torch.cat(raw_targets) if raw_targets else torch.empty(0)
        factor_tensor = torch.cat(factors) if factors else torch.empty(0)

        # Compute raw predictions
        if preds_tensor.numel() > 0 and factor_tensor.numel() == preds_tensor.numel():
            raw_pred = preds_tensor * factor_tensor
        else:
            raw_pred = torch.empty_like(preds_tensor)

        # === MSE METRICS (THE FIX) ===
        if preds_tensor.numel() >= 1:
            mse_norm = F.mse_loss(preds_tensor, norm_tensor).item()
        else:
            mse_norm = float("nan")

        if raw_pred.numel() == raw_tensor.numel() and raw_pred.numel() > 0:
            mse_raw = F.mse_loss(raw_pred, raw_tensor).item()
        else:
            mse_raw = float("nan")

        # Correlations (unchanged)
        if preds_tensor.numel() >= 2:
            corr_norm = torch.corrcoef(torch.stack([preds_tensor, norm_tensor]))[0, 1].item()
        else:
            corr_norm = float("nan")

        if raw_pred.numel() >= 2:
            corr_raw = torch.corrcoef(torch.stack([raw_pred, raw_tensor]))[0, 1].item()
        else:
            corr_raw = float("nan")

        # Save metrics
        epoch_metrics = {
            "epoch": float(epoch),
            "loss": epoch_loss / total_examples,
            "mse_norm": mse_norm,
            "mse_raw": mse_raw,
            "corr_norm": corr_norm,
            "corr_raw": corr_raw,
        }
        history.append(epoch_metrics)

        tqdm.write(
            f"Epoch {epoch:03d} | "
            f"Loss: {epoch_metrics['loss']:.4f} | "
            f"MSE(norm): {mse_norm:.4f} | "
            f"MSE(raw): {mse_raw:.4f} | "
            f"Corr(norm): {corr_norm:.4f} | "
            f"Corr(raw): {corr_raw:.4f}"
        )

        # Checkpoints
        if (
            checkpoint_path is not None
            and (epoch % max(checkpoint_interval, 1) == 0 or epoch == epochs)
        ):
            save_checkpoint(checkpoint_path, model, optimizer, epoch)

        # Update the top-level bar
        epoch_bar.set_postfix(loss=f"{loss:.4f}", corr=f"{corr_norm:.4f}")