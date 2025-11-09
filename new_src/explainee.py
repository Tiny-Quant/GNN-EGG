"""Explainee models and training loops for the generalized pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCN
from torch_geometric.nn import global_add_pool, global_mean_pool


class GeneralGCN(nn.Module):
    """A lightweight classifier used as the explainee network."""

    def __init__(
        self,
        node_features: int,
        hidden_channels: int,
        num_classes: int,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv = GCN(
            in_channels=node_features,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            act=nn.LeakyReLU(inplace=True),
            dropout=dropout,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.readout = nn.Linear(hidden_channels * 2, hidden_channels)
        self.out = nn.Linear(hidden_channels, num_classes)

    def forward(self, batch: Batch) -> torch.Tensor:
        h = self.conv(batch.x, batch.edge_index)
        pooled = torch.cat(
            [
                global_add_pool(h, batch=batch.batch),
                global_mean_pool(h, batch=batch.batch),
            ],
            dim=-1,
        )
        pooled = self.dropout(pooled)
        pooled = self.readout(pooled)
        pooled = F.relu(pooled)
        return self.out(pooled)


def train_explainee(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    epochs: int,
    device: torch.device,
) -> Dict[str, List[float]]:
    """Train the explainee network and return loss/accuracy history."""

    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.num_graphs
        history["train_loss"].append(running_loss / len(train_loader.dataset))

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                loss = criterion(logits, batch.y.view(-1))
                val_loss += loss.item() * batch.num_graphs
                preds = logits.argmax(dim=-1)
                correct += (preds == batch.y.view(-1)).sum().item()
                total += batch.num_graphs
        history["val_loss"].append(val_loss / len(val_loader.dataset))
        history["val_acc"].append(correct / max(total, 1))

    return history


class ActivationExtractor:
    """Utility to capture intermediate embeddings from a model."""

    def __init__(self, model: nn.Module, layer_names: Sequence[str]):
        self.model = model
        self.layer_names = set(layer_names)
        self.activations: Dict[str, torch.Tensor] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        for name, module in model.named_modules():
            if name in self.layer_names:
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name: str):
        def fn(module, inputs, output):
            self.activations[name] = output.detach()
        return fn

    def clear(self):
        self.activations = {}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def class_average_embeddings(
    model: nn.Module,
    loader: DataLoader,
    layer_names: Sequence[str],
    device: torch.device,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Compute per-class average embeddings for selected layers."""

    extractor = ActivationExtractor(model, layer_names)
    per_class: Dict[int, Dict[str, List[torch.Tensor]]] = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            extractor.clear()
            _ = model(batch)
            batch_labels = batch.y.view(-1).tolist()
            for idx, label in enumerate(batch_labels):
                if label not in per_class:
                    per_class[label] = {name: [] for name in layer_names}
                for name in layer_names:
                    activation = extractor.activations.get(name)
                    if activation is None:
                        continue
                    if activation.size(0) == batch.num_graphs:
                        per_class[label][name].append(activation[idx].detach().cpu())
                    else:
                        # Activation is node-level; pool using mean per graph.
                        graph_mask = batch.batch == idx
                        pooled = activation[graph_mask].mean(dim=0)
                        per_class[label][name].append(pooled.detach().cpu())

    extractor.close()

    averaged: Dict[int, Dict[str, torch.Tensor]] = {}
    for label, acts in per_class.items():
        averaged[label] = {}
        for name, tensors in acts.items():
            if not tensors:
                continue
            stacked = torch.stack(tensors)
            averaged[label][name] = stacked.mean(dim=0, keepdim=True)
    return averaged
