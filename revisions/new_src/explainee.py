# Fit a GCNClassifier to any dataset supported by dataAdapter.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from .dataAdapter import load_dataset

from typing import Optional

import torch
from torch import Tensor
from torch_geometric.utils import scatter
from torch_geometric.typing import Adj


def scatter_logmeanexp(x, index, dim=-1, dim_size=None, temperature=1):
    """
    Scattered smooth maximum function
    """
    index = index.to(x.device)
    return scatter(src=x.div(temperature).exp(),
                   index=index,
                   dim=dim,
                   dim_size=dim_size,
                   reduce="mean").log().mul(temperature)


def scatter_mean_weighted(x, index, dim=-1, dim_size=None, weight=None):
    """
    Weighted scatter mean
    """
    if weight is None:
        return scatter(x, index,
                       dim=dim,
                       dim_size=dim_size,
                       reduce="mean")
    return (scatter(x * weight[:, None], index, dim=dim, dim_size=dim_size, reduce="sum")
            / scatter(weight[:, None], index, dim=dim, dim_size=dim_size, reduce="sum"))


def scatter_sum_weighted(x, index, dim=-1, dim_size=None, weight=None):
    """
    Weighted scatter mean
    """
    if weight is None:
        return scatter(x, index,
                       dim=dim,
                       dim_size=dim_size,
                       reduce="sum")
    return scatter(x * weight[:, None], index, dim=dim, dim_size=dim_size, reduce="sum")


def scatter_max_weighted(x, index, dim=-1, dim_size=None, weight=None):
    """
    Weighted scatter mean
    """
    if weight is None:
        return scatter(x, index,
                       dim=dim,
                       dim_size=dim_size,
                       reduce="max")
    return scatter(x * weight[:, None], index, dim=dim, dim_size=dim_size, reduce="max")


def global_mean_pool_weighted(x: Tensor,
                              batch: Optional[Tensor],
                              size: Optional[int] = None,
                              node_weight: Optional[Tensor] = None) -> Tensor:
    if batch is None:
        return (x.mean(dim=0, keepdim=True) if node_weight is None
                else x.mul(node_weight[: None]).sum(dim=0, keepdim=True) / node_weight.sum())
    size = int(batch.max().item() + 1) if size is None else size
    return scatter_mean_weighted(x, batch, dim=0, dim_size=size, weight=node_weight)


def global_sum_pool_weighted(x: Tensor,
                             batch: Optional[Tensor],
                             size: Optional[int] = None,
                             node_weight: Optional[Tensor] = None) -> Tensor:
    if batch is None:
        return (x.sum(dim=0, keepdim=True) if node_weight is None
                else x.mul(node_weight[: None]).sum(dim=0, keepdim=True))
    size = int(batch.max().item() + 1) if size is None else size
    return scatter_sum_weighted(x, batch, dim=0, dim_size=size, weight=node_weight)


def global_max_pool_weighted(x: Tensor,
                             batch: Optional[Tensor],
                             size: Optional[int] = None,
                             node_weight: Optional[Tensor] = None) -> Tensor:
    if batch is None:
        return (x.max(dim=0, keepdim=True) if node_weight is None
                else x.mul(node_weight[: None]).max(dim=0, keepdim=True))
    size = int(batch.max().item() + 1) if size is None else size
    return scatter_max_weighted(x, batch, dim=0, dim_size=size, weight=node_weight)


def smooth_maximum_weight_propagation(edge_index: Adj,
                                      edge_weight: Tensor,
                                      size: Optional[int] = None,
                                      temperature: float = 0.05):
    size = int(edge_index.max().item() + 1) if size is None else size
    return scatter_logmeanexp(edge_weight.repeat(2),
                              index=torch.cat(tuple(edge_index)),
                              dim_size=size,
                              temperature=temperature)


class GCNClassifier(nn.Module):
    def __init__(self, hidden_channels, node_features, num_classes, num_layers=3, dropout=0):
        super().__init__()
        import torch_geometric as pyg

        self.conv = pyg.nn.GCN(
            in_channels=node_features,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            act=nn.LeakyReLU(inplace=True),
            dropout=dropout,
        )
        self.drop = nn.Dropout(p=dropout)
        self.lin = pyg.nn.Linear(hidden_channels * 2, hidden_channels)
        self.out = pyg.nn.Linear(hidden_channels, num_classes)

    def forward(self, batch=None, embeds=None, embeds_last=None, edge_weight=None, temperature=0.05):
        if embeds_last is None:
            if embeds is None:
                device = batch.x.device

                node_weight = (
                    None
                    if edge_weight is None
                    else smooth_maximum_weight_propagation(
                        batch.edge_index, edge_weight, size=len(batch.x), temperature=temperature,
                    )
                )

                # --- ensure everything is on the same device ---
                edge_index = batch.edge_index.to(device)
                edge_weight = None if edge_weight is None else edge_weight.to(device)
                node_weight = None if node_weight is None else node_weight.to(device)
                if batch.batch is not None: 
                    bvec = batch.batch.to(device)
                else: 
                    bvec = None

                h = self.conv(batch.x, edge_index, edge_weight=edge_weight)

                embeds = torch.cat(
                    [
                        global_sum_pool_weighted(h, batch=bvec, node_weight=node_weight),
                        global_mean_pool_weighted(h, batch=bvec, node_weight=node_weight),
                    ],
                    dim=1,
                )

            h = self.drop(embeds)
            h = self.lin(h)
            embeds_last = h.relu()

        h = self.out(embeds_last)
        return dict(
            logits=h,
            probs=F.softmax(h, dim=-1),
            embeds=embeds,
            embeds_last=embeds_last,
        )

########################################
#  Training logic
########################################

def fit_explainee(
    dataset_name,
    root="data",
    hidden=64,
    layers=3,
    dropout=0.0,
    epochs=50,
    batch_size=32,
    lr=1e-3,
    device="cuda"
):
    # 1. Load data
    dataset = load_dataset(dataset_name, root=root)

    # 2. Basic split
    num_graphs = len(dataset)
    perm = torch.randperm(num_graphs)
    train_end = int(0.8 * num_graphs)
    val_end = int(0.9 * num_graphs)

    train_ds = dataset[perm[:train_end]]
    val_ds = dataset[perm[train_end:val_end]]
    test_ds = dataset[perm[val_end:]]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # 3. Initialize model
    model = GCNClassifier(
        hidden_channels=hidden,
        node_features=len(dataset.NODE_CLS),
        num_classes=len(dataset.GRAPH_CLS),
        num_layers=layers,
        dropout=dropout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    # 4. Training loop
    for ep in range(epochs):
        model.train()
        total = 0

        for batch in train_loader:
            batch = batch.to(device)
            out = model(batch)
            loss = ce(out["logits"], batch.y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * batch.num_graphs

        # simple validation
        model.eval()
        with torch.no_grad():
            correct = 0
            count = 0
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                pred = out["logits"].argmax(dim=-1)
                correct += (pred == batch.y).sum().item()
                count += batch.num_graphs

        val_acc = correct / max(count, 1)
        print(f"[{ep+1:03d}/{epochs}] train_loss={total/train_end:.4f}  val_acc={val_acc:.3f}")

    # 5. Final test accuracy
    model.eval()
    with torch.no_grad():
        correct = 0
        count = 0
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out["logits"].argmax(dim=-1)
            correct += (pred == batch.y).sum().item()
            count += batch.num_graphs

    test_acc = correct / max(count, 1)
    print(f"Test Accuracy: {test_acc:.3f}")

    return model, dict(val_acc=val_acc, test_acc=test_acc)
