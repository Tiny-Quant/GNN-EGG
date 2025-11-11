# utils.py
#
# Graph visualization utilities for datasets loaded via dataAdapter.load_dataset.
# Uses NetworkX + Matplotlib. Supports node colors, labels, edge styles,
# and metadata attached directly to the dataset.

import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch 
from typing import Optional, Union


def _to_cpu(x):
    """Ensure tensor is on CPU."""
    if torch.is_tensor(x):
        return x.detach().cpu()
    return x


def plot_graph(data, dataset,
               node_size=400,
               font_size=8,
               edge_alpha=0.9,
               layout="spring",
               show_labels=True,
               figsize=(6, 6)):
    """
    Visualize a graph using dataset metadata.

    Args:
        data: a PyG Data object
        dataset: the dataset containing metadata (returned by load_dataset)
        node_size: default node size
        font_size: size of node label text
        edge_alpha: transparency of edges
        layout: "spring", "kamada", "circular"
        show_labels: whether to show node labels
        figsize: size of the plot
    """

    # -------- Convert PyG → NetworkX --------
    edge_index = _to_cpu(data.edge_index)
    x = _to_cpu(data.x)

    G = nx.Graph()
    num_nodes = x.shape[0]
    G.add_nodes_from(range(num_nodes))

    # Add edges
    for u, v in edge_index.t().tolist():
        G.add_edge(int(u), int(v))

    # -------- Node Colors --------
    # x is typically integer labels OR one-hot
    if x.dim() == 2 and x.dtype == torch.long:
        node_labels = x.squeeze().tolist()
    elif x.dim() == 2:
        node_labels = x.argmax(dim=-1).tolist()
    else:
        node_labels = [0 for _ in range(num_nodes)]

    # Assign colors using metadata (fallback → gray)
    node_colors = []
    for nl in node_labels:
        if hasattr(dataset, "NODE_COLOR") and nl in dataset.NODE_COLOR:
            node_colors.append(dataset.NODE_COLOR[nl])
        else:
            node_colors.append("gray")

    # -------- Edge widths --------
    edge_attrs = getattr(data, "edge_attr", None)
    edge_widths = []
    if edge_attrs is not None:
        e = _to_cpu(edge_attrs)
        if e.dim() == 2 and e.dtype == torch.long:
            edge_labels = e.squeeze().tolist()
        elif e.dim() == 2:
            edge_labels = e.argmax(dim=-1).tolist()
        else:
            edge_labels = [1] * G.number_of_edges()
    else:
        edge_labels = [1] * G.number_of_edges()

    for el in edge_labels:
        if hasattr(dataset, "EDGE_WIDTH") and el in dataset.EDGE_WIDTH:
            edge_widths.append(dataset.EDGE_WIDTH[el])
        else:
            edge_widths.append(2)

    # -------- Layout --------
    if layout == "spring":
        pos = nx.spring_layout(G)
    elif layout == "kamada":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G)

    # -------- Plotting --------
    plt.figure(figsize=figsize)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_size,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        edge_color="black",
        alpha=edge_alpha,
    )

    if show_labels:
        # Node labels based on dataset.NODE_CLS
        label_map = {}
        for idx, nl in enumerate(node_labels):
            if hasattr(dataset, "NODE_CLS") and nl in dataset.NODE_CLS:
                label_map[idx] = dataset.NODE_CLS[nl]
            else:
                label_map[idx] = str(nl)

        nx.draw_networkx_labels(
            G,
            pos,
            labels=label_map,
            font_size=font_size,
        )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def convert_hard_to_soft_edges(
    data: Union[Data, Batch],
    *,
    keep_edge_attr: bool = True,
) -> Union[Data, Batch]:
    """
    Convert a sparse PyG graph to a dense all-pairs edge_index with an aligned
    edge_weight (and optional edge_attr), without changing devices.
    """

    def _infer_device(obj: Data) -> torch.device:
        for key in ("x", "edge_index", "edge_attr", "edge_weight", "pos"):
            t = getattr(obj, key, None)
            if isinstance(t, torch.Tensor):
                return t.device
        # Fallback to CPU if nothing is present:
        return torch.device("cpu")

    if isinstance(data, Batch):
        # Preserve the original batch's device (based on any tensor it holds)
        batch_device = _infer_device(data)
        converted_list = [
            convert_hard_to_soft_edges(g, keep_edge_attr=keep_edge_attr)  # per-graph device is preserved inside
            for g in data.to_data_list()
        ]
        out = data.__class__.from_data_list(converted_list)
        return out.to(batch_device)

    if not isinstance(data, Data):
        raise TypeError("Expected a torch_geometric.data.Data or Batch instance")

    # ---- infer device & basic sizes ----
    device = _infer_device(data)
    edge_index = getattr(data, "edge_index", None)
    if edge_index is None:
        raise ValueError("The input graph does not contain an edge_index")
    edge_index = edge_index.to(device)

    num_nodes: Optional[int] = getattr(data, "num_nodes", None)
    if not num_nodes:
        x = getattr(data, "x", None)
        if isinstance(x, torch.Tensor) and x.numel() > 0:
            num_nodes = int(x.size(0))
        elif edge_index.numel() > 0:
            num_nodes = int(edge_index.max().item() + 1)
        else:
            num_nodes = 0

    # ---- build dense all-pairs index on the SAME device ----
    if num_nodes == 0:
        # Degenerate case: keep a minimal clone on the right device
        new_data = data.clone()
        new_data.edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        new_data.edge_weight = torch.empty(0, dtype=torch.float32, device=device)
        if keep_edge_attr and hasattr(new_data, "edge_attr"):
            delattr(new_data, "edge_attr")
        return new_data

    full_row = torch.arange(num_nodes, device=device).repeat_interleave(num_nodes)
    full_col = torch.arange(num_nodes, device=device).repeat(num_nodes)
    full_edge_index = torch.stack([full_row, full_col], dim=0)  # [2, n*n]

    full_edge_weight = torch.zeros(num_nodes * num_nodes, device=device, dtype=torch.float32)

    # ---- optional edge_attr handling ----
    src_edge_attr = getattr(data, "edge_attr", None)
    if src_edge_attr is not None:
        src_edge_attr = src_edge_attr.to(device)
        if src_edge_attr.dim() == 1:
            src_edge_attr = src_edge_attr.unsqueeze(-1)
        attr_dim = int(src_edge_attr.size(-1))
        full_edge_attr = torch.zeros(num_nodes * num_nodes, attr_dim, device=device, dtype=src_edge_attr.dtype)
    else:
        attr_dim = None
        full_edge_attr = None

    # ---- scatter source edges into dense slots (stay on device) ----
    if edge_index.numel() > 0:
        flat_ids = edge_index[0] * num_nodes + edge_index[1]  # [E] on device
        if src_edge_attr is not None:
            weights = src_edge_attr.mean(dim=-1).to(dtype=torch.float32)  # [E]
        else:
            weights = torch.ones_like(flat_ids, dtype=torch.float32, device=device)

        # Use scatter_reduce_ when available (PyTorch >= 1.12) – all on-device:
        if hasattr(full_edge_weight, "scatter_reduce_"):
            full_edge_weight.scatter_reduce_(0, flat_ids, weights, reduce="amax", include_self=True)
        else:
            full_edge_weight[flat_ids] = torch.maximum(full_edge_weight[flat_ids], weights)

        if full_edge_attr is not None:
            if hasattr(full_edge_attr, "scatter_reduce_"):
                index = flat_ids.unsqueeze(-1).expand(-1, attr_dim)  # [E, D]
                full_edge_attr.scatter_reduce_(0, index, src_edge_attr, reduce="amax", include_self=True)
            else:
                full_edge_attr[flat_ids] = torch.maximum(full_edge_attr[flat_ids], src_edge_attr)

    # ---- assemble output (stays on the same device) ----
    new_data = data.clone()
    new_data.edge_index = full_edge_index
    new_data.edge_weight = full_edge_weight
    if full_edge_attr is not None:
        new_data.edge_attr = full_edge_attr.view(-1) if attr_dim == 1 else full_edge_attr
    elif keep_edge_attr and hasattr(new_data, "edge_attr"):
        delattr(new_data, "edge_attr")

    # Ensure any lingering tensors are on the same device
    return new_data.to(device)