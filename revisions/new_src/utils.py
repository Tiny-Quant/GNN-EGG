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
    """Convert a PyG ``Data`` or ``Batch`` with a sparse edge index into one that
    stores a *dense* (complete) edge index with soft edge weights.

    This utility is primarily meant for similarity-learning models such as
    SimGNN that operate on fully connected graphs with an ``edge_weight``
    attribute.  The returned object is a cloned instance of ``data`` whose
    ``edge_index`` enumerates every possible pair of nodes.  Existing edges are
    assigned a weight of ``1`` (or the mean of their edge attributes when
    available) while missing edges receive a weight of ``0``.

    Args:
        data: The original PyG ``Data`` or ``Batch`` object.
        keep_edge_attr: When ``True`` and the original data contains
            ``edge_attr``, a dense attribute tensor aligned with the new edge
            index is produced.  Missing edges are padded with zeros so that
            downstream modules can continue to rely on ``edge_attr`` if
            required.

    Returns:
        A cloned ``Data`` object whose ``edge_index`` enumerates all
        ``num_nodes ** 2`` directed edges and that contains a new
        ``edge_weight`` attribute describing the soft connectivity.
    """

    if isinstance(data, Batch):
        # Recursively densify each component graph so that no artificial
        # cross-graph connections are introduced when operating on batched
        # inputs.
        converted = [
            convert_hard_to_soft_edges(item, keep_edge_attr=keep_edge_attr)
            for item in data.to_data_list()
        ]
        kwargs = {}
        follow_batch = getattr(data, "_follow_batch", None)
        exclude_keys = getattr(data, "_exclude_keys", None)
        if follow_batch:
            kwargs["follow_batch"] = follow_batch
        if exclude_keys:
            kwargs["exclude_keys"] = exclude_keys
        return data.__class__.from_data_list(converted, **kwargs)

    if not isinstance(data, Data):
        raise TypeError(
            "Expected a torch_geometric.data.Data or Batch instance"
        )

    edge_index = getattr(data, "edge_index", None)
    if edge_index is None:
        raise ValueError("The input graph does not contain an edge_index")

    num_nodes: Optional[int] = getattr(data, "num_nodes", None)
    if num_nodes is None or num_nodes == 0:
        x = getattr(data, "x", None)
        if x is not None:
            num_nodes = int(x.size(0))
        elif edge_index.numel() > 0:
            num_nodes = int(edge_index.max().item() + 1)
        else:
            num_nodes = 0

    device = edge_index.device
    full_row = torch.arange(num_nodes, device=device).repeat_interleave(num_nodes)
    full_col = torch.arange(num_nodes, device=device).repeat(num_nodes)
    full_edge_index = torch.stack([full_row, full_col], dim=0)

    full_edge_weight = torch.zeros(num_nodes * num_nodes, device=device, dtype=torch.float32)

    edge_attr = getattr(data, "edge_attr", None)
    attr_for_weight = None
    attr_dim = None
    full_edge_attr = None
    if edge_attr is not None:
        attr_for_weight = edge_attr.to(device=device, dtype=torch.float32)
    if keep_edge_attr and edge_attr is not None:
        edge_attr = edge_attr if edge_attr.dim() > 1 else edge_attr.unsqueeze(-1)
        attr_dim = int(edge_attr.size(-1))
        full_edge_attr = torch.zeros(num_nodes * num_nodes, attr_dim,
                                     device=device, dtype=edge_attr.dtype)

    if edge_index.numel() > 0:
        # Convert the original sparse edges into flat indices.
        flat_ids = edge_index[0] * num_nodes + edge_index[1]

        if attr_for_weight is not None:
            if attr_for_weight.dim() == 1:
                weights = attr_for_weight
            else:
                weights = attr_for_weight.mean(dim=-1)
        else:
            weights = torch.ones_like(flat_ids, dtype=torch.float32, device=device)

        if hasattr(full_edge_weight, "scatter_reduce_"):
            full_edge_weight.scatter_reduce_(
                0, flat_ids, weights, reduce="amax", include_self=True
            )
        else:
            for idx, weight in zip(flat_ids.tolist(), weights.tolist()):
                if weight > full_edge_weight[idx]:
                    full_edge_weight[idx] = weight

        if full_edge_attr is not None:
            if hasattr(full_edge_attr, "scatter_reduce_"):
                full_edge_attr.scatter_reduce_(
                    0,
                    flat_ids.unsqueeze(-1).expand(-1, attr_dim),
                    edge_attr,
                    reduce="amax",
                    include_self=True,
                )
            else:
                for idx, value in zip(flat_ids.tolist(), edge_attr.tolist()):
                    value_tensor = torch.as_tensor(value, dtype=full_edge_attr.dtype, device=device)
                    current = full_edge_attr[idx]
                    full_edge_attr[idx] = torch.maximum(current, value_tensor)

    new_data = data.clone()
    new_data.edge_index = full_edge_index
    new_data.edge_weight = full_edge_weight

    if full_edge_attr is not None:
        if attr_dim == 1:
            new_data.edge_attr = full_edge_attr.view(-1)
        else:
            new_data.edge_attr = full_edge_attr
    elif keep_edge_attr and hasattr(new_data, "edge_attr"):
        # Remove stale attribute to avoid mismatched shapes downstream.
        delattr(new_data, "edge_attr")

    return new_data