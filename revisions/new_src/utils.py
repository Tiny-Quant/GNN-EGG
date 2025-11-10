# utils.py
#
# Graph visualization utilities for datasets loaded via dataAdapter.load_dataset.
# Uses NetworkX + Matplotlib. Supports node colors, labels, edge styles,
# and metadata attached directly to the dataset.

import torch
import networkx as nx
import matplotlib.pyplot as plt


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
