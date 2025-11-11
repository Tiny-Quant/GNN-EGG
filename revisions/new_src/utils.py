# utils.py
#
# Graph visualization utilities for datasets loaded via dataAdapter.load_dataset.
# Uses NetworkX + Matplotlib. Supports node colors, labels, edge styles,
# and metadata attached directly to the dataset.

import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from typing import Any, List, Mapping, Optional, Sequence, Union


def _to_cpu(x):
    """Ensure tensor is on CPU."""
    if torch.is_tensor(x):
        return x.detach().cpu()
    return x


def plot_graph(
    data,
    dataset,
    node_size=400,
    font_size=8,
    edge_alpha=0.9,
    layout="spring",
    show_labels=True,
    figsize=(6, 6),
    *,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
):
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
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_size,
        ax=ax,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        edge_color="black",
        alpha=edge_alpha,
        ax=ax,
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
            ax=ax,
        )

    ax.set_axis_off()
    if created_fig:
        fig.tight_layout()
    if show:
        fig.canvas.draw_idle()
        plt.show()
    return ax


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


def eval_plot(
    explainee: torch.nn.Module,
    gen_graphs_0: Sequence[Data],
    gen_graphs_1: Sequence[Data],
    obs_graphs_0: Sequence[Data],
    obs_graphs_1: Sequence[Data],
    *,
    target_class: int,
    ged_model: torch.nn.Module,
    dataset: Optional[Any] = None,
    max_pairs: int = 5,
    batch_size: int = 32,
    device: Optional[Union[str, torch.device]] = None,
    layout: str = "spring",
    class_labels: Sequence[str] = ("Class 0", "Class 1"),
    plot_kwargs: Optional[Mapping[str, Any]] = None,
) -> None:
    """Visualise generated/observed graph pairs with model confidences and GED."""

    if ged_model is None:
        raise ValueError("ged_model must be provided to compute graph distances")

    plot_kwargs = dict(plot_kwargs or {})
    for reserved in ("ax", "show"):
        plot_kwargs.pop(reserved, None)

    gen_graphs_0 = list(gen_graphs_0)
    gen_graphs_1 = list(gen_graphs_1)
    obs_graphs_0 = list(obs_graphs_0)
    obs_graphs_1 = list(obs_graphs_1)

    if not gen_graphs_0 and not gen_graphs_1:
        raise ValueError("At least one generated graph must be provided")

    try:
        ged_device = next(ged_model.parameters()).device
    except StopIteration:  # pragma: no cover - defensive programming
        ged_device = torch.device("cpu")

    if device is not None:
        device = torch.device(device)

    def _predict_probs(graphs: Sequence[Data]) -> Sequence[float]:
        if len(graphs) == 0:
            return []

        loader = DataLoader(graphs, batch_size=batch_size)
        probs: List[float] = []
        with torch.inference_mode():
            was_training = explainee.training
            explainee.eval()
            try:
                for batch in loader:
                    if device is not None:
                        batch = batch.to(device)
                    prediction = explainee(batch)
                    if isinstance(prediction, Mapping):
                        if "probs" in prediction:
                            pred_probs = prediction["probs"]
                        elif "logits" in prediction:
                            pred_probs = prediction["logits"].softmax(dim=-1)
                        else:
                            raise KeyError(
                                "Prediction dictionary must contain 'probs' or 'logits'."
                            )
                    else:
                        pred_probs = prediction
                        if pred_probs.dim() == 1:
                            pred_probs = pred_probs.unsqueeze(0)
                        pred_probs = pred_probs.softmax(dim=-1)

                    pred_probs = pred_probs[:, target_class]
                    probs.extend(pred_probs.detach().cpu().tolist())
            finally:
                if was_training:
                    explainee.train()
        return probs

    def _pair_distance(gen_graph: Data, obs_graph: Data) -> float:
        from .graph_level_dist import neural_approx_ged_dist

        gen_soft = convert_hard_to_soft_edges(gen_graph)
        obs_soft = convert_hard_to_soft_edges(obs_graph)

        gen_batch = Batch.from_data_list([gen_soft])
        obs_batch = Batch.from_data_list([obs_soft])
        if ged_device is not None:
            gen_batch = gen_batch.to(ged_device)
            obs_batch = obs_batch.to(ged_device)

        was_training = ged_model.training
        try:
            module = neural_approx_ged_dist([obs_graph], ged_model)
            with torch.inference_mode():
                distance = module.model(gen_batch, obs_batch)
                distance = distance.mean()
        finally:
            if was_training:
                ged_model.train()
        return float(distance.detach().cpu().item())

    gen_probs_0 = _predict_probs(gen_graphs_0)
    gen_probs_1 = _predict_probs(gen_graphs_1)
    obs_probs_0 = _predict_probs(obs_graphs_0)
    obs_probs_1 = _predict_probs(obs_graphs_1)

    class_pairs = [
        (
            class_labels[0] if len(class_labels) > 0 else "Class 0",
            list(zip(gen_graphs_0, obs_graphs_0, gen_probs_0, obs_probs_0)),
        ),
        (
            class_labels[1] if len(class_labels) > 1 else "Class 1",
            list(zip(gen_graphs_1, obs_graphs_1, gen_probs_1, obs_probs_1)),
        ),
    ]

    for class_label, pairs in class_pairs:
        if not pairs:
            continue

        limit = min(max_pairs, len(pairs))
        fig, axes = plt.subplots(limit, 2, figsize=(10, 5 * limit))
        if limit == 1:
            axes = axes.reshape(1, 2)

        for row, (gen_graph, obs_graph, gen_prob, obs_prob) in enumerate(pairs[:limit]):
            distance = _pair_distance(gen_graph, obs_graph)

            ax_gen = axes[row, 0]
            ax_obs = axes[row, 1]

            plot_graph(
                gen_graph,
                dataset,
                layout=layout,
                ax=ax_gen,
                show=False,
                **plot_kwargs,
            )
            ax_gen.set_title(f"Generated (P={gen_prob:.3f})")

            plot_graph(
                obs_graph,
                dataset,
                layout=layout,
                ax=ax_obs,
                show=False,
                **plot_kwargs,
            )
            ax_obs.set_title(f"Observed (P={obs_prob:.3f})\nGED≈{distance:.3f}")

        fig.suptitle(f"{class_label} graph pairs", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        plt.show()
