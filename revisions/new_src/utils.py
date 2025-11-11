# utils.py
#
# Graph visualization utilities for datasets loaded via dataAdapter.load_dataset.
# Uses NetworkX + Matplotlib. Supports node colors, labels, edge styles,
# and metadata attached directly to the dataset.

import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from typing import Any, List, Mapping, Optional, Sequence, Union
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from types import SimpleNamespace


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

    return new_data


def eval_plot(
    explainee: torch.nn.Module,
    gen_graphs_0: Sequence[Data],
    gen_graphs_1: Sequence[Data],
    obs_graphs_0: Sequence[Data],
    obs_graphs_1: Sequence[Data],
    *,
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

    def _infer_module_device(module: torch.nn.Module) -> torch.device:
        for tensor in module.parameters():
            return tensor.device
        for tensor in module.buffers():
            return tensor.device
        return torch.device("cpu")

    plot_kwargs = dict(plot_kwargs or {})
    for reserved in ("ax", "show"):
        plot_kwargs.pop(reserved, None)

    gen_graphs_0 = list(gen_graphs_0)
    gen_graphs_1 = list(gen_graphs_1)
    obs_graphs_0 = list(obs_graphs_0)
    obs_graphs_1 = list(obs_graphs_1)

    if not gen_graphs_0 and not gen_graphs_1:
        raise ValueError("At least one generated graph must be provided")

    ged_device = _infer_module_device(ged_model)

    if device is not None:
        device = torch.device(device)
    else:
        device = _infer_module_device(explainee)

    def _predict_probs(graphs: Sequence[Data], target_class_idx: int) -> Sequence[float]:
        if len(graphs) == 0:
            return []

        loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
        probs: List[float] = []
        with torch.inference_mode():
            was_training = explainee.training
            explainee.eval()
            try:
                for batch in loader:
                    if device is not None:
                        batch = batch.to(device, non_blocking=True)
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

                    pred_probs = pred_probs[:, target_class_idx]
                    probs.extend(pred_probs.detach().cpu().tolist())
            finally:
                if was_training:
                    explainee.train()
        return probs

    def _pair_distance(gen_graph: Data, obs_graph: Data) -> float:
        from .graph_level_dist import neural_approx_ged_dist

        was_training = ged_model.training
        try:
            obs_ref = convert_hard_to_soft_edges(obs_graph)
            if ged_device is not None:
                obs_ref = obs_ref.to(ged_device)

            module = neural_approx_ged_dist([obs_ref], ged_model)

            gen_soft = convert_hard_to_soft_edges(gen_graph)
            gen_batch = Batch.from_data_list([gen_soft])
            if ged_device is not None:
                gen_batch = gen_batch.to(ged_device)

            with torch.inference_mode():
                distance = module(gen_batch)
        finally:
            if was_training:
                ged_model.train()
        return float(distance.detach().cpu().item())

    gen_probs_0 = _predict_probs(gen_graphs_0, 0)
    gen_probs_1 = _predict_probs(gen_graphs_1, 1)
    obs_probs_0 = _predict_probs(obs_graphs_0, 0)
    obs_probs_1 = _predict_probs(obs_graphs_1, 1)

    class_pairs = [
        (
            0,
            class_labels[0] if len(class_labels) > 0 else "Class 0",
            list(zip(gen_graphs_0, obs_graphs_0, gen_probs_0, obs_probs_0)),
        ),
        (
            1,
            class_labels[1] if len(class_labels) > 1 else "Class 1",
            list(zip(gen_graphs_1, obs_graphs_1, gen_probs_1, obs_probs_1)),
        ),
    ]

    class_pairs = [(idx, label, pairs) for idx, label, pairs in class_pairs if pairs]
    if not class_pairs:
        raise ValueError("No graph pairs available to plot.")

    limits = [min(max_pairs, len(pairs)) for _, _, pairs in class_pairs]
    max_limit = max(limits)

    if max_limit == 0:
        raise ValueError("max_pairs resulted in zero columns to plot.")

    fig_width = max(1, max_limit * 2) * 4.5
    fig_height = len(class_pairs) * 4.5
    fig, axes = plt.subplots(len(class_pairs), max_limit * 2, figsize=(fig_width, fig_height))

    axes_array = np.array(axes, copy=False)
    if axes_array.ndim == 1:
        axes_array = axes_array.reshape(1, -1)

    def _select_dataset(graph: Data, class_idx: int) -> Any:
        if isinstance(dataset, MappingABC):
            candidate = dataset.get(class_idx)
            if candidate is not None:
                return candidate
        elif isinstance(dataset, SequenceABC) and not isinstance(dataset, (str, bytes)):
            if class_idx < len(dataset):
                return dataset[class_idx]
        elif dataset is not None:
            return dataset

        meta_candidate = getattr(graph, "metadata", None)
        if meta_candidate is not None:
            if isinstance(meta_candidate, MappingABC):
                return SimpleNamespace(**meta_candidate)
            return meta_candidate

        dataset_attr = getattr(graph, "dataset", None)
        if dataset_attr is not None:
            return dataset_attr

        return dataset

    for row_idx, (class_idx, class_label, pairs) in enumerate(class_pairs):
        limit = limits[row_idx]
        row_axes = axes_array[row_idx]

        for pair_idx in range(max_limit):
            ax_gen = row_axes[pair_idx * 2]
            ax_obs = row_axes[pair_idx * 2 + 1]

            if pair_idx >= limit:
                ax_gen.set_visible(False)
                ax_obs.set_visible(False)
                continue

            gen_graph, obs_graph, gen_prob, obs_prob = pairs[pair_idx]
            distance = _pair_distance(gen_graph, obs_graph)

            dataset_gen = _select_dataset(gen_graph, class_idx)
            dataset_obs = _select_dataset(obs_graph, class_idx)

            plot_graph(
                gen_graph,
                dataset_gen,
                layout=layout,
                ax=ax_gen,
                show=False,
                **plot_kwargs,
            )
            ax_gen.set_title(f"Generated (P={gen_prob:.3f})")

            plot_graph(
                obs_graph,
                dataset_obs,
                layout=layout,
                ax=ax_obs,
                show=False,
                **plot_kwargs,
            )
            ax_obs.set_title(f"Observed (P={obs_prob:.3f})\nGED≈{distance:.3f}")

        first_axis = row_axes[0]
        first_axis.set_ylabel(class_label, rotation=90, fontsize=12, labelpad=40)

    fig.suptitle("Generated vs Observed graph pairs", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()