"""Evaluation utilities for graph generators.

This module implements several metrics that assess how well a graph generator
explains a target class by measuring model confidence, counterfactual impact,
robustness to perturbations, and similarity to real graphs.  The implementations
follow the procedures used in ``Generic_Results_Notebook.ipynb`` but are written
in a reusable form suitable for automated evaluation.
"""

from __future__ import annotations

import itertools
import random
from contextlib import contextmanager
import math
from typing import Callable, Dict, Mapping, MutableSequence, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader


Prediction = Mapping[str, Tensor]
MeanStd = Tuple[float, float]
DistanceFactory = Callable[[Sequence[Data]], torch.nn.Module]


@contextmanager
def _temporary_eval(module: torch.nn.Module):
    """Temporarily switch a module to ``eval`` mode during inference."""

    if not isinstance(module, torch.nn.Module):  # pragma: no cover - defensive
        yield
        return

    was_training = module.training
    try:
        module.eval()
        yield
    finally:  # pragma: no branch - simple restore logic
        if was_training:
            module.train()


@contextmanager
def _module_on_device(module: torch.nn.Module, device: Optional[torch.device]):
    """Move ``module`` to ``device`` for the duration of the context."""

    if (device is None) or (not isinstance(module, torch.nn.Module)):
        yield module
        return

    original_devices = {
        param.device for param in module.parameters(recurse=True)
    }
    original_devices.update(buffer.device for buffer in module.buffers(recurse=True))

    original_device = next(iter(original_devices), None)
    needs_restore = original_device is not None and original_device != device

    if needs_restore:
        module.to(device)

    try:
        yield module
    finally:
        if needs_restore:
            module.to(original_device)


def _detect_module_device(module: torch.nn.Module) -> Optional[torch.device]:
    """Infer the primary device associated with ``module``."""

    if not isinstance(module, torch.nn.Module):
        return None

    for param in module.parameters(recurse=True):
        return param.device

    for buffer in module.buffers(recurse=True):
        return buffer.device

    return None


def _detect_module_device(module: torch.nn.Module) -> Optional[torch.device]:
    """Infer the primary device associated with ``module``."""

    if not isinstance(module, torch.nn.Module):
        return None

    for param in module.parameters(recurse=True):
        return param.device

    for buffer in module.buffers(recurse=True):
        return buffer.device

    return None


def _mean_std(values: Tensor) -> MeanStd:
    """Return the mean and (population) standard deviation of a tensor."""

    if values.numel() == 0:
        raise ValueError("values must contain at least one element")
    values = values.float().view(-1)
    mean = values.mean().item()
    std = values.std(unbiased=False).item() if values.numel() > 1 else 0.0
    return mean, std


def _extract_probs(prediction: Prediction) -> Tensor:
    """Extract probability predictions from a model output mapping."""

    if "probs" in prediction:
        return prediction["probs"]
    if "logits" in prediction:
        return prediction["logits"].softmax(dim=-1)
    raise KeyError("Prediction dictionary must contain 'probs' or 'logits'.")


def _infer_num_nodes(graph: Data) -> int:
    num_nodes = getattr(graph, "num_nodes", None)
    if num_nodes is not None and num_nodes > 0:
        return int(num_nodes)
    x = getattr(graph, "x", None)
    if x is not None:
        return int(x.size(0))
    edge_index = getattr(graph, "edge_index", None)
    if edge_index is not None and edge_index.numel() > 0:
        return int(edge_index.max().item() + 1)
    return 0


def _edge_count(graph: Data) -> int:
    edge_index = getattr(graph, "edge_index", None)
    if edge_index is None:
        return 0
    return int(edge_index.size(1))


def _ensure_feature_tensor(
    graph: Data, feat_dim: int, *, attr: str = "x", count: Optional[int] = None
) -> Optional[Tensor]:
    if feat_dim == 0:
        return None
    tensor = getattr(graph, attr, None)
    if count is None:
        count = _infer_num_nodes(graph) if attr == "x" else _edge_count(graph)
    if tensor is None:
        return torch.zeros((count, feat_dim), dtype=torch.float32)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.size(-1) > feat_dim:
        tensor = tensor[..., :feat_dim]
    elif tensor.size(-1) < feat_dim:
        pad_shape = list(tensor.shape[:-1]) + [feat_dim - tensor.size(-1)]
        pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, pad], dim=-1)
    if tensor.size(0) != count:
        raise ValueError("Feature tensor count does not match inferred entity count")
    return tensor.float()


def _merge_graphs(base: Data, addition: Data) -> Data:
    """Create a disjoint union of ``base`` and ``addition`` graphs."""

    base_nodes = _infer_num_nodes(base)
    add_nodes = _infer_num_nodes(addition)

    base_edges = getattr(base, "edge_index", None)
    add_edges = getattr(addition, "edge_index", None)

    new_edge_indices: MutableSequence[Tensor] = []
    if base_edges is not None and base_edges.numel() > 0:
        new_edge_indices.append(base_edges.clone())
    if add_edges is not None and add_edges.numel() > 0:
        shifted = add_edges + base_nodes
        new_edge_indices.append(shifted)

    if new_edge_indices:
        edge_index = torch.cat(new_edge_indices, dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    feat_dim = 0
    base_x = getattr(base, "x", None)
    add_x = getattr(addition, "x", None)
    if base_x is not None:
        feat_dim = max(feat_dim, int(base_x.size(-1))) if base_x.dim() > 1 else 1
    if add_x is not None:
        feat_dim = max(feat_dim, int(add_x.size(-1))) if add_x.dim() > 1 else max(feat_dim, 1)

    x = None
    if feat_dim > 0:
        base_feat = _ensure_feature_tensor(base, feat_dim, count=base_nodes)
        add_feat = _ensure_feature_tensor(addition, feat_dim, count=add_nodes)
        x = torch.cat([base_feat, add_feat], dim=0)

    edge_feat_dim = 0
    base_attr = getattr(base, "edge_attr", None)
    add_attr = getattr(addition, "edge_attr", None)
    if base_attr is not None:
        edge_feat_dim = max(edge_feat_dim, int(base_attr.size(-1))) if base_attr.dim() > 1 else 1
    if add_attr is not None:
        edge_feat_dim = max(edge_feat_dim, int(add_attr.size(-1))) if add_attr.dim() > 1 else max(edge_feat_dim, 1)

    edge_attr = None
    if edge_feat_dim > 0:
        base_attr_tensor = _ensure_feature_tensor(base, edge_feat_dim, attr="edge_attr", count=_edge_count(base))
        add_attr_tensor = _ensure_feature_tensor(addition, edge_feat_dim, attr="edge_attr", count=_edge_count(addition))
        edge_attr = torch.cat([base_attr_tensor, add_attr_tensor], dim=0)

    base_weight = getattr(base, "edge_weight", None)
    add_weight = getattr(addition, "edge_weight", None)
    default_device = None
    if base_weight is not None:
        default_device = base_weight.device
    elif add_weight is not None:
        default_device = add_weight.device
    elif new_edge_indices:
        default_device = new_edge_indices[0].device

    weights: MutableSequence[Tensor] = []
    if base_edges is not None and base_edges.numel() > 0:
        if base_weight is not None:
            weights.append(base_weight.float())
        else:
            weights.append(
                torch.ones(
                    base_edges.size(1), dtype=torch.float32, device=default_device
                )
            )
    if add_edges is not None and add_edges.numel() > 0:
        if add_weight is not None:
            weights.append(add_weight.float())
        else:
            weights.append(
                torch.ones(
                    add_edges.size(1), dtype=torch.float32, device=default_device
                )
            )
    edge_weight = torch.cat(weights) if weights else None

    merged = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if edge_weight is not None:
        merged.edge_weight = edge_weight
    merged.num_nodes = base_nodes + add_nodes
    return merged


def _predict_target_probs(
    explainee: torch.nn.Module,
    graphs: Sequence[Data],
    target_class: int,
    *,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
) -> Tensor:
    loader = DataLoader(graphs, batch_size=batch_size)
    probs: MutableSequence[Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            if device is not None:
                batch = batch.to(device)
            pred = explainee(batch)
            batch_probs = _extract_probs(pred)
            probs.append(batch_probs[:, target_class].detach().cpu())
    return torch.cat(probs, dim=0)


def compute_target_class_probability(
    explainee: torch.nn.Module,
    graphs: Sequence[Data],
    target_class: int,
    *,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
) -> MeanStd:
    """Compute the Target Class Probability (TCP) metric."""

    if len(graphs) == 0:
        raise ValueError("graphs must contain at least one generated graph")

    with _temporary_eval(explainee):
        probs = _predict_target_probs(
            explainee, graphs, target_class, device=device, batch_size=batch_size
        )
    return _mean_std(probs)


def compute_counterfactual_delta(
    explainee: torch.nn.Module,
    generated_graphs: Sequence[Data],
    reference_graphs: Sequence[Data],
    target_class: int,
    *,
    device: Optional[torch.device] = None,
) -> MeanStd:
    """Compute the counterfactual delta metric.

    The metric measures how much inserting generated structures into
    opposite-class graphs increases the explainee's probability for the target
    class.  Positive values indicate that the generated motifs successfully push
    counterfactual examples towards the target class.
    """

    if len(generated_graphs) == 0:
        raise ValueError("generated_graphs must contain at least one graph")
    if len(reference_graphs) == 0:
        raise ValueError("reference_graphs must contain at least one graph")

    deltas: MutableSequence[Tensor] = []
    iterator = itertools.cycle(generated_graphs)

    with _temporary_eval(explainee), torch.inference_mode():
        for reference in reference_graphs:
            motif = next(iterator)
            base_prob = _predict_target_probs(
                explainee, [reference], target_class, device=device, batch_size=1
            )[0]
            merged = _merge_graphs(reference, motif)
            cf_prob = _predict_target_probs(
                explainee, [merged], target_class, device=device, batch_size=1
            )[0]
            deltas.append(cf_prob - base_prob)

    delta_tensor = torch.stack(deltas)
    return _mean_std(delta_tensor)


def _perturb_graph(graph: Data, rng: random.Random) -> Data:
    perturbed = graph.clone()
    edge_index = getattr(perturbed, "edge_index", None)
    num_nodes = _infer_num_nodes(perturbed)
    if edge_index is None or num_nodes < 2:
        return perturbed

    edge_index = edge_index.clone()
    num_edges = edge_index.size(1)
    remove_edge = num_edges > 0 and rng.random() < 0.5

    edge_attr = getattr(perturbed, "edge_attr", None)
    if edge_attr is not None:
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        edge_attr = edge_attr.clone()
    edge_weight = getattr(perturbed, "edge_weight", None)
    if edge_weight is not None:
        edge_weight = edge_weight.clone()

    if remove_edge and num_edges > 0:
        idx = rng.randrange(num_edges)
        mask = torch.ones(num_edges, dtype=torch.bool)
        mask[idx] = False
        edge_index = edge_index[:, mask]
        if edge_attr is not None:
            edge_attr = edge_attr[mask]
        if edge_weight is not None:
            edge_weight = edge_weight[mask]
    else:
        u = rng.randrange(num_nodes)
        v = rng.randrange(num_nodes)
        new_edge = edge_index.new_tensor([[u], [v]])
        edge_index = torch.cat([edge_index, new_edge], dim=1)
        if edge_attr is not None:
            feat_dim = edge_attr.size(-1)
            zeros = edge_attr.new_zeros((1, feat_dim))
            edge_attr = torch.cat([edge_attr, zeros], dim=0)
        if edge_weight is not None:
            one = edge_weight.new_ones((1,))
            edge_weight = torch.cat([edge_weight, one], dim=0)

    perturbed.edge_index = edge_index
    if edge_attr is not None:
        perturbed.edge_attr = edge_attr
    if edge_weight is not None:
        perturbed.edge_weight = edge_weight
    perturbed.num_nodes = num_nodes
    return perturbed


def compute_sensitivity_to_perturbations(
    explainee: torch.nn.Module,
    graphs: Sequence[Data],
    target_class: int,
    *,
    num_perturbations: int = 5,
    device: Optional[torch.device] = None,
    rng: Optional[random.Random] = None,
) -> MeanStd:
    """Measure sensitivity of generated graphs to random small perturbations."""

    if len(graphs) == 0:
        raise ValueError("graphs must contain at least one generated graph")
    if num_perturbations <= 0:
        raise ValueError("num_perturbations must be positive")

    rng = rng or random.Random()

    diffs: MutableSequence[Tensor] = []
    with _temporary_eval(explainee), torch.inference_mode():
        base_probs = _predict_target_probs(
            explainee, graphs, target_class, device=device, batch_size=32
        )
        for graph, base_prob in zip(graphs, base_probs):
            for _ in range(num_perturbations):
                perturbed = _perturb_graph(graph, rng)
                pert_prob = _predict_target_probs(
                    explainee, [perturbed], target_class, device=device, batch_size=1
                )[0]
                diffs.append((pert_prob - base_prob).abs())

    diff_tensor = torch.stack(diffs)
    return _mean_std(diff_tensor)


def _distance_per_graph(
    module: torch.nn.Module,
    graphs: Sequence[Data],
    *,
    device: Optional[torch.device] = None,
) -> Tensor:
    values: MutableSequence[Tensor] = []
    with torch.inference_mode():
        for graph in graphs:
            batch = Batch.from_data_list([graph])
            if device is not None:
                batch = batch.to(device)
            value = module(batch).detach().cpu()
            values.append(value.view(-1))
    return torch.cat(values)


def compute_average_distance_to_classes(
    generated_graphs: Sequence[Data],
    target_graphs: Sequence[Data],
    other_graphs: Sequence[Data],
    distance_factories: Mapping[str, DistanceFactory],
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, MeanStd]]:
    """Compute average distances between generated graphs and class datasets."""

    if len(generated_graphs) == 0:
        raise ValueError("generated_graphs must contain at least one graph")
    if len(target_graphs) == 0:
        raise ValueError("target_graphs must contain at least one graph")
    if len(other_graphs) == 0:
        raise ValueError("other_graphs must contain at least one graph")

    results: Dict[str, Dict[str, MeanStd]] = {}

    for name, factory in distance_factories.items():
        target_module = factory(target_graphs)
        other_module = factory(other_graphs)
        if device is not None:
            target_module = target_module.to(device)
            other_module = other_module.to(device)

        with _temporary_eval(target_module):
            target_values = _distance_per_graph(target_module, generated_graphs, device=device)
        with _temporary_eval(other_module):
            other_values = _distance_per_graph(other_module, generated_graphs, device=device)
        delta_values = other_values - target_values

        results[name] = {
            "target": _mean_std(target_values),
            "non_target": _mean_std(other_values),
            "delta": _mean_std(delta_values),
        }

    return results


def eval_summary(
    explainee: torch.nn.Module,
    gen_graphs_0: Sequence[Data],
    gen_graphs_1: Sequence[Data],
    obs_graphs_0: Sequence[Data],
    obs_graphs_1: Sequence[Data],
    dist_to_0: torch.nn.Module,
    dist_to_1: torch.nn.Module,
) -> float:
    device = _detect_module_device(explainee)

    with _module_on_device(explainee, device):
        tcp_0 = compute_target_class_probability(
            explainee, gen_graphs_0, 0, device=device
        )
        tcp_1 = compute_target_class_probability(
            explainee, gen_graphs_1, 1, device=device
        )
        cd_0 = compute_counterfactual_delta(
            explainee, gen_graphs_0, obs_graphs_1, 0, device=device
        )
        cd_1 = compute_counterfactual_delta(
            explainee, gen_graphs_1, obs_graphs_0, 1, device=device
        )
        perd_0 = compute_sensitivity_to_perturbations(
            explainee, gen_graphs_0, 0, device=device
        )
        perd_1 = compute_sensitivity_to_perturbations(
            explainee, gen_graphs_1, 1, device=device
        )

    class _OnlineStats:
        def __init__(self) -> None:
            self.count = 0
            self.mean = 0.0
            self.m2 = 0.0

        def update(self, value: float) -> None:
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (value - self.mean)

        def mean_std(self) -> Tuple[float, float]:
            if self.count == 0:
                return 0.0, 0.0
            variance = self.m2 / self.count if self.count > 1 else 0.0
            return self.mean, math.sqrt(variance)

    stats_0 = _OnlineStats()
    stats_1 = _OnlineStats()

    with _module_on_device(dist_to_0, device) as module_0, _module_on_device(
        dist_to_1, device
    ) as module_1, torch.inference_mode():
        for graph in gen_graphs_0:
            batch = Batch.from_data_list([graph])
            if device is not None:
                batch = batch.to(device)
            delta = module_0.evaluate(batch) - module_1.evaluate(batch)
            delta = delta.detach().float().cpu().view(-1)
            for value in delta:
                stats_0.update(float(value.item()))

        for graph in gen_graphs_1:
            batch = Batch.from_data_list([graph])
            if device is not None:
                batch = batch.to(device)
            delta = module_1.evaluate(batch) - module_0.evaluate(batch)
            delta = delta.detach().float().cpu().view(-1)
            for value in delta:
                stats_1.update(float(value.item()))

    del_dist_0_mean, del_dist_0_std = stats_0.mean_std()
    del_dist_1_mean, del_dist_1_std = stats_1.mean_std()

    print(f"Prediction Interval Class 0 {tcp_0[0]} +/- {tcp_0[1]}\n")
    print(f"Prediction Interval Class 1 {tcp_1[0]} +/- {tcp_1[1]}\n")
    print(f"Counterfactual Shift to Class 0 {cd_0[0]} +/- {cd_0[1]}\n")
    print(f"Counterfactual Shift to Class 1 {cd_1[0]} +/- {cd_1[1]}\n")
    print(f"Class 0 Perturbation Sensitivity {perd_0[0]} +/- {perd_0[1]}\n")
    print(f"Class 1 Perturbation Sensitivity {perd_1[0]} +/- {perd_1[1]}\n")
    print(
        f"Relative Distance to Class 0 {del_dist_0_mean} +/- {del_dist_0_std} \n"
    )
    print(
        f"Relative Distance to Class 1 {del_dist_1_mean} +/- {del_dist_1_std} \n"
    )

    score = (
        tcp_0[0]
        + tcp_1[0]
        + cd_0[0]
        + cd_1[0]
        - perd_0[0]
        - perd_1[0]
        - del_dist_0_mean
        - del_dist_1_mean
    )

    return float(score)
