"""Utilities for turning instance-level explanations into dataset-level insights.

This module serves as the glue layer for a four-cell workflow that is common in
our notebooks:

1. Train an explainee and configure a :class:`torch_geometric.explain.Explainer`
   with a user selected algorithm.
2. Run the explainer over a dataset and aggregate the instance-level masks into
   batches of canonical motifs per class.
3. Compute the quantitative summary metrics defined in :mod:`.eval`.
4. Visualise the generated motifs together with the observed class graphs via
   :func:`.utils.eval_plot`.

Every public helper exposes a minimal, well documented surface so that the
notebook code remains concise while still being flexible in the choice of
explainers, aggregation strategies, metrics and visualisation settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Union

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, Data
from torch_geometric.explain import Explainer, CaptumExplainer
from torch_geometric.explain.algorithm import DummyExplainer, ExplainerAlgorithm, GNNExplainer, PGExplainer
from torch_geometric.utils import subgraph

from .eval import eval_summary
from .utils import eval_plot

GraphList = Sequence[Data]
ExplanationStrategy = Callable[..., GraphList]

try:  # Torch Geometric 2.4+
    from torch_geometric.explain import Explanation
except ImportError:  # pragma: no cover - fallback for older versions
    Explanation = Any  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 1.  Explainer construction
# ---------------------------------------------------------------------------


class ExtractProbs(nn.Module):
    """Adapter that exposes an explainee with the interface expected by PyG.

    The wrapped model is assumed to accept a :class:`~torch_geometric.data.Batch`
    object and to return either a tensor of logits/probabilities or a mapping
    containing ``"probs"`` or ``"logits"``.  The adapter keeps the original
    behaviour but ensures that the explainer always receives probabilities.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        pyg_batch = Batch(x=x, edge_index=edge_index, batch=batch)
        if edge_attr is not None:
            pyg_batch.edge_attr = edge_attr
            pyg_batch.edge_weight = edge_attr

        prediction = self.model(pyg_batch, **kwargs)

        if isinstance(prediction, Mapping):
            if "probs" in prediction:
                probs = prediction["probs"]
            elif "logits" in prediction:
                probs = prediction["logits"].softmax(dim=-1)
            else:  # pragma: no cover - defensive
                raise KeyError("Prediction dictionary must contain 'probs' or 'logits'.")
        else:
            probs = prediction
            if probs.dim() == 1:
                probs = probs.unsqueeze(0)
            probs = probs.softmax(dim=-1)

        return probs


AlgorithmFactory = Callable[..., ExplainerAlgorithm]
AlgorithmSpec = Union[str, ExplainerAlgorithm, AlgorithmFactory]


def _build_algorithm(name: AlgorithmSpec, **kwargs: Any) -> ExplainerAlgorithm:
    """Resolve ``name`` into a concrete :class:`ExplainerAlgorithm` instance."""

    if isinstance(name, ExplainerAlgorithm):
        return name

    if callable(name) and not isinstance(name, str):
        return name(**kwargs)

    if not isinstance(name, str):  # pragma: no cover - defensive
        raise TypeError("algorithm must be a string, algorithm instance or factory")

    key = name.lower()
    if key in {"gnn", "gnnexplainer"}:
        return GNNExplainer(**kwargs)
    if key in {"pg", "pgexplainer"}:
        return PGExplainer(**kwargs)
    if key in {"dummy", "random"}:
        return DummyExplainer(**kwargs)
    if key in {"captum_saliency", "saliency"}:
        return CaptumExplainer(algorithm="Saliency")
    if key in {"captum_ig", "integrated_gradients", "ig"}:
        return CaptumExplainer(algorithm="IntegratedGradients")
    if key.startswith("captum"):
        algorithm = kwargs.pop("algorithm", "Saliency")
        return CaptumExplainer(algorithm=algorithm)

    raise ValueError(f"Unknown explainer algorithm '{name}'.")


def build_explainer(
    explainee: nn.Module,
    *,
    algorithm: AlgorithmSpec = "gnnexplainer",
    algorithm_kwargs: Optional[Mapping[str, Any]] = None,
    explanation_type: str = "model",
    node_mask_type: Optional[str] = "object",
    edge_mask_type: Optional[str] = "object",
    model_config: Optional[Mapping[str, Any]] = None,
    threshold_config: Optional[Mapping[str, Any]] = None,
) -> Explainer:
    """Create a ready-to-use :class:`Explainer` for the trained ``explainee``.

    Parameters
    ----------
    explainee:
        Trained model to be explained.
    algorithm:
        Either the name of a built-in algorithm (``"gnnexplainer"``,
        ``"pgexplainer"``, ``"captum"`` variants, ``"dummy"``) or a custom
        :class:`ExplainerAlgorithm`/callable.
    algorithm_kwargs:
        Optional configuration passed to the algorithm constructor.
    explanation_type, node_mask_type, edge_mask_type, model_config,
    threshold_config:
        Mirrors the arguments of :class:`Explainer` for notebook convenience.
    """

    adapter = ExtractProbs(explainee)
    algorithm_kwargs = dict(algorithm_kwargs or {})

    resolved_algorithm = _build_algorithm(algorithm, **algorithm_kwargs)

    model_config = dict(
        mode="binary_classification",
        task_level="graph",
        return_type="probs",
        **(model_config or {}),
    )

    return Explainer(
        model=adapter,
        algorithm=resolved_algorithm,
        explanation_type=explanation_type,
        model_config=model_config,
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
        threshold_config=threshold_config,
    )


# ---------------------------------------------------------------------------
# 2.  Aggregation helpers
# ---------------------------------------------------------------------------


def wl_hash(data: Data, hops: int = 2) -> int:
    """Simple Weisfeiler-Lehman hash used to canonicalise motifs."""

    edge_index = data.edge_index
    num_nodes = data.num_nodes

    if getattr(data, "x", None) is not None:
        labels: List[Any] = [tuple(data.x[i].tolist()) for i in range(num_nodes)]
    else:
        deg = torch.bincount(edge_index[0], minlength=num_nodes)
        labels = [int(d.item()) for d in deg]

    for _ in range(hops):
        new_labels = []
        for v in range(num_nodes):
            neigh = edge_index[1][edge_index[0] == v]
            neigh_labels = sorted(labels[int(u)] for u in neigh)
            combined = (labels[v], tuple(neigh_labels))
            new_labels.append(hash(combined))
        labels = new_labels

    return hash(tuple(sorted(labels)))


def _topk_edge_components(data: Data, edge_mask: Tensor, top_p: float = 0.1) -> List[Data]:
    """Return connected components spanned by the top ``p``% of edges."""

    edge_mask = edge_mask.detach().float().view(-1)
    num_edges = edge_mask.numel()
    if num_edges == 0:
        return []

    k = max(1, int(num_edges * top_p))
    top_idx = torch.topk(edge_mask, k).indices
    edge_index = data.edge_index[:, top_idx]

    nodes = edge_index.unique().tolist()
    adj: MutableMapping[int, List[int]] = {int(n): [] for n in nodes}
    for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))

    visited = set()
    components: List[List[int]] = []
    for node in nodes:
        if node in visited:
            continue
        stack = [node]
        comp: List[int] = []
        visited.add(node)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adj[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        components.append(comp)

    motifs: List[Data] = []
    for comp_nodes in components:
        node_idx = torch.tensor(comp_nodes, dtype=torch.long)
        ei, _ = subgraph(node_idx, data.edge_index, relabel_nodes=True)
        x = data.x[node_idx] if getattr(data, "x", None) is not None else None
        motif = Data(x=x, edge_index=ei, num_nodes=len(comp_nodes))
        motif.original_node_indices = node_idx
        motifs.append(motif)
    return motifs


def _node_threshold_components(data: Data, node_mask: Tensor, threshold: float = 0.5) -> List[Data]:
    mask = node_mask.detach().float().view(-1)
    selected = (mask >= threshold).nonzero(as_tuple=False).view(-1)
    if selected.numel() == 0:
        return []
    ei, _ = subgraph(selected, data.edge_index, relabel_nodes=True)
    x = data.x[selected] if getattr(data, "x", None) is not None else None
    motif = Data(x=x, edge_index=ei, num_nodes=int(selected.numel()))
    motif.original_node_indices = selected
    return [motif]


def _normalize_mask(mask: Optional[Tensor]) -> Optional[Tensor]:
    if mask is None:
        return None
    mask = mask.detach().float().view(-1)
    if mask.numel() == 0:
        return mask
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


def _resolve_strategy(name_or_callable: Union[str, ExplanationStrategy]) -> ExplanationStrategy:
    if callable(name_or_callable) and not isinstance(name_or_callable, str):
        return name_or_callable

    if not isinstance(name_or_callable, str):  # pragma: no cover - defensive
        raise TypeError("aggregation strategy must be a callable or string")

    key = name_or_callable.lower()
    if key in {"wl_topk", "wl", "graphframerx"}:
        def strategy(data: Data, explanation: Explanation, *, top_p: float = 0.1, wl_hops: int = 2, **_: Any) -> GraphList:
            edge_mask = explanation.edge_mask
            if edge_mask is None:
                node_mask = explanation.node_mask
                if node_mask is None:
                    return []
                edge_mask = node_mask[data.edge_index[0]] * 0.5 + node_mask[data.edge_index[1]] * 0.5
            edge_mask = _normalize_mask(edge_mask)
            motifs = _topk_edge_components(data, edge_mask, top_p=top_p)
            unique: Dict[int, Data] = {}
            for motif in motifs:
                unique.setdefault(wl_hash(motif, hops=wl_hops), motif)
            return list(unique.values())
        return strategy

    if key in {"node_threshold", "saliency", "gradient"}:
        def strategy(data: Data, explanation: Explanation, *, threshold: float = 0.5, **_: Any) -> GraphList:
            node_mask = explanation.node_mask
            if node_mask is None:
                return []
            node_mask = _normalize_mask(node_mask)
            return _node_threshold_components(data, node_mask, threshold=threshold)
        return strategy

    raise ValueError(f"Unknown aggregation strategy '{name_or_callable}'.")


@dataclass
class AggregationResult:
    """Container bundling motif batches with metadata for downstream cells."""

    by_class: Dict[int, Batch]
    raw_by_class: Dict[int, List[Data]]

    def graphs_for(self, class_id: int) -> GraphList:
        batch = self.by_class.get(class_id)
        if batch is None:
            return []
        return batch.to_data_list()


def _detect_module_device(module: Optional[nn.Module]) -> torch.device:
    if module is None:
        return torch.device("cpu")
    for param in module.parameters():
        return param.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def _predict_class_from_model(model: nn.Module, graph: Data, device: torch.device) -> int:
    if isinstance(graph, Batch):
        batch = graph
    else:
        batch = Batch.from_data_list([graph])
    with torch.inference_mode():
        batch = batch.to(device)
        prediction = model(batch)
        if isinstance(prediction, Mapping):
            if "probs" in prediction:
                probs = prediction["probs"]
            elif "logits" in prediction:
                probs = prediction["logits"].softmax(dim=-1)
            else:
                raise KeyError("Prediction dictionary must contain 'probs' or 'logits'.")
        else:
            probs = prediction
            if probs.dim() == 1:
                probs = probs.unsqueeze(0)
            probs = probs.softmax(dim=-1)
    return int(probs.argmax(dim=-1).item())


def aggregate_instance_explanations(
    dataset: Iterable[Data],
    explainer: Explainer,
    *,
    explainee: Optional[nn.Module] = None,
    strategy: Union[str, ExplanationStrategy] = "wl_topk",
    strategy_kwargs: Optional[Mapping[str, Any]] = None,
    device: Optional[Union[str, torch.device]] = None,
    class_getter: Optional[Callable[[Data], int]] = None,
) -> AggregationResult:
    """Aggregate explanations into motif batches grouped by predicted class.

    Parameters
    ----------
    dataset:
        Iterable of :class:`~torch_geometric.data.Data` objects to explain.
    explainer:
        Configured :class:`Explainer` returned by :func:`build_explainer`.
    explainee:
        Original model used to predict classes.  Required if ``class_getter`` is
        not supplied.
    strategy / strategy_kwargs:
        Aggregation scheme.  ``"wl_topk"`` follows the GraphFramEx-style
        pipeline (top edge components + WL hashing).  ``"node_threshold``
        aggregates salient node clusters.  Custom callables are also supported.
    class_getter:
        Optional callable that maps a :class:`Data` object to its class label.
        When omitted the predicted label from ``explainee`` is used.
    """

    if class_getter is None and explainee is None:
        raise ValueError("Either explainee or class_getter must be provided.")

    if device is not None:
        device = torch.device(device)
    else:
        device = _detect_module_device(explainee)
    strategy_fn = _resolve_strategy(strategy)
    strategy_kwargs = dict(strategy_kwargs or {})

    motif_lists: Dict[int, List[Data]] = {}

    for graph in dataset:
        data = graph
        data = data.to(device)
        num_nodes = getattr(data, "num_nodes", None)
        if num_nodes is None and getattr(data, "x", None) is not None:
            num_nodes = data.x.size(0)
        batch_vec = getattr(
            data,
            "batch",
            torch.zeros(num_nodes or 0, dtype=torch.long, device=device),
        )
        edge_attr = getattr(data, "edge_attr", getattr(data, "edge_weight", None))

        explanation = explainer(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=edge_attr,
            batch=batch_vec,
        )

        data_cpu = data.cpu()
        explanation_cpu = explanation.cpu()

        if class_getter is None:
            class_id = _predict_class_from_model(explainee, data_cpu, device)
        else:
            class_id = class_getter(data_cpu)

        motifs = strategy_fn(data_cpu, explanation_cpu, **strategy_kwargs)
        if not motifs:
            continue
        motif_lists.setdefault(class_id, []).extend(motifs)

    batch_by_class: Dict[int, Batch] = {}
    for cls, motifs in motif_lists.items():
        if motifs:
            batch_by_class[cls] = Batch.from_data_list(motifs)

    return AggregationResult(by_class=batch_by_class, raw_by_class=motif_lists)


# ---------------------------------------------------------------------------
# 3.  Metric + plotting wrappers
# ---------------------------------------------------------------------------


def _ensure_graph_sequence(graphs: Union[Batch, GraphList, None]) -> List[Data]:
    if graphs is None:
        return []
    if isinstance(graphs, Batch):
        return graphs.to_data_list()
    return list(graphs)


def run_eval_summary(
    explainee: nn.Module,
    aggregation: AggregationResult,
    *,
    observed_class_0: Union[Batch, GraphList],
    observed_class_1: Union[Batch, GraphList],
    dist_to_0: nn.Module,
    dist_to_1: nn.Module,
) -> float:
    """Convenience wrapper that plugs motif batches into :func:`eval_summary`."""

    gen_graphs_0 = aggregation.graphs_for(0)
    gen_graphs_1 = aggregation.graphs_for(1)
    obs_0 = _ensure_graph_sequence(observed_class_0)
    obs_1 = _ensure_graph_sequence(observed_class_1)

    return eval_summary(
        explainee,
        gen_graphs_0,
        gen_graphs_1,
        obs_0,
        obs_1,
        dist_to_0,
        dist_to_1,
    )


def plot_eval(
    explainee: nn.Module,
    aggregation: AggregationResult,
    *,
    observed_class_0: Union[Batch, GraphList],
    observed_class_1: Union[Batch, GraphList],
    ged_model: nn.Module,
    dataset: Optional[Any] = None,
    **plot_kwargs: Any,
) -> None:
    """Visualise aggregated motifs next to observed class graphs."""

    gen_graphs_0 = aggregation.graphs_for(0)
    gen_graphs_1 = aggregation.graphs_for(1)

    return eval_plot(
        explainee=explainee,
        gen_graphs_0=gen_graphs_0,
        gen_graphs_1=gen_graphs_1,
        obs_graphs_0=_ensure_graph_sequence(observed_class_0),
        obs_graphs_1=_ensure_graph_sequence(observed_class_1),
        ged_model=ged_model,
        dataset=dataset,
        **plot_kwargs,
    )


__all__ = [
    "build_explainer",
    "aggregate_instance_explanations",
    "run_eval_summary",
    "plot_eval",
    "wl_hash",
    "AggregationResult",
]
