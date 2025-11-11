import math
import random
from typing import List, Sequence

import torch
from torch import nn
from torch_geometric.data import Batch, Data

class dummyDist(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cont_data):
        print(cont_data)

        return 1

class neural_approx_ged_dist(nn.Module):

    def __init__(self, data, model):
        super().__init__()
        self.data = data
        self.model = model.eval()
       
    def _sample_obs_data(self, gen_graph):
        target_n = gen_graph.num_graphs

        indices = random.choices(range(len(self.data)), k=target_n)
        chosen_graphs = [self.data[i] for i in indices]

        return Batch.from_data_list(chosen_graphs)

    def forward(self, cont_data): 
        obs_data = self._sample_obs_data(cont_data)

        dist = self.model(cont_data, obs_data)

        return dist.mean()


class mcs_soft_graph_dist(nn.Module):
    """Differentiable relaxation of a maximum common subgraph distance.

    The module mirrors :class:`neural_approx_ged_dist` in spirit – it samples a
    batch of observed graphs from ``data`` and compares them to the input
    ``cont_data``.  The distance between two graphs is estimated through a
    differentiable soft assignment between their nodes and by measuring the
    amount of overlapping edge weight under this soft matching.  The overlap is
    normalised into ``[0, 1]`` so that the returned value is a proper distance
    (``0`` for identical graphs, approaching ``1`` for dissimilar ones).

    Args:
        data: Sequence of reference graphs.  When ``forward`` is called the
            module samples a batch of graphs of the same size as ``cont_data``
            from this collection, analogous to :class:`neural_approx_ged_dist`.
        temperature: Softmax temperature used for the node matching logits. A
            lower temperature results in assignments closer to a hard matching.
        sinkhorn_iters: Number of Sinkhorn normalisation iterations used to
            transform the similarity matrix into a doubly-stochastic matching
            matrix.
        eps: Numerical stability constant used when normalising the overlap.
    """

    def __init__(
        self,
        data: Sequence[Data],
        *,
        temperature: float = 0.1,
        sinkhorn_iters: int = 10,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if sinkhorn_iters <= 0:
            raise ValueError("sinkhorn_iters must be a positive integer")

        self.data = list(data)
        if len(self.data) == 0:
            raise ValueError("data must contain at least one reference graph")

        self.temperature = float(temperature)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.eps = float(eps)

    def _sample_obs_data(self, gen_graph: Batch) -> Batch:
        target_n = gen_graph.num_graphs

        indices = random.choices(range(len(self.data)), k=target_n)
        chosen_graphs = [self.data[i] for i in indices]

        return Batch.from_data_list(chosen_graphs)

    def forward(self, cont_data: Batch) -> torch.Tensor:
        """Compute the mean soft-MCS distance between generated and observed graphs."""

        if not isinstance(cont_data, Batch):
            raise TypeError("cont_data must be a torch_geometric.data.Batch instance")

        device = None
        if hasattr(cont_data, "edge_weight") and cont_data.edge_weight is not None:
            device = cont_data.edge_weight.device
        elif hasattr(cont_data, "x") and cont_data.x is not None:
            device = cont_data.x.device

        obs_data = self._sample_obs_data(cont_data)
        if device is not None:
            obs_data = obs_data.to(device)

        cont_list = cont_data.to_data_list()
        obs_list = obs_data.to_data_list()

        distances: List[torch.Tensor] = []
        for generated_graph, observed_graph in zip(cont_list, obs_list):
            distances.append(self._pair_distance(generated_graph, observed_graph))

        stacked = torch.stack(distances)
        return stacked.mean()

    def _pair_distance(self, g1: Data, g2: Data) -> torch.Tensor:
        adj1 = self._get_dense_adjacency(g1)
        adj2 = self._get_dense_adjacency(g2)

        if adj1.numel() == 0 or adj2.numel() == 0:
            # Handle empty graphs gracefully: if either graph has no nodes the
            # overlap is trivially zero and the distance defaults to ``1``.
            return torch.tensor(1.0, device=adj1.device if adj1.numel() > 0 else adj2.device)

        features1 = self._get_node_features(g1, adj1)
        features2 = self._get_node_features(g2, adj2)

        match_matrix = self._soft_matching(features1, features2)

        adj1_soft = adj1.clamp_min(0.0)
        adj2_soft = adj2.clamp_min(0.0)

        # Align the adjacency matrices using the soft assignment.  ``match`` is
        # interpreted as mapping nodes of ``g1`` to nodes of ``g2``.
        aligned_2 = match_matrix @ adj2_soft @ match_matrix.transpose(-1, -2)
        aligned_1 = match_matrix.transpose(-1, -2) @ adj1_soft @ match_matrix

        overlap_12 = (adj1_soft * aligned_2).sum()
        overlap_21 = (adj2_soft * aligned_1).sum()

        union_base = adj1_soft.sum() + adj2_soft.sum()
        union_12 = torch.clamp(union_base - overlap_12, min=self.eps)
        union_21 = torch.clamp(union_base - overlap_21, min=self.eps)

        overlap_score = 0.5 * (
            overlap_12 / (union_12 + self.eps) + overlap_21 / (union_21 + self.eps)
        )

        overlap_score = overlap_score.clamp(0.0, 1.0)
        return 1.0 - overlap_score

    def _get_dense_adjacency(self, graph: Data) -> torch.Tensor:
        edge_weight = getattr(graph, "edge_weight", None)
        if edge_weight is None:
            raise ValueError(
                "Graph does not contain edge_weight. Ensure graphs are converted to a dense representation."
            )

        edge_weight = edge_weight.float()
        num_nodes = getattr(graph, "num_nodes", None)
        if num_nodes is None or num_nodes == 0:
            x = getattr(graph, "x", None)
            if x is not None:
                num_nodes = int(x.size(0))
            else:
                num_nodes = int(round(math.sqrt(edge_weight.numel())))

        if num_nodes == 0:
            return edge_weight.new_zeros((0, 0))

        expected = num_nodes * num_nodes
        if edge_weight.numel() != expected:
            raise ValueError(
                "edge_weight does not represent a dense adjacency matrix: "
                f"expected {expected} values but found {edge_weight.numel()}"
            )

        return edge_weight.view(num_nodes, num_nodes)

    def _get_node_features(self, graph: Data, adjacency: torch.Tensor) -> torch.Tensor:
        features = getattr(graph, "x", None)
        if features is not None:
            return features.float()

        # Fall back to simple structural features derived from the adjacency
        # matrix (node degrees).  Using the adjacency ensures that gradients can
        # flow back to edge weights even when explicit node features are absent.
        degrees = adjacency.sum(dim=-1, keepdim=True)
        return degrees

    def _soft_matching(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        logits = features1 @ features2.transpose(-1, -2)
        logits = logits / self.temperature

        # Apply Sinkhorn normalisation to obtain a doubly stochastic matrix that
        # serves as a soft correspondence between nodes of the two graphs.  The
        # procedure remains differentiable because it only consists of smooth
        # normalisation steps.
        for _ in range(self.sinkhorn_iters):
            logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)

        return torch.exp(logits)

 
