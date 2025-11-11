import random
from typing import Callable, List, Optional, Sequence

import torch
from torch import nn
from torch_geometric.data import Batch, Data

from .utils import convert_hard_to_soft_edges


class _BaseGraphLevelDistance(nn.Module):
    """Utility base class shared by graph-level distance modules."""

    def __init__(self, data: Sequence[Data]) -> None:
        super().__init__()
        try:
            length = len(data)
        except TypeError as exc:  # pragma: no cover - defensive programming
            raise TypeError(
                "data must define __len__ so graphs can be sampled"
            ) from exc
        if length == 0:
            raise ValueError("data must contain at least one reference graph")
        self.data = data

    def _sample_obs_data(
        self,
        gen_graph: Batch,
        *,
        transform: Optional[Callable[[Data], Data]] = None,
    ) -> Batch:
        target_n = gen_graph.num_graphs

        indices = random.choices(range(len(self.data)), k=target_n)
        chosen_graphs: List[Data] = []
        for i in indices:
            graph = self.data[i]
            if transform is not None:
                graph = transform(graph)
            chosen_graphs.append(graph)

        return Batch.from_data_list(chosen_graphs)

    @staticmethod
    def _infer_num_nodes(graph: Data) -> int:
        num_nodes = getattr(graph, "num_nodes", None)
        if num_nodes is None or num_nodes == 0:
            x = getattr(graph, "x", None)
            if x is not None:
                num_nodes = int(x.size(0))
            else:
                raise ValueError(
                    "Unable to infer the number of nodes. Provide graph.x or graph.num_nodes."
                )
        return int(num_nodes)

    @classmethod
    def _dense_adjacency_from_edge_weight(cls, graph: Data) -> torch.Tensor:
        edge_weight = getattr(graph, "edge_weight", None)
        if edge_weight is None:
            raise ValueError(
                "Graph does not contain edge_weight. Ensure graphs are converted to a dense representation."
            )

        edge_weight = edge_weight.float()
        num_nodes = cls._infer_num_nodes(graph)
        if num_nodes == 0:
            return edge_weight.new_zeros((0, 0))

        expected_with_loops = num_nodes * num_nodes
        expected_without_loops = num_nodes * (num_nodes - 1)
        numel = edge_weight.numel()

        if numel == expected_with_loops:
            return edge_weight.view(num_nodes, num_nodes)

        if numel != expected_without_loops:
            raise ValueError(
                "edge_weight does not represent a supported dense adjacency matrix: "
                f"expected {expected_with_loops} (with self-loops) or {expected_without_loops} "
                f"(without self-loops) values but found {numel}"
            )

        mask = torch.ones(
            (num_nodes, num_nodes), dtype=torch.bool, device=edge_weight.device
        )
        mask.fill_diagonal_(False)

        adjacency = edge_weight.new_zeros((num_nodes * num_nodes,))
        adjacency = adjacency.masked_scatter(mask.view(-1), edge_weight)
        return adjacency.view(num_nodes, num_nodes)

    @staticmethod
    def _graph_device(graph) -> Optional[torch.device]:
        edge_weight = getattr(graph, "edge_weight", None)
        if edge_weight is not None:
            return edge_weight.device
        features = getattr(graph, "x", None)
        if features is not None:
            return features.device
        return None

class dummyDist(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cont_data):
        print(cont_data)

        return 1

class neural_approx_ged_dist(_BaseGraphLevelDistance):

    def __init__(self, data: Sequence[Data], model: nn.Module):
        super().__init__(data)
        self.model = model.eval()

    def forward(self, cont_data):
        cont_data = convert_hard_to_soft_edges(cont_data) 

        device = self._graph_device(cont_data)

        obs_data = self._sample_obs_data(
            cont_data, transform=convert_hard_to_soft_edges
        )
        if device is not None:
            obs_data = obs_data.to(device)
        
        dist = self.model(cont_data, obs_data)

        return dist.mean()

    @torch.no_grad()
    def evaluate(self, cont_data):
        cont_data = convert_hard_to_soft_edges(cont_data) 

        device = self._graph_device(cont_data)

        obs_data = self._sample_obs_data(
            cont_data, transform=convert_hard_to_soft_edges
        )
        if device is not None:
            obs_data = obs_data.to(device)
        
        dist = self.model(cont_data, obs_data)

        return dist

class mcs_soft_graph_dist(_BaseGraphLevelDistance):
    """Differentiable relaxation of a maximum common subgraph distance.

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
        super().__init__(data)
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if sinkhorn_iters <= 0:
            raise ValueError("sinkhorn_iters must be a positive integer")

        self.temperature = float(temperature)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.eps = float(eps)

    def forward(self, cont_data: Batch) -> torch.Tensor:
        """Compute the mean soft-MCS distance between generated and observed graphs."""
        cont_data = convert_hard_to_soft_edges(cont_data) 
        
        if not isinstance(cont_data, Batch):
            raise TypeError("cont_data must be a torch_geometric.data.Batch instance")

        device = self._graph_device(cont_data)

        obs_data = self._sample_obs_data(
            cont_data, transform=convert_hard_to_soft_edges
        )
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
        adj1 = self._dense_adjacency_from_edge_weight(g1)
        adj2 = self._dense_adjacency_from_edge_weight(g2)

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


class spectral_dist(_BaseGraphLevelDistance):
    r"""Spectral distance between generated and observed graphs.

    This module compares (a subset of) the eigen-spectra of either the adjacency
    matrix :math:`A`, the (unnormalized) Laplacian :math:`L=D-A`, or the
    normalized Laplacian :math:`\mathcal{L}=I-D^{-1/2} A D^{-1/2}`.

    Interpretation
    --------------
    - Small values (near 0) indicate the two graphs have similar global
      structure as captured by the chosen spectrum (e.g., similar connectivity,
      cut structure, community signals).
    - Larger values indicate global structural mismatch.
    - Because we differentiate through the eigendecomposition of matrices
      constructed from ``edge_weight``, gradients flow to the edge weights,
      enabling end-to-end training.

    Args:
        data: Sequence of reference graphs. On ``forward``, a batch of the same
            size as ``cont_data`` is sampled (with replacement) from this pool.
        which: One of {"laplacian", "norm_laplacian", "adjacency"} selecting the
            matrix whose spectrum to compare. Default: "laplacian".
        k: Number of eigenvalues to compare. If ``None`` (default), uses
            ``min(n1, n2)`` per pair. If set and larger than either graph size,
            spectra are zero-padded.
        p: The :math:`\ell_p` norm used to compare spectra. Default: 2.0.
        symmetrize: If ``True``, symmetrize the dense adjacency as
            ``0.5 * (A + A.T)`` before constructing (normalized) Laplacians.
        eps: Numerical stability constant for degree inverses.

    Returns:
        A scalar tensor containing the mean distance across the batch.
    """

    def __init__(
        self,
        data: Sequence[Data],
        *,
        which: str = "laplacian",
        k: Optional[int] = None,
        p: float = 2.0,
        symmetrize: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(data)

        valid = {"laplacian", "norm_laplacian", "adjacency"}
        if which not in valid:
            raise ValueError(f"`which` must be one of {valid}, got {which!r}")
        if k is not None and k <= 0:
            raise ValueError("k must be a positive integer or None")
        if p <= 0:
            raise ValueError("p must be positive")

        self.which = which
        self.k = k
        self.p = float(p)
        self.symmetrize = bool(symmetrize)
        self.eps = float(eps)

    def forward(self, cont_data: Batch) -> torch.Tensor:
        cont_data = convert_hard_to_soft_edges(cont_data) 

        if not isinstance(cont_data, Batch):
            raise TypeError("cont_data must be a torch_geometric.data.Batch instance")

        device = self._graph_device(cont_data)

        obs_data = self._sample_obs_data(
            cont_data, transform=convert_hard_to_soft_edges
        )
        if device is not None:
            obs_data = obs_data.to(device)

        cont_list = cont_data.to_data_list()
        obs_list = obs_data.to_data_list()

        distances: List[torch.Tensor] = []
        for g_gen, g_obs in zip(cont_list, obs_list):
            distances.append(self._pair_distance(g_gen, g_obs))

        return torch.stack(distances).mean()

    def _pair_distance(self, g1: Data, g2: Data) -> torch.Tensor:
        # Build dense adjacencies from possibly loop-less edge_weight
        A1 = self._dense_adjacency_from_edge_weight(g1).clamp_min(0.0)
        A2 = self._dense_adjacency_from_edge_weight(g2).clamp_min(0.0)

        if A1.numel() == 0 or A2.numel() == 0:
            # If either graph has no nodes, treat distance as maximal unit cost.
            return torch.tensor(1.0, device=A1.device if A1.numel() else A2.device)

        if self.symmetrize:
            A1 = 0.5 * (A1 + A1.transpose(-1, -2))
            A2 = 0.5 * (A2 + A2.transpose(-1, -2))

        # Select matrix to spectrally compare
        if self.which == "adjacency":
            M1, M2 = A1, A2
        elif self.which == "laplacian":
            M1, M2 = self._laplacian(A1), self._laplacian(A2)
        else:  # "norm_laplacian"
            M1, M2 = self._normalized_laplacian(A1), self._normalized_laplacian(A2)

        # Compute eigenvalues (symmetric -> eigh), ascending order
        # Note: torch.linalg.eigh is differentiable for symmetric inputs.
        e1 = torch.linalg.eigh(M1).eigenvalues
        e2 = torch.linalg.eigh(M2).eigenvalues

        # Align spectra (truncate or pad with zeros) to a common length
        k = self.k
        if k is None:
            k = min(e1.numel(), e2.numel())
            e1_k = e1.narrow(0, 0, k)
            e2_k = e2.narrow(0, 0, k)
        else:
            e1_k = self._pad_or_truncate(e1, k)
            e2_k = self._pad_or_truncate(e2, k)

        # Compare with L_p norm, normalized by k to keep scale stable across sizes
        diff = e1_k - e2_k
        dist = diff.abs().pow(self.p).sum().pow(1.0 / self.p)
        dist = dist / (k + self.eps)

        return dist

    @staticmethod
    def _laplacian(A: torch.Tensor) -> torch.Tensor:
        deg = A.sum(dim=-1)
        L = torch.diag_embed(deg) - A
        return L

    def _normalized_laplacian(self, A: torch.Tensor) -> torch.Tensor:
        deg = A.sum(dim=-1).clamp_min(self.eps)
        d_inv_sqrt = deg.pow(-0.5)
        D_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        I = torch.eye(A.size(-1), device=A.device, dtype=A.dtype)
        # L_sym = I - D^{-1/2} A D^{-1/2}
        return I - (D_inv_sqrt @ A @ D_inv_sqrt)

    @staticmethod
    def _pad_or_truncate(evals: torch.Tensor, k: int) -> torch.Tensor:
        n = evals.numel()
        if n == k:
            return evals
        if n > k:
            return evals.narrow(0, 0, k)
        # pad with zeros at the end (smallest eigenvalues first for L/Adj)
        pad = evals.new_zeros(k - n)
        return torch.cat([evals, pad], dim=0)


class wl_graph_kernel_dist(_BaseGraphLevelDistance):
    r"""Differentiable Weisfeiler–Lehman (WL) graph *kernel* distance.

    This implements a *soft* WL-type kernel by performing T rounds of
    differentiable message passing (a WL-style update) over dense
    (soft-weighted) adjacencies, then comparing graph-level embeddings
    with a normalized kernel. The distance is:

        d(G1, G2) = 1 - (1/T) * Σ_t k(g1^(t), g2^(t)),

    where k is a cosine kernel in [0, 1] and g^(t) is the pooled embedding at
    iteration t. Gradients flow through the aggregations, so ``edge_weight``
    can be optimized end-to-end.

    Interpretation
    --------------
    - Small values (near 0): graphs are similar under WL-like subtree patterns.
    - Large values (near 1): graphs are dissimilar in their multi-hop structure.
    - Uses only the dense adjacency (and optionally node features via a single
      scalar mixing parameter) so that gradients flow to ``edge_weight``.

    Args:
        data: Pool of reference graphs; on ``forward`` we sample a batch the same
            size as ``cont_data`` (with replacement).
        num_iterations: WL iterations (including t=0 representation in the kernel).
        hidden_dim: Hidden dimensionality of node embeddings.
        readout: Graph readout; one of {"mean", "sum"}.
        symmetrize: If True, use 0.5 * (A + Aᵀ).
        dropout: Dropout applied after each update (0 disables).
        use_layer_norm: If True, applies LayerNorm to node embeddings each iter.
        eps: Numerical stability constant.

    Notes
    -----
    - Node features (if present) are incorporated as a single scalar shift to the
      degree signal using a learned mixing parameter ``alpha``. This keeps input
      dimensionality fixed (1) while still leveraging features in a stable way.
    - To compare across iterations, we average cosine similarities from t=0..T.
    """

    def __init__(
        self,
        data: Sequence[Data],
        *,
        num_iterations: int = 3,
        hidden_dim: int = 64,
        readout: str = "mean",
        symmetrize: bool = True,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(data)

        if num_iterations <= 0:
            raise ValueError("num_iterations must be a positive integer")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if readout not in {"mean", "sum"}:
            raise ValueError("readout must be 'mean' or 'sum'")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")

        self.num_iterations = int(num_iterations)
        self.hidden_dim = int(hidden_dim)
        self.readout = readout
        self.symmetrize = bool(symmetrize)
        self.eps = float(eps)

        # Input is 1D (degree + optional feature scalar via alpha)
        self.input_proj = nn.Linear(1, hidden_dim)
        self.updates_self = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_iterations)
        )
        self.updates_neigh = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_iterations)
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lns = (
            nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(self.num_iterations))
            if use_layer_norm
            else None
        )

        # Learned scalar mixing of node features (if present) into degree signal.
        # h0 = degree + alpha * mean(x, dim=1)
        self.alpha = nn.Parameter(torch.tensor(0.0))


    def forward(self, cont_data: Batch) -> torch.Tensor:
        cont_data = convert_hard_to_soft_edges(cont_data)

        if not isinstance(cont_data, Batch):
            raise TypeError("cont_data must be a torch_geometric.data.Batch instance")

        # --- unify module & data device ---
        # pick a reliable device from inputs
        if getattr(cont_data, "x", None) is not None:
            device = cont_data.x.device
        elif getattr(cont_data, "edge_weight", None) is not None:
            device = cont_data.edge_weight.device
        else:
            # fall back to CPU
            device = torch.device("cpu")

        # move the whole WL module (parameters/buffers) if needed
        if next(self.parameters()).device != device:
            self.to(device)

        # sample observed graphs and move them too
        obs_data = self._sample_obs_data(cont_data, transform=convert_hard_to_soft_edges)
        obs_data = obs_data.to(device)

        cont_list = cont_data.to(device).to_data_list()
        obs_list = obs_data.to_data_list()

        distances: List[torch.Tensor] = []
        for g_gen, g_obs in zip(cont_list, obs_list):
            distances.append(self._pair_distance(g_gen, g_obs))

        return torch.stack(distances).mean()


    def _pair_distance(self, g1: Data, g2: Data) -> torch.Tensor:
        # Build dense adjacencies (supports with/without self-loops)
        A1 = self._dense_adjacency_from_edge_weight(g1).clamp_min(0.0)
        A2 = self._dense_adjacency_from_edge_weight(g2).clamp_min(0.0)

        if A1.numel() == 0 or A2.numel() == 0:
            return torch.tensor(1.0, device=A1.device if A1.numel() else A2.device)

        if self.symmetrize:
            A1 = 0.5 * (A1 + A1.transpose(-1, -2))
            A2 = 0.5 * (A2 + A2.transpose(-1, -2))

        # Initial node signals (1D): degree + alpha * mean(x)
        h1_0 = self._initial_signal(g1, A1)  # [n1, 1]
        h2_0 = self._initial_signal(g2, A2)  # [n2, 1]

        # Project to hidden
        z1 = self.act(self.input_proj(h1_0))
        z2 = self.act(self.input_proj(h2_0))

        # Collect per-iteration graph embeddings (include t=0)
        g1_embeds = [self._readout(z1)]
        g2_embeds = [self._readout(z2)]

        # WL-style updates
        for t in range(self.num_iterations):
            # Neighbor aggregation using dense adjacency (differentiable w.r.t. edge_weight)
            m1 = A1 @ z1
            m2 = A2 @ z2

            # Linear transforms + nonlinearity
            z1 = self.updates_self[t](z1) + self.updates_neigh[t](m1)
            z2 = self.updates_self[t](z2) + self.updates_neigh[t](m2)

            if self.lns is not None:
                z1 = self.lns[t](z1)
                z2 = self.lns[t](z2)

            z1 = self.act(z1)
            z2 = self.act(z2)

            z1 = self.dropout(z1)
            z2 = self.dropout(z2)

            g1_embeds.append(self._readout(z1))
            g2_embeds.append(self._readout(z2))

        # Cosine kernel in [0, 1]: k = (cos + 1) / 2
        sims = []
        for e1, e2 in zip(g1_embeds, g2_embeds):
            sim = self._cosine_sim01(e1, e2)
            sims.append(sim)

        k_mean = torch.stack(sims).mean()  # in [0, 1]
        dist = 1.0 - k_mean  # map similarity to distance in [0, 1]
        return dist

    def _initial_signal(self, g: Data, A: torch.Tensor) -> torch.Tensor:
        deg = A.sum(dim=-1, keepdim=True)  # [n, 1]
        x = getattr(g, "x", None)
        if x is not None:
            x_mean = x.float().mean(dim=-1, keepdim=True)  # [n, 1]
            return deg + self.alpha * x_mean
        return deg

    def _readout(self, Z: torch.Tensor) -> torch.Tensor:
        if self.readout == "mean":
            return Z.mean(dim=0, keepdim=True)  # [1, d]
        else:  # "sum"
            # Normalize by node count to keep scale comparable across sizes
            return Z.sum(dim=0, keepdim=True) / (Z.size(0) + self.eps)

    def _cosine_sim01(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a, b: [1, d]
        a = a.view(-1)
        b = b.view(-1)
        denom = (a.norm(p=2) * b.norm(p=2)).clamp_min(self.eps)
        cos = (a @ b) / denom  # in [-1, 1]
        return (cos + 1.0) * 0.5  # map to [0, 1]
