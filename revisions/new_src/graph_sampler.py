"""Disclaimer: Adapted from GNNInterpreter's code base."""

from __future__ import annotations

import random
from typing import Literal, Optional

import networkx as nx
import torch
import torch.distributions as dist
import torch.nn as nn
import torch_geometric as pyg


class GraphSampler(nn.Module):
    """An i.i.d. Binomial graph sampler supporting GPU execution."""

    def __init__(
        self,
        max_nodes: Optional[int] = None,
        num_node_cls: Optional[int] = None,
        num_edge_cls: Optional[int] = None,
        nodes: Optional[list[int]] = None,
        edges: Optional[list[tuple[int, int]]] = None,
        G: nx.Graph | None = None,
        learn_node_feat: bool = False,
        learn_edge_feat: bool = False,
        temperature: float = 1,
    ) -> None:
        super().__init__()

        if G:
            G = nx.convert_node_labels_to_integers(G)
            nodes = [G.nodes[i]["label"] for i in range(G.number_of_nodes())]
            edges = list(G.edges)

        self.n = max_nodes or (len(nodes) if nodes is not None else 0)
        self.k = num_node_cls or (max(nodes) + 1 if nodes is not None else 1)
        self.l = num_edge_cls
        self.nodes = nodes or self._gen_random_cls(self.n, self.k)
        self.edges = edges or self._gen_complete_edges(self.n)
        self.edge_cls = (
            self._gen_random_cls(len(self.edges), self.l) if (self.l and nodes is None) else None
        )
        self.tau = float(temperature)

        # Anchor parameter to keep track of the active device.
        self._dev_param = nn.Parameter(torch.empty(0), requires_grad=False)

        # Main parameters; they'll migrate with ``module.to(device)``.
        self.omega = nn.Parameter(torch.empty(self.m))
        if learn_node_feat:
            self.xi = nn.Parameter(torch.empty(self.n, self.k))
        else:
            self.xi = None
        if learn_edge_feat and self.l:
            self.eta = nn.Parameter(torch.empty(self.m, self.l))
        else:
            self.eta = None

        # Static indices registered as buffers so they track device transfers.
        edge_index = self._build_edge_index(self.edges)
        pair_index = self._build_pair_index(edge_index, self.m)
        self.register_buffer("edge_index_buf", edge_index)
        self.register_buffer("pair_index_buf", pair_index)

        self.param_list = ["omega", "theta", "theta_pairs"]
        if self.xi is not None:
            self.param_list.extend(["xi", "p"])
        if self.eta is not None:
            self.param_list.extend(["eta", "q"])

        self.init()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self._dev_param.device

    @property
    def m(self) -> int:
        return len(self.edges)

    @property
    def edge_index(self) -> torch.Tensor:
        return self.edge_index_buf

    @property
    def pair_index(self) -> torch.Tensor:
        return self.pair_index_buf

    @property
    def theta(self) -> torch.Tensor:
        return torch.sigmoid(self.omega)

    @property
    def p(self) -> torch.Tensor:
        return torch.softmax(self.xi, dim=1)

    @property
    def q(self) -> torch.Tensor:
        return torch.softmax(self.eta, dim=1)

    @property
    def expected_m(self) -> float:
        return self.theta.sum().item()

    @property
    def theta_pairs(self) -> torch.Tensor:
        return self.theta[self.pair_index]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
        undirected = list(edges)
        directed = undirected + [(j, i) for (i, j) in undirected]
        return torch.tensor(directed, dtype=torch.long).t().contiguous()

    @staticmethod
    def _build_pair_index(edge_index_directed: torch.Tensor, m: int) -> torch.Tensor:
        undirected = edge_index_directed.t()[:m]
        pairs: list[tuple[int, int]] = []
        for i in range(m - 1):
            for j in range(i + 1, m):
                if undirected[i, 0].item() == undirected[j, 0].item():
                    pairs.append((i, j))
        if not pairs:
            return torch.empty(2, 0, dtype=torch.long)
        return torch.tensor(pairs, dtype=torch.long).t().contiguous()

    @staticmethod
    def _gen_random_cls(n: int, k: int) -> list[int]:
        return random.choices(range(k), k=n)

    @staticmethod
    def _gen_complete_edges(n: int) -> list[tuple[int, int]]:
        return [(i, j) for i in range(n) for j in range(n) if i < j]

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def init(self, G: nx.Graph | None = None, eps: float = 1e-4) -> None:
        dev = self.device

        if G is None:
            theta = torch.rand(self.m, device=dev)
        else:
            undirected_edges = self.edge_index.t()[: self.m].tolist()
            vals = []
            for (u, v) in undirected_edges:
                present = ((u, v) in G.edges) or ((v, u) in G.edges)
                vals.append(1 - eps if present else eps)
            theta = torch.tensor(vals, device=dev, dtype=torch.float32)
        self.omega.data = torch.logit(theta)

        if self.xi is not None:
            if G is None:
                p = dist.Dirichlet(torch.ones(self.k, device=dev)).sample([self.n])
            else:
                eye = torch.eye(self.k, device=dev)
                undiff = eye * (1 - 2 * eps) + eps
                p = torch.stack([undiff[G.nodes[i]["label"]] for i in range(self.n)], dim=0)
            self.xi.data = torch.log(p)

        if self.eta is not None:
            if G is None:
                q = dist.Dirichlet(torch.ones(self.l, device=dev)).sample([self.m])
            else:
                eye = torch.eye(self.l, device=dev)
                undiff = eye * (1 - 2 * eps) + eps
                undirected_edges = self.edge_index.t()[: self.m].tolist()
                rows = []
                for (u, v) in undirected_edges:
                    if (u, v) in G.edges:
                        rows.append(undiff[G.edges[(u, v)]["label"]])
                    elif (v, u) in G.edges:
                        rows.append(undiff[G.edges[(v, u)]["label"]])
                    else:
                        rows.append(torch.zeros(self.l, device=dev) + eps)
                q = torch.stack(rows, dim=0)
            self.eta.data = torch.log(q)

    # ------------------------------------------------------------------
    # Sampling utilities
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, torch.Tensor]:
        return {item: getattr(self, item) for item in self.param_list}

    def sample_eps(self, target: torch.Tensor, seed: int | None = None, expected: bool = False) -> torch.Tensor:
        if expected:
            return torch.ones_like(target, device=target.device) / 2
        if seed is not None:
            torch.manual_seed(seed)
        else:
            torch.seed()
        return torch.rand_like(target, device=target.device)

    def sample_A(self, seed: int | None = None, expected: bool = False) -> torch.Tensor:
        eps = self.sample_eps(self.omega, seed=seed, expected=expected)
        logistic = torch.logit(eps)
        return torch.sigmoid((self.omega + logistic) / self.tau)

    def sample_X(self, seed: int | None = None, expected: bool = False) -> torch.Tensor:
        dev = self.device
        if self.xi is not None:
            eps = self.sample_eps(self.xi, seed=seed, expected=expected)
            gumbel = -torch.log(-torch.log(eps))
            return torch.softmax((self.xi + gumbel) / self.tau, dim=1)
        return torch.eye(self.k, device=dev)[self.nodes]

    def sample_E(self, seed: int | None = None, expected: bool = False) -> Optional[torch.Tensor]:
        dev = self.device
        if self.eta is not None:
            eps = self.sample_eps(self.eta, seed=seed, expected=expected)
            gumbel = -torch.log(-torch.log(eps))
            return torch.softmax((self.eta + gumbel) / self.tau, dim=1)
        if self.l:
            if self.edge_cls is None:
                raise RuntimeError("Edge classes are not initialised.")
            return torch.eye(self.l, device=dev)[self.edge_cls]
        return None

    # ------------------------------------------------------------------
    # Forward sampling
    # ------------------------------------------------------------------
    def forward(
        self,
        k: int = 1,
        mode: Literal["continuous", "discrete", "both"] = "continuous",
        seed: int | None = None,
        expected: bool = False,
    ) -> pyg.data.Batch | tuple[pyg.data.Batch, pyg.data.Batch]:
        dev = self.device
        X = self.sample_X(seed=seed, expected=expected)
        A_undirected = self.sample_A(seed=seed, expected=expected)
        E_undirected = self.sample_E(seed=seed, expected=expected)

        A_dir = torch.cat([A_undirected, A_undirected], dim=0)
        E_dir = None if E_undirected is None else torch.cat([E_undirected, E_undirected], dim=0)

        def make_data() -> pyg.data.Data:
            return pyg.data.Data(
                x=X.clone(),
                edge_index=self.edge_index.clone(),
                edge_weight=A_dir.clone(),
                edge_attr=None if E_dir is None else E_dir.clone(),
            )

        def make_disc_data() -> pyg.data.Data:
            x_disc = torch.eye(self.k, device=dev)[X.argmax(dim=-1)].float()
            ew_disc = (A_dir > 0.5).float()
            if E_dir is not None:
                e_disc = torch.eye(self.l, device=dev)[E_dir.argmax(dim=-1)].float()
            else:
                e_disc = None
            return pyg.data.Data(
                x=x_disc,
                edge_index=self.edge_index.clone(),
                edge_weight=ew_disc,
                edge_attr=e_disc,
            )

        cont_data = None
        disc_data = None
        if mode in ["continuous", "both"]:
            cont_data = pyg.data.Batch.from_data_list([make_data() for _ in range(k)])
        if mode in ["discrete", "both"]:
            disc_data = pyg.data.Batch.from_data_list([make_disc_data() for _ in range(k)])

        if mode == "both":
            assert cont_data is not None and disc_data is not None
            return cont_data, disc_data
        return cont_data if mode == "continuous" else disc_data

    # ------------------------------------------------------------------
    # Thresholded sampling
    # ------------------------------------------------------------------
    def sample_by_threshold(self, threshold: float) -> nx.Graph:
        mask = self.theta >= threshold
        undirected_ei = self.edge_index.t()[: self.m]
        chosen = undirected_ei[mask].detach().cpu().tolist()

        G = nx.Graph(chosen)
        if G.number_of_nodes() == 0:
            raise Exception("Empty graph!")
        nx.set_node_attributes(G, 0, name="label")

        if self.xi is not None:
            node_cls = self.xi.argmax(dim=1).detach().cpu().tolist()
            nx.set_node_attributes(G, {v: {"label": node_cls[v]} for v in range(self.n)})

        if self.eta is not None:
            edge_cls = self.eta.argmax(dim=1).detach().cpu().tolist()
            nx.set_edge_attributes(
                G,
                {(u, v): {"label": c} for (u, v), c in zip(self.edges, edge_cls)},
            )

        return G

    # ------------------------------------------------------------------
    # Misc utilities
    # ------------------------------------------------------------------
    def _bernoulli_threshold(self) -> torch.Tensor:
        return torch.rand_like(self.theta, device=self.device)

    def _top_k_threshold(self, k: int) -> torch.Tensor:
        return self.theta.sort()[0][-k]

    def sample(self, threshold: float = 0.5, k: int | None = None, bernoulli: bool = False) -> nx.Graph:
        if k is not None:
            threshold = self._top_k_threshold(k=k)
        if bernoulli:
            threshold = self._bernoulli_threshold()
        return self.sample_by_threshold(threshold=threshold)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location=self.device))