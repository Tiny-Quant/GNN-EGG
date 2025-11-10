"""
ged_dataset.py

Create, load, and serve synthetic Graph Edit Distance (GED) datasets
for training neural GED approximators (e.g., SimGNN).

Depends on:
- graph_perturb.generate_perturbed_pairs
"""

from __future__ import annotations
import os
import random
import copy
from typing import List, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx


# ---------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------

def _sample_node_from_graphs(graphs) -> Dict[str, Any]:
    """Sample a random node and its attributes from a list of graphs."""
    g = random.choice(graphs)
    if g.number_of_nodes() == 0:
        return {"feat": torch.zeros(1)}
    node = random.choice(list(g.nodes()))
    return g.nodes[node]

def _sample_edge_from_graphs(graphs) -> Dict[str, Any]:
    """Sample a random edge (u, v) and its attributes from a list of graphs."""
    g = random.choice(graphs)
    if g.number_of_edges() == 0:
        return {}
    u, v = random.choice(list(g.edges()))
    return g.edges[u, v]

# ---------------------------------------------------------------------
# Core perturbation routine
# ---------------------------------------------------------------------

def _perturb_graph(
    G,
    graphs,
    node_add_prob=1.0,
    node_remove_prob=0.25,
    edge_add_prob=0.0,
    edge_remove_prob=0.1,
    max_edges_per_new_node=2,
) -> Tuple[nx.Graph, Dict[str, int]]:
    """
    Perturb a given graph G by randomly adding/removing nodes and edges.

    - New nodes/edges are sampled from other graphs in `graphs` for realism.
    - Ensures the perturbed graph remains connected (reconnects components if needed).
    - Automatically converts directed graphs to undirected for consistency.
    - Removes self-loops and isolated nodes before returning.

    Returns:
        g (nx.Graph): the perturbed graph
        edit_counts (dict): dictionary with counts for each edit type
    """
    # Work on a copy and ensure undirected representation
    g = copy.deepcopy(G)
    if g.is_directed():
        g = g.to_undirected()

    edit_counts = {
        "node_add": 0,
        "node_remove": 0,
        "edge_add": 0,
        "edge_remove": 0,
        "total": 0,
    }

    # --- Node removal ---
    removable_nodes = list(g.nodes())
    for n in removable_nodes:
        if len(g.nodes()) > 1 and random.random() < node_remove_prob:
            g.remove_node(n)
            edit_counts["node_remove"] += 1

    # --- Edge removal ---
    edges_to_remove = [e for e in g.edges() if random.random() < edge_remove_prob]
    g.remove_edges_from(edges_to_remove)
    edit_counts["edge_remove"] += len(edges_to_remove)

    # --- Node + edge additions ---
    num_new_nodes = int(len(G.nodes()) * node_add_prob)
    for _ in range(num_new_nodes):
        new_node_id = max(g.nodes()) + 1 if len(g.nodes()) > 0 else 0
        sampled_node_attr = _sample_node_from_graphs(graphs)
        g.add_node(new_node_id, **sampled_node_attr)
        edit_counts["node_add"] += 1

        # Connect the new node to existing ones
        existing_nodes = list(g.nodes())
        existing_nodes.remove(new_node_id)
        if len(existing_nodes) > 0:
            num_edges_to_add = random.randint(1, min(max_edges_per_new_node, len(existing_nodes)))
            neighbors = random.sample(existing_nodes, num_edges_to_add)
            for v in neighbors:
                sampled_edge_attr = _sample_edge_from_graphs(graphs)
                g.add_edge(new_node_id, v, **sampled_edge_attr)
                edit_counts["edge_add"] += 1

    # --- Optional additional edge additions ---
    possible_edges = list(nx.non_edges(g))
    random.shuffle(possible_edges)
    for (u, v) in possible_edges:
        if random.random() < edge_add_prob:
            sampled_edge_attr = _sample_edge_from_graphs(graphs)
            g.add_edge(u, v, **sampled_edge_attr)
            edit_counts["edge_add"] += 1

    # --- Remove self-loops and isolates ---
    g.remove_edges_from(nx.selfloop_edges(g))
    g.remove_nodes_from(list(nx.isolates(g)))

    # If removals left us empty, force a single node to keep the graph valid
    if g.number_of_nodes() == 0:
        sampled_node_attr = _sample_node_from_graphs(graphs)
        g.add_node(0, **sampled_node_attr)

    # --- Reconnect if disconnected ---
    if g.number_of_nodes() > 1:
        g_undirected = g.to_undirected()
        if not nx.is_connected(g_undirected):
            comps = list(nx.connected_components(g_undirected))
            for i in range(len(comps) - 1):
                u = random.choice(list(comps[i]))
                v = random.choice(list(comps[i + 1]))
                sampled_edge_attr = _sample_edge_from_graphs(graphs)
                g.add_edge(u, v, **sampled_edge_attr)
                edit_counts["edge_add"] += 1

    # --- Final accounting ---
    edit_counts["total"] = (
        edit_counts["node_add"]
        + edit_counts["node_remove"]
        + edit_counts["edge_add"]
        + edit_counts["edge_remove"]
    )

    return g, edit_counts


def nx_to_pyg(G: nx.Graph) -> Data:
    """
    Convert a NetworkX graph to a PyTorch Geometric Data object.

    - Reindexes nodes to a contiguous 0..N-1 range (required by PyG)
    - Preserves per-node 'feat' as x
    - Maps edges through the new index mapping
    - Handles empty-graph and no-edge cases safely
    """
    # Ensure undirected simple graph view (we treat graphs as undirected in GED)
    if G.is_directed():
        G = G.to_undirected()

    # Make a stable ordering of nodes and build an id->idx mapping
    ordered_nodes = list(G.nodes())
    n2i = {n: i for i, n in enumerate(ordered_nodes)}

    # Build node features in that order
    x_list = []
    for n in ordered_nodes:
        feat = G.nodes[n].get("feat", torch.zeros(1))
        if not isinstance(feat, torch.Tensor):
            feat = torch.tensor(feat, dtype=torch.float32)
        x_list.append(feat.flatten().float())

    if len(x_list) > 0:
        x = torch.stack(x_list, dim=0)
    else:
        x = torch.zeros((0, 1), dtype=torch.float32)

    # Map edges to the contiguous indices
    if G.number_of_edges() > 0 and len(ordered_nodes) > 0:
        mapped_edges = [(n2i[u], n2i[v]) for (u, v) in G.edges() if u in n2i and v in n2i]
        if len(mapped_edges) > 0:
            edge_index = torch.tensor(mapped_edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


# ---------------------------------------------------------------------
# Dataset-level generation
# ---------------------------------------------------------------------

def generate_perturbed_pairs(
    dataset: List[Data],
    n_perturbations: int = 5,
    base_node_add_prob: float = 0.75,
    base_node_remove_prob: float = 0.75,
    base_edge_add_prob: float = 0.1,
    base_edge_remove_prob: float = 0.1,
    max_edges_per_new_node: int = 4,
    sample_mode: str = "beta",
) -> List[Tuple[Data, Data, float]]:
    """
    Generate (graph_i, perturbed_graph_j, ged_label) pairs.
    Guarantees both graphs are valid and returns exactly n_perturbations per base graph
    (via a small retry loop with a safe fallback).
    """
    pairs: List[Tuple[Data, Data, float]] = []

    # Convert PyG graphs -> NetworkX
    nx_graphs: List[nx.Graph] = []
    for data in dataset:
        G = to_networkx(data, node_attrs=["x"] if hasattr(data, "x") else None)
        for n in G.nodes:
            if "x" in G.nodes[n]:
                G.nodes[n]["feat"] = G.nodes[n].pop("x")
        nx_graphs.append(G)

    for G in nx_graphs:
        for _ in range(n_perturbations):
            # Randomize perturbation scales
            if sample_mode == "beta":
                scale = lambda b: float(b * np.random.beta(0.5, 2.0))
            elif sample_mode == "uniform":
                scale = lambda b: float(b * np.random.uniform(0.0, 1.0))
            else:
                raise ValueError("sample_mode must be 'beta' or 'uniform'")

            # --- Retry a few times to avoid empty/invalid perturbed graphs ---
            max_retries = 3
            for attempt in range(max_retries):
                Gp, edits = _perturb_graph(
                    G,
                    nx_graphs,
                    node_add_prob=scale(base_node_add_prob),
                    node_remove_prob=scale(base_node_remove_prob),
                    edge_add_prob=scale(base_edge_add_prob),
                    edge_remove_prob=scale(base_edge_remove_prob),
                    max_edges_per_new_node=max_edges_per_new_node,
                )
                if Gp.number_of_nodes() > 0:
                    break
            else:
                # Fallback: keep count consistent; use original and 0 edits
                Gp = copy.deepcopy(G)
                edits = {"total": 0, "node_add": 0, "node_remove": 0, "edge_add": 0, "edge_remove": 0}

            if Gp.number_of_edges() == 0:
                nodes = list(Gp.nodes())
                if len(nodes) >= 2:
                    # add a single undirected edge between first two nodes
                    u, v = nodes[0], nodes[1]
                    sampled_edge_attr = _sample_edge_from_graphs(nx_graphs)
                    Gp.add_edge(u, v, **sampled_edge_attr)
                    edits["edge_add"] = edits.get("edge_add", 0) + 1
                    edits["total"] = edits.get("total", 0) + 1
                elif len(nodes) == 1:
                    # add a self-loop so edge_index is non-empty; connectivity holds
                    (u,) = nodes
                    Gp.add_edge(u, u)
                    edits["edge_add"] = edits.get("edge_add", 0) + 1
                    edits["total"] = edits.get("total", 0) + 1
            
            pairs.append((nx_to_pyg(G), nx_to_pyg(Gp), float(edits["total"])))

    return pairs

# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

def plot_graph_pair(pyg_graph: Data, pyg_perturbed: Data, label: str) -> None:
    """Visualize an original and perturbed graph side-by-side."""
    G1, G2 = to_networkx(pyg_graph), to_networkx(pyg_perturbed)
    nodes_G1, nodes_G2 = set(G1.nodes()), set(G2.nodes())
    added_nodes, removed_nodes = nodes_G2 - nodes_G1, nodes_G1 - nodes_G2
    edges_G1, edges_G2 = {frozenset(e) for e in G1.edges()}, {frozenset(e) for e in G2.edges()}
    added_edges, removed_edges = edges_G2 - edges_G1, edges_G1 - edges_G2

    pos_common = nx.kamada_kawai_layout(G1.subgraph(nodes_G1 & nodes_G2))
    pos_G1, pos_G2 = pos_common.copy(), pos_common.copy()

    if added_nodes:
        pos_G2.update(nx.kamada_kawai_layout(G2.subgraph(added_nodes)))
    if removed_nodes:
        pos_G1.update(nx.kamada_kawai_layout(G1.subgraph(removed_nodes)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    nx.draw(
        G1, pos_G1, with_labels=True,
        node_color=["red" if n in removed_nodes else "skyblue" for n in G1.nodes()],
        edge_color=["red" if frozenset(e) in removed_edges else "gray" for e in G1.edges()],
        ax=axes[0],
    )
    axes[0].set_title("Original Graph")

    nx.draw(
        G2, pos_G2, with_labels=True,
        node_color=["green" if n in added_nodes else "lightcoral" for n in G2.nodes()],
        edge_color=["green" if frozenset(e) in added_edges else "gray" for e in G2.edges()],
        ax=axes[1],
    )
    axes[1].set_title("Perturbed Graph")

    if label is not None:
        # # keep integer-like display when you pass ints/strings
        # if isinstance(ged_label, (int, np.integer)):
        #     title = f"Approx. GED (edit count): {ged_label}"
        # else:
        #     title = f"Approx. GED (edit count): {ged_label:.3f}"
        fig.suptitle(label, fontsize=14)

    plt.tight_layout()
    plt.show()


class GEDDataset(Dataset):
    """
    Dataset wrapper around (graph_i, graph_j, {"raw": ..., "norm": ..., "factor": ...}) tuples.
    Each element includes both the raw and normalized GED values and the normalization factor.
    """

    def __init__(self, pairs: List[Tuple[Data, Data, Dict[str, float]]]) -> None:
        super().__init__()
        self.pairs: List[Tuple[Data, Data, Dict[str, float]]] = []
        for g1, g2, label_info in pairs:
            # Backward-compatible load
            if isinstance(label_info, dict):
                label_info = {
                    "raw": float(label_info.get("raw", 0.0)),
                    "norm": float(label_info.get("norm", 0.0)),
                    "factor": float(label_info.get("factor", 1.0)),
                }
            else:
                label_info = {"raw": float(label_info), "norm": float(label_info), "factor": 1.0}
            self.pairs.append((g1, g2, label_info))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[Data, Data, torch.Tensor]:
        g1, g2, label_info = self.pairs[idx]
        return g1, g2, torch.tensor(label_info["norm"], dtype=torch.float32)

    def get_raw_ged(self, idx: int) -> float:
        """Access the raw (unnormalized) GED value."""
        return self.pairs[idx][2]["raw"]

    def get_norm_factor(self, idx: int) -> float:
        """Access the normalization factor."""
        return self.pairs[idx][2]["factor"]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.pairs, path)

    @classmethod
    def load(cls, path: str) -> "GEDDataset":
        pairs: List[Tuple[Data, Data, Dict[str, float]]] = torch.load(path, weights_only=False)
        return cls(pairs)


def collate_pairs(
    batch: List[Tuple[Data, Data, torch.Tensor]]
) -> Tuple[Batch, Batch, torch.Tensor]:
    g1_list = [item[0] for item in batch]
    g2_list = [item[1] for item in batch]
    labels = torch.stack([item[2] for item in batch]).float()

    batch1 = Batch.from_data_list(g1_list)
    batch2 = Batch.from_data_list(g2_list)
    return batch1, batch2, labels


def create_and_save_ged_dataset(
    base_dataset: List[Data],
    out_path: str,
    n_perturbations: int = 5,
    node_add_prob: float = 0.1,
    node_remove_prob: float = 0.05,
    edge_add_prob: float = 0.05,
    edge_remove_prob: float = 0.05,
    sample_mode: str = "beta",
    max_edges_per_new_node: int = 4,
    alpha: float = 0.5,
) -> GEDDataset:
    """
    Generate and save a synthetic GED dataset, storing both raw and normalized values.
    """

    print(f"Generating synthetic GED dataset with {n_perturbations} perturbations per graph...")

    pairs = generate_perturbed_pairs(
        base_dataset,
        n_perturbations=n_perturbations,
        base_node_add_prob=node_add_prob,
        base_node_remove_prob=node_remove_prob,
        base_edge_add_prob=edge_add_prob,
        base_edge_remove_prob=edge_remove_prob,
        sample_mode=sample_mode,
        max_edges_per_new_node=max_edges_per_new_node,
    )

    enriched_pairs: List[Tuple[Data, Data, Dict[str, float]]] = []
    for g1, g2, ged_raw in pairs:
        node_term = g1.num_nodes + g2.num_nodes
        edge_term = g1.num_edges + g2.num_edges
        norm_factor = (alpha * node_term + (1 - alpha) * edge_term) / 2.0
        ged_norm = ged_raw / max(norm_factor, 1.0)

        label_info = {
            "raw": float(ged_raw),
            "norm": float(ged_norm),
            "factor": float(norm_factor),
        }

        g1.raw_ged = ged_raw
        g2.raw_ged = ged_raw
        g1.norm_factor = norm_factor
        g2.norm_factor = norm_factor

        enriched_pairs.append((g1, g2, label_info))

    dataset = GEDDataset(enriched_pairs)
    if out_path is not None: 
        dataset.save(out_path)
        print(f"Saved GED dataset (raw + normalized) to {out_path} ({len(dataset)} pairs)")
    return dataset

def ensure_ged_dataset(base_graphs: List[Data], out_path: str, alpha=0.5, overwrite=False, n_perturbations=3) -> str:
    """Ensure a GED dataset exists on disk; create if missing."""
    if os.path.exists(out_path) and not overwrite:
        print(f"📂 Using existing GED dataset: {out_path}")
        return out_path
    print(f"🧩 Creating new GED dataset at {out_path} ...")
    create_and_save_ged_dataset(base_graphs, out_path, alpha=alpha, n_perturbations=n_perturbations)
    return out_path

if __name__ == "__main__":
    from src.datasets.mutag import load_mutag

    data = load_mutag()
    subset = data[:5]
    out = "data/mutag_ged_smoketest.pt"
    ds = create_and_save_ged_dataset(subset, out_path=out, n_perturbations=2)
    print("First sample (raw, norm, factor):", ds.get_raw_ged(0), ds[0][2].item(), ds.get_norm_factor(0))
