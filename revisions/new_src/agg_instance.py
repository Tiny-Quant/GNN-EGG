import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
from collections import defaultdict
import numpy as np

from torch_geometric.explain import Explainer, CaptumExplainer
from torch_geometric.explain.algorithm import GNNExplainer, PGExplainer

# --------------------------------------------------------
# 1. Simple WL hash (works for motifs up to ~10 nodes)
# --------------------------------------------------------

def wl_hash(data: Data, hops: int = 2):
    edge_index = data.edge_index
    N = data.num_nodes

    # If node features exist, use them; otherwise use degree
    if hasattr(data, "x") and data.x is not None:
        labels = [tuple(data.x[i].tolist()) for i in range(N)]
    else:
        deg = torch.bincount(edge_index[0], minlength=N)
        labels = [int(d.item()) for d in deg]

    for _ in range(hops):
        new_labels = []
        for v in range(N):
            neigh = edge_index[1][edge_index[0] == v]
            neigh_labels = sorted(labels[u] for u in neigh)
            combined = (labels[v], tuple(neigh_labels))
            new_labels.append(hash(combined))
        labels = new_labels

    # Graph-level hash
    return hash(tuple(sorted(labels)))


# --------------------------------------------------------
# 2. Extract top-p% edges → return connected components
# --------------------------------------------------------

def extract_motif_components(data: Data, edge_mask, p=0.1):
    E = edge_mask.size(0)
    k = max(1, int(E * p))

    # top edges by mask score
    idx = torch.topk(edge_mask, k).indices

    edge_index = data.edge_index[:, idx]

    # find connected components
    # node set involved in the selected edges
    nodes = edge_index.unique().tolist()

    # adjacency map for DFS
    adj = defaultdict(list)
    for u, v in zip(edge_index[0], edge_index[1]):
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))

    visited = set()
    comps = []

    for node in nodes:
        if node in visited:
            continue
        stack = [node]
        comp = []
        visited.add(node)

        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adj[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)

        comps.append(comp)

    # Build motif subgraphs
    motifs = []
    for comp_nodes in comps:
        comp_nodes_tensor = torch.tensor(comp_nodes, dtype=torch.long)
        ei, ea = subgraph(comp_nodes_tensor, data.edge_index, edge_attr=None)
        motifs.append(Data(x=None if not hasattr(data, "x") else data.x[comp_nodes_tensor],
                           edge_index=ei,
                           num_nodes=len(comp_nodes)))
    return motifs


# --------------------------------------------------------
# 3. Instance-level explainer hook
# --------------------------------------------------------

class ExtractProbs(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        out = self.model(
            batch=Batch(x=x, edge_index=edge_index, batch=batch),
            embeds=None,
            embeds_last=None,
            edge_weight=edge_attr, 
        )
        return out["probs"]        

def run_instance_explainer(
    model,
    data,
    method="gnnexplainer",
    device=None,
    epochs=200,
):
    """
    Returns:
       edge_mask: [E]
       node_mask: [N]
    """

    model.eval()
    device = device or next(model.parameters()).device
    data = data.to(device)

    model_forward = ExtractProbs(model) 

    # Prepare fields
    x = data.x
    edge_index = data.edge_index
    edge_weight = getattr(data, "edge_weight", None)
    batch_vec = getattr(
        data, "batch",
        torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    )


    # =============================================================
    # Construct Explainer with correct model_config
    # =============================================================
    method = method.lower()

    # ----------------------------
    # 1) GNNExplainer
    # ----------------------------
    if method in ["gnnexplainer", "gnn"]:
        algorithm = GNNExplainer(epochs=epochs)

        explainer = Explainer(
            model=model_forward,
            algorithm=algorithm,
            explanation_type="model",
            node_mask_type="object",
            edge_mask_type="object",
            model_config=dict(
                mode="binary_classification",
                task_level="graph",
                return_type="probs",        
            )
        )

    # ----------------------------
    # 2) PGExplainer
    # ----------------------------
    elif method in ["pgexplainer", "pg"]:
        algorithm = PGExplainer(epochs=epochs, lr=0.003)

        explainer = Explainer(
            model=model_forward,
            algorithm=algorithm,
            explanation_type="model",
            edge_mask_type="object",
            node_mask_type=None,
            model_config=dict(
                mode="binary_classification",
                task_level="graph",
                return_type="probs",        
            )
        )

    # ----------------------------
    # 3) Captum-based explainers
    # ----------------------------
    elif method in ["saliency", "grad", "integrated", "ig", "integrated_gradients"]:

        algorithm = "Saliency" if method in ["saliency", "grad"] else "IntegratedGradients"

        algorithm = CaptumExplainer(algorithm=algorithm)

        explainer = Explainer(
            model=model_forward,
            algorithm=algorithm,
            explanation_type="model",
            node_mask_type="object",
            edge_mask_type=None,           # we'll compute edge mask manually
            model_config=dict(
                mode="binary_classification",
                task_level="graph",
                return_type="probs",      # <--- FIX
            )
        )

    else:
        raise ValueError(f"Unknown explainer '{method}'.")


    # =============================================================
    # Apply explainer (modern API)
    # =============================================================
    explanation = explainer(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        batch=batch_vec,
    )

    # =============================================================
    # Extract masks
    # =============================================================
    node_mask = explanation.node_mask
    edge_mask = explanation.edge_mask

    # If no edge mask (e.g. Captum), derive from node mask
    if edge_mask is None:
        src, dst = edge_index
        edge_mask = 0.5 * (node_mask[src] + node_mask[dst])

    # Normalize
    edge_mask = (edge_mask - edge_mask.min()) / (edge_mask.max() - edge_mask.min() + 1e-8)
    node_mask = (node_mask - node_mask.min()) / (node_mask.max() - node_mask.min() + 1e-8)

    return edge_mask.cpu(), node_mask.cpu()


# --------------------------------------------------------
# 4. Class prediction function
# --------------------------------------------------------

def predict_class(model, data):
    out = model(data.x, data.edge_index, data.batch)
    return int(out.argmax(dim=-1).item())


# --------------------------------------------------------
# 5. Aggregation Pipeline
# --------------------------------------------------------

def aggregate_motifs_by_class(
    dataset,
    model,
    explainer_name="gnnexplainer",
    p=0.10,          # top p% edges used to extract motifs
    wl_hops=2,       # WL hashing radius
    device=None,
):
    """
    Returns:
        dict: class_id -> list of PyG Data subgraphs (motifs)
    """

    model.eval()
    device = device or next(model.parameters()).device

    motifs_by_class = defaultdict(list)

    for data in dataset:
        data = data.to(device)

        # --------------------------------------------------
        # 1. Run instance-level explainer → edge_mask
        # --------------------------------------------------
        edge_mask, node_mask = run_instance_explainer(
            model=model,
            data=data,
            method=explainer_name,
            device=device,
        )

        # normalize mask
        edge_mask = (edge_mask - edge_mask.min()) / (edge_mask.max() - edge_mask.min() + 1e-8)

        # --------------------------------------------------
        # 2. Extract motif components from this graph
        # --------------------------------------------------
        comps = extract_motif_components(data.cpu(), edge_mask.cpu(), p=p)

        # --------------------------------------------------
        # 3. Determine predicted class
        # --------------------------------------------------
        pred_logits = model(data)["logits"]
        c = int(pred_logits.argmax(dim=-1).item())

        # --------------------------------------------------
        # 4. Canonicalize & store motifs
        # --------------------------------------------------
        for sub in comps:
            key = wl_hash(sub, hops=wl_hops)
            motifs_by_class[c].append(sub)   # store motif directly

    return motifs_by_class
