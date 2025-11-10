# dataAdapter.py
#
# Unified dataset loader for TU datasets + BA-Shapes + BA-2Motifs.
# The returned object is the dataset itself (PyG dataset) with metadata
# directly attached as attributes.

from __future__ import annotations

import os
import string
from typing import Dict, Any

import torch
from torch_geometric.datasets import TUDataset, BAShapes
import torch_geometric.utils as pyg_utils
import torch_geometric.data as pyg_data
from torch_geometric.datasets import BA2MotifDataset

########################################
#        Supported Datasets
########################################

SUPPORTED_DATASETS: Dict[str, str] = {
    # TU datasets (graph classification)
    "MUTAG": "TU",
    "NCI1": "TU",
    "PROTEINS": "TU",
    "PTC_MR": "TU",
    "PTC_FR": "TU",
    "PTC_MM": "TU",
    "PTC_FM": "TU",
    "COX2": "TU",
    "COX2_MD": "TU",
    "BZR": "TU",
    "BZR_MD": "TU",
    "DHFR": "TU",
    "DHFR_MD": "TU",
    "ER_MD": "TU",

    # Synthetic datasets
    "BA-2MOTIFS": "BA2MOTIFS",
}


########################################
#       Dataset-Specific Metadata
########################################
# This is "best effort" based on documentation.
# Any missing or mismatched metadata will be auto-filled to match the
# dimensionality / number of classes of the underlying tensors.

DATASET_METADATA: Dict[str, Dict[str, Any]] = {
    # MUTAG: well-defined atoms & bonds.
    "MUTAG": {
        "NODE_CLS": {
            0: 'C',   # carbon
            1: 'N',   # nitrogen
            2: 'O',   # oxygen
            3: 'F',   # fluorine
            4: 'I',   # iodine
            5: 'Cl',  # chlorine
            6: 'Br',  # bromine
        },
        "NODE_COLOR": {
            0: 'orange',
            1: 'magenta',
            2: 'green',
            3: 'blue',
            4: 'cyan',
            5: 'red',
            6: 'yellowgreen',
        },
        "GRAPH_CLS": {
            0: 'nonmutagen',
            1: 'mutagen',
        },
        "EDGE_CLS": {
            0: 'aromatic',
            1: 'single',
            2: 'double',
            3: 'triple',
        },
        "EDGE_WIDTH": {
            0: 3,
            1: 2,
            2: 4,
            3: 6,
        },
    },

    # PROTEINS: graphs are proteins; label = enzyme vs non-enzyme.
    # Nodes = secondary structure elements (SSEs), roughly 3 types.
    "PROTEINS": {
        "NODE_CLS": {
            0: "helix",
            1: "sheet",
            2: "turn",
        },
        "GRAPH_CLS": {
            0: "non-enzyme",
            1: "enzyme",
        },
        # Edge types vary in literature; leave EDGE_CLS empty and
        # let auto-filler handle based on edge_attr dimensionality.
    },

    # NCI1: small molecules with binary class (active vs inactive).
    "NCI1": {
        # Atom / bond semantics are not standardized in docs,
        # so use generic placeholders; auto-filler will fix lengths.
        "NODE_CLS": {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
        },
        "GRAPH_CLS": {
            0: "class0",
            1: "class1",
        },
    },

    # PTC variants: carcinogenicity in different settings.
    "PTC_MR": {
        "GRAPH_CLS": {
            0: "non-carcinogenic",
            1: "carcinogenic",
        },
    },
    "PTC_FR": {
        "GRAPH_CLS": {
            0: "class0",
            1: "class1",
        },
    },
    "PTC_MM": {
        "GRAPH_CLS": {
            0: "class0",
            1: "class1",
        },
    },
    "PTC_FM": {
        "GRAPH_CLS": {
            0: "class0",
            1: "class1",
        },
    },

    "BA-2MOTIFS" : {
        "NODE_CLS": {0: "base"},
        "NODE_COLOR": {0: "gray"},
        "GRAPH_CLS": {
            0: "house",
            1: "cycle",
        },
    }, 

    # Catch-all defaults
    "DEFAULT": {
        "NODE_CLS": {},
        "NODE_COLOR": {},
        "GRAPH_CLS": {},
        "EDGE_CLS": {},
        "EDGE_WIDTH": {},
    },
}


########################################
#              Loaders
########################################

def _load_tu(name: str, root: str) -> TUDataset:
    return TUDataset(root=root, name=name)

def _load_ba_2motifs(root: str):
    ba_root = os.path.join(root, "BA2Motif")
    return BA2MotifDataset(root=ba_root)

########################################
#          Counting Helpers
########################################

def _infer_node_feature_dim(dataset) -> int:
    """Infer the *feature dimension* of x, robust to shape [N] vs [N,1]."""
    x = getattr(dataset[0], "x", None)
    if x is None:
        return 0
    if x.dim() == 1:
        # Single scalar feature (e.g. label index or attribute).
        return 1
    return int(x.size(-1))


def _infer_edge_feature_dim(dataset) -> int:
    """Infer the *feature dimension* of edge_attr, robust to shape [E] vs [E,1]."""
    e = getattr(dataset[0], "edge_attr", None)
    if e is None:
        return 0
    if e.dim() == 1:
        return 1
    return int(e.size(-1))


def _infer_graph_classes(dataset) -> int:
    """Infer number of graph-level classes from y."""
    ys = []
    for data in dataset:
        if getattr(data, "y", None) is None:
            continue
        # y may be shape [1] or [1, num_classes] etc.
        y_flat = data.y.view(-1)[0]
        ys.append(int(y_flat.item()))
    if not ys:
        return 0
    return max(ys) + 1


########################################
#    Metadata Auto-Fill / Consistency
########################################

def _make_alpha_labels(n: int, prefix: str = "") -> Dict[int, str]:
    """Create labels: A, B, C, ... then prefix+index if > 26."""
    labels = {}
    for i in range(n):
        if i < len(string.ascii_uppercase):
            labels[i] = f"{prefix}{string.ascii_uppercase[i]}"
        else:
            labels[i] = f"{prefix}{i}"
    return labels


def _ensure_metadata_dimensionality(dataset, name: str) -> None:
    """
    Ensure NODE_CLS, EDGE_CLS, and GRAPH_CLS exist and that their
    lengths match the dimensionality / number of classes inferred
    from the actual tensors.

    - NODE_CLS length == node feature dimension
    - EDGE_CLS length == edge feature dimension
    - GRAPH_CLS length == number of graph classes
    """
    # Infer dimensions
    node_feat_dim = _infer_node_feature_dim(dataset)
    edge_feat_dim = _infer_edge_feature_dim(dataset)
    num_graph_classes = _infer_graph_classes(dataset)

    dataset.num_node_features_inferred = node_feat_dim
    dataset.num_edge_features_inferred = edge_feat_dim
    dataset.num_graph_classes = num_graph_classes

    # Ensure attributes exist
    if not hasattr(dataset, "NODE_CLS") or not isinstance(dataset.NODE_CLS, dict):
        dataset.NODE_CLS = {}
    if not hasattr(dataset, "EDGE_CLS") or not isinstance(dataset.EDGE_CLS, dict):
        dataset.EDGE_CLS = {}
    if not hasattr(dataset, "GRAPH_CLS") or not isinstance(dataset.GRAPH_CLS, dict):
        dataset.GRAPH_CLS = {}

    # --- Fix NODE_CLS ---
    if node_feat_dim > 0:
        if len(dataset.NODE_CLS) != node_feat_dim:
            # If we had explicit metadata for this dataset but the length
            # doesn't match, we overwrite with A,B,C... to avoid shape bugs.
            dataset.NODE_CLS = _make_alpha_labels(node_feat_dim, prefix="N_")
    else:
        dataset.NODE_CLS = {}

    # --- Fix EDGE_CLS ---
    if edge_feat_dim > 0:
        if len(dataset.EDGE_CLS) != edge_feat_dim:
            dataset.EDGE_CLS = _make_alpha_labels(edge_feat_dim, prefix="E_")
        # Reasonable default for EDGE_WIDTH if lengths differ:
        if not hasattr(dataset, "EDGE_WIDTH") or not isinstance(dataset.EDGE_WIDTH, dict):
            dataset.EDGE_WIDTH = {}
        if len(dataset.EDGE_WIDTH) != edge_feat_dim:
            # Just scale from 2 upwards.
            dataset.EDGE_WIDTH = {i: 2 + i for i in range(edge_feat_dim)}
    else:
        dataset.EDGE_CLS = {}
        if not hasattr(dataset, "EDGE_WIDTH"):
            dataset.EDGE_WIDTH = {}

    # --- Fix GRAPH_CLS ---
    if num_graph_classes > 0:
        if len(dataset.GRAPH_CLS) != num_graph_classes:
            dataset.GRAPH_CLS = _make_alpha_labels(num_graph_classes, prefix="Y_")
    else:
        dataset.GRAPH_CLS = {}


########################################
#          Public Loader API
########################################

def load_dataset(name: str, root: str = "data"):
    """
    Load a PyG dataset and attach metadata directly onto it.

    Returns
    -------
    dataset : torch_geometric.data.Dataset
        A dataset object with additional attributes:
          - NODE_CLS, NODE_COLOR (optional)
          - EDGE_CLS, EDGE_WIDTH (optional)
          - GRAPH_CLS
          - num_node_features_inferred
          - num_edge_features_inferred
          - num_graph_classes
          - split_by_class() method
    """
    key = name.upper()

    if key not in SUPPORTED_DATASETS:
        raise ValueError(f"Dataset '{name}' not supported. "
                         f"Supported: {list(SUPPORTED_DATASETS.keys())}")

    dtype = SUPPORTED_DATASETS[key]

    # --- Load dataset ---
    if dtype == "TU":
        dataset = _load_tu(key, root)
    elif dtype == "BA2MOTIFS":
        dataset = _load_ba_2motifs(root)
    else:
        raise ValueError(f"Unknown dataset type '{dtype}' for '{key}'.")

    # --- Attach initial metadata (best-effort semantics) ---
    md = DATASET_METADATA.get(key, DATASET_METADATA["DEFAULT"])
    dataset.NODE_CLS = dict(md.get("NODE_CLS", {}))
    dataset.NODE_COLOR = dict(md.get("NODE_COLOR", {}))
    dataset.GRAPH_CLS = dict(md.get("GRAPH_CLS", {}))
    dataset.EDGE_CLS = dict(md.get("EDGE_CLS", {}))
    dataset.EDGE_WIDTH = dict(md.get("EDGE_WIDTH", {}))

    # --- Make metadata consistent with actual tensor dims ---
    _ensure_metadata_dimensionality(dataset, key)

    # --- Utility method: split by class ---
    def split_by_class(self):
        """
        Return a list of sub-datasets, one per graph-level class.
        Uses GRAPH_CLS length (kept consistent with y) to split.
        """
        if self.num_graph_classes == 0:
            return [self]

        # Build a tensor of graph labels
        ys = torch.tensor(
            [int(getattr(data, "y", torch.tensor([0])).view(-1)[0].item())
             for data in self],
            dtype=torch.long,
        )
        subsets = []
        for c in range(self.num_graph_classes):
            idx = (ys == c).nonzero(as_tuple=True)[0]
            subsets.append(self[idx])
        return subsets

    dataset.split_by_class = split_by_class.__get__(dataset)

    return dataset
