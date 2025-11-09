"""Dataset helpers for the generalized GNN-EGG pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import random_split

from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import (
    GNNBenchmarkDataset,
    Planetoid,
    TUDataset,
)
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj


@dataclass
class DatasetSplits:
    """Light-weight container for train/validation/test splits."""

    train: Sequence[Data]
    val: Sequence[Data]
    test: Sequence[Data]


def _attempt_dataset(build_fn: Callable[[], Iterable[Data]]) -> Optional[Iterable[Data]]:
    try:
        dataset = build_fn()
    except Exception:
        return None
    if dataset is None or len(dataset) == 0:
        return None
    return dataset


def load_dataset(
    name: str,
    root: str = "data",
    transform=None,
    pre_transform=None,
    **kwargs,
) -> Iterable[Data]:
    """Load a PyG dataset by name with a few smart fall-backs.

    Parameters
    ----------
    name:
        Name of the dataset. The helper first tries to load the dataset
        through :class:`~torch_geometric.datasets.TUDataset`, then
        :class:`~torch_geometric.datasets.Planetoid`, and finally the
        :class:`~torch_geometric.datasets.GNNBenchmarkDataset` catalogue.
    root:
        Root directory passed to the dataset constructors.
    transform, pre_transform:
        Optional transforms applied during data loading.
    **kwargs:
        Additional keyword arguments forwarded to the dataset constructors.
    """

    trial_builders = [
        lambda: TUDataset(
            root=root,
            name=name,
            transform=transform,
            pre_transform=pre_transform,
            **kwargs,
        ),
        lambda: Planetoid(
            root=root,
            name=name,
            transform=transform,
            pre_transform=pre_transform,
            **kwargs,
        ),
        lambda: GNNBenchmarkDataset(
            root=root,
            name=name,
            split="train",
            transform=transform,
            pre_transform=pre_transform,
            **kwargs,
        ),
    ]

    errors = []
    for builder in trial_builders:
        dataset = _attempt_dataset(builder)
        if dataset is not None:
            return dataset
        errors.append(builder.__name__)

    raise ValueError(
        f"Unable to load dataset '{name}'. Tried constructors: {errors}."
    )


def stratified_split(
    dataset: Sequence[Data],
    train_ratio: float,
    val_ratio: float,
    seed: int = 0,
) -> DatasetSplits:
    """Create deterministic stratified splits for graph datasets."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1.")

    y = torch.stack([data.y.view(-1) for data in dataset]).view(-1)
    num_classes = int(y.max().item() + 1)

    generator = torch.Generator().manual_seed(seed)
    per_class_indices: List[List[int]] = [[] for _ in range(num_classes)]
    for idx, label in enumerate(y.tolist()):
        per_class_indices[label].append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for class_indices in per_class_indices:
        if not class_indices:
            continue
        class_tensor = torch.tensor(class_indices)
        perm = class_tensor[torch.randperm(len(class_tensor), generator=generator)]

        train_end = int(len(perm) * train_ratio)
        val_end = train_end + int(len(perm) * val_ratio)

        train_indices.extend(perm[:train_end].tolist())
        val_indices.extend(perm[train_end:val_end].tolist())
        test_indices.extend(perm[val_end:].tolist())

    train = [dataset[idx] for idx in train_indices]
    val = [dataset[idx] for idx in val_indices]
    test = [dataset[idx] for idx in test_indices]

    return DatasetSplits(train=train, val=val, test=test)


def random_dataset_split(
    dataset: Sequence[Data],
    train_ratio: float,
    val_ratio: float,
    seed: int = 0,
) -> DatasetSplits:
    """Random (non-stratified) dataset split."""

    dataset_len = len(dataset)
    train_len = int(dataset_len * train_ratio)
    val_len = int(dataset_len * val_ratio)
    test_len = dataset_len - train_len - val_len

    if test_len <= 0:
        raise ValueError("train_ratio and val_ratio leave no room for testing")

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        list(dataset),
        [train_len, val_len, test_len],
        generator=generator,
    )

    return DatasetSplits(
        train=list(train_set),
        val=list(val_set),
        test=list(test_set),
    )


def make_loaders(
    splits: DatasetSplits,
    batch_size: int,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Convenience function returning standard PyG :class:`DataLoader`s."""

    train_loader = DataLoader(
        splits.train,
        batch_size=batch_size,
        shuffle=shuffle_train,
    )
    val_loader = DataLoader(splits.val, batch_size=batch_size)
    test_loader = DataLoader(splits.test, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def max_nodes(dataset: Sequence[Data]) -> int:
    """Return the maximum number of nodes across a dataset."""

    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")
    return max(data.num_nodes for data in dataset)


def infer_feature_dimensions(dataset: Sequence[Data]) -> Tuple[int, int]:
    """Infer node and edge feature dimensionalities from the dataset."""

    adapter = GeneratorAdapter(dataset)
    spec = adapter.spec
    return spec.total_node_features, spec.total_edge_features


def dense_adjacency(data: Data, max_nodes: int, device: torch.device) -> torch.Tensor:
    """Return a dense adjacency matrix padded to ``max_nodes``."""

    adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)
    adj = adj.squeeze(0).to(device)
    pad_nodes = max_nodes - adj.size(0)
    if pad_nodes > 0:
        adj = torch.nn.functional.pad(adj, (0, pad_nodes, 0, pad_nodes))
    return adj
@dataclass(frozen=True)
class AdapterSpec:
    """Normalized description of dataset characteristics for the generator."""

    max_nodes: int
    num_cont_node_feats: Optional[int]
    dis_node_blocks: Tuple[int, ...]
    num_cont_edge_feats: Optional[int]
    dis_edge_blocks: Tuple[int, ...]

    @property
    def total_node_features(self) -> int:
        return (self.num_cont_node_feats or 0) + sum(self.dis_node_blocks)

    @property
    def total_edge_features(self) -> int:
        return (self.num_cont_edge_feats or 0) + sum(self.dis_edge_blocks)


def _is_binary_column(column: torch.Tensor, atol: float = 1e-6) -> bool:
    """Return ``True`` if a feature column represents binary/one-hot data."""

    rounded = column.round()
    if not torch.allclose(column, rounded, atol=atol):
        return False
    unique_vals = torch.unique(rounded)
    if unique_vals.numel() == 0:
        return False
    return bool(torch.all((unique_vals == 0) | (unique_vals == 1)))


def _find_blocks_and_continuous(
    features: torch.Tensor, *, block_hints: Optional[List[List[int]]] = None
) -> Tuple[int, List[int]]:
    """Split features into continuous dims and one-hot categorical blocks."""

    if features.dim() != 2:
        raise ValueError("Expected features to be a 2D tensor")

    binary_mask = [_is_binary_column(features[:, i]) for i in range(features.size(1))]
    continuous = 0
    blocks: List[int] = []
    run_index = 0
    i = 0
    while i < len(binary_mask):
        if binary_mask[i]:
            start = i
            while i < len(binary_mask) and binary_mask[i]:
                i += 1
            run_length = i - start
            if block_hints and run_index < len(block_hints):
                hint = block_hints[run_index]
                if sum(hint) != run_length:
                    raise ValueError(
                        "Block hint does not match the length of the binary run"
                    )
                blocks.extend(hint)
            else:
                blocks.append(run_length)
            run_index += 1
        else:
            continuous += 1
            i += 1
    return continuous, blocks


def _stack_feature_matrix(
    data_list: Sequence[Data], attr: str
) -> Optional[torch.Tensor]:
    tensors: List[torch.Tensor] = []
    for data in data_list:
        value = getattr(data, attr, None)
        if value is None:
            continue
        if value.dim() == 1:
            value = value.unsqueeze(-1)
        if value.dim() != 2:
            raise ValueError(f"Expected {attr} to be 1D or 2D tensor")
        tensors.append(value)
    if not tensors:
        return None
    return torch.cat(tensors, dim=0)


class GeneratorAdapter:
    """Infer generator metadata from a dataset of PyG ``Data`` objects."""

    def __init__(
        self,
        dataset: Union[Dataset, Iterable[Data]],
        node_block_hints: Optional[List[List[int]]] = None,
        edge_block_hints: Optional[List[List[int]]] = None,
    ) -> None:
        if (
            hasattr(dataset, "__len__")
            and len(dataset) > 0
            and hasattr(dataset, "__getitem__")
        ):
            data_list = [dataset[i] for i in range(len(dataset))]
        else:
            data_list = list(dataset)
        if not data_list:
            raise ValueError("Dataset must not be empty.")

        max_nodes = 0
        for data in data_list:
            if hasattr(data, "x") and data.x is not None:
                max_nodes = max(max_nodes, int(data.x.size(0)))
            elif hasattr(data, "num_nodes") and data.num_nodes is not None:
                max_nodes = max(max_nodes, int(data.num_nodes))

        stacked_nodes = _stack_feature_matrix(data_list, "x")
        if stacked_nodes is not None:
            num_cont_node_feats, dis_node_blocks_list = _find_blocks_and_continuous(
                stacked_nodes, block_hints=node_block_hints
            )
            dis_node_blocks = tuple(dis_node_blocks_list)
            if num_cont_node_feats == 0:
                num_cont_node_feats = None
        else:
            num_cont_node_feats = None
            dis_node_blocks = tuple()

        stacked_edges = _stack_feature_matrix(data_list, "edge_attr")
        if stacked_edges is not None:
            num_cont_edge_feats, dis_edge_blocks_list = _find_blocks_and_continuous(
                stacked_edges, block_hints=edge_block_hints
            )
            dis_edge_blocks = tuple(dis_edge_blocks_list)
            if num_cont_edge_feats == 0:
                num_cont_edge_feats = None
        else:
            num_cont_edge_feats = None
            dis_edge_blocks = tuple()

        self._spec = AdapterSpec(
            max_nodes=max_nodes,
            num_cont_node_feats=num_cont_node_feats,
            dis_node_blocks=dis_node_blocks,
            num_cont_edge_feats=num_cont_edge_feats,
            dis_edge_blocks=dis_edge_blocks,
        )

    @property
    def spec(self) -> AdapterSpec:
        return self._spec

