"""Configuration objects for the generalized GNN-EGG pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import torch


@dataclass
class DatasetConfig:
    """Parameters describing how to load and split a PyG dataset."""

    name: str
    root: str = "data"
    transform: Optional[object] = None
    pre_transform: Optional[object] = None
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    stratified: bool = True
    seed: int = 42


@dataclass
class ExplaineeConfig:
    """Hyper-parameters for the surrogate GNN that the generator explains."""

    hidden_channels: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 50
    batch_size: int = 32
    layer_names: Sequence[str] = field(
        default_factory=lambda: ("conv", "readout", "out")
    )


@dataclass
class GeneratorConfig:
    """Arguments used when building the :class:`EggGeneric` generator."""

    max_node_size: Optional[int] = None
    cont_node_feats: Optional[int] = None
    dis_node_feats: Optional[Tuple[int, ...]] = None
    cont_edge_feats: Optional[int] = None
    dis_edge_feats: Optional[Tuple[int, ...]] = None
    temp: float = 1.0
    batch_size: int = 1
    allow_self_loops: bool = True
    loss_weights: torch.Tensor = field(
        default_factory=lambda: torch.tensor([1.0, 1.0, 1.0, 1.0])
    )


@dataclass
class ExperimentConfig:
    """Composite configuration for end-to-end experiments."""

    dataset: DatasetConfig
    explainee: ExplaineeConfig = field(default_factory=ExplaineeConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    target_class: int = 0
    device: Optional[torch.device] = None
    extra_loss_terms: Iterable["LossTermConfig"] = field(default_factory=list)


@dataclass
class LossTermConfig:
    """Metadata describing a custom pairwise loss between batches."""

    name: str
    weight: float = 1.0
    # Callable imported lazily to avoid circular import in type checking.
    fn: Optional[object] = None

    def build(self):
        if self.fn is None:
            raise ValueError(
                "LossTermConfig.fn must be assigned a callable that accepts "
                "(generated_batch, observed_batch)."
            )
        return self
