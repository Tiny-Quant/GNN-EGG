"""Extendable loss utilities for the generalized trainer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch_geometric.data import Batch


BatchLossFn = Callable[[Batch, Batch], torch.Tensor]


@dataclass
class LossTerm:
    """Description of an additional pairwise loss."""

    name: str
    fn: BatchLossFn
    weight: float = 1.0

    def __call__(self, generated: Batch, observed: Batch) -> torch.Tensor:
        value = self.fn(generated, observed)
        if not torch.is_tensor(value):
            raise TypeError("Custom loss functions must return a torch.Tensor")
        return value
