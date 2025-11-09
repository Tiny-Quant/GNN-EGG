"""Utilities for running GNN-EGG experiments on generic PyG datasets."""

from importlib import import_module
from typing import Any

__all__ = ["GNNEggExperiment", "ExperimentConfig"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module(".workflow", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
