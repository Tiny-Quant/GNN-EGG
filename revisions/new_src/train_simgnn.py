"""Training script for SimGNN on GED datasets."""
from __future__ import annotations

import argparse
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from .simgnn import SimGNN
from .ged_dataset import GEDDataset, collate_pairs
from .utils import convert_hard_to_soft_edges


def _infer_input_dim(dataset: GEDDataset) -> int:
    sample = dataset[0]
    sample_g1 = sample[0]
    if sample_g1.x is None:
        raise ValueError("Dataset graphs must include node features in `x`")
    return int(sample_g1.x.size(-1))


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SimGNN on a GED dataset")
    parser.add_argument("--dataset-path", required=True, help="Path to a saved GEDDataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--tensor-channels", type=int, default=16)
    parser.add_argument(
        "--mlp-hidden-dims",
        type=int,
        nargs="*",
        default=[128, 64],
        help="Hidden layer sizes for the regression head",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="checkpoints/simgnn_ged.pt",
        help="Where to store the model checkpoint",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save a checkpoint every N epochs",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device",
    )
    return parser.parse_args(args=args)


def main(args: Sequence[str] | None = None) -> None:
    config = parse_args(args)

    dataset = GEDDataset.load(config.dataset_path)
    input_dim = _infer_input_dim(dataset)

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_pairs,
    )

    device = torch.device(config.device)
    model = SimGNN(
        in_channels=input_dim,
        hidden_channels=config.hidden_dim,
        num_layers=config.num_layers,
        tensor_channels=config.tensor_channels,
        mlp_hidden_dims=config.mlp_hidden_dims,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    train_simgnn(
        model,
        loader,
        optimizer,
        epochs=config.epochs,
        device=device,
        checkpoint_path=config.checkpoint_path,
        checkpoint_interval=config.checkpoint_interval,
    )


if __name__ == "__main__":
    main()
