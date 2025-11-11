"""Utilities for training graph generators during Ray Tune sweeps."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx

from .criteria import BudgetPenalty, ClassScoreCriterion, EmbeddingCriterion, WeightedCriterion
from .graph_sampler import GraphSampler
from .trainer import Trainer


def _build_optimizer(
    sampler: GraphSampler,
) -> tuple[torch.optim.Optimizer, Callable[[int], None]]:
    optimizer = torch.optim.SGD(sampler.parameters(), lr=1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
    return optimizer, scheduler


def train_generator(
    cls_idx: int,
    *,
    data,
    mean_embeds,
    explainee,
    nn_ged,
    max_common_sub,
    spectral_dist_fn,
    wl_graph_kernel_dist_fn,
    max_nodes: int,
    device,
    w_pred: float = 1.0,
    w_embed: float = 1.0,
    w_ged: float = 1.0,
    w_msc: float = 1.0,
    w_spec: float = 1.0,
    w_wl: float = 1.0,
    iterations: int = 20,
    target_prob_min: float = 0.9,
    target_prob_max: float = 1.0,
    target_size: int = 30,
    w_budget_init: float = 0.5,
    w_budget_inc: float = 1.1,
    w_budget_dec: float = 0.95,
    k_samples: int = 32,
    threshold: float = 0.5,
):
    """Train the generator for a single class and return a sampled explanation graph."""

    sampler = GraphSampler(
        max_nodes=max_nodes,
        num_node_cls=len(data.NODE_CLS),
        num_edge_cls=len(data.EDGE_CLS),
        temperature=0.15,
        learn_node_feat=len(data.NODE_CLS) > 0,
        learn_edge_feat=len(data.EDGE_CLS) > 0,
    )

    optimizer, scheduler = _build_optimizer(sampler)

    trainer = Trainer(
        sampler=sampler,
        discriminator=explainee,
        criterion=WeightedCriterion(
            [
                dict(
                    key="logits",
                    criterion=ClassScoreCriterion(class_idx=cls_idx, mode="maximize"),
                    weight=w_pred,
                ),
                dict(
                    key="embeds",
                    criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_idx]),
                    weight=w_embed,
                ),
                dict(key="cont_data", criterion=nn_ged, weight=w_ged),
                dict(key="cont_data", criterion=max_common_sub, weight=w_msc),
                dict(key="cont_data", criterion=spectral_dist_fn, weight=w_spec),
                dict(key="cont_data", criterion=wl_graph_kernel_dist_fn, weight=w_wl),
            ]
        ),
        optimizer=optimizer,
        scheduler=scheduler,
        dataset=data,
        budget_penalty=BudgetPenalty(budget=20, order=2, beta=1),
        device=device,
    )

    trainer.train(
        iterations=iterations,
        target_probs={cls_idx: (target_prob_min, target_prob_max)},
        target_size=target_size,
        w_budget_init=w_budget_init,
        w_budget_inc=w_budget_inc,
        w_budget_dec=w_budget_dec,
        k_samples=k_samples,
    )

    example = trainer.evaluate(threshold=threshold)
    example = from_networkx(example)

    if "label" in example:
        example.x = F.one_hot(example.label, num_classes=len(data.NODE_CLS)).float()

    if "edge_label" in example:
        example.edge_attr = F.one_hot(
            example.edge_label, num_classes=len(data.EDGE_CLS)
        ).float()

    example.y = torch.tensor(cls_idx).float()
    return example


__all__ = ["train_generator"]
