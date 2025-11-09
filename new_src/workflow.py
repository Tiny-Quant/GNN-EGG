"""High-level workflow for running GNN-EGG on PyG benchmark datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from egg_models.egg_generic import EggGeneric

from .config import ExperimentConfig, LossTermConfig
from .data import (
    DatasetSplits,
    infer_feature_dimensions,
    load_dataset,
    make_loaders,
    max_nodes,
    stratified_split,
)
from .explainee import GeneralGCN, class_average_embeddings, train_explainee
from .losses import LossTerm
from .trainer import GenericGNNEggTrainer


@dataclass
class ExperimentArtifacts:
    """Artifacts returned after running an experiment."""

    dataset_splits: DatasetSplits
    explainee: nn.Module
    generator: EggGeneric
    trainer: GenericGNNEggTrainer
    history: Dict[str, List[float]]
    class_embeddings: Dict[int, Dict[str, torch.Tensor]]


class GNNEggExperiment:
    """Utility class orchestrating the generalized pipeline."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.device = config.device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def prepare_data(self) -> DatasetSplits:
        cfg = self.config.dataset
        dataset = load_dataset(
            cfg.name,
            root=cfg.root,
            transform=cfg.transform,
            pre_transform=cfg.pre_transform,
        )
        if cfg.stratified:
            splits = stratified_split(
                dataset,
                train_ratio=cfg.train_ratio,
                val_ratio=cfg.val_ratio,
                seed=cfg.seed,
            )
        else:
            from .data import random_dataset_split

            splits = random_dataset_split(
                dataset,
                train_ratio=cfg.train_ratio,
                val_ratio=cfg.val_ratio,
                seed=cfg.seed,
            )
        self._ensure_node_features(splits)
        return splits

    @staticmethod
    def _ensure_node_features(splits: DatasetSplits) -> None:
        """Guarantee that each graph contains node features."""

        for subset in (splits.train, splits.val, splits.test):
            for data in subset:
                if getattr(data, "x", None) is None:
                    data.x = torch.ones((data.num_nodes, 1), dtype=torch.float32)

    def build_explainee(self, splits: DatasetSplits) -> nn.Module:
        node_dim, _ = infer_feature_dimensions(splits.train)
        if node_dim == 0:
            node_dim = max(data.num_nodes for data in splits.train)
        num_classes = int(
            max(data.y.item() for data in splits.train) + 1
        )
        model = GeneralGCN(
            node_features=node_dim,
            hidden_channels=self.config.explainee.hidden_channels,
            num_classes=num_classes,
            num_layers=self.config.explainee.num_layers,
            dropout=self.config.explainee.dropout,
        )
        return model

    def train_explainee(self, model: nn.Module, splits: DatasetSplits) -> Dict[str, List[float]]:
        loaders = make_loaders(
            splits,
            batch_size=self.config.explainee.batch_size,
        )
        train_loader, val_loader, _ = loaders
        optimizer = Adam(
            model.parameters(),
            lr=self.config.explainee.lr,
            weight_decay=self.config.explainee.weight_decay,
        )
        history = train_explainee(
            model,
            train_loader,
            val_loader,
            optimizer,
            epochs=self.config.explainee.epochs,
            device=self.device,
        )
        return history

    def _class_embeddings(
        self,
        model: nn.Module,
        splits: DatasetSplits,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        train_loader = DataLoader(
            splits.train,
            batch_size=self.config.explainee.batch_size,
            shuffle=False,
        )
        return class_average_embeddings(
            model,
            train_loader,
            self.config.explainee.layer_names,
            device=self.device,
        )

    def build_generator(
        self,
        splits: DatasetSplits,
        node_feature_dim: int,
        edge_feature_dim: int,
    ) -> EggGeneric:
        gen_cfg = self.config.generator
        #max_node_size = gen_cfg.max_node_size or max_nodes(splits.train)
        largest_graph = max_nodes(splits.train)
        if gen_cfg.max_node_size is None:
            max_node_size = largest_graph
        else:
            if gen_cfg.max_node_size < largest_graph:
                raise ValueError(
                    "GeneratorConfig.max_node_size "
                    f"({gen_cfg.max_node_size}) is smaller than the largest "
                    f"training graph ({largest_graph}). Increase the value "
                    "or leave it unset to auto-detect."
                )
            max_node_size = gen_cfg.max_node_size
        node_feats = (
            gen_cfg.cont_node_feats
            if gen_cfg.cont_node_feats is not None
            else node_feature_dim
        )
        edge_feats = (
            gen_cfg.cont_edge_feats
            if gen_cfg.cont_edge_feats is not None
            else (edge_feature_dim if edge_feature_dim > 0 else None)
        )
        generator = EggGeneric(
            max_node_size=max_node_size,
            cont_node_feats=node_feats,
            dis_node_feats=gen_cfg.dis_node_feats,
            cont_edge_feats=edge_feats,
            dis_edge_feats=gen_cfg.dis_edge_feats,
            temp=gen_cfg.temp,
            batch_size=gen_cfg.batch_size,
            allow_self_loops=gen_cfg.allow_self_loops,
        )
        generator.to(self.device)
        generator.train()
        return generator

    def _build_loss_terms(self) -> Sequence[LossTerm]:
        return [LossTerm(cfg.name, cfg.fn, cfg.weight) for cfg in self.config.extra_loss_terms]

    def run(self) -> ExperimentArtifacts:
        splits = self.prepare_data()
        explainee = self.build_explainee(splits)
        history = self.train_explainee(explainee, splits)
        class_embeds = self._class_embeddings(explainee, splits)

        node_dim, edge_dim = infer_feature_dimensions(splits.train)
        if node_dim == 0:
            node_dim = max(data.num_nodes for data in splits.train)
        generator = self.build_generator(splits, node_dim, edge_dim)

        num_classes = int(
            max(data.y.item() for data in splits.train) + 1
        )
        target = torch.zeros(num_classes, device=self.device)
        target[self.config.target_class] = 1.0
        uninfo = torch.full((num_classes,), 1.0 / num_classes, device=self.device)

        target_embeds = class_embeds.get(self.config.target_class, {})
        other_embeds: Dict[str, torch.Tensor] = {}
        for name in target_embeds.keys():
            collected = [
                embeds[name]
                for label, embeds in class_embeds.items()
                if label != self.config.target_class and name in embeds
            ]
            if collected:
                other_embeds[name] = torch.stack(collected).mean(dim=0)

        optimizer = Adam(generator.parameters(), lr=1e-3)

        trainer = GenericGNNEggTrainer(
            model=generator,
            explainee=explainee.to(self.device),
            target=target,
            uninfo_target=uninfo,
            obs_data_list=splits.train,
            optimizer=optimizer,
            loss_term_weights=self.config.generator.loss_weights,
            node_feature_dim=node_dim,
            edge_feature_dim=edge_dim,
            avg_embed_targets=target_embeds,
            avg_embed_other_class=other_embeds,
            extra_losses=self._build_loss_terms(),
        )

        return ExperimentArtifacts(
            dataset_splits=splits,
            explainee=explainee,
            generator=generator,
            trainer=trainer,
            history=history,
            class_embeddings=class_embeds,
        )
