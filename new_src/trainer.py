"""Trainer extensions for running GNN-EGG on generic datasets."""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from egg_models.egg_generic import EggGeneric, EggGenericTrainer
from utils import misc

from .losses import LossTerm


class GenericGNNEggTrainer(EggGenericTrainer):
    """Specialised trainer that supports custom loss terms and formatting."""

    def __init__(
        self,
        model: EggGeneric,
        explainee,
        target: torch.Tensor,
        uninfo_target: torch.Tensor,
        obs_data_list: List[Data],
        optimizer: torch.optim.Optimizer,
        loss_term_weights: torch.Tensor,
        node_feature_dim: int,
        edge_feature_dim: int,
        tensorboard_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        avg_embed_targets: Optional[dict] = None,
        avg_embed_other_class: Optional[dict] = None,
        cont_node_indices: Optional[Tuple] = None,
        dis_node_indices: Optional[Tuple] = None,
        cont_edge_indices: Optional[Tuple] = None,
        dis_edge_indices: Optional[Tuple] = None,
        extra_losses: Optional[Sequence[LossTerm]] = None,
        **kwargs,
    ) -> None:
        extra_losses = list(extra_losses or [])
        extra_weights = torch.tensor([term.weight for term in extra_losses])
        if extra_weights.numel() == 0:
            full_weights = loss_term_weights
        else:
            full_weights = torch.cat([loss_term_weights, extra_weights])
        super().__init__(
            model=model,
            explainee=explainee,
            target=target,
            uninfo_target=uninfo_target,
            obs_data_list=obs_data_list,
            optimizer=optimizer,
            tensorboard_path=tensorboard_path,
            checkpoint_path=checkpoint_path,
            loss_term_weights=full_weights,
            avg_embed_targets=avg_embed_targets,
            avg_embed_other_class=avg_embed_other_class,
            cont_node_indices=cont_node_indices,
            dis_node_indices=dis_node_indices,
            cont_edge_indices=cont_edge_indices,
            dis_edge_indices=dis_edge_indices,
            **kwargs,
        )
        self.base_loss_terms = len(loss_term_weights)
        self.loss_term_names: List[str] = [
            "Prediction",
            "Other Embeddings",
            "Sparsity",
            "Structural",
        ] + [term.name for term in extra_losses]
        self.extra_losses = extra_losses
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self._full_edge_template = self._build_full_edge_template(
            self.model.max_node_size
        ).to(self.model.device_param.device)

    @staticmethod
    def _build_full_edge_template(max_nodes: int) -> torch.Tensor:
        indices = torch.cartesian_prod(
            torch.arange(max_nodes), torch.arange(max_nodes)
        ).t()
        return indices.unsqueeze(0)  # [1, 2, max_nodes**2]

    def ex_to_egg(self, obs_batch: Batch) -> List[torch.Tensor]:
        data_list = obs_batch.to_data_list()
        device = self.model.device_param.device
        max_nodes = self.model.max_node_size
        edge_template = self._full_edge_template
        obs_X = []
        obs_E = []
        for data in data_list:
            x = data.x if data.x is not None else torch.zeros(
                data.num_nodes, self.node_feature_dim
            )
            if x.size(-1) < self.node_feature_dim:
                x = F.pad(x, (0, self.node_feature_dim - x.size(-1)))
            x = x.to(device)
            pad_nodes = max_nodes - x.size(0)
            if pad_nodes > 0:
                x = F.pad(x, (0, 0, 0, pad_nodes))
            obs_X.append(x)

            adj = torch.zeros((max_nodes, max_nodes), device=device)
            adj_indices = (
                data.edge_index.to(device)
                if data.edge_index.numel() > 0
                else torch.empty((2, 0), dtype=torch.long, device=device)
            )
            if adj_indices.numel() > 0:
                adj[adj_indices[0], adj_indices[1]] = 1.0

            if self.edge_feature_dim > 0:
                dense_attr = torch.zeros(
                    (max_nodes, max_nodes, self.edge_feature_dim), device=device
                )
                if data.edge_attr is not None:
                    edge_attr = data.edge_attr.to(device)
                    if edge_attr.size(-1) < self.edge_feature_dim:
                        edge_attr = F.pad(
                            edge_attr,
                            (0, self.edge_feature_dim - edge_attr.size(-1)),
                        )
                    if adj_indices.numel() > 0:
                        dense_attr[adj_indices[0], adj_indices[1]] = edge_attr
            else:
                dense_attr = torch.zeros(
                    (max_nodes, max_nodes, 0), device=device
                )

            if self.edge_feature_dim > 0:
                flat_edge_attr = dense_attr.view(max_nodes ** 2, -1)
                adjacency_weights = adj.view(max_nodes ** 2, 1)
                obs_E.append(
                    torch.cat([flat_edge_attr, adjacency_weights], dim=-1)
                )
            else:
                obs_E.append(adj.view(max_nodes ** 2))

        obs_X_tensor = torch.stack(obs_X)
        obs_E_tensor = torch.stack(obs_E)
        full_edges = edge_template.repeat(len(data_list), 1, 1).to(device)
        return [obs_X_tensor, full_edges, obs_E_tensor]

    def egg_to_egg(self, generated: dict) -> List[torch.Tensor]:
        gen_X = misc.concat_possible_none_tensors(
            generated["cont_node_feats"], generated["dis_node_feats"], dim=-1
        )
        gen_E = misc.concat_possible_none_tensors(
            generated["cont_edge_feats"], generated["dis_edge_feats"], dim=-1
        )
        gen_E = misc.concat_possible_none_tensors(
            gen_E, generated.get("edge_weights"), dim=-1
        )
        return [gen_X, generated["full_edge_indices"], gen_E]

    def compute_loss_terms(
        self,
        generated: dict,
        obs_batch: Batch,
        gen_ex_format: Batch,
        gen_egg_format: List[torch.Tensor],
        obs_egg_format: List[torch.Tensor],
    ) -> torch.Tensor:
        base_terms = super().compute_loss_terms(
            generated,
            obs_batch,
            gen_ex_format,
            gen_egg_format,
            obs_egg_format,
        )
        if not self.extra_losses:
            return base_terms
        extra_terms = []
        for term in self.extra_losses:
            extra_terms.append(term(gen_ex_format, obs_batch))
        return torch.cat(
            [base_terms, torch.stack(extra_terms).to(base_terms.device)]
        )

    def train_one_epoch(self):
        self.optimizer.zero_grad()
        running_total_loss = 0.0
        running_total_loss_terms = torch.zeros_like(self.loss_term_weights)

        self.create_data_loader()
        for i, obs_batch in enumerate(
            self.obs_data_loader
        ):
            obs_batch = obs_batch.to(self.model.device_param.device)
            with torch.autocast(
                device_type=self.model.device_param.device.type,
                dtype=torch.float16,
                enabled=self.auto_mixed_precision,
            ):
                self.optimizer.zero_grad()
                generated = self.model()
                gen_ex_format = self.egg_to_ex(generated)
                gen_egg_format = self.egg_to_egg(generated)
                obs_egg_format = self.ex_to_egg(obs_batch)
                try:
                    loss_terms = self.compute_loss_terms(
                        generated,
                        obs_batch,
                        gen_ex_format,
                        gen_egg_format,
                        obs_egg_format,
                    )
                except Exception as exc:  # pragma: no cover - defensive logging
                    print(f"Encountered error {exc} in batch {i}; skipping batch")
                    del generated, gen_ex_format, gen_egg_format, obs_egg_format
                    torch.cuda.empty_cache()
                    continue

                if self.auto_mixed_precision:
                    loss_terms = loss_terms / self.batches_per_param

                with torch.no_grad():
                    running_total_loss_terms += loss_terms * self.loss_term_weights

                total_loss = loss_terms @ self.loss_term_weights

            self.scaler.scale(total_loss).backward(
                retain_graph=self.retain_comp_graph
            )

            if (i + 1) % self.batches_per_param == 0:
                if self.auto_mixed_precision:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                if self.grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()

            running_total_loss += total_loss.item()

        results = {
            "total_loss": running_total_loss / max(len(self.obs_data_loader), 1)
        }
        avg_loss_terms = running_total_loss_terms / max(
            len(self.obs_data_loader), 1
        )
        for i, name in enumerate(self.loss_term_names):
            results[name] = avg_loss_terms[i]
        return results
