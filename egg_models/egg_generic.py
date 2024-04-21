# %% Dependencies
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm

import torch
import torch.nn as nn 
import torch.nn.functional as F
#from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader, GraphSAINTRandomWalkSampler
from torch_geometric.data import Data, Batch


import pygmtools as pygm
from pygmtools.utils import dense_to_sparse
pygm.set_backend('pytorch')

from egg_models.generic_layers import (
    ContFeatMatrix, ConcreteLayer, BinaryConcrete
)
from egg_models.base_trainer import BaseTrainer
from egg_models.egg_generic_losses import(
    PredLossBatched, EdgePenalty, StructuralLoss, GEDasMatchLoss
)
from utils import misc

# %% Generator Model 
class EggGeneric(nn.Module):
    def __init__(self, max_node_size: int, 
                 cont_node_feats: Optional[int] = None, 
                 dis_node_feats: Optional[Tuple[int]] = None, 
                 cont_edge_feats: Optional[int] = None, 
                 dis_edge_feats: Optional[Tuple[int]] = None,
                 temp = 1.0, 
                 batch_size=1, 
                 allow_self_loops=True): 

        super(EggGeneric, self).__init__()

        # Parameter Checks: 
        if cont_node_feats is None and dis_node_feats is None:
            raise ValueError("Need at least one of `cont_node_feats` or " +  
                             "`dis_node_feats`")

        self.max_node_size = max_node_size
        self.cont_node_feats = cont_node_feats      
        self.dis_node_feats = dis_node_feats
        self.cont_edge_feats = cont_edge_feats
        self.dis_edge_feats = dis_edge_feats
        self.temp = temp
        self.batch_size = batch_size
        self.allow_self_loops = allow_self_loops

        # Layers 
        if self.cont_node_feats is not None:
            self.ContNodeFeats = ContFeatMatrix(
                self.batch_size, self.max_node_size, self.cont_node_feats, 
            )

        if self.dis_node_feats is not None: 
            self.DisNodeFeats = nn.ModuleList()
            for i, dist_feats in enumerate(self.dis_node_feats):
                self.DisNodeFeats.append(
                    ConcreteLayer(
                        self.batch_size, self.max_node_size, dist_feats, 
                        self.temp
                    )
                )

        if self.cont_edge_feats is not None: 
            self.ContEdgeFeats = ContFeatMatrix(
                self.batch_size, self.max_node_size**2, self.cont_edge_feats
            )

        if self.dis_edge_feats is not None: 
            self.DisEdgeFeats = nn.ModuleList()
            for i, dist_feats in enumerate(self.dis_edge_feats):
                self.DisEdgeFeats.append(
                    ConcreteLayer(
                        self.batch_size, self.max_node_size**2, dist_feats, 
                        self.temp
                    )
                )

        self.AdjacencyMatrix = BinaryConcrete(
            self.batch_size, self.max_node_size, self.max_node_size, 
            self.temp, self.allow_self_loops
        )

        self.device_param = nn.Parameter(torch.empty(0))
    
    def forward(self):
        """
        returns: dict
            b = batch_size
            n = max_node_size
            f1 = cont_node_feats
            f2 = dis_node_feats 
            f3 = cont_edge_feats
            f4 = dis_edge_feats
            - cont_node_feats: torch.Size([b, n, f1])
            - dis_node_feats: torch.Size([b, n, sum(f2)])
            - cont_edge_feats: torch.Size([b, n**2, f3])
            - dis_edge_feats: torch.Size([b, n, sum(f4)])
            - full_edge_indices: torch.Size([b, 2, n**2])
            - adjacency_matrix: torch.Size([b, n, n])
            - C_x_logLik: torch.Size([b, len(f2)])
            - C_e_logLik: torch.Size([b, len(f4)])
            - A_logLik: torch.Size([b])
        """
        if self.cont_node_feats is not None:
            X = self.ContNodeFeats()
        else: 
            X = None

        if self.dis_node_feats is not None: 
            C_x , C_x_logLik = [], []
            for layer in self.DisNodeFeats:
                out_C_x, out_C_x_logLik = layer()
                C_x.append(out_C_x)
                C_x_logLik.append(out_C_x_logLik)
            C_x = torch.cat(C_x, dim=2)
            C_x_logLik = torch.stack(C_x_logLik, dim=1)
        else: 
            C_x = None 
            C_x_logLik = torch.tensor([0.0]).repeat(self.batch_size)

        if self.cont_edge_feats is not None:
            E = self.ContEdgeFeats()
        else: 
            E = None

        if self.dis_edge_feats is not None: 
            C_e , C_e_logLik = [], []
            for layer in self.DisEdgeFeats:
                out_C_e, out_C_e_logLik = layer()
                C_e.append(out_C_e)
                C_e_logLik.append(out_C_e_logLik)
            C_e = torch.cat(C_e, dim=2)
            C_e_logLik = torch.stack(C_e_logLik, dim=1)
        else: 
            C_e = None
            C_e_logLik = torch.tensor([0.0]).repeat(self.batch_size)
    
        A, A_logLik = self.AdjacencyMatrix()
        edge_indices, edge_weights, _ = dense_to_sparse(A)
        edge_indices = edge_indices.transpose(1, 2)

        results_dict = {'cont_node_feats': X, 
                        'dis_node_feats': C_x, 
                        'cont_edge_feats': E, 
                        'dis_edge_feats': C_e, 
                        'full_edge_indices': edge_indices, 
                        'adjacency_matrix': A, 
                        'edge_weights': edge_weights, 
                        'C_x_logLik': C_x_logLik, 
                        'C_e_logLik': C_e_logLik, 
                        'A_logLik': A_logLik,
        }

        return results_dict

# %% Trainer
class EggGenericTrainer(BaseTrainer):
    """
    Trainer class for EggGeneric models that implements the 3 part loss 
    function describe in the paper (cite). 

    This class handles the loss calculation, data loading and formatting, and 
    the within epoch training logic. 
    """
    def __init__(self, 
                 model: EggGeneric, 
                 explainee: nn.Module, 
                 target: torch.Tensor, 
                 uninfo_target: torch.Tensor, 
                 obs_data_list: list, 
                 optimizer: torch.optim.Optimizer, 
                 tensorboard_path: str, 
                 checkpoint_path: str,
                 save_every=1, 
                 avg_embed_targets: Optional[Dict[str, torch.Tensor]]=None, 
                 cont_node_indices: Optional[Tuple]=None, 
                 dis_node_indices: Optional[Tuple]=None,
                 cont_edge_indices: Optional[Tuple]=None, 
                 dis_edge_indices: Optional[Tuple]=None, 
                 loss_term_weights=torch.tensor([1.0, 1.0, 1.0]), 
                 edge_budget=None, 
                 reinforce_pred=False, 
                 reinforce_struct=False,
                 sub_sampler="default", 
                 repeat_sampling=False, 
                 batches_per_param=1, 
                 grad_norm: Optional[float]=None, 
                 use_embeddings=True, 
                 auto_mixed_precision=False):

        super().__init__(model, optimizer, 
                         tensorboard_path, checkpoint_path, save_every)

        self.batch_size = self.model.batch_size
        
        # Explainee parameters: 
        self.explainee = explainee
        self.target = target
        self.gamma = F.cross_entropy(self.target, uninfo_target)
        self.obs_data_list = obs_data_list

        # Loss term parameters: 
        self.loss_term_weights = (
            loss_term_weights.to(self.model.device_param.device)
        )

        if edge_budget is None:
            self.edge_budget = self.model.max_node_size
        else: 
            self.edge_budget = edge_budget

        self.reinforce_pred = reinforce_pred
        self.reinforce_struct = reinforce_struct

        self.batches_per_param = batches_per_param
        self.sub_sampler = sub_sampler
        self.repeat_sampling = repeat_sampling
        self.grad_norm = grad_norm
        self.auto_mixed_precision = auto_mixed_precision

        # Create loss functions:  
        self.pred_loss_fn = PredLossBatched(self.target, self.explainee,
            avg_embed_targets=avg_embed_targets
        )

        self.edge_loss_fn = EdgePenalty(self.edge_budget)

        self.GED_fn = GEDasMatchLoss(self.model.max_node_size, 
                                     # Should match ex_to_egg and
                                     # egg_to_egg format indices.
                                     cont_node_indices, 
                                     dis_node_indices,  
                                     cont_edge_indices, 
                                     dis_edge_indices)

        self.struct_loss_fn = StructuralLoss(self.GED_fn, self.explainee, 
                                             self.gamma, self.target, 
                                             use_embeddings=use_embeddings)

    def egg_to_ex(self, generated: dict) -> Batch: 
        """
        Default method for formatting the generated egg output to the data 
        type taken by the explainee model. 
        """
        if generated['dis_node_feats'] is not None: 
            X = misc.concat_one_hot_to_labels(generated['dis_node_feats'], 
                                              indices=self.model.dis_node_feats) 
        
        if generated['cont_node_feats'] is not None: 
            X = misc.concat_possible_none_tensors(generated['cont_node_feats'], 
                                                  X, dim=-1)

        if generated['dis_edge_feats'] is not None: 
            E = misc.concat_one_hot_to_labels(generated['dis_edge_feats'], 
                                              indices=self.model.dis_edge_feats)
        else: 
            E = None

        if generated['cont_edge_feats'] is not None: 
            E = misc.concat_possible_none_tensors(generated['cont_edge_feats'], 
                                                  E, dim=-1) 
        
        edge_attr = misc.concat_possible_none_tensors(E, generated['edge_weights'], 
                                              dim=-1)

        A_hard = generated['adjacency_matrix'] >= 0.5
        edge_indices = dense_to_sparse(A_hard)[0].transpose(1, 2)

        data_list = [Data(X, A, E) 
                     for (X, A, E) in zip(
                         X.unbind(), edge_indices.unbind(), 
                         edge_attr.unbind()
                     )] 

        return Batch.from_data_list(data_list)

    def ex_to_egg(self, obs_batch) -> List[torch.tensor]: 
        """
        Default method to converting the explainee's training data to a format
        that can be compared with the generated output. 
        """
        # TODO: Write default function
        return None

    def egg_to_egg(self, generated: dict) -> List[torch.tensor]:
        """
        Default method for post-processing the generated output for comparison 
        with the transformed training data. 
        """

        gen_X = misc.concat_possible_none_tensors(
                generated['cont_node_feats'], 
                generated['dis_node_feats'], 
                dim=-1
        )

        gen_A = generated['full_edge_indices']

        gen_E = misc.concat_possible_none_tensors(
                generated['cont_edge_feats'], 
                generated['dis_edge_feats'], 
                dim=-1
        )

        gen_E = misc.concat_possible_none_tensors(
            gen_E, generated['edge_weights'], 
            dim=-1
        )

        return [gen_X, gen_A, gen_E]

    def create_data_loader(self):
        """
        Default method for creating the dataloader used for within epoch 
        training. Expected to load graphs for the explainee's format. 

        sub_sampler is expected to be a partial function where the only missing 
        parameter is a torch geometric data object. 
        """
        if not hasattr(self, 'obs_data_loader') or self.repeat_sampling: 

            if self.sub_sampler is None: 
                self.obs_data_loader = (
                    DataLoader(self.obs_data_list, 
                               batch_size=self.batch_size, 
                               shuffle=True, drop_last=True)
                )

            elif self.sub_sampler == "default":
                sub_samples = [] 
                for graph in self.obs_data_list:  

                    sub_sampler = (
                        GraphSAINTRandomWalkSampler(graph, 
                                                    batch_size=1,
                                                    walk_length=400, 
                                                    sample_coverage=400, 
                                                    log=False, 
                                                    num_steps=self.batch_size)
                    )


                    for batch in sub_sampler: 
                        sub_samples.append(batch)

                self.obs_data_loader = (
                    DataLoader(sub_samples, 
                               batch_size=self.batch_size, 
                               shuffle=True, drop_last=True)
                )

            elif callable(self.sub_sampler): 
                sub_samples = [] 
                for graph in self.obs_data_list:  

                    sub_sampler = self.sub_sampler(data=graph)

                    for batch in sub_sampler: 
                        sub_samples.append(batch)

                self.obs_data_loader = (
                    DataLoader(sub_samples, 
                               batch_size=self.batch_size, 
                               shuffle=True, drop_last=True)
                )

            else: 
                raise ValueError(f'{self.sub_sampler} is an invalid sub-sampler.')

    def compute_loss_terms(self, 
                           generated: dict, 
                           obs_batch: Batch, 
                           gen_ex_format: Batch, 
                           gen_egg_format: List[torch.tensor], 
                           obs_egg_format: List[torch.tensor]) -> torch.tensor:

        pred_loss, gen_act, gen_act_batch = (
            self.pred_loss_fn(gen_ex_format)
        )

        if self.reinforce_pred:
            pred_loss = pred_loss.mean() + (
                ((1 / pred_loss) @ -generated['C_x_logLik']).sum() + 
                ((1 / pred_loss) @ -generated['C_e_logLik']).sum() + 
                ((1 / pred_loss) @ -generated['A_logLik']).sum()
            )
        
        else: 
            pred_loss = pred_loss.mean()

        edge_loss = self.edge_loss_fn(self.model.AdjacencyMatrix.probs)

        struct_loss = self.struct_loss_fn(gen_egg_format, obs_egg_format, 
                                          obs_batch, gen_act, gen_act_batch)

        if self.reinforce_struct:
            struct_loss = struct_loss.mean() + (
                ((1 / struct_loss) @ -generated['C_x_logLik']).sum() + 
                ((1 / struct_loss) @ -generated['C_e_logLik']).sum() + 
                ((1 / struct_loss) @ -generated['A_logLik']).sum()
            )

        else: 
            struct_loss = struct_loss.mean()

        return torch.stack([pred_loss, edge_loss, struct_loss])

    def train_one_epoch(self): 

        self.optimizer.zero_grad() 
        running_total_loss = 0
        running_total_loss_terms = torch.zeros_like(self.loss_term_weights)

        self.create_data_loader()
        for i, obs_batch in enumerate(tqdm(self.obs_data_loader, 
                                        desc="Observed Data", leave=False)): 
            obs_batch.to(self.model.device_param.device)

            with torch.autocast(device_type=self.model.device_param.device.type, 
                                dtype=torch.float16, 
                                enabled=self.auto_mixed_precision): 
                generated = self.model()
                gen_ex_format = self.egg_to_ex(generated)
                gen_egg_format = self.egg_to_egg(generated)
                obs_egg_format = self.ex_to_egg(obs_batch)

                loss_terms = self.compute_loss_terms(generated, obs_batch, 
                                                    gen_ex_format, 
                                                    gen_egg_format,  
                                                    obs_egg_format)

                if self.auto_mixed_precision:
                    # Gradient accumulation scaling. 
                    # Reference: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation. 
                    loss_terms = loss_terms / self.batches_per_param

                with torch.no_grad():
                    running_total_loss_terms += (
                        loss_terms * self.loss_term_weights
                    )

                total_loss = loss_terms @ self.loss_term_weights

            self.scaler.scale(total_loss).backward()

            if (i + 1) % self.batches_per_param == 0:

                # Gradient clipping. 
                # Reference: https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#advanced-topics
                if self.auto_mixed_precision: 
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                if self.grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_norm
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            running_total_loss += total_loss.item()

        results = {
            'total_loss': running_total_loss / len(self.obs_data_loader)
        }

        avg_loss_terms = running_total_loss_terms / len(self.obs_data_loader)

        loss_term_names = ["Prediction", "Sparsity", "Structural"]
        for i, name in enumerate(loss_term_names):
            results[name] = avg_loss_terms[i]

        return results
