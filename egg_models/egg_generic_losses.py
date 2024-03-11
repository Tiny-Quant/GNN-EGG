from functools import partial 
from typing import Dict, List, Tuple, Callable, Optional

import torch 
import torch.nn as nn 
import torch.nn.functional as F

import torch_geometric as pyg

import pygmtools as pygm
pygm.set_backend('pytorch')
from pygmtools.utils import build_aff_mat

from utils import misc

# %% Helper Functions
def activation_hook(model: nn.Module,
                    layer_names: List[str]) -> Tuple[Dict[str, torch.Tensor], 
                                                     Callable]:
    """
    Establishes forward hooks to extract a names activations from a model. 
    Returns a dictionary of activations and a function to remove the hooks.
    """
    activations = {}

    def hook(module, input, output, name):
        activations[name] = output.detach()

    hooks = []
    for name, module in model.named_modules():
        if name in layer_names:
            hook_fn = (
                lambda module, input, output, name=name: 
                    hook(module, input, output, name)
            )
            hooks.append(module.register_forward_hook(hook_fn))

    def remove_hooks():
        for hook in hooks:
            hook.remove()

    return activations, remove_hooks

def dict_cos_dist(dict1: Dict[str, torch.Tensor], 
                  dict2: Dict[str, torch.Tensor], 
                  batch_indices: torch.Tensor, 
                  act_pool_func: Callable=pyg.nn.global_mean_pool, 
                  agg_func: Callable=torch.sum):
    """
    Returns the aggregated cosine distance by batch between two dictionaries 
    of tensors on matching keys. 
    Tensors contribute 2 if opposite, 1 is orthogonal, and 0 is same. 
    """
    agg_cos_dist = []
    for key in set(dict1.keys()) & set(dict2.keys()): 
        tensor1 = act_pool_func(dict1[key], batch=batch_indices)
        tensor2 = dict2[key].expand_as(tensor1).to(tensor1.device)

        cos_sim = F.cosine_similarity(tensor1, tensor2, dim=1)

        agg_cos_dist.append(1 - cos_sim) # [B]
    
    return agg_func(torch.stack(agg_cos_dist), dim=0)

# %%
class PredLossBatched(nn.Module):
    """
    Implements the prediction loss term explaining the paper (cite). 
    """
    def __init__(self, 
                 target: torch.Tensor, 
                 explainee: nn.Module,
                 criterion = nn.CrossEntropyLoss(reduction='none'), 
                 avg_embed_targets: Dict[str, torch.Tensor]=None):
        super(PredLossBatched, self).__init__()

        self.target = target
        self.explainee = explainee
        self.criterion = criterion
        self.avg_embed_targets = avg_embed_targets
    
    def forward(self, batch):
        if self.avg_embed_targets is not None: 
            activations, remove_hooks = (
                activation_hook(self.explainee, self.avg_embed_targets.keys())
            ) 
            explainee_pred = self.explainee(batch)

            loss = self.criterion(explainee_pred, 
                                  (self.target.expand_as(explainee_pred).
                                    to(explainee_pred.device))
            )
            
            loss = loss + dict_cos_dist(activations, self.avg_embed_targets, 
                                        batch_indices=batch.batch)

            remove_hooks()

            return loss, activations

        else: 
            explainee_pred = self.explainee(batch)
            loss = self.criterion(explainee_pred, 
                                self.target.expand_as(explainee_pred))
            return loss, None

# %%
class EdgePenalty(nn.Module):
    """
    Implements the edge penalty of sparsity loss term in (cite). 
    """
    def __init__(self, edge_budget = 0):
        super(EdgePenalty, self).__init__()

        self.edge_budget = edge_budget
        
    def forward(self, edge_probs: torch.Tensor) -> torch.Tensor:
        #flat_edge_probs = edge_probs.flatten(start_dim=1) # All after batch.
        L2_pen = torch.norm(edge_probs, p=2)
        budget_pen = (
            (F.softplus(edge_probs.sum() - self.edge_budget)) ** 2
        ) 

        return L2_pen + budget_pen

# %%
class GEDasMatchLoss(nn.Module):
    # TODO: Consider the case where indices are None. 
    def __init__(self, node_size, 
                 cont_node_indices: tuple, 
                 dis_node_indices: tuple,
                 cont_edge_indices: tuple, 
                 dis_edge_indices: tuple, 
                 cont_edit_weight=0.25, 
                 dis_edit_weight=0.5, 
                 grad_strength=1e-1, 
                 device=torch.device(0)):
        super(GEDasMatchLoss, self).__init__()
        self.max_gen_nodes = node_size

        self.cont_node_indices = cont_node_indices
        self.dis_node_indices = dis_node_indices
        self.cont_edge_indices = cont_edge_indices
        self.dis_edge_indices = dis_edge_indices

        self.cont_edit_weight = cont_edit_weight
        self.dis_edit_weight = dis_edit_weight

        self.grad_strength = grad_strength

        self.device = device

    def cont_edit_aff_fn(self, 
                         feat1: torch.tensor, 
                         feat2: torch.tensor) -> torch.tensor:
        """
        feat1: [b, n1, f]
        feat2: [b, n2, f]
        return: [b, n1, n2]
        """

        feat1_norm = F.normalize(feat1, p=2, dim=-1)
        feat2_norm = F.normalize(feat2, p=2, dim=-1)

        cos_sim_mat = torch.einsum('bij, bkj -> bik', 
                                feat1_norm, feat2_norm)

        return -1 * (1 - cos_sim_mat)
    
    def dis_edit_aff_fn(self, 
                        feat1: torch.tensor, 
                        feat2: torch.tensor) -> torch.tensor:
        """
        feat1: [b, n1, f]
        feat2: [b, n2, f]
        return: [b, n1, n2]
        """

        delta = feat1.unsqueeze(2) - feat2.unsqueeze(1)

        dis_edit_cost = (delta / (delta + self.grad_strength)).mean(dim=-1)

        return -1 * dis_edit_cost

    def mixed_edit_aff_fn(self, 
                          feat1: torch.tensor, feat2:torch.tensor,
                          cont_indices: tuple, dis_indices: tuple,
                         ) -> torch.tensor:
        """
        Computes the -1 * edit distance or the edit affinity for feature vectors 
        with continuous and discrete values. By default the return values range 
        from [-1, 0]. 

        Indices should be given as [start, end)

        feat1: [b, n1, f]
        feat2: [b, n2, f]
        return: [b, n1, n2]
        """

        # Shape: [batch, nodes, dis_index]
        #cont_start, cont_end = cont_indices
        #cont_feat1 = feat1[:, :, cont_start:cont_end] 
        #cont_feat2 = feat2[:, :, cont_start:cont_end]
        cont_feat1 = misc.subset_tensor(feat1, cont_indices, dim=-1)
        cont_feat2 = misc.subset_tensor(feat2, cont_indices, dim=-1)

        cont_edit_aff = self.cont_edit_aff_fn(cont_feat1, cont_feat2)

        #dis_start, dis_end = dis_indices
        #dis_feat1 = feat1[:, :, dis_start:dis_end]
        #dis_feat2 = feat2[:, :, dis_start:dis_end]
        dis_feat1 = misc.subset_tensor(feat1, dis_indices, dim=-1)
        dis_feat2 = misc.subset_tensor(feat2, dis_indices, dim=-1)

        dis_edit_aff = self.dis_edit_aff_fn(dis_feat1, dis_feat2)

        edit_aff = (self.cont_edit_weight * cont_edit_aff + 
                    self.dis_edit_weight * dis_edit_aff)

        return edit_aff

    def forward(self, gen_X, gen_A, gen_E, obs_X, obs_A, obs_E,): 
        node_edit_aff_fn = partial(self.mixed_edit_aff_fn, 
                                   cont_indices = self.cont_node_indices, 
                                   dis_indices = self.dis_node_indices)
        
        edge_edit_aff_fn = partial(self.mixed_edit_aff_fn, 
                                   cont_indices = self.cont_edge_indices, 
                                   dis_indices = self.dis_edge_indices)

        aff_mat = build_aff_mat(
            node_feat1=gen_X, 
            edge_feat1=gen_E, 
            connectivity1=gen_A.transpose(1, 2), 
            node_feat2=obs_X, 
            edge_feat2=obs_E, 
            connectivity2=obs_A.transpose(1, 2),  
            node_aff_fn=node_edit_aff_fn, 
            edge_aff_fn=edge_edit_aff_fn
        )

        match_mat = pygm.rrwm(aff_mat, 
                              n1max=self.max_gen_nodes, 
                              n2max=aff_mat.shape[1] // self.max_gen_nodes)

        dis_match_mat = pygm.hungarian(match_mat)

        score = pygm.utils.compute_affinity_score(dis_match_mat, aff_mat)

        return -1 * score # Returns a positive upper bound of GED.  