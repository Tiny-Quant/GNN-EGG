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
        activations[name] = output #.detach()

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
                  batch_indices1: torch.Tensor, 
                  batch_indices2=None,  
                  act_pool_func: Callable=pyg.nn.global_mean_pool, 
                  expand2=True, 
                  agg_func: Callable=torch.mean, 
                  batch_size=None):
    """
    Returns the aggregated cosine distance by batch between two dictionaries 
    of tensors on matching keys. 
    Tensors contribute 2 if opposite, 1 is orthogonal, and 0 is same. 
    """
    agg_cos_dist = []
    for key in set(dict1.keys()) & set(dict2.keys()): 
        tensor1 = act_pool_func(dict1[key], 
                                batch=batch_indices1, size=batch_size)
        if expand2: 
            tensor2 = dict2[key].expand_as(tensor1).to(tensor1.device)
        else: 
            tensor2 = act_pool_func(dict2[key], 
                                    batch=batch_indices2, size=batch_size)

        cos_sim = F.cosine_similarity(tensor1, tensor2, dim=1)

        agg_cos_dist.append(1 - cos_sim) # [B]

    assert torch.all(agg_func(torch.stack(agg_cos_dist), dim=0) >= -1e-5)
    
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
            explainee_pred = F.softmax(self.explainee(batch), dim=-1)

            loss = self.criterion(explainee_pred, 
                                  (self.target.expand_as(explainee_pred).
                                    to(explainee_pred.device))
            )
            
            loss = loss + dict_cos_dist(activations, self.avg_embed_targets, 
                                        batch_indices1=batch.batch)

            remove_hooks()

            return loss, activations, batch.batch

        else: 
            explainee_pred = F.softmax(self.explainee(batch), dim=-1)
            loss = self.criterion(explainee_pred, 
                                  self.target.expand_as(explainee_pred).
                                  to(explainee_pred.device)
            )
            return loss, None, None

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
    """
    Computes a positive upper bound on the number of edits required to make 
    G2 subgraph isomorphic (ie deletes cost 0) to G1. 
    """
    # TODO: Consider the case where indices are None. 
    def __init__(self, node_size, 
                 cont_node_indices: tuple, 
                 dis_node_indices: tuple,
                 cont_edge_indices: tuple, 
                 dis_edge_indices: tuple, 
                 cont_edit_weight=0.25, 
                 dis_edit_weight=0.5, 
                 #grad_strength=1e-1, 
                 #device=torch.device(0)
                 ):
        super(GEDasMatchLoss, self).__init__()
        self.max_gen_nodes = node_size

        self.cont_node_indices = cont_node_indices
        self.dis_node_indices = dis_node_indices
        self.cont_edge_indices = cont_edge_indices
        self.dis_edge_indices = dis_edge_indices

        self.cont_edit_weight = cont_edit_weight
        self.dis_edit_weight = dis_edit_weight

        # self.grad_strength = grad_strength

        # self.device = device

    def cont_edit_aff_fn(self, 
                         feat1: torch.tensor, 
                         feat2: torch.tensor) -> torch.tensor:
        """
        Computes the normalized cosine distance between batches of feature 
        vectors. The returned tensor values should be -2 for opposite, -1 for 
        orthogonal or 0 for same.  

        feat1: [b, n1, f]
        feat2: [b, n2, f]
        return: [b, n1, n2]
        """

        feat1_norm = F.normalize(feat1, p=2, dim=-1)
        feat2_norm = F.normalize(feat2, p=2, dim=-1)

        cos_sim_mat = torch.einsum('bij, bkj -> bik', 
                                   feat1_norm, feat2_norm)

        # tol = 1e-5
        # assert (-1 * (1 - cos_sim_mat) >= -2. - tol).all()
        # assert (-1 * (1 - cos_sim_mat) <= 0. + tol).all()

        return -1 * (1 - cos_sim_mat)
    
    def dis_edit_aff_fn(self, 
                        feat1: torch.tensor, 
                        feat2: torch.tensor) -> torch.tensor:
        """
        Computes the un-normalized cosine distance between batches of one-hot 
        (or binary[0, 1]) features vectors. The returned tensor values should 
        be close to -1 for differing classes and 0 for similarly classes. 
        The values do not reach -2 because unordered discrete features cannot 
        be "opposites" of each other.

        feat1: [b, n1, f]
        feat2: [b, n2, f]
        return: [b, n1, n2]
        """

        cos_sim_mat = torch.einsum('bij, bkj -> bik', 
                                   feat1, feat2)

        # tol = 1e-5
        # assert (-1 * (1 - cos_sim_mat) >= -1. - tol).all()
        # assert (-1 * (1 - cos_sim_mat) <= 0. + tol).all()

        return -1 * (1 - cos_sim_mat)

    def mixed_edit_aff_fn(self, 
                          feat1: torch.tensor, feat2:torch.tensor,
                          cont_indices: tuple, dis_indices: tuple,
                         ) -> torch.tensor:
        """
        Computes the -1 * edit distance or the edit affinity for feature vectors 
        with continuous and discrete values. By default the return values range 
        from [-1, 0]. 

        feat1: [b, n1, f]
        feat2: [b, n2, f]
        return: [b, n1, n2]
        """

        # Shape: [batch, nodes, dis_index]
        #cont_start, cont_end = cont_indices
        #cont_feat1 = feat1[:, :, cont_start:cont_end] 
        #cont_feat2 = feat2[:, :, cont_start:cont_end]
        # Set to zero if no features.
        if cont_indices is not None:
            cont_feat1 = misc.subset_tensor(feat1, cont_indices, dim=-1)
            cont_feat2 = misc.subset_tensor(feat2, cont_indices, dim=-1)

            cont_edit_aff = self.cont_edit_aff_fn(cont_feat1, cont_feat2)

        else:
            cont_edit_aff = 0

        #dis_start, dis_end = dis_indices
        #dis_feat1 = feat1[:, :, dis_start:dis_end]
        #dis_feat2 = feat2[:, :, dis_start:dis_end]
        # Set to zero if no features.
        if dis_indices is not None:
            dis_feat1 = misc.subset_tensor(feat1, dis_indices, dim=-1)
            dis_feat2 = misc.subset_tensor(feat2, dis_indices, dim=-1)

            dis_edit_aff = self.dis_edit_aff_fn(dis_feat1, dis_feat2)

        else:
            dis_edit_aff = 0

        edit_aff = (self.cont_edit_weight * cont_edit_aff + 
                    self.dis_edit_weight * dis_edit_aff)

        return edit_aff

    def get_aff_mat(self, gen_X, gen_A, gen_E, obs_X, obs_A, obs_E,): 
        node_edit_aff_fn = partial(self.mixed_edit_aff_fn, 
                                   cont_indices = self.cont_node_indices, 
                                   dis_indices = self.dis_node_indices)
        
        edge_edit_aff_fn = partial(self.mixed_edit_aff_fn, 
                                   cont_indices = self.cont_edge_indices, 
                                   dis_indices = self.dis_edge_indices)

        # Collect graph size information - avoids weird bugs.
        self.n1 = (
            torch.tensor([gen_X.shape[1]]).expand(gen_X.shape[0]).
                to(gen_X.device)
        )
        self.ne1 = (
            torch.tensor([gen_A.shape[2]]).expand(gen_A.shape[0]). 
                to(gen_A.device)
        )
        self.n2 = (
            torch.tensor([obs_X.shape[1]]).expand(obs_X.shape[0]). 
                to(obs_X.device)
        )
        self.ne2 = (
            torch.tensor([obs_A.shape[2]]).expand(obs_A.shape[0]). 
                to(obs_A.device)
        )
    
        aff_mat = build_aff_mat(
            node_feat1=gen_X, 
            edge_feat1=gen_E, 
            connectivity1=gen_A.transpose(1, 2), 
            node_feat2=obs_X, 
            edge_feat2=obs_E, 
            connectivity2=obs_A.transpose(1, 2),  
            node_aff_fn=node_edit_aff_fn, 
            edge_aff_fn=edge_edit_aff_fn,
            n1=self.n1, 
            ne1=self.ne1, 
            n2=self.n2, 
            ne2=self.ne2
        )

        return aff_mat

    def get_dis_match_mat(self, aff_mat):

        match_mat = pygm.rrwm(aff_mat, n1=self.n1, n2=self.n2)

        dis_match_mat = pygm.hungarian(match_mat)

        return dis_match_mat

    def forward(self, *args): 
        
        aff_mat = self.get_aff_mat(*args)

        dis_match_mat = self.get_dis_match_mat(aff_mat)

        score = pygm.utils.compute_affinity_score(dis_match_mat, aff_mat)

        assert torch.all(-1 * score >= -1e-5)

        return -1 * score # Returns a positive upper bound of GED.  

# %%
class StructuralLoss(nn.Module):
    """
    Implements the structural loss term in (cite). 
    """
    def __init__(self, 
                 GED_fn: nn.Module, 
                 explainee: nn.Module, 
                 gamma: torch.Tensor, 
                 target: torch.Tensor, 
                 criterion = nn.CrossEntropyLoss(reduction='none'), 
                 use_embeddings=True, 
                 uninfo_pen=-1): 
        super(StructuralLoss, self).__init__()

        self.GED_fn = GED_fn
        self.explainee = explainee
        self.gamma = gamma
        self.target = target
        self.criterion = criterion
        self.use_embeddings = use_embeddings
        self.uninfo_pen = uninfo_pen # TODO: Remove argument. 

    def forward(self, 
                gen_egg: List[torch.Tensor], obs_egg: List[torch.Tensor], 
                obs_ex, 
                gen_acts: Optional[Dict[str, torch.Tensor]]=None, 
                gen_acts_batch: Optional[torch.Tensor]=None):

        approx_GED = self.GED_fn(*gen_egg, *obs_egg)

        if gen_acts is not None and self.use_embeddings: 
            #with torch.no_grad():
            activations, remove_hooks = (
                activation_hook(self.explainee, gen_acts.keys())
            ) 
            explainee_pred = F.softmax(self.explainee(obs_ex), dim=-1)
            remove_hooks()

            # TODO: Look into auto batch_size calc problems in pyg pooling.
            batch_size = gen_egg[0].shape[0]
            embed_loss = dict_cos_dist(activations, gen_acts, 
                                       batch_indices1=obs_ex.batch, 
                                       batch_indices2=gen_acts_batch, 
                                       expand2=False, 
                                       batch_size=batch_size)

        else: 
            with torch.no_grad():
                explainee_pred = F.softmax(self.explainee(obs_ex), dim=-1)
                embed_loss = torch.tensor(0.)
        
        with torch.no_grad():
            omega = (self.gamma - 
                self.criterion(
                    explainee_pred, 
                    self.target.expand_as(explainee_pred).
                        to(explainee_pred.device)
                )
            ) ** 3

        return omega * (approx_GED + embed_loss)
        