import torch 
import torch.nn as nn 
import torch.nn.functional as F

from functools import partial 

import pygmtools as pygm
pygm.set_backend('pytorch')
from pygmtools.utils import build_aff_mat

from utils import misc

# %%

class PredLossBatched(nn.Module):
    def __init__(self, target: torch.Tensor, 
                 explainee: nn.Module,
                 criterion = nn.BCELoss(reduction='none')):

        super(PredLossBatched, self).__init__()
        self.target = target
        self.explainee = explainee
        self.criterion = criterion
    
    def forward(self, batch):
        explainee_pred = torch.softmax(self.explainee(batch), dim=1)
        loss = self.criterion(explainee_pred, 
                              self.target.expand_as(explainee_pred))

        return loss

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