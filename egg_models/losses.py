
# %% Dependencies:
from typing import List
from functools import partial

import numpy as np 

import multiprocessing as mp 
from multiprocessing import Pool

import networkx as nx
from networkx import graph_edit_distance

import torch 
import torch.nn as nn 
import pygmtools as pygm
from pygmtools.utils import build_aff_mat, gaussian_aff_fn
pygm.set_backend('pytorch')

from utils.ceograph import NucleiData, nuclei_to_nx

# %%
class PredLoss(nn.Module):
    def __init__(self, target, criterion, explainee):
        super(PredLoss, self).__init__()
        self.target = target
        self.criterion = criterion
        self.explainee = explainee

    def pred_loss_fn(self, example):
        try: 
            explainee_pred = torch.softmax(self.explainee(example), dim=0)

            return self.criterion(explainee_pred, self.target)
        
        except Exception as e:
            # To-do: potential bug - how to get gradient in failed pred case?
            coin_flip = torch.tensor([0.5, 0.5]).to(self.target.device)
            return self.criterion(coin_flip, self.target)

    def forward(self, examples):
        pred_losses = torch.stack([
            self.pred_loss_fn(example) for example in examples
        ])

        return pred_losses

# %%
class EditLoss(nn.Module):
    def __init__(self, obs: list):
        super(EditLoss, self).__init__()
        self.obs = obs

    @staticmethod
    def node_strict_type_match(node_dict_1, node_dict_2): 

        # Quick return false if features names don't match. 
        if not(set(node_dict_1) & set(node_dict_2)):
            return 0

        if node_dict_1['node_features'][0] == node_dict_2['node_features'][0]:
            return 1

        else: 
            return 0 

    @staticmethod
    def single_edit_distance(ob: nx.DiGraph, G: nx.DiGraph,
                            node_match=None):
        node_match = node_match or EditLoss.node_strict_type_match

        dist = graph_edit_distance(
            ob, G, 
            node_match=node_match, 
            node_del_cost=lambda x: 0, 
            edge_del_cost=lambda x: 0,
            upper_bound=50, 
            timeout=120
        )

        if dist is None: 
            return torch.tensor([50.0])
        else:
            return torch.tensor([dist])
    
    @staticmethod
    def pairwise_edit_distance(Gs: List[nx.DiGraph], obs: List[nx.DiGraph], 
                            dist_fn=None):
        dist_fn = dist_fn or EditLoss.single_edit_distance

        with Pool() as pool:
            distances = pool.starmap(dist_fn, zip(Gs, obs))
        
        return torch.stack(distances).squeeze()

    def forward(self, graph_list):
        with torch.no_grad():
            detached_graph_list = [graph.detach().cpu()
                                   for graph in graph_list]
            with Pool() as pool:
                nx_graph_list = pool.map(nuclei_to_nx, detached_graph_list)
            nx_graph_list = nx_graph_list * len(self.obs) #padded computation.

            obs_padded = [ob for ob in self.obs 
                          for _ in range(len(graph_list))]

            return EditLoss.pairwise_edit_distance(nx_graph_list, obs_padded)

# %%
class MatchingLoss(nn.Module):
    def __init__(self, obs: list):
        self.obs_node_matrix = obs[0]
        self.obs_edge_matrix = obs[1]
        self.obs_adj_matrix = obs[2]

    def forward(self, X, C_x, A, E):
        # X [b, nodes, cont_feat], C_x[b, nodes] 
        # A[b, 2, edges], E[B, edges, feats]

        # cat X and C_x -> [b, nodes, feat + 1]
        node_matrix = torch.cat(
            X, C_x.unsqueeze(-1), dim=-1
        ) 

        aff_matrix = build_aff_mat(
            node_feat1=node_matrix, 
            edge_feat1=E, 
            connectivity1=A.transpose(1, 2), 
            node_feat2=self.obs_node_matrix, 
            edge_feat2=self.obs_edge_matrix, 
            connectivity2=self.obs_adj_matrix, 
            node_aff_fn=gaussian_aff_fn, 
            edge_aff_fn=gaussian_aff_fn,
        )

        # matching_matrix = 
        

