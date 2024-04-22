
# %% Dependencies:
from typing import List
from functools import partial
import gc

import numpy as np 
from tqdm.auto import tqdm

import multiprocessing as mp 
from multiprocessing import Pool

import networkx as nx
from networkx import graph_edit_distance

import torch 
import torch.nn as nn 

import torch_geometric as pyg
from torch_geometric.data import Data 

import pygmtools as pygm
from pygmtools.utils import build_aff_mat, gaussian_aff_fn, compute_affinity_score
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
class PredLossBatched(nn.Module):
    def __init__(self, target: torch.Tensor, 
                 criterion: nn.Module, explainee: nn.Module):
        super(PredLossBatched, self).__init__()
        self.target = target
        self.criterion = criterion
        self.explainee = explainee
    
    def forward(self, nuclei_batch):
        explainee_pred = torch.softmax(self.explainee(nuclei_batch), dim=1)
        loss = self.criterion(explainee_pred, 
                              self.target.expand_as(explainee_pred))

        return loss
        #return torch.tensor([loss]).to(self.target.device)

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
            timeout=300
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
    def __init__(self, node_size, device=torch.device(0)):
        super(MatchingLoss, self).__init__()
        self.device = device 
        self.max_gen_nodes = node_size

    def forward(self, obs_batch, X, A, E):
        obs_X = obs_batch['obs_X'].to(self.device)
        obs_A = obs_batch['obs_A'].to(self.device)
        obs_E = obs_batch['obs_E'].to(self.device)

        aff_matrix = build_aff_mat(
            node_feat1=X, 
            edge_feat1=E, 
            connectivity1=A.transpose(1,2), 
            node_feat2=obs_X, 
            edge_feat2=obs_E, 
            connectivity2=obs_A, 
            node_aff_fn=gaussian_aff_fn, 
            edge_aff_fn=gaussian_aff_fn, 
            # n2 = max_nodes_2 # sometimes need to resolve bug - I think for unbatched. 
        )

        matching = pygm.hungarian(
            pygm.rrwm(aff_matrix, n1max = self.max_gen_nodes, 
                      n2max = aff_matrix.shape[1] // self.max_gen_nodes)
        )

        return 1 / compute_affinity_score(matching, aff_matrix)

# %%
# class AffinityScore(nn.Module):
#     def __init__(self, optimizer):
#         super(AffinityScore, self).__init__()
#         self.optimizer = optimizer

#     def score(self, example: Data, ob: Data):

#         print(torch.cuda.memory_summary())

#         example.to(torch.device(0))
#         ob.to(torch.device(0))

#         aff_matrix = build_aff_mat(
#             node_feat1=example.x, 
#             edge_feat1=example.edge_attr, 
#             connectivity1=example.edge_index.t(),
#             node_feat2=ob.x, 
#             edge_feat2=ob.edge_attr, 
#             connectivity2=ob.edge_index.t(), 
#             node_aff_fn=gaussian_aff_fn, 
#             edge_aff_fn=gaussian_aff_fn, 
#             n2=[ob.x.shape[0]]
#         )
        
#         matching = pygm.hungarian(
#             pygm.rrwm(aff_matrix, n1 = example.x.shape[0], n2 = ob.x.shape[0])
#         )

#         matching_score = compute_affinity_score(matching, aff_matrix)
#         print(matching_score)
#         matching_score = matching_score.cpu().detach()
#         # ret = matching_score.cpu().item()
#         #matching_score.backward(retain_graph=True) 
#         #self.optimizer.step()

#         # Free vram
#         example.cpu()
#         ob.cpu()
#         aff_matrix.cpu()
#         matching.cpu()
#         del example, ob, aff_matrix, matching
#         torch.cuda.synchronize()
#         gc.collect()
#         torch.cuda.empty_cache()

#         print(matching_score)

#         return matching_score

#     def forward(self, examples: List[Data], obs: List[Data]):
#         scores = [self.score(example, ob) 
#                   for example in tqdm(examples, desc="Generated")
#                   for ob in tqdm(obs, desc="Observed")]

#         return torch.stack(scores)
