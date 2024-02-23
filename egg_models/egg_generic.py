# %% Dependencies

from typing import Optional, Tuple

import torch
import torch.nn as nn 

import pygmtools as pygm
from pygmtools.utils import dense_to_sparse
pygm.set_backend('pytorch')

from egg_models.generic_layers import (
    ContFeatMatrix, ConcreteLayer, BinaryConcrete
)

# %% Generator Model 

class EggGeneric(nn.Module):
    def __init__(self, max_node_size: int, 
                 cont_node_feats: Optional[int] = None, 
                 dis_node_feats: Optional[Tuple[int]] = None, 
                 cont_edge_feats: Optional[int] = None, 
                 dis_edge_feats: Optional[Tuple[int]] = None,
                 temp = 1.0, 
                 batch_size=1): 

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
                        self.batch_size, self.max_node_size, dist_feats, 
                        self.temp
                    )
                )

        self.AdjacencyMatrix = BinaryConcrete(
            self.batch_size, self.max_node_size, self.max_node_size, 
            self.temp
        )

        self.device_param = nn.Parameter(torch.empty(0))
    
    def forward(self):
        """
        returns: dict
            - cont_node_feats: torch.Size([b, n, f_1])
            - dis_node_feats: torch.Size([b, n, total_cats])
            - cont_edge_feats: torch.Size([b, e, f_2])
            - dis_edge_feats: torch.Size([b, n, total_cats])
            - full_edge_indices: torch.Size([b, 2, e])
            - C_x_logLik: torch.Size([b, f_3])
            - C_e_logLik: torch.Size([b, f_4])
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
            C_x, C_x_logLik = None, None

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
            C_e, C_e_logLik = None, None
    
        A, A_logLik = self.AdjacencyMatrix()
        edge_indices, edge_weights, _ = dense_to_sparse(A)
        edge_indices = edge_indices.transpose(1, 2)

        results_dict = {'cont_node_feats': X, 
                        'dis_node_feats': C_x, 
                        'cont_edge_feats': E, 
                        'dis_edge_feats': C_e, 
                        'full_edge_indices': edge_indices, 
                        'C_x_logLik': C_x_logLik, 
                        'C_e_logLik': C_e_logLik, 
                        'A_logLik': A_logLik,
        }

        return results_dict
