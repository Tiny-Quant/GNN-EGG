# %%
import torch 
import torch.nn as nn 
# import torch.distributions as td

# import torch_geometric as pyg
from torch_geometric.data import Data, Batch

import pygmtools as pygm
from pygmtools.utils import dense_to_sparse
pygm.set_backend('pytorch')

from egg_models.generic_layers import ContFeatMatrix, ConcreteLayer, BinaryConcrete
from utils.ceograph import assign_edge_type, NucleiData, load_model, clear_iso_nodes

# %%
class EggSoft(nn.Module):
    def __init__(self, node_size, cont_node_feat, node_types, cont_edge_feat,
                 temp_1=1.0, temp_2=1.0,
                 batch_size=1): 
        super(EggSoft, self).__init__()
        self.node_size = node_size
        self.cont_node_feat = cont_node_feat       
        self.node_types = node_types
        self.cont_edge_feat = cont_edge_feat
        self.temp_1 = temp_1
        self.temp_2 = temp_2
        self.batch_size = batch_size

        # Layers 
        self.ContNodeFeats = ContFeatMatrix(
            self.batch_size, self.node_size, self.cont_node_feat, 
        )
        self.DisNodeFeats = ConcreteLayer(
            self.batch_size, self.node_size, self.node_types, 
            self.temp_1
        )
        self.AdjacencyMatrix = BinaryConcrete(
            self.batch_size, self.node_size, self.node_size, 
            self.temp_2
        )
        self.ContEdgeFeats = ContFeatMatrix(
            self.batch_size, self.node_size**2, self.cont_edge_feat
        )

        self.device_param = nn.Parameter(torch.empty(0))
    
    def forward(self):
        X = self.ContNodeFeats()
        
        C_x, C_x_logLik = self.DisNodeFeats()
        #C_x = C_x.to(self.device_param.device)
        # print(C_x.device, self.device_param.device)

        A, A_logLik = self.AdjacencyMatrix()
        #A.to(self.device_param.device)

        E = self.ContEdgeFeats()

        # No Diff - for pred_loss: 
        with torch.no_grad():
            C_x_hard = torch.argmax(C_x, dim=-1) + 1
            A_hard = (A >= 0.5)
            edge_indices_hard = dense_to_sparse(A_hard)[0].transpose(1, 2)
            edge_types_hard = assign_edge_type(
                edge_indices_hard, C_x_hard
            ).to(self.device_param.device)
            E_hard = E.narrow(dim=1, start=0, length=edge_indices_hard.shape[2])
            edge_attr_hard = torch.cat((edge_types_hard, E_hard), dim=-1)

        # Diff - for discriminators:
        # print(X.device, C_x.device)
        node_matrix = torch.cat((X, C_x), dim=-1)
        edge_indices, edge_weights, _ = dense_to_sparse(A)
        edge_indices = edge_indices.transpose(1, 2)
        edge_attr = assign_edge_type(
            edge_indices, C_x_hard
        ).to(self.device_param.device)
        edge_attr = torch.cat((edge_attr, E, edge_weights), dim=-1)

        # Wrap 
        nuclei_list = [clear_iso_nodes(NucleiData(X, C_x, A, E))
                        for (X, C_x, A, E) in zip(
                            X.unbind(), C_x_hard.unbind(), edge_indices_hard.unbind(), 
                            edge_attr_hard.unbind()
                      )]
        #nuclei_batch = Batch().from_data_list(nuclei_batch)

        # data_list = [Data(X, A, E) 
        #               for (X, A, E) in zip(
        #                 node_matrix.unbind(), edge_indices.unbind(), 
        #                 edge_attr.unbind()
        #             )]
        # data_batch = Batch().from_data_list(data_batch)

        return (nuclei_list, 
                node_matrix, edge_indices, edge_attr, 
                C_x_logLik, A_logLik)
