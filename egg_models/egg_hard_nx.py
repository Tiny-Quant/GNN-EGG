# %% Dependencies

# torch 
import torch 
import torch.nn as nn 

import pygmtools as pygm
# TODO: check padding method. 
from pygmtools.utils import dense_to_sparse
pygm.set_backend('pytorch')

# within 
from egg_models import generic_layers
from utils.ceograph import assign_edge_type, assign_edge_type_par

# %%
class EggHardNx(nn.Module):
    def __init__(self, node_size, cont_node_feat, node_types, cont_edge_feat,
                 batch_size=1): 
        super(EggHardNx, self).__init__()
        self.node_size = node_size
        self.cont_node_feat = cont_node_feat       
        self.node_types = node_types
        self.cont_edge_feat = cont_edge_feat
        self.batch_size = batch_size

        # Layers 
        self.ContNodeFeats = generic_layers.ContFeatMatrix(
            self.batch_size, self.node_size, self.cont_node_feat
        )
        self.DisNodeFeats = generic_layers.CatFeatVector(
            self.batch_size, self.node_size, self.node_types
        )
        self.AdjacencyMatrix = generic_layers.BinaryMatrix(
            self.batch_size, self.node_size, self.node_size
        )
        self.ContEdgeFeats = generic_layers.ContFeatMatrix(
            self.batch_size, self.node_size**2, self.cont_edge_feat
        )

        self.device_param = nn.Parameter(torch.empty(0))
    
    def forward(self, parallel=False):
        X = self.ContNodeFeats()

        C_x, C_x_logLik = self.DisNodeFeats()
        C_x = C_x + 1 # ceograph cats are 1 indexed.

        A, A_logLik = self.AdjacencyMatrix()
        A = dense_to_sparse(A)[0].transpose(1, 2)

        E = self.ContEdgeFeats()
        E = E.narrow(dim=1, start=0, length=A.shape[2]) # undo layer padding.

        if parallel: 
            # Parallel always returns to the cpu for some reason.
            C_e = assign_edge_type_par(A, C_x).to(self.device_param.device)
        else: 
            C_e = assign_edge_type(A, C_x).to(self.device_param.device)

        E = torch.cat((C_e, E), dim=-1)

        return X, C_x, A, E, C_x_logLik, A_logLik 
