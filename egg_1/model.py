# %% Dependencies
import torch 
import torch.distributions as td
import torch.nn as nn 
from torch_geometric.utils import dense_to_sparse

import sys 
sys.path.append("../scripts/ceograph/")
from ceograph import get_edge_type

# %%
class ContNodeFeats(nn.Module):
    def __init__(self, node_size, cont_node_feat):
        super(ContNodeFeats, self).__init__()
        self.node_size = node_size
        self.cont_node_feat = cont_node_feat       
        self.cont_node_feat_1 = nn.Linear(self.node_size, 
            self.cont_node_feat * self.node_size)

    def forward(self):
        Z = torch.randn(self.node_size)
        X = self.cont_node_feat_1(Z)
        X = X.view(-1, self.cont_node_feat)
        return X

# %%
class DisNodeFeat(nn.Module): 
    def __init__(self, node_size, cell_types):
        super(DisNodeFeat, self).__init__()
        self.node_size = node_size
        self.cell_types = cell_types

        self.cell_logits = nn.Parameter(
            nn.init.xavier_normal_( # Glorot initialization. 
                torch.empty((1, self.cell_types))
            )
        )

    def forward(self, batch_size=None):
        self.dist = td.Categorical(logits=self.cell_logits)
        self.sample = self.dist.sample([batch_size, self.node_size]).squeeze()
        return self.sample + 1 # category indexing starts at 1. 

# %%
class AdjacencyMatrix(nn.Module):
    def __init__(self, node_size):
        super(AdjacencyMatrix, self).__init__()
        self.node_size = node_size

        self.edge_logits = nn.Parameter(
            nn.init.xavier_normal_( # Glorot initialization. 
                torch.empty((self.node_size, self.node_size))
            )
        )

    def forward(self, batch_size=None):
        self.dist = td.Bernoulli(logits=self.edge_logits)
        self.sample = self.dist.sample([batch_size]).squeeze()
        return self.sample


# %% Full Model 
class EGG(nn.Module):
    def __init__(self, node_size, cont_node_feat, cell_types, cont_edge_feat, 
                 batch_size=None): 
        super(EGG, self).__init__()
        self.node_size = node_size
        self.cont_node_feat = cont_node_feat       
        self.cell_types = cell_types
        self.cont_edge_feat = cont_edge_feat
        self.batch_size = batch_size

        # Sub-Generator models.
        self.ContNodeFeats = ContNodeFeats(self.node_size, self.cont_node_feat)
        self.DisNodeFeat = DisNodeFeat(self.node_size, self.cell_types)
        self.AdjacencyMatrix = AdjacencyMatrix(self.node_size)

    def forward(self):
        # Sub-Generator models.
        X = self.ContNodeFeats()

        C_x = self.DisNodeFeat(self.batch_size)

        A = self.AdjacencyMatrix(self.batch_size)

        if self.batch_size is not None: 
            A = [dense_to_sparse(x)[0] for x in torch.unbind(A)]
        else: 
            A = dense_to_sparse(A)[0]

        # Discrete Edge Features:
        edges = list(zip(A.long()[0], A.long()[1]))
        e_c = list(map(lambda x: get_edge_type(x, C_x.int()), edges))
        e_c = torch.tensor(e_c)
        e_c = e_c.reshape(-1, 1)

        # Continuous Edge Features
        Z = torch.randn(self.node_size**2, 1)
        W = torch.normal(mean=0, std=1, size=(1, self.cont_edge_feat))
        E = Z @ W
        E = E[:A.shape[1]] # match number of edges. 

        # Combines continuous and discrete node features.
        edge_features = torch.cat((e_c, E), dim=-1) 
        
        return X, C_x, A, edge_features

# %%
