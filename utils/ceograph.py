import numpy as np
import networkx as nx

import torch 
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, NNConv, global_max_pool
from torch_geometric.data import Data
from torch_scatter import scatter_mean

class EdgeNN(nn.Module):
    """
    Design: embedding according to edge type, and then modulated by edge features.
    """
    def __init__(self, in_channels, out_channels, n_edge_types=36,
        device=torch.device(0)):
        super(EdgeNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.edge_type_embedding = nn.Embedding(n_edge_types, out_channels)
        self.fc_h = nn.Linear(in_channels, out_channels)
        self.fc_g = nn.Linear(in_channels, out_channels)
        # self.bn = nn.BatchNorm(out_channels, eps=0.001)
        
        self.device = device

    def forward(self, x):
        """
        Args:
            x: [batch_size, n_edges, 1(edge type) + in_channels]
        
        return: [batch_size, out_channels]
        """
        y = self.edge_type_embedding(x[..., 0].clone().detach().type(torch.long).to(self.device))
        h = self.fc_h(x[..., 1:(self.in_channels + 1)].clone().detach().type(torch.float).to(self.device))
        g = self.fc_g(x[..., 1:(self.in_channels + 1)].clone().detach().type(torch.float).to(self.device))
        y = y * h + g
        # x = self.bn(x)
        return F.relu(y, inplace=True)
        
class NucleiNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, batch=True, edge_features=2, 
        n_edge_types=36, device=torch.device(0)):
        """
        Args:
            in_channels: No. of node features
            out_channels: No. of output node features (e.g., No. classes for classification)
            batch: True if from DataLoader; False if single Data object
            edge_features: No. of edge features (excluding edge type)
            n_edge_types: No. of edge types
        """
        super(NucleiNet, self).__init__()
        self.batch=batch
        
        self.conv1 = NNConv(in_channels, 10, EdgeNN(edge_features, in_channels*10, n_edge_types=n_edge_types), 
                            aggr='mean', root_weight=True, bias=True)
        self.conv2 = NNConv(10, 10, EdgeNN(edge_features, 10*10, n_edge_types=n_edge_types), 
                            aggr='mean', root_weight=True, bias=True)
        self.conv3 = NNConv(10, out_channels, EdgeNN(edge_features, 10*out_channels, n_edge_types=n_edge_types), 
                            aggr='mean', root_weight=True, bias=True)
        
        self.device = device
        
    def forward(self, data):
        """
        Args:
            data: Data in torch_geometric.data
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        '''
        if self.batch:
            x = global_max_pool(x, batch=data.batch).to(device)
        else:
            x = global_max_pool(x, batch=torch.tensor(np.zeros(x.shape[0]), dtype=torch.long).to(device))
        '''
        # cell_type = data.cell_type
        # gate = data.cell_type
        gate = torch.eq(data.cell_type, 1).clone().detach().requires_grad_(False).type(torch.long).to(self.device)
        # print(np.unique(data.batch.cpu().numpy()))  ## [0, 1, 2, ... BATCH_SIZE-1]
        # print(gate * data.batch)  ## [[0]s, [1]s, .... [BATCH_SIZE - 1]s]
        if self.batch:
            _batch_size = data.batch[-1] + 1
            x = scatter_mean(x, gate * (data.batch+1), dim=0).to(self.device)[1:_batch_size+1, :]  # Keep the batches
            # print(x.shape)  ## [BATCH_SIZE, 2]
        else:
            x = scatter_mean(x, gate, dim=0).to(self.device)[1, :]
            # print(x.shape)  ## [2]
        
        # return F.log_softmax(x, dim=1)
        return x

class NucleiData(Data):
    """Add some attributes to Data object.
    
    Args:
        * All args mush be torch.tensor. So string is not supported.
        x: Matrix for nodes
        cell_type: cell type
        edge_index: 2*N matrix
        edge_attr: edge type
        y: Label
        pid: Patient ID
        region_id: Region ID, range from 0~99
    """
    def __init__(self, x=None, cell_type=None, edge_index=None, edge_attr=None, y=None, pos=None, pid=None, region_id=None):
        super().__init__(x, edge_index, edge_attr, y, pos)
        self.cell_type = cell_type
        self.pid = pid
        self.region_id = region_id
        
    def __repr__(self):
        info = ['{}={}'.format(key, self.size_repr(item)) for key, item in self]
        return '{}({})'.format(self.__class__.__name__, ', '.join(info))
    
    @staticmethod
    def size_repr(value):
        if torch.is_tensor(value):
            return list(value.size())
        elif isinstance(value, int) or isinstance(value, float) or isinstance(value, str):
            return [1]
        else:
            raise ValueError('Unsupported attribute type.')

def get_nuclei_orientation_diff(edge, nuclei_orientation):
    return np.cos(nuclei_orientation[edge[0]] - nuclei_orientation[edge[1]])

def load_model(path, device=torch.device(0)): 
    model = NucleiNet(11, 2, batch=False)
    model.to(device)
    model.load_state_dict(torch.load(
            path, 
            map_location=device
        )
    )
    return model

def nuclei_to_nx(data: NucleiData) -> nx.DiGraph: 
    '''
    Helper function for converting Nuclei Data to NetworkX while retaining features.
    '''
    G = nx.DiGraph()   

    # Add nodes with features
    for i in range(data.num_nodes):
        node_feats = np.hstack((data.cell_type[i].numpy(), data.x[i].numpy()))
        G.add_node(i, node_features=node_feats)

    # Add edges with features
    for i in range(data.num_edges):
        src, tgt = data.edge_index[0, i].item(), data.edge_index[1, i].item()
        G.add_edge(src, tgt, edge_features=data.edge_attr[i].numpy())

    return G