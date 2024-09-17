from typing import Optional, List
import numpy as np
import networkx as nx

import torch 
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp 
from torch.multiprocessing import Pool

import torch_geometric as pyg
from torch_geometric.utils import remove_isolated_nodes, remove_self_loops
from torch_geometric.nn import GCNConv, NNConv, global_max_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import remove_isolated_nodes
from torch_scatter import scatter_mean

import pygmtools as pygm
from pygmtools.utils import dense_to_sparse, build_batch
pygm.set_backend('pytorch')

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

def get_edge_type(edge, cell_type):
    """
    Args: 
        edge: (in_cell_index, out_cell_index, 1/edge_length)
        
    Returns:
        edge type index.
        0: 1-1; 1: 1-2; 2: 1-3; 3: 1-4; 4: 1-5; 5: 1-6
        6: 2-1; 7: 2-2; 8: 2-3; 9: 2-4; 10: 2-5; 11: 2-6
        12: 3-1; 13: 3-2; 14: 3-3; 15: 3-4; 16: 3-5; 17: 3-6
        18: 4-1; 19: 4-2; 20: 4-3; 21: 4-4; 22: 4-5; 23: 4-6
        24: 5-1; 25: 5-2; 26: 5-3; 27: 5-4; 28: 5-5; 29: 5-6
        30: 6-1; 31: 6-2; 32: 6-3; 33: 6-4; 34: 6-5; 35: 6-6
    """
    mapping = {"1-1": 0, "1-2": 1, "1-3": 2, "1-4": 3, "1-5": 4, "1-6": 5,
                   "2-1": 6, "2-2": 7, "2-3": 8, "2-4": 9, "2-5": 10, "2-6": 11,
                   "3-1": 12, "3-2": 13, "3-3": 14, "3-4": 15, "3-5": 16, "3-6": 17,
                   "4-1": 18, "4-2": 19, "4-3": 20, "4-4": 21, "4-5": 22, "4-6": 23,
                   "5-1": 24, "5-2": 25, "5-3": 26, "5-4": 27, "5-5": 28, "5-6": 29,
                   "6-1": 30, "6-2": 31, "6-3": 32, "6-4": 33, "6-5": 34, "6-6": 35}
    return mapping['{}-{}'.format(cell_type[edge[0]].item(), 
                                  cell_type[edge[1]].item())]

def get_edge_type_app(A, C_x):
    edges = list(zip(A.long()[0], A.long()[1]))
    e_c = list(map(lambda x: get_edge_type(x, C_x.int()), edges))
    e_c = torch.tensor(e_c).unsqueeze(-1)

    return e_c

def assign_edge_type_par(A, C_x):
    '''
    Assigns edge types to batches in parallel. 
    A - Batched sparse adjacency matrix : tensor [batch, 2, #edges]
    C_x - Batched node cell types : tensor [batch, #nodes]
    '''
    with torch.no_grad():
        mp.set_start_method('spawn', force=True)
        with Pool() as pool: 
            e_c = torch.stack(
                pool.starmap(get_edge_type_app, 
                            zip(A.unbind(), C_x.unbind()))
            ) 
            
    return e_c

def assign_edge_type(A, C_x):
    with torch.no_grad():
        e_c = torch.stack(
            [get_edge_type_app(A, C_x) for (A, C_x) in 
             zip(A.unbind(), C_x.unbind())]
        )
    
    return e_c


# def load_model(path, device=torch.device(0), batched=False): 
#     model = NucleiNet(11, 2, batch=batched)
#     model.to(device)
#     model.load_state_dict(torch.load(
#             path, 
#             map_location=device
#         )
#     )
#     return model

def nuclei_to_nx(data: NucleiData) -> nx.DiGraph: 
    '''
    Helper function for converting Nuclei Data to NetworkX while retaining features.
    '''
    G = nx.DiGraph()   

    # Add nodes with features
    for i in range(data.num_nodes):
        node_feats = np.hstack((data.cell_type[i].cpu().detach().numpy(), 
                                data.x[i].cpu().detach().numpy()))
        G.add_node(i, node_features=node_feats)

    # Add edges with features
    for i in range(data.num_edges):
        src, tgt = data.edge_index[0, i].item(), data.edge_index[1, i].item()
        G.add_edge(src, tgt, 
                   edge_features=data.edge_attr[i].cpu().detach().numpy())

    return G

# def nuclei_to_data(G: NucleiData):
#     node_matrix = nn.functional.one_hot(G.cell_type - 1, 6)
#     node_matrix = torch.cat((G.x, node_matrix), dim=-1)

#     edge_index = pyg.utils.to_dense_adj(G.edge_index) + 1e-8
#     edge_index, edge_weights, _ = dense_to_sparse(edge_index)
#     edge_index = edge_index.transpose(1, 2).squeeze(0)

#     edge_attr = G.edge_attr
#     zero_pad = torch.zeros((edge_index.shape[1] - edge_attr.shape[0], 
#                             edge_attr.shape[1]))
#     edge_attr = torch.cat((edge_attr, zero_pad), dim=0)
#     edge_attr = torch.cat((edge_attr, edge_weights.squeeze(0)), dim=-1)


#     return Data(node_matrix, edge_index, edge_attr)

def nuclei_to_data(G: NucleiData, num_cell_types: int):
    node_matrix = nn.functional.one_hot(G.cell_type - 1, num_cell_types)
    node_matrix = torch.cat((G.x, node_matrix), dim=-1)

    edge_index = torch_geometric.utils.to_dense_adj(G.edge_index) # + 1e-8
    edge_index, edge_weights, _ = dense_to_sparse(edge_index)
    edge_index = edge_index.transpose(1, 2).squeeze(0)

    edge_attr = G.edge_attr
    edge_attr = torch.cat((edge_attr, edge_weights.squeeze(0)), dim=-1)

    return Data(node_matrix, edge_index, edge_attr)

# def clear_iso_nodes(example: NucleiData, 
#                     num_nodes: Optional[int] = None) -> NucleiData: 
#     '''
#     Should not lose grad - checked in debugging.ipynb. 
#     '''
#     edge_index, edge_attr, mask = (
#         remove_isolated_nodes(example.edge_index, 
#                               example.edge_attr, 
#                               num_nodes=num_nodes)
#     )
#     example_masked = NucleiData(
#         x = example.x[mask], 
#         edge_index = edge_index, 
#         cell_type = example.cell_type[mask], 
#         edge_attr = edge_attr,
#     )

#     return example_masked

def clean_gen_graph(gen: NucleiData) -> NucleiData:

    edge_index = gen.edge_index
    edge_attr = gen.edge_attr

    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

    edge_index, edge_attr, mask = (
        remove_isolated_nodes(edge_index, edge_attr, num_nodes=gen.x.shape[0])
    )

    # Uninformative values for when all nodes get cleared. 
    # Needed to maintain mini-batch information. 
    if gen.x[mask].shape[0] == 0:
        x = gen.x * 0.0
        cell_types = torch.ones_like(gen.cell_type) * gen.cell_type.mode().values
    else: 
        x = gen.x[mask]
        cell_types = gen.cell_type[mask]

    gen_cleaned = NucleiData(
        x = x, 
        edge_index = edge_index, 
        cell_type = cell_types,
        edge_attr = edge_attr,
    )

    return gen_cleaned

# class ObsDataset(Dataset):
#     def __init__(self, data):
#         self.data = data
    
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         graph_batch = self.data[index]
#         X = graph_batch.x 
#         A = graph_batch.edge_index.t()
#         E = graph_batch.edge_attr
#         return {'obs_X': X, 'obs_A': A, 'obs_E': E}

# def pygm_collate_fn(batch):
#     # Assuming you have a custom padding function that pads and creates batches
#     # Modify this function according to your specific padding logic
#     X_batch = build_batch([item['obs_X'] for item in batch])
#     A_batch = build_batch([item['obs_A'] for item in batch])
#     E_batch = build_batch([item['obs_E'] for item in batch])

#     return {'obs_X': X_batch, 'obs_A': A_batch, 'obs_E': E_batch}

# def get_obs_loader(raw: List[NucleiData], node_limit=1000, batch_size=1, 
#                    shuffle=True):
#     node_limited = []
#     for graph in raw:
#         if graph.x.shape[0] <= node_limit: 
#             temp = nuclei_to_data(graph)
#             node_limited.append(temp)
#     obs_dataset = ObsDataset(node_limited) 
#     obs_loader = DataLoader(obs_dataset, batch_size=batch_size, 
#                             collate_fn=pygm_collate_fn, shuffle=shuffle, 
#                             drop_last=True)

#     return obs_loader

# def get_nuclei_batch(X, C_x, A, E):

#     nuclei_list = [clear_iso_nodes(NucleiData(X, C_x, A, E))
#                     for (X, C_x, A, E) in zip(
#                         X.unbind(), C_x.unbind(), A.unbind(), 
#                         E.unbind()
#                   )]

#     return Batch().from_data_list(nuclei_list)

def egg_to_ex(generated: dict):
    """
    Implementation of egg_to_ex for the trainer of an EggGeneric model 
    for lung ceograph data. 
    """
    with torch.no_grad():
        C_x_hard = torch.argmax(
            generated['dis_node_feats'], dim=-1
        ) + 1

        A_hard = (generated['adjacency_matrix'] >= 0.5)
        edge_index = dense_to_sparse(A_hard)[0].transpose(1, 2)
        
        edge_types = (assign_edge_type(edge_index, C_x_hard).
                        to(edge_index.device))

        E_hard = (generated['cont_edge_feats'].
                    narrow(dim=1, start=0, length=edge_index.shape[2])
        )
        
        edge_attr = torch.cat((edge_types, E_hard), dim=-1)

    X = generated['cont_node_feats']
    
    nuclei_list = [
        clean_gen_graph(
            NucleiData(x=X, cell_type=C_x, edge_index=A, edge_attr=E)
        )
        for (X, C_x, A, E) in zip(
            X.unbind(), C_x_hard.unbind(), 
            edge_index.unbind(), edge_attr.unbind()
        )
    ]

    nuclei_batch = Batch.from_data_list(nuclei_list)

    return nuclei_batch

def ex_to_egg(obs_batch, num_cell_types) -> List[torch.tensor]:
    """
    Implementation of ex_to_egg for the trainer of an EggGeneric model 
    for lung ceograph data. 
    """

    data_list = [nuclei_to_data(graph, num_cell_types)
                for graph in obs_batch.to_data_list()
    ]

    X_list = [graph.x for graph in data_list]
    A_list = [graph.edge_index for graph in data_list]
    E_list = [graph.edge_attr for graph in data_list]

    return [build_batch(X_list), build_batch(A_list), build_batch(E_list)]

def egg_to_egg(generated: dict) -> List[torch.tensor]: 
    """
    Implementation of egg_to_egg for the trainer of an EggGeneric model 
    for lung ceograph data. 
    """
    with torch.no_grad():
        C_x_hard = torch.argmax(
            generated['dis_node_feats'], dim=-1
        ) + 1

        A = generated['adjacency_matrix']
        edge_index = dense_to_sparse(A)[0].transpose(1, 2)
        
        edge_types = (assign_edge_type(edge_index, C_x_hard).
                        to(edge_index.device))

    gen_X = torch.cat([
        generated['cont_node_feats'], 
        generated['dis_node_feats']
    ], dim=-1)

    gen_A = generated['full_edge_indices']

    gen_E = torch.cat(
        [edge_types.to(gen_A.device), 
         generated['cont_edge_feats'], 
         generated['edge_weights']
        ], dim=-1
    )

    return [gen_X, gen_A, gen_E]