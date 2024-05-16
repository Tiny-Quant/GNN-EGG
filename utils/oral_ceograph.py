
import os
import re
import sys 
sys.path.append("..")

from typing import List

# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np
import numexpr as ne
import pandas as pd
import sklearn.neighbors as skgraph
from scipy import sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import DataLoader
import torch_geometric
from torch_geometric.utils import remove_isolated_nodes, remove_self_loops
from torch_geometric.data import Data, InMemoryDataset, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, NNConv, global_max_pool
from torch_scatter import scatter_mean

import pygmtools as pygm
from pygmtools.utils import dense_to_sparse, build_batch
pygm.set_backend('pytorch')

# Global Constants 
PATCH_SIZE = 2048

x_ave = np.array([2.15e+03, 2.21e+03, # area, convex_area, 
                   7.13e-01, 7.51e-01, 2.15e+02, 6.32e+01,  # eccentricity, extent, filled_area, major_axis_length
                   4.08e+01, 1.59e+01, 1.77e+02, 8.68e-01,  # minor_axis_length, pa_ratio, perimeter, probability
                   9.69e-01])  # solidity

x_std = np.array([1289.57, 1321.28,  # area, convex_area,
                   0.155, 0.0583, 1289.57, 20.08,  # eccentricity, extent, filled_area, major_axis_length
                   13.35, 2.257, 53.41, 0.0825,  # minor_axis_length, pa_ratio, perimeter, probability
                   0.021])   # solidity

nuclei_dir = "/project/DPDS/Xiao_lab/shared/Xinyi/201908_HN/results/20210607_unet_mrcnn_cell_info"

def read_data(): 

    nuclei_dir = "/project/DPDS/Xiao_lab/shared/Xinyi/201908_HN/results/20210607_unet_mrcnn_cell_info"
    info_df = pd.read_csv("../data/slides/HN/EPOC full cohort info.csv", index_col="file ID")

    # Remove "Status for analysis == Exclude"
    info_df = info_df.loc[info_df['Status for analysis'] == "ok", :]

    # Remove NAN time & event
    info_df = info_df.loc[np.logical_not(pd.isna(info_df['cancer (1= oral cancer or death, 0=censor)'])), :]

    # Remove "Carcinoma? == yes"
    info_df = info_df.loc[np.logical_or(np.logical_and(info_df['Carcinoma?'] == "no", info_df['First group.1'] == "no"), 
                                        info_df['First group.1'] == "yes"), :]

    # Indexes of train and test
    indexes = info_df.index
    train_indexes = info_df.index[info_df['First group.1'] == "yes"].values
    # test_indexes = info_df.index[info_df['First group.1'] == "no"].values
    train_indexes.sort()

    return train_indexes

def get_info_df():
    #return pd.read_csv("../data/slides/HN/2_HN_dysplasia-abs_cos_v2_test_in_no_carcinoma_epoch_15_info_df.csv", index_col="file ID")
    return pd.read_csv("../data/slides/HN/EPOC full cohort info.csv", index_col="file ID")

def get_edge_type(edge, cell_type):
    """
    Args: 
        edge: (in_cell_index, out_cell_index, 1/edge_length)
        
    Returns:
        edge type index.
        0: 1-1; 1: 1-2; 2: 1-3; 3: 1-4; 
        4: 2-1; 5: 2-2; 6: 2-3; 7: 2-4; 
        8: 3-1; 9: 3-2; 10: 3-3; 11: 3-4;
        12: 4-1; 13: 4-2; 14: 4-3; 15: 4-4
    """
    mapping = {"1-1": 0, "1-2": 1, "1-3": 2, "1-4": 3, 
                   "2-1": 4, "2-2": 5, "2-3": 6, "2-4": 7, 
                   "3-1": 8, "3-2": 9, "3-3": 10, "3-4": 11, 
                   "4-1": 12, "4-2": 13, "4-3": 14, "4-4": 15}
    return mapping['{}-{}'.format(cell_type[edge[0]].item(), 
                                  cell_type[edge[1]].item())]

def get_nuclei_orientation_diff(edge, nuclei_orientation):
    return np.abs(np.cos(nuclei_orientation[edge[0]] - nuclei_orientation[edge[1]]))

class NucleiData(Data):
    """Add some attributes to Data object.
    
    Args:
        * All args mush be torch.tensor. So string is not supported.
        x: Matrix for nodes
        edge_index: 2*N matrix
        edge_attr: edge type
        y: Label
        pid: Patient ID
        time: time to cancer
        event: cancer or not
    """
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, pos=None, 
                 cell_type=None,
                 pid=None, time=None, event=None, 
                 coord_x=None, coord_y=None):
        super().__init__(x, edge_index, edge_attr, y, pos)
        self.cell_type = cell_type
        self.pid = pid
        self.time = time
        self.event = event
        self.coord_x = coord_x
        self.coord_y = coord_y
        
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

class Dataset(torch.utils.data.Dataset):
    __initialized = False
    def __init__(self, indexes, info_df, augmentation=False):
        """
        Args:
            indexes: list of info_df.index
            info_df: information
            augmentation: True for augmentation
        """
        self.indexes = indexes
        self.info_df = info_df
        self.augmentation = augmentation
        self.__initialized = True

    def __len__(self):
        """Denotes the number of samples"""
        return len(self.indexes)
    
    def __getitem__(self, index, coords=None):
        """Generate one item of data.
        """
        return self.__data_generation(index, coords)
    
    def __data_generation(self, index, coords=None):
        """Generation & augmentation
        
        Args:
            coords: for patch extraction (left upper side)
        """
        pid = index
        index = self.indexes[index]
        
        # Construct data
        cell_summary = pd.read_csv(os.path.join(nuclei_dir, "cell_summary_"+index+".csv"))
        cell_summary = cell_summary.loc[cell_summary['cell_type'] != 0, :]
        coordinates_x = cell_summary['coordinate_x']
        coordinates_y = cell_summary['coordinate_y']

        # Get patch summary
        if coords is None:
            # Random select a nuclei from 1, 2, 4
            while True:
                selected = np.random.choice(cell_summary.index[np.isin(cell_summary['cell_type'], [1, 2, 4])].values)
                coord_x_start = cell_summary.loc[selected, "coordinate_x"]
                coord_x_start -= PATCH_SIZE/2
                coord_x_end = coord_x_start + PATCH_SIZE
                coord_y_start = cell_summary.loc[selected, "coordinate_y"]
                coord_y_start -= PATCH_SIZE/2
                coord_y_end = coord_y_start + PATCH_SIZE
                expr = "(coordinates_x >= coord_x_start) & (coordinates_x <= coord_x_end) " +\
                       "& (coordinates_y >= coord_y_start) & (coordinates_y <= coord_y_end)"
                patch_summary = cell_summary[ne.evaluate(expr)]
                if np.sum(np.isin(patch_summary['cell_type'], [1, 2, 4])) > 50:
                    break
        else:
            coord_x_start = coords[0]
            coord_x_end = coord_x_start + PATCH_SIZE
            coord_y_start = coords[1]
            coord_y_end = coord_y_start + PATCH_SIZE
            expr = "(coordinates_x >= coord_x_start) & (coordinates_x <= coord_x_end) " +\
                   "& (coordinates_y >= coord_y_start) & (coordinates_y <= coord_y_end)"
            patch_summary = cell_summary[ne.evaluate(expr)]
                
        # Create 8 nearest neighbors graph
        graph = skgraph.kneighbors_graph(np.array(patch_summary.loc[:, ['coordinate_x', 'coordinate_y']]), 
                                         n_neighbors=8, mode='distance')
        I, J, V = sp.find(graph)
        edges = list(zip(I, J, 1/V))
        edge_index = np.transpose(np.array(edges)[:, 0:2])
        x = (np.array(patch_summary.loc[:, ['area', 'convex_area', 
                                           'eccentricity', 'extent', 'filled_area', 
                                            'major_axis_length', 'minor_axis_length', 'pa_ratio', 
                                            'perimeter', 'probability', 'solidity']]) - x_ave)/x_std
        cell_type = np.array(patch_summary['cell_type'])
        orientation = np.array(patch_summary['orientation'])

        # Edge features
        edge_type = list(map(lambda x: get_edge_type(x, cell_type), edges))
        nuclei_orientation = list(map(lambda x: get_nuclei_orientation_diff(x, orientation), edges))
        edge_attr = np.transpose(np.array([edge_type, nuclei_orientation, 1/V]))
        
        event = self.info_df.loc[index, "cancer (1= oral cancer or death, 0=censor)"]
        y = event
        time = self.info_df.loc[index, "TimeTocancerA"]

        # TODO: Implement class / size filter. 

        data = NucleiData(x=torch.tensor(x, dtype=torch.float), 
                          edge_index=torch.tensor(edge_index, dtype=torch.long),
                          edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                          y=torch.tensor([[y]], dtype=torch.long),
                          cell_type = torch.tensor(cell_type, dtype=torch.long),
                          pid=torch.tensor([[pid]]),
                          time=torch.tensor([[time]]), 
                          event=torch.tensor([[event]]), 
                          coord_x=torch.tensor([[coord_x_start]]), 
                          coord_y=torch.tensor([[coord_y_start]]))
        
        # Augmentation
        if self.augmentation:
            # Aug x
            # torch.seed()
            data.x += torch.randn_like(data.x) / 100
            # Aug edge_attr[:, 1] (orientation)
            # torch.seed()
            data.edge_attr[:, 1] += torch.randn_like(data.edge_attr[:, 1]) / 100
            data.edge_attr[:, 1] = torch.clamp(data.edge_attr[:, 1], -1, 1)
            # Aug edge_attr[:, 2] (1/V)
            # torch.seed()
            data.edge_attr[:, 2] += torch.randn_like(data.edge_attr[:, 2]) / 10000
            data.edge_attr[:, 2] = torch.clamp(data.edge_attr[:, 2], 0, 1)
        else:
            data.edge_attr[:, 2] = torch.clamp(data.edge_attr[:, 2], 0, 1)
            
        return data

def get_obs_loader(batch_size):
    train_indexes = read_data()
    info_df = get_info_df()

    train_set = Dataset(train_indexes, info_df, augmentation=True)

    return DataLoader(train_set, batch_size, shuffle=True)

class EdgeNN(nn.Module):
    """
    Design: embedding according to edge type, and then modulated by edge features.
    """
    def __init__(self, in_channels, out_channels, n_edge_types=16, 
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
        y = self.edge_type_embedding(x[..., 0].type(torch.long).to(self.device))
        h = self.fc_h(x[..., 1:(self.in_channels + 1)].type(torch.float).to(self.device))
        g = self.fc_g(x[..., 1:(self.in_channels + 1)].type(torch.float).to(self.device))
        y = y * h + g
        # x = self.bn(x)
        return F.relu(y, inplace=True)

class NucleiNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, batch=True, edge_features=2, n_edge_types=16, 
                 device=torch.device(0)):
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
        
        self.conv1 = NNConv(in_channels, 20, EdgeNN(edge_features, in_channels*20, n_edge_types=n_edge_types), 
                            aggr='mean', root_weight=True, bias=True)
        self.conv2 = NNConv(20, 20, EdgeNN(edge_features, 20*20, n_edge_types=n_edge_types), 
                            aggr='mean', root_weight=True, bias=True)
        self.conv3 = NNConv(20, 20, EdgeNN(edge_features, 20*20, n_edge_types=n_edge_types), 
                            aggr='mean', root_weight=True, bias=True)
        self.conv4 = NNConv(20, out_channels, EdgeNN(edge_features, 20*out_channels, n_edge_types=n_edge_types), 
                            aggr='mean', root_weight=True, bias=True)

        self.device = device
        
    def forward(self, data):
        """
        Args:
            data: Data in torch_geometric.data
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x_conv1 = self.conv1(x, edge_index, edge_attr)
        x_relu1 = F.relu(x_conv1)
        x_relu1 = F.dropout(x_relu1, training=self.training)
        x_conv2 = self.conv2(x_relu1, edge_index, edge_attr)
        x_relu2 = F.relu(x_conv2)
        x_relu2 = F.dropout(x_relu2, training=self.training)
        x_conv3 = self.conv3(x_relu2, edge_index, edge_attr)
        x_relu3 = F.relu(x_conv3)
        x_relu3 = F.dropout(x_relu3, training=self.training)
        x_conv4 = self.conv4(x_relu3, edge_index, edge_attr)
        x_relu4 = F.relu(x_conv4) # TODO: check if vector.
        '''
        if self.batch:
            x = global_max_pool(x, batch=data.batch).to(self.device)
        else:
            x = global_max_pool(x, batch=torch.tensor(np.zeros(x.shape[0]), dtype=torch.long).to(self.device))
        '''
        # cell_type = data.cell_type
        # gate = data.cell_type
        gate_1 = torch.eq(data.cell_type, 1).clone().detach().requires_grad_(False).type(torch.long).to(self.device)
        gate_2 = torch.eq(data.cell_type, 2).clone().detach().requires_grad_(False).type(torch.long).to(self.device)
        gate_4 = torch.eq(data.cell_type, 4).clone().detach().requires_grad_(False).type(torch.long).to(self.device)
        gate = gate_1 | gate_2 | gate_4
        # print(np.unique(data.batch.cpu().numpy()))  ## [0, 1, 2, ... BATCH_SIZE-1]
        # print(gate * data.batch)  ## [[0]s, [1]s, .... [BATCH_SIZE - 1]s]
        if self.batch:
            _batch_size = data.batch[-1] + 1
            x_out = scatter_mean(x_relu4, gate * (data.batch+1), dim=0).to(self.device)[1:_batch_size+1, :]  
            # Keep the batches (0 is False in gate)
            # print(x.shape)  ## [BATCH_SIZE, 2]
        else:
            x_out = scatter_mean(x_relu4, gate, dim=0).to(self.device)[1, :]
            # print(x.shape)  ## [2]
        
        return x_out
        #return F.log_softmax(x_out, dim=1)
        #return x_conv1, x_relu1, x_conv2, x_relu2, x_conv3, x_relu3, x_conv4, x_relu4, x_out

def nuclei_to_data(G: NucleiData, num_cell_types: int):
    node_matrix = nn.functional.one_hot(G.cell_type - 1, num_cell_types)
    node_matrix = torch.cat((G.x, node_matrix), dim=-1)

    edge_index = torch_geometric.utils.to_dense_adj(G.edge_index) # + 1e-8
    edge_index, edge_weights, _ = dense_to_sparse(edge_index)
    edge_index = edge_index.transpose(1, 2).squeeze(0)

    edge_attr = G.edge_attr
    # zero_pad = torch.zeros((edge_index.shape[1] - edge_attr.shape[0], 
    #                         edge_attr.shape[1])).to(edge_attr.device)
    # edge_attr = torch.cat((edge_attr, zero_pad), dim=0)
    edge_attr = torch.cat((edge_attr, edge_weights.squeeze(0) - 1e-8), dim=-1)

    return Data(node_matrix, edge_index, edge_attr)

def get_edge_type_app(A, C_x):
    edges = list(zip(A.long()[0], A.long()[1]))
    e_c = list(map(lambda x: get_edge_type(x, C_x.int()), edges))
    e_c = torch.tensor(e_c).unsqueeze(-1)

    return e_c

def assign_edge_type(A, C_x):
    with torch.no_grad():
        e_c = torch.stack(
            [get_edge_type_app(A, C_x) for (A, C_x) in 
             zip(A.unbind(), C_x.unbind())]
        )
    
    return e_c

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

def egg_to_ex(generated: dict):
    """
    Implementation of egg_to_ex for the trainer of an EggGeneric model 
    for oral ceograph data. 
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
    for oral ceograph data. 
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
    for oral ceograph data. 
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
