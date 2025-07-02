from typing import List

import torch 
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F

import torch_geometric as pyg
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import remove_isolated_nodes, remove_self_loops

import pygmtools
from pygmtools.utils import dense_to_sparse, build_batch
pygmtools.set_backend('pytorch')

from utils import misc

class GCN1(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(7, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

class GCN2(torch.nn.Module):
    def __init__(self, hidden_channels, node_features, num_classes, 
                 num_layers=3, dropout=0):
        super().__init__()
        self.conv = pyg.nn.GCN(in_channels=node_features,
                               hidden_channels=hidden_channels,
                               num_layers=num_layers,
                               act=nn.LeakyReLU(inplace=True),
                               dropout=dropout)
        self.drop = nn.Dropout(p=dropout)
        self.lin = pyg.nn.Linear(hidden_channels*2, hidden_channels)
        self.out = pyg.nn.Linear(hidden_channels, num_classes)

    def forward(self, batch):

        h = self.conv(batch.x, batch.edge_index)

        embeds = torch.cat([
            pyg.nn.global_add_pool(h, batch=batch.batch),
            pyg.nn.global_mean_pool(h, batch=batch.batch),
        ], dim=1)

        h = self.drop(embeds)
        h = self.lin(h)
        h = h.relu()
        h = self.out(h)

        return h

# def clean_gen_graph(gen: pyg.data.Data) -> pyg.data.Data: 
#     """
#     Removes self loops and isolated nodes. 
#     """
#     edge_index = gen.edge_index
#     edge_attr = gen.edge_attr

#     edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

#     edge_index, edge_attr, mask = (
#         remove_isolated_nodes(edge_index, edge_attr, num_nodes=gen.x.shape[0])
#     )

#     # Zero node features if all cleared. 
#     if gen.x[mask].shape[0] == 0:
#         x = gen.x * 0.0
#     else: 
#         x = gen.x[mask]

#     gen_cleaned = pyg.data.Data(
#         x = x, 
#         edge_index = edge_index, 
#         edge_attr = edge_attr,
#     )

#     return gen_cleaned

def clean_gen_graph(gen: pyg.data.Data) -> pyg.data.Data: 
    """
    Removes self-loops and isolated nodes from a PyG gen object.
    Adjusts node features, edge index, edge attributes, and any other
    node-level attributes like 'pos' or 'batch'.
    """
    # Step 1: Remove self-loops
    edge_index, edge_attr = remove_self_loops(gen.edge_index, gen.edge_attr)

    # Step 2: Remove isolated nodes
    edge_index, edge_attr, node_mask = remove_isolated_nodes(
        edge_index, edge_attr, num_nodes=gen.num_nodes
    )

    # Step 3: Handle all-isolated case (empty graph)
    if node_mask.sum() == 0:
        x = torch.zeros_like(gen.x)
        new_gen = pyg.data.Data(
            x=x, 
            edge_index=torch.empty((2, 0), 
            dtype=torch.long, device=x.device)
        )
        if edge_attr is not None:
            new_gen.edge_attr = edge_attr.new_empty((0,))
        return new_gen

    # Step 4: Reindex node indices in edge_index
    new_index_map = -torch.ones(
        gen.num_nodes, dtype=torch.long, device=edge_index.device
    )
    new_index_map[node_mask] = torch.arange(
        node_mask.sum(), device=edge_index.device
    )
    edge_index = new_index_map[edge_index]

    # Step 5: Build cleaned gen
    new_gen = pyg.data.Data()
    new_gen.edge_index = edge_index
    new_gen.edge_attr = edge_attr

    # Copy and filter node-level attributes
    for key, value in gen.items():
        if key in ['edge_index', 'edge_attr']:
            continue
        if isinstance(value, torch.Tensor) and value.size(0) == gen.num_nodes:
            new_gen[key] = value[node_mask]
        else:
            new_gen[key] = value  # global attributes or unrelated

    return new_gen

def egg_to_ex(generated: dict):
    """
    Implementation of egg_to_ex for the trainer of an EggGeneric model 
    for MUTAG data. 
    """

    with torch.no_grad():
        A_hard = (generated['adjacency_matrix'] >= 0.5)
        edge_index = dense_to_sparse(A_hard)[0].transpose(1, 2)

    X = generated['dis_node_feats']

    E = generated['dis_edge_feats'].narrow(dim=1, start=0, 
                                           length=edge_index.shape[2])

    data_list = [
        clean_gen_graph(pyg.data.Data(x = X, edge_index = A, edge_attr = E))
        for (X, A, E) in zip(X.unbind(), edge_index.unbind(), E.unbind())
    ]

    return pyg.data.Batch.from_data_list(data_list)

def edge_relaxer(graph: pyg.data.Data) -> pyg.data.Data: 
    """
    Coverts a sparse adjacency matrix to be fully connected and converts 
    the original edge indices to edge weights in the edge feature matrix. 
    """
    
    edge_index = pyg.utils.to_dense_adj(graph.edge_index)
    edge_index, edge_weights, _ = dense_to_sparse(edge_index)
    edge_index = edge_index.transpose(1, 2).squeeze(0)
    graph.edge_index = edge_index

    edge_attr = graph.edge_attr
    # zero_pad = torch.zeros((edge_index.shape[1] - edge_attr.shape[0], 
    #                         edge_attr.shape[1])).to(edge_attr.device)
    # edge_attr = torch.cat((edge_attr, zero_pad), dim=0)
    #edge_attr = torch.cat((edge_attr, edge_weights.squeeze(0)), dim=-1)
    edge_attr = misc.concat_possible_none_tensors(
        edge_attr, edge_weights.squeeze(0), dim=-1
    )
    graph.edge_attr = edge_attr

    return graph

def ex_to_egg(obs_batch: pyg.data.Batch) -> List[torch.tensor]:
    """
    Implementation of ex_to_egg for the trainer of an EggGeneric model 
    for MUTAG data. 
    """
    
    data_list = [edge_relaxer(graph) for graph in obs_batch.to_data_list()] 
    
    X_list = [graph.x for graph in data_list]
    A_list = [graph.edge_index for graph in data_list]
    E_list = [graph.edge_attr for graph in data_list]

    return [build_batch(X_list), build_batch(A_list), build_batch(E_list)]

def egg_to_egg(generated: dict) -> List[torch.tensor]:
    """
    Default method for post-processing the generated output for comparison 
    with the transformed training data. 
    """

    gen_X = misc.concat_possible_none_tensors(
            generated['cont_node_feats'], 
            generated['dis_node_feats'], 
            dim=-1
    )

    gen_A = generated['full_edge_indices']

    gen_E = misc.concat_possible_none_tensors(
            generated['cont_edge_feats'], 
            generated['dis_edge_feats'], 
            dim=-1
    )

    gen_E = misc.concat_possible_none_tensors(
        gen_E, generated['edge_weights'], 
        dim=-1
    )

    return [gen_X, gen_A, gen_E]




