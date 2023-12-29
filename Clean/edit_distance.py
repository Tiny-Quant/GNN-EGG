# %% Edit Distance
from typing import List

import networkx as nx
from networkx import graph_edit_distance

import torch_geometric

import numpy as np 
import multiprocessing as mp 
from multiprocessing import Pool
from functools import partial

import sys
sys.path.append("../scripts/ceograph/")
from ceograph import NucleiNet, NucleiData

mp.set_start_method('spawn', force=True)

# Helper function for converting Nuclei Data to NetworkX while retaining features.
def nuclei_to_nx(data: NucleiData) -> nx.DiGraph: 

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


def node_strict_type_match(node_dict_1, node_dict_2): 

    # Quick return false if features names don't match. 
    if not(set(node_dict_1) & set(node_dict_2)):
        return 0

    if node_dict_1['node_features'][0] == node_dict_2['node_features'][0]:
        return 1

    else: 
        return 0 

def single_edit_distance(ob: nx.DiGraph, G: nx.DiGraph,
                         node_match=node_strict_type_match):
    dist = graph_edit_distance(
        ob, G, 
        node_match=node_match, 
        node_del_cost=lambda x: 0, 
        edge_del_cost=lambda x: 0,
        upper_bound=50, 
        timeout=60
    )

    if dist is None: 
        return torch.tensor([50.0])
    else:
        return torch.tensor([dist])

def list_edit_distance(G: nx.DiGraph, obs: List[nx.DiGraph], 
                       dist_fn=single_edit_distance):
    with Pool() as pool: 
        distances = pool.map(partial(dist_fn, G=G), obs)
    
    return torch.stack(distances).mean()
