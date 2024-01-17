import sys

import numpy as np 
import torch_geometric
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# import ceograph

def color_cell_graph(graph, seed=100):
    network = torch_geometric.utils.to_networkx(graph)

    cell_type_data = graph.cell_type.cpu().numpy()
    edge_type_data = graph.edge_attr[:,0].cpu().numpy()

    mycolors = np.array(["lime", "red", "blue", "magenta", "yellow", "cyan"])
    mycolor = mycolors[cell_type_data - 1]

    pos = nx.spring_layout(network, seed=100)

    plt.figure(figsize=(10, 10))
    nx.draw_networkx(network, pos = pos, with_labels=False, 
        node_color=mycolor, node_size=25)

def masked_cell_graph(graph, node_mask, edge_mask, seed=100):
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cmap = cm.RdBu
    network = torch_geometric.utils.to_networkx(graph)

    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    node_mask = node_mask.cpu().numpy()
    edge_mask = edge_mask.cpu().numpy()
    mycolor2 = m.to_rgba(node_mask)
    edge_color = m.to_rgba(edge_mask)

    pos = nx.spring_layout(network, seed=100)

    plt.figure(figsize=(10, 10))
    plt.colorbar(m)
    nx.draw_networkx(network, pos = pos, with_labels=False, 
        node_color=mycolor2, node_size=25, edge_color=edge_color)


    
