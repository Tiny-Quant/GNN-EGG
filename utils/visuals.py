import sys
import os 

import numpy as np 
import numexpr as ne
import pandas as pd
import torch_geometric as pyg
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utils.segmentation_functions import extract_patch_by_location

def color_cell_graph(graph, seed=100):
    network = pyg.utils.to_networkx(graph)

    cell_type_data = graph.cell_type.cpu().detach().numpy()
    edge_type_data = graph.edge_attr[:,0].cpu().detach().numpy()

    mycolors = np.array(["lime", "red", "blue", "magenta", "yellow", "cyan"])
    mycolor = mycolors[cell_type_data - 1]

    pos = nx.spring_layout(network, seed=100)

    plt.figure(figsize=(10, 10))
    nx.draw_networkx(network, pos = pos, with_labels=False, 
        node_color=mycolor, node_size=25)

def masked_cell_graph(graph, node_mask, edge_mask, seed=100):
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cmap = cm.RdBu
    network = pyg.utils.to_networkx(graph)

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

def get_patch_positional_info(cell_summary_path: str, 
                              patch_size: int, coords):
    """
    """

    cell_summary = pd.read_csv(cell_summary_path)   
    cell_summary = cell_summary.loc[cell_summary['cell_type'] != 0, :]
    coordinates_x = cell_summary['coordinate_x']
    coordinates_y = cell_summary['coordinate_y']
    coord_x_start = coords[0]
    coord_x_end = coord_x_start + patch_size
    coord_y_start = coords[1]
    coord_y_end = coord_y_start + patch_size
    expr = (
        "(coordinates_x >= coord_x_start) & (coordinates_x <= coord_x_end) " + 
        "& (coordinates_y >= coord_y_start) & (coordinates_y <= coord_y_end)"
    )
    patch_summary = cell_summary[ne.evaluate(expr)]

    coordinate_x = patch_summary['coordinate_x'] - coords[0]
    coordinate_y = patch_summary['coordinate_y'] - coords[1]

    return [patch_summary, coordinate_x, coordinate_y]

def get_node_pos(coordinate_x, coordinate_y):
    pos = {}
    for i in range(coordinate_x.shape[0]):
        pos[i] = [coordinate_x.values[i], 
                    coordinate_y.values[i]]
    return pos

def plot_slide_background(slide_image_path, patch_size, coords):
    slide_file = os.path.join(slide_image_path)
    image = extract_patch_by_location(slide_file, 
                                    location=np.array(coords, dtype=int),
                                    patch_size=(patch_size, patch_size))
    image = np.array(image)[..., :3]

    return image

def vis_cell_graph_and_slide(cell_summary_path: str, 
                             slide_image_path: str, 
                             graph_data_obj, 
                             patch_size: int,
                             color_palette, ax=None):
    """
    """
    coords = [graph_data_obj.coord_x.item(), graph_data_obj.coord_y.item()]

    patch_summary, coordinate_x, coordinate_y = get_patch_positional_info(
        cell_summary_path, patch_size, coords
    )

    f = plt.figure(figsize=(8, 8))

    if slide_image_path is not None: 
        image = plot_slide_background(slide_image_path, patch_size, coords)
    else:
        image = None

    cell_colors = color_palette[patch_summary['cell_type'] - 1]

    pos = {}
    for i in range(patch_summary.shape[0]):
        pos[i] = [coordinate_x.values[i], 
                    coordinate_y.values[i]]

    if ax is None: 
        graph_plot = nx.draw(pyg.utils.to_networkx(graph_data_obj), pos=pos, 
                             with_labels=False, 
                             node_color=cell_colors, node_size=10)
    else:
        graph_plot = nx.draw(pyg.utils.to_networkx(graph_data_obj), pos=pos, 
                             with_labels=False, 
                             node_color=cell_colors, node_size=10, ax=ax)

    return graph_plot, image

def vis_connected_graph(
        obs_graph_1,
        cell_summary_path_1, 
        slide_image_path_1, 
        obs_graph_2,
        cell_summary_path_2, 
        slide_image_path_2, 
        dis_match_mat1, 
        dis_match_mat2, 
    ):
    """

    """

    fig, axes = plt.subplots(4, 1, figsize=(8, 32))



    nx.draw(pyg.utils.to_networkx(obs_graph_1), )

    return None