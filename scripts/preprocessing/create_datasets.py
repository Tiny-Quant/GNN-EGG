# Imports
import sys 
import os 
import re

import numpy as np
import pandas as pd
from scipy import sparse as sp

import torch 
import sklearn.neighbors as skgraph

sys.path.append("../ceograph/")
from ceograph import NucleiData, get_edge_type, get_nuclei_orientation_diff 

# Raw data file paths.
nlst_data_dir = "/project/DPDS/Xiao_lab/shared/deep_learning_SW_RR/nuclei_segmentation/output/NLST_ADC_100_patches_updated/"
tcga_luad_data_dir = "/project/DPDS/Xiao_lab/shared/deep_learning_SW_RR/nuclei_segmentation/output/TCGA_LUAD_100_patches_updated/"                 
tcga_lusc_data_dir = "/project/DPDS/Xiao_lab/shared/deep_learning_SW_RR/nuclei_segmentation/output/TCGA_LUSC_100_patches/" 

# Centering and scaling factors. 
x_ave = np.array([2.62e+02, 2.76e+02, # area, convex_area, 
                   7.44e-01, 7.02e-01, 2.62e+02, 2.32e+01,  # eccentricity, extent, filled_area, major_axis_length
                   1.32e+01, 1.54e+01, 5.97e+01, 8.63e-01,  # minor_axis_length, pa_ratio, perimeter, probability
                   9.52e-01])  # solidity

x_std = np.array([2.34e+02, 2.49e+02,  # area, convex_area,
                   1.61e-01, 1.04e-01, 2.34e+02, 1.13e+01,  # eccentricity, extent, filled_area, major_axis_length
                   5.31e+00, 4.07e+00, 2.76e+01, 7.25e-02,  # minor_axis_length, pa_ratio, perimeter, probability
                   3.78e-02])   # solidity

tcga_luad_slide_names = os.listdir(tcga_luad_data_dir)
tcga_luad_dataset = []
for i, slide_name in enumerate(tcga_luad_slide_names):
    slide_dir = os.path.join(tcga_luad_data_dir, slide_name)
    cell_summary_file = [_ for _ in os.listdir(slide_dir)
                        if re.search("cell_summary_(20|40)X_100patches.csv", _) is not None]
    if not len(cell_summary_file):
        print("cell summary file not found")
        continue
    
    cell_summary_file = cell_summary_file[0]
    if os.path.exists(os.path.join(tcga_luad_data_dir, slide_name, cell_summary_file)):
        print(i, slide_name)
        cell_summary = pd.read_csv(os.path.join(tcga_luad_data_dir, slide_name, cell_summary_file))
        # patch_coords = pd.read_csv(os.path.join(tcga_luad_data_dir, slide_name, "patch_summary_40X_100patches.csv"))
        
        for region_id in range(100):
            try:
                patch_summary = cell_summary[cell_summary['n_patch'] == region_id]
                # coord_x, coord_y = patch_coords.loc[region_id, ["coordinate_x", "coordinate_y"]].values
                if sum(patch_summary['cell_type'] == 1) < 20:
                    # Only process the patches with >= 20 tumor cells
                    continue
                    # raise Exception("Too few tumor cells.")
                
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
                
                data = NucleiData(x=torch.tensor(x, dtype=torch.float), 
                                  cell_type=torch.tensor(cell_type, dtype=torch.long),
                                  edge_index=torch.tensor(edge_index, dtype=torch.long),
                                  edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                                  y=torch.tensor([[0]]),
                                  pid=torch.tensor([[i]]),
                                  region_id=torch.tensor([[region_id]]))
                tcga_luad_dataset.append(data)
            except Exception as e:
                print(e)
                continue
    break

torch.save(tcga_luad_dataset, "../../data/slides/LUDA/TCGA_LUAD_8neighbors_100regions_1024_1024_updated_2.pt")

