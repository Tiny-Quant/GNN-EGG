import typing
from typing import List

import torch
import torch.nn as nn 
from torch.optim import Adam
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils.data import DataLoader
torch.autograd.set_detect_anomaly(True)

import torch_geometric as pyg
from torch_geometric.data import Batch

# Adds the repo directory to the import paths.
import sys
from os.path import dirname, abspath
repo_dir = dirname(dirname(abspath(__file__)))
sys.path.append(repo_dir)
import dill as pickle
import argparse
import json
import numpy as np

import pygmtools as pygm
from pygmtools.utils import build_batch, dense_to_sparse
pygm.set_backend('pytorch')

from egg_models.egg_generic import EggGeneric, EggGenericTrainer
# from egg_models.egg_soft import EggSoft, EggSoftTrainer
# from egg_models.losses import PredLossBatched, MatchingLoss
from utils import oral_ceograph
from utils.oral_ceograph import NucleiData, assign_edge_type

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True 
np.random.seed(SEED)

# %%
if __name__ == '__main__':
    
    # Terminal Arguments
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        # from repo dir
        '--path_to_json_config', 
        type=str, 
        default="config.json"
    )
    parser.add_argument(
        '--resume_path', 
        type=str
    )
    opt = parser.parse_args()

    # Training level arguments.
    config_path = opt.path_to_json_config
    with open(config_path) as f:
        config_data = json.load(f)
        
    for key in config_data.keys():
        value = config_data.get(key)
        
        globals()[key] = value

    device = torch.device(0)

    # Model Level Parameters
    PATH_TO_OBS_DATA = ""

    with open(PATH_TO_OBS_DATA) as f:
        obs_data_list = pickle.load(f)

    MAX_NODE_SIZE = 25
    CONT_NODE_FEATS = 11
    DIS_NODE_FEATS = (4, )
    CONT_EDGE_FEATS = 2
    # DIS_EDGE_FEATS = None

    class SpecificTrainer(EggGenericTrainer): 
        def __init__(self, 
                    model: EggGeneric, 
                    obs_data_list: list, 
                    optimizer: torch.optim.Optimizer, 
                    loss_term_weights: torch.tensor, 
                    tensorboard_path: str, 
                    checkpoint_path: str, 
                    save_every=1, 
                    sub_sampler="default", 
                    repeat_sampling=False, 
                    batches_per_param=1):

            super().__init__(model, optimizer, 
                            tensorboard_path, checkpoint_path, save_every)

            self.batch_size = self.model.batch_size

            self.obs_data_list = obs_data_list
            self.loss_term_weights = loss_term_weights
            self.batches_per_param = batches_per_param
            self.sub_sampler = sub_sampler
            self.repeat_sampling = repeat_sampling

        def egg_to_ex(self, generated: dict):
            with torch.no_grad():
                C_x_hard = torch.argmax(
                    generated['dis_node_feats'], dim=-1
                ) + 1

                A_hard = (generated['adjacency_matrix'] >= 0.5)
                edge_index = dense_to_sparse(A_hard)[0].transpose(1, 2)
                
                edge_types = assign_edge_type(edge_index, C_x_hard) 

                E_hard = (generated['cont_edge_feats'].
                            narrow(dim=1, start=0, length=edge_index.shape[2])
                )
                
                edge_attr = torch.cat((edge_types, E_hard), dim=-1)

            X = generated['cont_node_feats']

            nuclei_list = [NucleiData(x=X, cell_type=C_x, edge_index=A, edge_attr=E)
                        for (X, C_x, A, E) in zip(
                            X.unbind(), C_x_hard.unbind(), 
                            edge_index.unbind(), edge_attr.unbind()
                        )
            ]

            # TODO: Remove isolated nodes. 
            # TODO: Remove self loops.

            nuclei_batch = Batch.from_data_list(nuclei_list)

            return nuclei_batch

    def ex_to_egg(self, obs_batch) -> List[torch.tensor]:

        data_list = [oral_ceograph.nuclei_to_data(graph, DIS_NODE_FEATS[0])
                    for graph in obs_batch.to_data_list()
        ]

        X_list = [graph.x for graph in data_list]
        A_list = [graph.edge_index for graph in data_list]
        E_list = [graph.edge_attr for graph in data_list]

        return [build_batch(X_list), build_batch(A_list), build_batch(E_list)]
        
    def egg_to_egg(self, generated: dict) -> List[torch.tensor]: 
        """
        Post-processor to fix formatting for loss terms.
        """

        with torch.no_grad():
            C_x_hard = torch.argmax(
                generated['dis_node_feats'], dim=-1
            ) + 1

            A = generated['adjacency_matrix']
            edge_index = dense_to_sparse(A)[0].transpose(1, 2)
            
            edge_types = assign_edge_type(edge_index, C_x_hard) 

        gen_X = torch.cat([
            generated['cont_node_feats'], 
            generated['dis_node_feats']
        ], dim=-1)

        gen_A = generated['full_edge_indices']

        gen_E = torch.cat([
        edge_types.to(gen_A.device), 
        generated['cont_edge_feats'], 
        generated['edge_weights']
        ], dim=-1)

        return [gen_X, gen_A, gen_E]

    # Load explainee. 
    explainee = oral_ceograph.NucleiNet(CONT_NODE_FEATS, CONT_EDGE_FEATS, 
                                        batch=True)

    explainee.to(device)
    explainee.load_state_dict(torch.load(
            "",
            map_location=device
        )
    )
    explainee.eval()

    # Get explainee target. 

    # Get explainee DataLoader. 

    generator = EggGeneric(max_node_size=MAX_NODE_SIZE, 
                           cont_node_feats=CONT_NODE_FEATS,
                           dis_node_feats=DIS_NODE_FEATS, 
                           cont_edge_feats=CONT_EDGE_FEATS, 
                           temp = temp, 
                           batch_size=batch_size)
    generator.to(device)
    generator.train()

    # Dynamically create the optimizer class
    optimizer_class_str = "optim." + optimizer_name
    optimizer_class = getattr(torch.optim, optimizer_name, None)

    if optimizer_class is not None:
        optimizer = optimizer_class(generator.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Define trainer.  
    trainer = SpecificTrainer(generator, explainee, obs_data_list, 
                              optimizer, tensorboard_path, checkpoint_path, 
                              save_every=1, batches_per_param=1)
    
    trainer.train(num_epochs, opt.resume_path, None)

    