# %% Dependencies
import typing
from typing import List

import torch
import torch.nn as nn 
from torch.optim.optimizer import Optimizer as Optimizer
torch.autograd.set_detect_anomaly(True)

# Adds the repo directory to the import paths.
import sys
from os.path import dirname, abspath
repo_dir = dirname(dirname(abspath(__file__)))
sys.path.append(repo_dir)

import dill as pickle
import argparse
import json
import numpy as np

from egg_models.egg_generic import EggGeneric, EggGenericTrainer

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

    ############################################################################
    ## config.json #############################################################
    ### Any variables that you want to be easily changeable from run to run ####
    ### should be placed in config.json. #######################################
    ############################################################################

    # Loads arguments from config.json file.
    config_path = opt.path_to_json_config
    with open(config_path) as f:
        config_data = json.load(f)
        
    for key in config_data.keys():
        value = config_data.get(key)
        
        globals()[key] = value

    # Sets the torch device. #TODO: Make this more generic. 
    device = torch.device(0)

    ############################################################################
    ## Observed Data ###########################################################
    ### Write any code necessary to load your observed data samples here. ######
    ############################################################################
    PATH_TO_OBS_DATA = "" # Expects a dill/pickled list of data objects.
    with open(PATH_TO_OBS_DATA) as f:
        obs_data_list = pickle.load(f)

    ############################################################################
    ## Load Explainee Model ####################################################
    ### Write any code necessary to load your explainee model here. ############
    ############################################################################
    explainee = None 

    explainee.to(device)
    explainee.load_state_dict(torch.load(
            "path/to/explainee.pt", 
            map_location=device
        )
    )
    explainee.eval()
    

    ############################################################################
    ## Get Explainee Targets ###################################################
    ### Write any code necessary to define the targeted output of your ######### 
    ### explainee model ie what values should be return when generated graphs ## 
    ### are passed to your model? ##############################################
    ############################################################################
    target = None
    avg_class_embedding = None

    def get_explainee_embedding(): 
        """

        """
        pass
    
    ############################################################################
    ## Generator Parameters ####################################################
    ############################################################################
    MAX_NODE_SIZE = None # Int. 
    CONT_NODE_FEATS = None # Int. 
    DIS_NODE_FEATS = None # Tuple(cats, )
    CONT_EDGE_FEATS = None # Int.
    DIS_EDGE_FEATS = None # Tuple(cats, )

    ############################################################################
    ## Data Processing Functions ###############################################
    ############################################################################
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
            """
            """
            pass 

        def ex_to_egg(self, obs_batch) -> List[torch.tensor]:
            """
            """
            pass 

        def egg_to_egg(self, generated: dict) -> List[torch.tensor]: 
            """
            Post-processor to fix formatting for loss terms.
            """

    ############################################################################
    ## No changes necessary below. #############################################
    ############################################################################
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
    