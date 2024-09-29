# %% Dependencies
import typing
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn 
from torch.optim.optimizer import Optimizer as Optimizer
torch.autograd.set_detect_anomaly(True)

import pygmtools as pygm 

# Adds the repo directory to the import paths.
import sys
from os.path import dirname, abspath
repo_dir = dirname(dirname(abspath(__file__)))
sys.path.append(repo_dir)

import dill as pickle
import argparse
import json
import numpy as np
from functools import partial 

from egg_models.egg_generic import EggGeneric, EggGenericTrainer

from utils import mutag_helper

# fix random seeds for reproducibility
SEED = 123123
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
    parser.add_argument(
        '--target',
        type=int, 
        default=1
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
    #PATH_TO_OBS_DATA = "" # Expects a dill/pickled list of data objects.
    PATH_TO_OBS_DATA = "data/explainees/MUTAG/MUTAG_train_data_list_dropped.pkl"
    with open(PATH_TO_OBS_DATA, 'rb') as f:
        obs_data_list = pickle.load(f)

    ############################################################################
    ## Load Explainee Model ####################################################
    ### Write any code necessary to load your explainee model here. ############
    ############################################################################
    explainee = mutag_helper.GCN2(64, 5, 2)

    explainee.to(device)
    explainee.load_state_dict(torch.load(
            "data/explainees/MUTAG/gcn_200_dropped.pt", 
            map_location=device
        ), 
        strict=False
    )
    explainee.eval()

    ############################################################################
    ## Get Explainee Targets ###################################################
    ### Write any code necessary to define the targeted output of your ######### 
    ### explainee model ie what values should be return when generated graphs ## 
    ### are passed to your model? ##############################################
    ############################################################################
    uninfo_target = torch.tensor([0.5, 0.5])

    with open("/data/explainees/MUTAG/MUTAG_train_avg_embedding_dict_0_dropped.pkl", 'rb') as f:
        avg_0_embedding = pickle.load(f)

    with open("/data/explainees/MUTAG/MUTAG_train_avg_embedding_dict_1_dropped.pkl", 'rb') as f:
        avg_1_embedding = pickle.load(f)

    if not opt.target: 
        target = torch.tensor([1., 0.])
        avg_class_embedding = avg_0_embedding
        avg_embed_other_class = avg_1_embedding
        with open("/data/explainees/MUTAG/MUTAG_test_data_list_0_dropped.pkl", 'rb') as f:
            test_data = pickle.load(f)

    else:  
        target = torch.tensor([0., 1.])
        avg_class_embedding = avg_1_embedding
        avg_embed_other_class = avg_0_embedding
        with open("/data/explainees/MUTAG/MUTAG_test_data_list_1_dropped.pkl", 'rb') as f:
            test_data = pickle.load(f)
    
    ############################################################################
    ## Generator Parameters ####################################################
    ############################################################################
    #MAX_NODE_SIZE = 25 # Int. 
    CONT_NODE_FEATS = None # Int. 
    DIS_NODE_FEATS = (5, ) # Tuple(cats, )
    CONT_EDGE_FEATS = None # Int.
    DIS_EDGE_FEATS = (3, ) # Tuple(cats, )

    CONT_NODE_INDICES = None # (slice(0, 11), )
    DIS_NODE_INDICES = (slice(0, 6), )
    CONT_EDGE_INDICES = None # (slice(1, 3), ) # 0 is connection type and is deterministic. 
    DIS_EDGE_INDICES = (slice(0, 3), 3)


    generator = EggGeneric(max_node_size=MAX_NODE_SIZE, 
                           cont_node_feats=CONT_NODE_FEATS,
                           dis_node_feats=DIS_NODE_FEATS, 
                           cont_edge_feats=CONT_EDGE_FEATS, 
                           temp = temp, 
                           batch_size=batch_size)
    generator.to(device)
    generator.train()

    # Dynamically create the optimizer class
    #optimizer_class_str = "optim." + optimizer_name
    optimizer_class = getattr(torch.optim, optimizer_name, None)

    if optimizer_class is not None:
        optimizer = optimizer_class(generator.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Dynamically create QAP solver class
    if QAP_solver_name == "identity":
        QAP_solver = "identity"
    else:
        #QAP_solver_str = "pygm" + QAP_solver_name
        QAP_solver = getattr(pygm, QAP_solver_name, None)

    # Define trainer.  
    trainer = EggGenericTrainer(
        model=generator, explainee=explainee, 
        target=target, uninfo_target=uninfo_target, 
        avg_embed_targets=avg_class_embedding, 
        avg_embed_other_class=avg_embed_other_class, 
        loss_term_weights=torch.tensor(
            [pred_loss_weight, embed_other_weight, 
             edge_loss_weight, struct_loss_weight]
        ), 
        obs_data_list=obs_data_list, 
        sub_sampler=sub_sampling_strat, 
        cont_node_indices=CONT_NODE_INDICES, 
        dis_node_indices=DIS_NODE_INDICES, 
        cont_edge_indices=CONT_EDGE_INDICES, 
        dis_edge_indices=DIS_EDGE_INDICES, 
        dis_imp_ratio=dis_imp_ratio, 
        QAP_solver=QAP_solver, 
        use_egg_size=use_egg_size, 
        optimizer=optimizer, 
        batches_per_param=batches_per_param,
        auto_mixed_precision=auto_mixed_precision, 
        tensorboard_path=tensorboard_path, checkpoint_path=checkpoint_path, 
        save_every=save_every
    )
    
    ############################################################################
    ## Data Processing Functions ###############################################
    ############################################################################
    trainer.egg_to_ex = mutag_helper.egg_to_ex
    trainer.ex_to_egg = mutag_helper.ex_to_egg
    trainer.egg_to_egg = mutag_helper.egg_to_egg

    trainer.train(num_epochs, opt.resume_path, None)
    