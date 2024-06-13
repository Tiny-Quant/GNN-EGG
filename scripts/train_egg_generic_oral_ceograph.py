# %% Dependencies
import typing
from typing import List, Optional, Tuple, Dict

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

from utils import oral_ceograph

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
    PATH_TO_OBS_DATA = "data/slides/HN/HN_train_data_list.pkl"
    with open(PATH_TO_OBS_DATA, 'rb') as f:
        obs_data_list = pickle.load(f)

    ############################################################################
    ## Load Explainee Model ####################################################
    ### Write any code necessary to load your explainee model here. ############
    ############################################################################
    explainee = oral_ceograph.NucleiNet(11, 2, batch=True)

    explainee.to(device)
    explainee.load_state_dict(torch.load(
            "data/explainees/HN/epoch_15.pt", 
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

    if target_class == 0: 
        target = torch.tensor([1., 0.])
        PATH_TO_CLASS_EMBEDDINGS = (
            "data/explainees/HN/HN_test_avg_embedding_dict_0.pkl"
        )
    elif target_class == 1:
        target = torch.tensor([0., 1.])
        PATH_TO_CLASS_EMBEDDINGS = (
            "data/explainees/HN/HN_test_avg_embedding_dict_1.pkl"
        )

    with open(PATH_TO_CLASS_EMBEDDINGS, 'rb') as f:
        avg_class_embedding = pickle.load(f)
    
    ############################################################################
    ## Generator Parameters ####################################################
    ############################################################################
    #MAX_NODE_SIZE = 25 # Int. 
    CONT_NODE_FEATS = 11 # Int. 
    DIS_NODE_FEATS = (4, ) # Tuple(cats, )
    CONT_EDGE_FEATS = 2 # Int.
    #DIS_EDGE_FEATS = None # Tuple(cats, )

    CONT_NODE_INDICES=(slice(0, 11), )
    DIS_NODE_INDICES=(slice(11, 16), )
    CONT_EDGE_INDICES=(slice(1, 3), )
    DIS_EDGE_INDICES=(3, )

    ############################################################################
    ## Data Processing Functions ###############################################
    ############################################################################
    class SpecificTrainer(EggGenericTrainer): 
        def __init__(self, 
                    model: EggGeneric, 
                    explainee: nn.Module, 
                    target: torch.Tensor, 
                    uninfo_target: torch.Tensor, 
                    obs_data_list: List, 
                    optimizer: torch.optim.Optimizer, 
                    loss_term_weights: torch.Tensor, 
                    tensorboard_path: str, 
                    checkpoint_path: str,
                    save_every=1, 
                    avg_embed_targets: Optional[Dict[str, torch.Tensor]]=None, 
                    cont_node_indices: Optional[Tuple]=None, 
                    dis_node_indices: Optional[Tuple]=None,
                    cont_edge_indices: Optional[Tuple]=None, 
                    dis_edge_indices: Optional[Tuple]=None, 
                    dis_imp_ratio: float=1.0, 
                    edge_budget=None, 
                    reinforce_pred=False, 
                    reinforce_struct=False,
                    sub_sampler="default", 
                    repeat_sampling=False, 
                    batches_per_param=1,
                    auto_mixed_precision=False): 
            super().__init__(
                model=model, explainee=explainee, 
                target=target, uninfo_target=uninfo_target,
                obs_data_list=obs_data_list, optimizer=optimizer, 
                loss_term_weights=loss_term_weights,
                tensorboard_path=tensorboard_path, 
                checkpoint_path=checkpoint_path, save_every=save_every,
                avg_embed_targets=avg_embed_targets, 
                cont_node_indices=cont_node_indices,
                dis_node_indices=dis_node_indices, 
                cont_edge_indices=cont_edge_indices,
                dis_edge_indices=dis_edge_indices, 
                dis_imp_ratio=dis_imp_ratio, 
                edge_budget=edge_budget, 
                reinforce_pred=reinforce_pred,reinforce_struct=reinforce_struct, 
                sub_sampler=sub_sampler, repeat_sampling=repeat_sampling,
                batches_per_param=batches_per_param, 
                auto_mixed_precision=auto_mixed_precision, 
            )

        def egg_to_ex(self, generated: dict):
            """
            """
            return oral_ceograph.egg_to_ex(generated)

        def ex_to_egg(self, obs_batch) -> List[torch.tensor]:
            """
            """
            return oral_ceograph.ex_to_egg(obs_batch, 
                                           self.model.dis_node_feats[0])

        def egg_to_egg(self, generated: dict) -> List[torch.tensor]: 
            """
            Post-processor to fix formatting for loss terms.
            """
            return oral_ceograph.egg_to_egg(generated)

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
    trainer = SpecificTrainer(
        model=generator, explainee=explainee, 
        target=target, uninfo_target=uninfo_target, 
        avg_embed_targets=avg_class_embedding, 
        loss_term_weights=torch.tensor(
            [pred_loss_weight, edge_loss_weight, struct_loss_weight]
        ), 
        obs_data_list=obs_data_list, 
        sub_sampler=sub_sampling_strat, 
        cont_node_indices=CONT_NODE_INDICES, 
        dis_node_indices=DIS_NODE_INDICES, 
        cont_edge_indices=CONT_EDGE_INDICES, 
        dis_edge_indices=DIS_EDGE_INDICES, 
        dis_imp_ratio=dis_imp_ratio, 
        optimizer=optimizer, 
        batches_per_param=batches_per_param,
        auto_mixed_precision=auto_mixed_precision, 
        tensorboard_path=tensorboard_path, checkpoint_path=checkpoint_path, 
        save_every=save_every
    )
    
    trainer.train(num_epochs, opt.resume_path, None)
    