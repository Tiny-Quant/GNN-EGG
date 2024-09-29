# %% Dependencies
import typing
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn 
from torch.optim.optimizer import Optimizer as Optimizer
torch.autograd.set_detect_anomaly(True)

import ray 
from ray import tune 
from ray.tune.schedulers import ASHAScheduler
from ray import train

import torch_geometric as pyg
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

    target = opt.target

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

    with open("./data/explainees/MUTAG/MUTAG_train_avg_embedding_dict_0_dropped.pkl", 'rb') as f:
        avg_0_embedding = pickle.load(f)

    with open("./data/explainees/MUTAG/MUTAG_train_avg_embedding_dict_1_dropped.pkl", 'rb') as f:
        avg_1_embedding = pickle.load(f)

    if not target:
        target = torch.tensor([1., 0.])
        avg_class_embedding = avg_0_embedding
        avg_embed_other_class = avg_1_embedding
        with open("./data/explainees/MUTAG/MUTAG_test_data_list_0_dropped.pkl", 'rb') as f:
            test_data = pickle.load(f)

    else: 
        target = torch.tensor([0., 1.])
        avg_class_embedding = avg_1_embedding
        avg_embed_other_class = avg_0_embedding
        with open("./data/explainees/MUTAG/MUTAG_test_data_list_1_dropped.pkl", 'rb') as f:
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
    DIS_NODE_INDICES = (slice(0, 5), )
    CONT_EDGE_INDICES = None # (slice(1, 3), ) # 0 is connection type and is deterministic. 
    DIS_EDGE_INDICES = (slice(0, 3), 3)

    # Tunable hyper parameters

    config = {
        "max_node_size": tune.choice([5, 10, 20, 30]),
        "temp": tune.choice([0.1, 0.15, 0.2]), 
        "lr": tune.choice([1e-4, 1e-3, 1e-2]),  
        "l1_weight": tune.choice([0.0, 0.5, 1.0]),
        "l2_weight": tune.choice([0.0, 0.5, 1.0]),
        "l3_weight": tune.choice([0.0]),
        "l4_weight": tune.choice([0.0, 0.5, 1.0]),
        "use_egg_size": tune.choice([True, False])
    }

    train_data_ref = ray.put(obs_data_list)
    test_data_ref = ray.put(test_data)

    def train_function(config, train_data_ref, test_data_ref):

        generator = EggGeneric(max_node_size=config['max_node_size'], 
                            cont_node_feats=CONT_NODE_FEATS,
                            dis_node_feats=DIS_NODE_FEATS, 
                            cont_edge_feats=CONT_EDGE_FEATS, 
                            dis_edge_feats=DIS_EDGE_FEATS, 
                            temp = config['temp'], 
                            batch_size=16)
        generator.to(device)
        generator.train()

        optimizer = torch.optim.RMSprop(generator.parameters(), lr=config['lr'])

        # Define trainer.  
        trainer = EggGenericTrainer(
            model=generator, explainee=explainee, 
            target=target, uninfo_target=uninfo_target, 
            avg_embed_targets=avg_class_embedding, 
            avg_embed_other_class=avg_embed_other_class, 
            loss_term_weights=torch.tensor(
                [config['l1_weight'], config['l2_weight'],
                config['l3_weight'],config['l4_weight']]
            ), 
            obs_data_list=train_data_ref, 
            cont_node_indices=CONT_NODE_INDICES, 
            dis_node_indices=DIS_NODE_INDICES, 
            cont_edge_indices=CONT_EDGE_INDICES, 
            dis_edge_indices=DIS_EDGE_INDICES, 
            QAP_solver=pygm.ngm, 
            use_egg_size=config['use_egg_size'], 
            optimizer=optimizer, 
        )
        
        ############################################################################
        ## Data Processing Functions ###############################################
        ############################################################################
        trainer.egg_to_ex = mutag_helper.egg_to_ex
        trainer.ex_to_egg = mutag_helper.ex_to_egg
        trainer.egg_to_egg = mutag_helper.egg_to_egg

        trainer.train(25, opt.resume_path, None)
    
        # Mean Prediction 
        running_pred = 0
        for i in range(100):
            with torch.no_grad():
                gen_ex = mutag_helper.egg_to_ex(trainer.model())
                pred = torch.softmax(explainee(gen_ex), dim=-1).mean(dim=0)
                running_pred += pred[target.argmax().item()].item()
        mean_pred = running_pred / 100
        
        test_loader = pyg.loader.DataLoader(
            test_data_ref, 16, drop_last=True
        )

        # Mean Edit
        running_GED = 0
        running_Density = 0
        for batch in test_loader:
            batch.to(device)
            generated = generator()
            gen_ex = mutag_helper.egg_to_ex(generated)
            gen_egg = mutag_helper.egg_to_egg(generated)
            obs_egg = mutag_helper.ex_to_egg(batch)

            GED = trainer.GED_fn(*gen_egg, *obs_egg) / egg_size
            running_GED += GED.mean().item()

            egg_size = torch.tensor([
                graph.x.shape[0] + graph.edge_index.shape[1] 
                for graph in gen_ex.to_data_list()
            ]).to(device)
            max_size = (gen_egg[0].shape[1] + gen_egg[2].shape[1])
            running_Density += (egg_size / max_size).mean().item()

        mean_GED = running_GED / len(test_loader)    
        mean_Density = running_Density / len(test_loader)

        train.report(
            {"mean_pred": mean_pred, 
             "mean_GED": mean_GED, 
             "mean_Density": mean_Density, 
             "tune_metric": mean_pred + mean_GED
            }
        ) 
    

    tune_scheduler = ASHAScheduler(
        metric="tune_metric", 
        mode="max",
        max_t=10, 
    )

    result = tune.run(
        tune.with_parameters(
            train_function, 
            train_data_ref=train_data_ref, 
            test_data_ref=test_data_ref,
        ), 
        config=config,
        num_samples=100,  
        scheduler=tune_scheduler, 
        resources_per_trial={"cpu": 6, "gpu": 0.1}
    )

    df = result.results_df
    df_name = "MUTAG_tuned_results_dropped" + target + ".csv"
    df.to_csv("results/MUTAG/" + df_name, index=False)
        