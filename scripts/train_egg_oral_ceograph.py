

import torch
import torch.nn as nn 
from torch.optim import Adam
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils.data import DataLoader
torch.autograd.set_detect_anomaly(True)

# Adds the repo directory to the import paths.
import sys
from os.path import dirname, abspath
repo_dir = dirname(dirname(abspath(__file__)))
sys.path.append(repo_dir)
import pickle
import argparse
import json
import numpy as np

from egg_models.egg_generic import EggGeneric, EggGenericTrainer
# from egg_models.egg_soft import EggSoft, EggSoftTrainer
# from egg_models.losses import PredLossBatched, MatchingLoss
from utils import ceograph

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
    MAX_NODE_SIZE = 25
    CONT_NODE_FEATS = 11
    DIS_NODE_FEATS = (6)
    CONT_EDGE_FEATS = 2
    # DIS_EDGE_FEATS = None

    class SpecificTrainer(EggGenericTrainer): 
        def __init__(self, 
                     model: EggGeneric, 
                     obs_loader: DataLoader, 
                     optimizer: Optimizer, 
                     tensorboard_path: str, 
                     checkpoint_path: str, 
                     save_every=1, 
                     batches_per_param=1):
            super().__init__(model, 
                             obs_loader, 
                             optimizer, 
                             tensorboard_path, 
                             checkpoint_path, 
                             save_every, 
                             batches_per_param)
        
        def egg_to_ex(self, generated: dict):
            return super().egg_to_ex(generated)

        def ex_to_egg(self):
            return super().ex_to_egg()

        def compute_loss_terms(self) -> torch.tensor:
            return super().compute_loss_terms()

    # Load explainee. 
    explainee = ceograph.NucleiNet(CONT_NODE_FEATS, CONT_EDGE_FEATS, batch=True)

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
    trainer = SpecificTrainer(generator, explainee, obs_loader, 
                              optimizer, tensorboard_path, checkpoint_path, 
                              save_every=1, batches_per_param=1)
    
    trainer.train(num_epochs, opt.resume_path, None)

    