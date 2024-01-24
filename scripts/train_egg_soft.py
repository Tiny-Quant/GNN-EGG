# %%
import torch
import torch.nn as nn 
from torch.optim import Adam
torch.autograd.set_detect_anomaly(True)

# Adds the repo directory to the import paths.
import sys
from os.path import dirname, abspath
repo_dir = dirname(dirname(abspath(__file__)))
sys.path.append(repo_dir)
import pickle
import argparse
import json

from egg_models.egg_soft import EggSoft, EggSoftTrainer
from egg_models.losses import PredLossBatched, MatchingLoss
from utils import ceograph

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

    config_path = opt.path_to_json_config
    with open(config_path) as f:
        config_data = json.load(f)

    num_epochs = config_data["num_epochs"]
    learning_rate = config_data["learning_rate"]
    optimizer_name = config_data["optimizer_name"]
    batch_size = config_data["batch_size"]
    max_nodes = config_data["max_nodes"]
    obs_node_limit = config_data["obs_node_limit"]
    samples_per_param = config_data["samples_per_param"]
    target_name = config_data["target_name"]
    lambda_1 = config_data["pred_weight"]
    lambda_2 = config_data["struct_weight"]
    lambda_3 = config_data["sparsity_weight"]
    obs_path = config_data["obs_path"]
    tensorboard_path = config_data["tensorboard_path"]
    checkpoint_path = config_data["checkpoint_path"]
    save_every = config_data["save_every"]
    profile_dir = config_data.get('profile_dir') 

    device = torch.device(0)

    if target_name == "ad":
        target = torch.tensor([1.0, 0.0]).to(device)
    else: 
        target = torch.tensor([0.0, 1.0]).to(device)

    with open(obs_path, 'rb') as f:
        obs = pickle.load(f)

    obs_loader = ceograph.get_obs_loader(obs, node_limit=obs_node_limit, 
                                         batch_size=batch_size)

    CONT_NODE_FEATS = 11
    CELL_TYPES = 6
    CONT_EDGE_FEAT = 2
    explainee = ceograph.NucleiNet(CONT_NODE_FEATS, CONT_EDGE_FEAT, batch=True)
    explainee.to(device)
    explainee.load_state_dict(torch.load(
            "data/explainees/ceograph/epoch_263.pt", # TODO: path parameter.
            map_location=device
        )
    )
    explainee.eval()

    generator = EggSoft(node_size=max_nodes, cont_node_feat=CONT_NODE_FEATS, 
                        node_types=CELL_TYPES, cont_edge_feat=CONT_EDGE_FEAT, 
                        batch_size=batch_size)

    generator.to(device) 
    generator.train()

    pred_loss_fn = PredLossBatched(target=target, criterion=nn.BCELoss(reduction='none'), 
                                   explainee=explainee)

    struct_loss_fn = MatchingLoss(node_size=max_nodes)

    # Dynamically create the optimizer class
    optimizer_class_str = "optim." + optimizer_name
    optimizer_class = getattr(torch.optim, optimizer_name, None)

    if optimizer_class is not None:
        optimizer = optimizer_class(generator.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    trainer = EggSoftTrainer(generator, explainee, obs_loader, 
                             target, pred_loss_fn, struct_loss_fn, optimizer,
                             tensorboard_path, checkpoint_path, save_every,
                             samples_per_param=samples_per_param) 

    trainer.train(num_epochs, opt.resume_path, profile_dir)
