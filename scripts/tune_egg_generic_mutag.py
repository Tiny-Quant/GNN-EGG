# %%
import os
import sys
from os.path import dirname, abspath
repo_dir = dirname(dirname(abspath(__file__)))
sys.path.append(repo_dir)

import dill as pickle
import numpy as np
import pandas as pd
from functools import partial
import argparse

import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.nn as GCNConv

import pygmtools as pygm
import networkx as nx

import ray 
from ray import tune 
from ray.tune.schedulers import ASHAScheduler
from ray import train

from egg_models import egg_generic_losses
from egg_models.egg_generic import EggGeneric
from egg_models.egg_generic import EggGeneric
from egg_models.egg_generic_losses import GEDasMatchLoss, EdgePenalty

from utils import mutag_helper
from utils import visuals

# %%
# fix random seeds for reproducibility.
SEED = 123123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True 
np.random.seed(SEED)

# Meta-Data
device = torch.device(0)    
num_epochs = 50

# %%
# Load explainee model.
explainee = mutag_helper.GCN2(hidden_channels=64, node_features=5, num_classes=2)
explainee.to(device)
explainee.load_state_dict(torch.load(
        "data/explainees/MUTAG/gcn_200_dropped.pt", 
        map_location=device
    ), 
    strict=False
)
explainee.eval()

# %%
# Load cleaned datasets.
# See MUTAG.ipynb for code used to generate the data. 

base_path = "data/explainees/MUTAG/"

with open(base_path + "MUTAG_train_data_list_dropped.pkl", "rb") as f:
    train_data_list = pickle.load(f)

with open(os.path.join(base_path, "MUTAG_train_data_list_1_dropped.pkl"), "rb") as f:
    train_data_list_1 = pickle.load(f)

with open(base_path + "MUTAG_train_data_list_0_dropped.pkl", "rb") as f:
    train_data_list_0 = pickle.load(f)

with open(base_path + "MUTAG_test_data_list_dropped.pkl", "rb") as f:
    test_data_list = pickle.load(f)

with open(base_path + "MUTAG_test_data_list_1_dropped.pkl", "rb") as f:
    test_data_list_1 = pickle.load(f)

with open(base_path + "MUTAG_test_data_list_0_dropped.pkl", "rb") as f:
    test_data_list_0 = pickle.load(f)

with open(base_path + "MUTAG_train_avg_embedding_dict_0_dropped.pkl", 'rb') as f: 
    avg_class_embedding_0 = pickle.load(f)

with open(base_path + "MUTAG_train_avg_embedding_dict_1_dropped.pkl", 'rb') as f: 
    avg_class_embedding_1 = pickle.load(f)

#explainee_ref = ray.put(explainee)
train_data_ref = ray.put(train_data_list + test_data_list)
test_data_ref = ray.put(train_data_list_0 + test_data_list_0)

# %%
def train_function(config, target, explainee, train_data_ref, test_data_ref):
    
    train_data = train_data_ref
    test_data = test_data_ref

    generator = EggGeneric(
        max_node_size=config["max_node_size"], 
        cont_node_feats=None, 
        dis_node_feats=(5, ), # num cats.
        cont_edge_feats=None, 
        dis_edge_feats=(3, ), # num cats.
        temp=config["temp"], 
        batch_size=config["batch_size"]
    )
    generator.to(device)

    loader = pyg.loader.DataLoader(train_data,  
                                batch_size=config["batch_size"], drop_last=True)

    # L1
    pred_loss_fn = egg_generic_losses.PredLossBatched(
        target, explainee
    )

    GED_fn = egg_generic_losses.GEDasMatchLoss(
        node_size=config["max_node_size"], 
        cont_node_indices=None, 
        dis_node_indices=(slice(0, 4), ), 
        cont_edge_indices=None, 
        dis_edge_indices=(slice(0, 3), 3),
        QAP_solver=pygm.ngm, 
    )

    edge_pen_fn = EdgePenalty(
        edge_budget=config["budget"] * config["max_node_size"]
    )

    optimizer = torch.optim.RMSprop(generator.parameters(), lr=config["lr"])  
    lr_schedule = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=config["lr_decay"]
    )

    for i in range(num_epochs):
        running_loss = 0.0
        for obs_batch in loader:
            obs_batch.to(device)

            gen_graph = generator()

            # Convert to format accepted by explainee.
            gen_ex = mutag_helper.egg_to_ex(gen_graph) 

            # Convert to format accepted by L2 and L3. 
            gen_egg = mutag_helper.egg_to_egg(gen_graph)

            # Convert to format accepted by L2 and L3. 
            obs_egg = mutag_helper.ex_to_egg(obs_batch)

            # Compute L1.
            loss_1, _, _ = pred_loss_fn(gen_ex) 
            loss_1 = config["l1_weight"] * loss_1 # Lambda_1

            with torch.no_grad():
                explainee_pred = F.softmax(explainee(obs_batch), dim=-1)
                omega = (explainee_pred @ 
                    target.to(explainee_pred.device) - 0.5
                )

            if config['var_egg_size']: 
                egg_size = torch.tensor([
                    graph.x.shape[0] + graph.edge_index.shape[1] 
                    for graph in gen_ex.to_data_list()
                ]).to(device)
            else: 
                egg_size = (gen_egg[0].shape[1] + gen_egg[2].shape[1])

            # Compute L2. 
            loss_2 = GED_fn(*gen_egg, *obs_egg) / egg_size # / (gen_egg[0].shape[1] + gen_egg[2].shape[1])

            # Compute L3. 
            loss_3 = config["l2_weight"] * omega * loss_2 # Lambda_2

            edge_pen = (
                config["edge_pen_weight"] * 
                edge_pen_fn(generator.AdjacencyMatrix.probs)
            )

            loss = loss_1 + loss_3

            loss = loss.mean()

            running_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

        lr_schedule.step()

    # Mean Prediction 
    running_pred = 0
    for i in range(100):
        with torch.no_grad():
            gen_ex = mutag_helper.egg_to_ex(generator())
            pred = torch.softmax(explainee(gen_ex), dim=-1).mean(dim=0)
            running_pred += pred[target.argmax().item()].item()
    mean_pred = running_pred / 100
    
    test_loader = pyg.loader.DataLoader(
        test_data, batch_size=config["batch_size"], drop_last=True
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

        GED = GED_fn(*gen_egg, *obs_egg) / egg_size
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
        {"loss": loss.item(), 
         "mean_pred": mean_pred, 
         "mean_GED": mean_GED, 
         "mean_Density": mean_Density}
    ) 

if __name__ == '__main__':

    # Terminal Arguments
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        # from repo dir
        '--target', 
        type=int, 
        default=1
    )
    opt = parser.parse_args()

    target_selection = opt.target
    if target_selection:
        target = torch.tensor([0.0, 1.0])
    else: 
        target = torch.tensor([1.0, 0.0])

    config = {
        "max_node_size": tune.choice([5, 10, 20, 30]),
        "temp": tune.choice([0.1, 0.15, 0.2]), 
        "batch_size": tune.choice([16]), 
        "lr": tune.choice([1e-4, 1e-3]),  
        "lr_decay": tune.choice([0.1, 1.0]), 
        "l1_weight": tune.choice([0.0, 0.5, 1.0]), 
        "l2_weight": tune.choice([0.0, 0.5, 1.0]),
        "var_egg_size": tune.choice([True, False]),
        "edge_pen_weight": tune.choice([0.0, 1e-4, 1e-2]), 
        "budget": tune.choice([0.0, 1.0, 2.0])
    }

    tune_scheduler = ASHAScheduler(
        metric="mean_pred", 
        mode="max",
        max_t=10, 
    )

    # print(type(train_data_ref))
    # print(ray.get(train_data_ref))

    result = tune.run(
        tune.with_parameters(
            train_function, 
            target=target, 
            explainee=explainee, 
            train_data_ref=train_data_ref, 
            test_data_ref=test_data_ref
        ), 
        config=config,
        num_samples=100,  
        scheduler=tune_scheduler, 
        resources_per_trial={"cpu": 6, "gpu": 0.1}
    )

    df = result.results_df
    print(df)
    df_name = "MUTAG_tuned_results" + target_selection + ".csv"
    df.to_csv(df_name, index=False)
    df.to_csv("results/MUTAG/" + df_name, index=False)
