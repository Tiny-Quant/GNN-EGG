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
from egg_models.egg_generic_losses import (
    GEDasMatchLoss, activation_hook, dict_cos_dist
)

from utils import mutag_helper
from utils import visuals

# %%
# fix random seeds for reproducibility.
SEED = 123123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True 
np.random.seed(SEED)

# %% Helper Functions
def sample_graph_dist(n, min_nodes, max_nodes, 
                      cont_node_loc, cont_edge_loc, 
                      cont_node_sd, cont_edge_sd,
                      prob_X_d, prob_E_d, prob_A, 
                      cluster_lab):
    samples = []

    # Define feature distributions. 

    X_c_dist = td.Normal(cont_node_loc, cont_node_sd)
    E_c_dist = td.Normal(cont_edge_loc, cont_edge_sd) 
    X_d_dist = td.Bernoulli(prob_X_d)
    E_d_dist = td.Bernoulli(prob_E_d)
    A_dist = td.Bernoulli(prob_A)

    for i in range(n):
        # Sample the number of nodes from a discrete uniform.
        num_nodes = torch.randint(min_nodes, max_nodes, (1, )).item()

        # Sample the adjacency matrix.
        A = A_dist.sample(torch.Size([num_nodes, num_nodes]))
        A, _ = pyg.utils.dense_to_sparse(A)
        num_edges = A.shape[1]

        # Sample continuous features.
        X_c = X_c_dist.sample(torch.Size([num_nodes]))
        E_c = E_c_dist.sample(torch.Size([num_edges]))

        # Sample discrete_feature.
        X_d = X_d_dist.sample(torch.Size([num_nodes]))
        E_d = E_d_dist.sample(torch.Size([num_edges]))

        samples.append(
            pyg.data.Data(x = torch.stack((X_c, X_d), 1), 
                          edge_attr=torch.stack((E_c, E_d), 1), 
                          edge_index = A, y = cluster_lab)
        )

    return samples

class EdgeNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeNN, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, E):
        E = F.relu(self.fc1(E))
        E = self.dropout(E)
        E = F.relu(self.fc2(E))
        E = torch.mean(E, dim=0)

        return E

class NNConv(nn.Module):
    def __init__(self):
        super(NNConv, self).__init__()

        self.nnConv1 = pyg.nn.NNConv(in_channels=2, out_channels=10, 
            nn=EdgeNN(in_channels=2, out_channels=20), 
            aggr='mean', root_weight = True        
        )

        self.nnConv2 = pyg.nn.NNConv(in_channels=10, out_channels=5, 
            nn=EdgeNN(in_channels=2, out_channels=50), 
            aggr='mean', root_weight = True        
        )

        self.nnConv3 = pyg.nn.NNConv(in_channels=5, out_channels=2, 
            nn=EdgeNN(in_channels=2, out_channels=10), 
            aggr='mean', root_weight = True        
        )

    def forward(self, data):
        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index

        x = self.nnConv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.nnConv2(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.nnConv3(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = pyg.nn.global_mean_pool(x, data.batch)
        return x

def train_embeddings(model, data_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(10):
        for i, batch in enumerate(data_loader):
            optimizer.zero_grad() 

            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()

            optimizer.step()

def extract_avg_embeddings_only_2(model, data_loader):
    activation_names = ["nnConv2"]
    train_avg_embedding_0 = {"nnConv2": []}
    train_avg_embedding_1 = {"nnConv2": []}
    pool_func = pyg.nn.global_mean_pool

    num_class_1 = 0
    num_class_0 = 0

    correct_1 = 0
    correct_0 = 0
    
    model.eval()

    for i, batch in enumerate(data_loader): 
        acts, remove_hooks = activation_hook(model, activation_names)
        pred_class = torch.argmax(model(batch), dim=1)
        remove_hooks()

        if batch.y.item() == 1:
            num_class_1 += 1

            for name in activation_names: 
                embed = pool_func(acts[name], batch.batch)
                train_avg_embedding_1[name].append(
                    embed.detach().to('cpu')
                )

            if pred_class.item() == 1: 
                correct_1 += 1

        elif batch.y.item() == 0: 
            num_class_0 += 1

            for name in activation_names:
                embed = pool_func(acts[name], batch.batch)
                train_avg_embedding_0[name].append(
                    embed.detach().to('cpu')
                )

            if pred_class.item() == 0: 
                correct_0 += 1

        assert batch.y.item() == 1 or batch.y.item() == 0

    acc_1 = correct_1 / num_class_1
    acc_0 = correct_0 / num_class_0

    print(f"Class 1 Accuracy: {acc_1} Class 0 Accuracy {acc_0}")

    for name in activation_names:
        train_avg_embedding_1[name] = torch.stack(
            train_avg_embedding_1[name]
        ).mean(dim=0)

        train_avg_embedding_0[name] = torch.stack(
            train_avg_embedding_0[name]
        ).mean(dim=0)

    return [train_avg_embedding_0, train_avg_embedding_1]

def get_embed_dist_only_2(model, data, avg_embeds):

    activation_names = ["nnConv2"]
    distance = 0

    model.eval()
    loader = pyg.loader.DataLoader(data)

    for batch in loader: 
        acts, remove_hooks = activation_hook(model, activation_names)
        out = model(batch)
        remove_hooks()

        distance += (dict_cos_dist(
            acts, avg_embeds, batch.batch
        ) / 2).item()
    
    return distance / len(loader)

def get_GED_between(data1, data2, ged_fn):
    avg_ged = 0
    for (graph_0, graph_1) in zip(data1, data2):

        # smaller graph first: 
        smaller = (
            (graph_1.x.shape[0] + graph_1.edge_index.shape[1]) < 
            (graph_0.x.shape[0] + graph_0.edge_index.shape[1])
        )
        if smaller:
            temp = graph_0
            graph_0 = graph_1
            graph_1 = temp 

        ged = ged_fn(
            graph_0.x.unsqueeze(0), graph_0.edge_index.unsqueeze(0), 
                graph_0.edge_attr.unsqueeze(0), 
            graph_1.x.unsqueeze(0), graph_1.edge_index.unsqueeze(0),
                graph_1.edge_attr.unsqueeze(0), 
        )

        avg_ged += (
            ged / (graph_0.x.shape[0] + graph_0.edge_index.shape[1])
        ).item()

    return avg_ged / len(data1)

def get_GED_within(data, GED_fn):

    avg_GED = 0
    for i in range(0, len(data), 2): 
        graph_0 = data[i]
        graph_1 = data[i + 1]

        # Smaller Graph First: 
        smaller = (
            (graph_1.x.shape[0] + graph_1.edge_index.shape[1]) < 
            (graph_0.x.shape[0] + graph_0.edge_index.shape[1])
        )
        if smaller:
            temp = graph_0
            graph_0 = graph_1
            graph_1 = temp 

        GED = GED_fn(
            graph_0.x.unsqueeze(0), graph_0.edge_index.unsqueeze(0), 
                graph_0.edge_attr.unsqueeze(0), 
            graph_1.x.unsqueeze(0), graph_1.edge_index.unsqueeze(0),
                graph_1.edge_attr.unsqueeze(0), 
        )

        avg_GED += (
            GED / (graph_0.x.shape[0] + graph_0.edge_index.shape[1])
        ).item()

    return avg_GED / (len(data) / 2)

# %% Constants
c0_n_min = 75
c0_n_max = 100

c1_n_min = 75
c1_n_max = 100

c2_n_min = 10
c2_n_max = 35

TRAINING_SAMPLES = 50
TESTING_SAMPLES = 50 # has to be even for paired comparisons.

base_distribution = partial(sample_graph_dist, 
    cont_edge_loc=0, 
    cont_node_sd=1, cont_edge_sd=1, 
    prob_X_d=0.25, prob_E_d=0.25, 
    prob_A=1e-2
)

GED_fn = GEDasMatchLoss(
    1, (0, ), (1, ), (0, ), (1, ), QAP_solver=pygm.ngm
)

# %%
def simulation_fn(config): 

    c2_loc = config['offset']
    c0_loc = config['mu0'] + c2_loc
    c1_loc = -1 * c0_loc
    c3_loc = c0_loc * config['r']
    c4_loc = -1 * c3_loc

    cluster_0_dist = partial(base_distribution, 
        cont_node_loc=c0_loc, cluster_lab=torch.tensor([0]), 
        min_nodes=c0_n_min, max_nodes=c0_n_max, 
    )

    cluster_1_dist = partial(base_distribution, 
        cont_node_loc=c1_loc, cluster_lab=torch.tensor([1]), 
        min_nodes=c1_n_min, max_nodes=c1_n_max, 
    )

    cluster_2_dist = partial(base_distribution, 
        cont_node_loc=c2_loc, cluster_lab=torch.tensor([2]), 
        min_nodes=c2_n_min, max_nodes=c2_n_max, 
    )

    cluster_3_dist = partial(base_distribution, 
        cont_node_loc=c3_loc, cluster_lab=torch.tensor([3]), 
        min_nodes=c2_n_min, max_nodes=c2_n_max, 
    )

    cluster_4_dist = partial(base_distribution, 
        cont_node_loc=c4_loc, cluster_lab=torch.tensor([4]), 
        min_nodes=c2_n_min, max_nodes=c2_n_max, 
    )

    c0_train = cluster_0_dist(n = TRAINING_SAMPLES)
    c1_train = cluster_1_dist(n = TRAINING_SAMPLES)

    train_data_loader = pyg.loader.DataLoader(
        c0_train + c1_train, shuffle=True
    )

    model = NNConv()

    train_embeddings(model, train_data_loader)

    avg_c0_embeds_only_2, avg_c1_embeds_only_2 = extract_avg_embeddings_only_2(
        model, train_data_loader
    )

    c0_test = cluster_0_dist(n = TESTING_SAMPLES)
    c1_test = cluster_1_dist(n = TESTING_SAMPLES)
    c2 = cluster_2_dist(n = TESTING_SAMPLES)
    c3 = cluster_3_dist(n = TESTING_SAMPLES)
    c4 = cluster_4_dist(n = TESTING_SAMPLES)

    # Metrics 
    # 1. Between distribution shouldn't be biased towards either obs cluster. 
    m1 = (
        get_embed_dist_only_2(model, c2, avg_c0_embeds_only_2) - 
        get_embed_dist_only_2(model, c2, avg_c1_embeds_only_2)
    )
    m2 = (
        get_GED_between(c0_test, c2, GED_fn) - 
        get_GED_between(c1_test, c2, GED_fn)
    )

    # 2. Far to the positive side should be reasonable far from both obs clusters. 
    m3 = (
        get_embed_dist_only_2(model, c3, avg_c0_embeds_only_2) - 
        get_embed_dist_only_2(model, c3, avg_c1_embeds_only_2)
    )
    m4 = (
        get_GED_between(c0_test, c3, GED_fn) - 
        get_GED_between(c1_test, c3, GED_fn)
    )

    # 3. Far to the negative side shouldn't differ from far to the positive side. 
    m5 = (
        get_embed_dist_only_2(model, c4, avg_c0_embeds_only_2) - 
        get_embed_dist_only_2(model, c4, avg_c1_embeds_only_2)
    )
    m6 = (
        get_GED_between(c0_test, c4, GED_fn) - 
        get_GED_between(c1_test, c4, GED_fn)
    )

    # Checks 
    # 1. c0 is far from c1. 
    check1 = (
        get_embed_dist_only_2(model, c0_test, avg_c1_embeds_only_2) - 
        get_embed_dist_only_2(model, c0_test, avg_c0_embeds_only_2)
    )
    check2 = (
       get_GED_between(c0_test, c1_test, GED_fn)  - 
       get_GED_between(c0_test, c0_test, GED_fn)
    )

    train.report(
        {"m1": m1, "m2": m2, "m3": m3, "m4": m4, "m5": m5, "m6": m6, 
         "check1": check1, "check2": check2}
    )

# %%
if __name__ == '__main__':
    
    config = {
        "offset": tune.uniform(-1, 1),
        "mu0": tune.uniform(1, 3), 
        "r": tune.uniform(1, 3), 
    }

    # tune_scheduler = ASHAScheduler(
    #     metric="m1", mode="min", max_t=10
    # )

    result = tune.run(
        simulation_fn, 
        config=config, 
        num_samples=100,
        #scheduler=tune_scheduler, 
        resources_per_trial={"cpu": 6, "gpu": 0.1}
    )

    df = result.results_df
    print(df)
    df.to_csv("simulation_tuned_results.csv", index=False)
    df.to_csv("results/simulation/simulation_tuned_results.csv", index=False)