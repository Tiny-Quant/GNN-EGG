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
from pygmtools.utils import(
    build_aff_mat, build_batch
)
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

device = torch.device(0)

# %%
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
        A = A_dist.sample(torch.Size([num_nodes, num_nodes])) + 1e-4
        A, edge_weights = pyg.utils.dense_to_sparse(A)
        num_edges = A.shape[1]

        # Sample continuous features.
        X_c = X_c_dist.sample(torch.Size([num_nodes]))
        E_c = E_c_dist.sample(torch.Size([num_edges]))

        # Sample discrete_feature.
        X_d = X_d_dist.sample(torch.Size([num_nodes]))
        E_d = E_d_dist.sample(torch.Size([num_edges]))
        E_d = torch.stack((E_d, edge_weights - 1e-4), dim=1)

        samples.append(
            pyg.data.Data(x = torch.stack((X_c, X_d), 1), 
                          edge_attr=torch.cat((E_c.unsqueeze(-1), E_d), -1), 
                          edge_index = A, y = cluster_lab)
        )

    return samples

# %%
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
            nn=EdgeNN(in_channels=3, out_channels=20), 
            aggr='mean', root_weight = True        
        )

        self.nnConv2 = pyg.nn.NNConv(in_channels=10, out_channels=5, 
            nn=EdgeNN(in_channels=3, out_channels=50), 
            aggr='mean', root_weight = True        
        )

        self.nnConv3 = pyg.nn.NNConv(in_channels=5, out_channels=2, 
            nn=EdgeNN(in_channels=3, out_channels=10), 
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
        ) / 2)
    
    return distance / len(loader)

# %%
def edit_aff_fn(feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:

    feat1_norm = F.normalize(feat1, p=2, dim=-1)
    feat2_norm = F.normalize(feat2, p=2, dim=-1)

    cos_sim_mat = torch.einsum('bij, bkj -> bik', 
                                feat1_norm, feat2_norm)

    return -1 * (1 - cos_sim_mat)

def to_egg(data_list): 

    X_list = [graph.x for graph in data_list]
    A_list = [graph.edge_index for graph in data_list]
    E_list = [graph.edge_attr for graph in data_list]

    return [build_batch(X_list), build_batch(A_list), build_batch(E_list)]

def GED_loss(gen, obs) -> torch.Tensor:

    gen_X, gen_A, gen_E = gen
    obs_X, obs_A, obs_E = obs

    # Collect graph size information - avoids weird bugs.
    n1 = (
        torch.tensor([gen_X.shape[1]]).expand(gen_X.shape[0]).
            to(gen_X.device)
    )
    ne1 = (
        torch.tensor([gen_A.shape[2]]).expand(gen_A.shape[0]). 
            to(gen_A.device)
    )
    n2 = (
        torch.tensor([obs_X.shape[1]]).expand(obs_X.shape[0]). 
            to(obs_X.device)
    )
    ne2 = (
        torch.tensor([obs_A.shape[2]]).expand(obs_A.shape[0]). 
            to(obs_A.device)
    )

    aff_mat = build_aff_mat(
        node_feat1=gen_X, 
        edge_feat1=gen_E, 
        connectivity1=gen_A.transpose(1, 2), 
        node_feat2=obs_X, 
        edge_feat2=obs_E, 
        connectivity2=obs_A.transpose(1, 2),  
        node_aff_fn=edit_aff_fn, 
        edge_aff_fn=edit_aff_fn,
        n1=n1, 
        ne1=ne1, 
        n2=n2, 
        ne2=ne2
    )

    match_mat = pygm.ngm(aff_mat, n1=n1, n2=n2)
    dis_match_mat = pygm.hungarian(match_mat)

    GED = -1 * pygm.utils.compute_affinity_score(dis_match_mat, aff_mat)

    return GED

class Generator(nn.Module): 
    def __init__(self, batch_size, min_nodes, max_nodes, temp=0.15): 

        super(Generator, self).__init__()

        self.batch_size = batch_size
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.temp = temp

        self.cont_node_loc = nn.Parameter(
            nn.init.uniform_(
                torch.tensor(0.0), 
                -1.0, 1.0
            )
        )

        self.cont_edge_loc = nn.Parameter(
            nn.init.uniform_(
                torch.tensor(0.0), 
                -1.0, 1.0
            )
        )
        self.prob_X_d = nn.Parameter(
            nn.init.uniform_(
                torch.tensor(0.0), 
                0.0, 1.0
            )
        )
        self.prob_E_d = nn.Parameter(
            nn.init.uniform_(
                torch.tensor(0.0), 
                0.0, 1.0
            )
        )
        self.prob_A = nn.Parameter(
            nn.init.uniform_(
                torch.tensor(0.0), 
                0.0, 1.0
            )
        )

    def forward(self): 

        self.prob_X_d.data.clamp_(0.0, 1.0)
        self.prob_E_d.data.clamp_(0.0, 1.0)
        self.prob_A.data.clamp_(0.0, 1.0)

        X_c_dist = td.Normal(loc=self.cont_node_loc, scale=1)         
        E_c_dist = td.Normal(loc=self.cont_edge_loc, scale=1)         
        X_d_dist = td.RelaxedBernoulli(self.temp, probs=self.prob_X_d)
        E_d_dist = td.RelaxedBernoulli(self.temp, probs=self.prob_E_d) 
        A_dist = td.RelaxedBernoulli(self.temp, probs=self.prob_A)

        batch = []

        for i in range(self.batch_size): 
            num_nodes = torch.randint(self.min_nodes, self.max_nodes, (1, )).item()

            A = A_dist.rsample(torch.Size([num_nodes, num_nodes])) + 1e-4
            A, edge_weights = pyg.utils.dense_to_sparse(A)
            num_edges = A.shape[1]

            X_c = X_c_dist.rsample(torch.Size([num_nodes]))
            E_c = E_c_dist.rsample(torch.Size([num_edges]))

            X_d = X_d_dist.rsample(torch.Size([num_nodes]))
            E_d = E_d_dist.rsample(torch.Size([num_edges]))
            E_d = torch.stack([E_d, edge_weights - 1e-4], dim=1)

            batch.append(
                pyg.data.Data(
                    x = torch.stack((X_c, X_d), 1), 
                    edge_attr=torch.cat((E_c.unsqueeze(-1), E_d), -1), 
                    edge_index = A)
            )

        return batch 

# %%
def sim_fn(config):

    # Sample training data. 
    ## Set gt parameters.   
    c0_node_loc = config['mu0'] + config['offset']
    c1_node_loc = -1 * c0_node_loc + config['mu1_noise']
    ood_node_loc = c0_node_loc * config['r']

    base_distribution = partial(sample_graph_dist, 
        cont_edge_loc=config['cont_edge_loc'], 
        cont_node_sd=1, cont_edge_sd=1, 
        prob_X_d=config['prob_X_d'], prob_E_d=config['prob_E_d'], 
        prob_A=config['prob_A'], 
    )
    cluster_0_dist = partial(base_distribution, 
        cont_node_loc=c0_node_loc, cluster_lab=torch.tensor([0]), 
        min_nodes=75, max_nodes=100, 
    )
    cluster_1_dist = partial(base_distribution, 
        cont_node_loc=c1_node_loc, cluster_lab=torch.tensor([1]), 
        min_nodes=75, max_nodes=100, 
    )

    c0_train = cluster_0_dist(n = 100)
    c1_train = cluster_1_dist(n = 100)
    train_data_loader = pyg.loader.DataLoader(
        c0_train + c1_train, shuffle=True
    )

    explainee = NNConv()
    train_embeddings(explainee, train_data_loader)
    avg_embeds_0, avg_embeds_1 = extract_avg_embeddings_only_2(
        explainee, train_data_loader
    )

    generator_embed = Generator(10, 10, 35)
    generator_GED = Generator(10, 10, 35)

    generator_embed.cont_node_loc = nn.Parameter(torch.tensor(ood_node_loc))
    generator_GED.cont_node_loc = nn.Parameter(torch.tensor(ood_node_loc))

    if not config['rand_init']:
        
        generator_embed.cont_edge_loc = nn.Parameter(
            torch.tensor(config['cont_edge_loc'])
        )
        generator_embed.prob_X_d = nn.Parameter(
            torch.tensor(config['prob_X_d'])
        )
        generator_embed.prob_E_d = nn.Parameter(
            torch.tensor(config['prob_E_d'])
        )
        generator_embed.prob_A = nn.Parameter(
            torch.tensor(config['prob_A'])
        )
        
        generator_GED.cont_edge_loc = nn.Parameter(
            torch.tensor(config['cont_edge_loc'])
        )
        generator_GED.prob_X_d = nn.Parameter(
            torch.tensor(config['prob_X_d'])
        )
        generator_GED.prob_E_d = nn.Parameter(
            torch.tensor(config['prob_E_d'])
        )
        generator_GED.prob_A = nn.Parameter(
            torch.tensor(config['prob_A'])
        )

    # Train the generator based on embedding distance loss. 
    optimizer_embed = torch.optim.RMSprop(
        generator_embed.parameters(), lr=config['lr']
    )
    lr_scheduler_embed = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_embed, gamma=config['lr_decay']
    )
    for epoch in range(25): 
        optimizer_embed.zero_grad()

        gen_graph_list = generator_embed()

        loss = get_embed_dist_only_2(explainee, gen_graph_list, avg_embeds_0)

        loss.backward()

        optimizer_embed.step()

        lr_scheduler_embed.step()
    
    # Train the generator based on GED loss. 

    optimizer_GED = torch.optim.RMSprop(
        generator_GED.parameters(), lr=config['lr']
    )
    lr_scheduler_GED = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_GED, gamma=config['lr_decay']
    )
    obs_data_loader = pyg.loader.DataLoader(
        c0_train, batch_size=10, shuffle=True
    )
    for epoch in range(25): 
        for obs_graph in obs_data_loader: 

            optimizer_GED.zero_grad()

            gen = to_egg(generator_GED())
            obs = to_egg(obs_graph.to_data_list())

            loss = GED_loss(gen, obs).mean()
            loss.backward()

            optimizer_GED.step()

        lr_scheduler_GED.step()
    
    # Collect metrics.  

    cont_node_loc_bias_embed = (
        generator_embed.cont_node_loc.data - c0_node_loc
    ).item()
    cont_edge_loc_bias_embed = (
        generator_embed.cont_edge_loc.data - config['cont_edge_loc'] 
    ).item()
    prob_X_d_bias_embed = (
        generator_embed.prob_X_d.data - config['prob_X_d']
    ).item()
    prob_E_d_bias_embed = (
        generator_embed.prob_E_d.data - config['prob_E_d']
    ).item()
    prob_A_bias_embed = (
        generator_embed.prob_A.data - config['prob_A']
    ).item()

    cont_node_loc_bias_GED = (
        generator_GED.cont_node_loc.data - c0_node_loc
    ).item()
    cont_edge_loc_bias_GED = (
        generator_GED.cont_edge_loc.data - config['cont_edge_loc'] 
    ).item()
    prob_X_d_bias_GED = (
        generator_GED.prob_X_d.data - config['prob_X_d']
    ).item()
    prob_E_d_bias_GED = (
        generator_GED.prob_E_d.data - config['prob_E_d']
    ).item()
    prob_A_bias_GED = (
        generator_GED.prob_A.data - config['prob_A']
    ).item()

    # Compute checks. 
    generator_embed.eval()
    pred_embed = 0
    for i in range(100):
        with torch.no_grad(): 
            gen_graph_list = generator_embed()
            gen_graph_batch = pyg.data.Batch.from_data_list(gen_graph_list)
            pred = torch.softmax(explainee(gen_graph_batch), dim=-1).mean(dim=0)
            pred_embed += pred[0].item()
    pred_embed = pred_embed / 100

    generator_GED.eval()
    pred_GED = 0
    for i in range(100):
        with torch.no_grad(): 
            gen_graph_list = generator_GED()
            gen_graph_batch = pyg.data.Batch.from_data_list(gen_graph_list)
            pred = torch.softmax(explainee(gen_graph_batch), dim=-1).mean(dim=0)
            pred_GED += pred[0].item()
    pred_GED = pred_GED / 100

    # Report metrics and checks. 
    train.report({
        'cont_node_loc_bias_embed': cont_node_loc_bias_embed,
        'cont_edge_loc_bias_embed': cont_edge_loc_bias_embed,
        'prob_X_d_bias_embed': prob_X_d_bias_embed,
        'prob_E_d_bias_embed': prob_E_d_bias_embed,
        'prob_A_bias_embed': prob_A_bias_embed,

        'cont_node_loc_bias_GED': cont_node_loc_bias_GED,
        'cont_edge_loc_bias_GED': cont_edge_loc_bias_GED,
        'prob_X_d_bias_GED': prob_X_d_bias_GED,
        'prob_E_d_bias_GED': prob_E_d_bias_GED,
        'prob_A_bias_GED': prob_A_bias_GED,

        'pred_embed': pred_embed, 
        'pred_GED': pred_GED
    })


# %%
if __name__ == '__main__':

    config = {
        "offset": tune.uniform(-1, 1),
        "mu0": tune.uniform(1, 3), 
        "mu1_noise": tune.uniform(0, 1), 
        "r": tune.uniform(2, 5), 
        "rand_init": tune.choice([True, False]), 
        "cont_edge_loc": tune.uniform(-1, 1), 
        "prob_X_d": tune.uniform(0, 1), 
        "prob_E_d": tune.uniform(0, 1),
        'prob_A': tune.uniform(0, 1), 
        'lr': tune.choice([1e-1, 1e-2, 1e-3]), 
        'lr_decay': tune.choice([0.1, 1.0])
    }

    result = tune.run(
        sim_fn, 
        config=config, 
        num_samples=100,
        resources_per_trial={"cpu": 3, "gpu": 0.05}
    )

    df = result.results_df
    print(df)
    df.to_csv("results/simulation/sim_tunned_parameter_bias_2.csv", index=False)
