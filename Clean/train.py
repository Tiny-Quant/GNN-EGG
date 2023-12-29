# %% Dependencies
import random 
import time
import pickle

import torch
import torch.nn as nn 
from torch.optim import Adam
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

# Set the multiprocessing start method to 'spawn'
mp.set_start_method('spawn', force=True)

import sys 
sys.path.append("../scripts/ceograph/")
from ceograph import NucleiData, load_model

from model import EGG
from utils import clear_iso_nodes
from dataloader import EGG_Loader

# Load ad obs 
path = "../data/slides/LUDA/ad_train_nx_100.pkl"

with open(path, 'rb') as f:
    ad_train_nx_100 = pickle.load(f)

# %% Training Config

# Reproducibility 
random.seed(0)
torch.manual_seed(0)
device = torch.device(0)

# Model to be explained. 
explainee = load_model(path = "../data/trained/epoch_263.pt", device=device)

# Model Parameters:
max_nodes = 50
cont_node_feats = 11
cell_types = 5
cont_edge_feat = 2

EGG_Model = EGG(node_size=max_nodes, cont_node_feat=cont_node_feats, 
                cell_types=cell_types, cont_edge_feat=cont_edge_feat)

# Training Parameters:
num_epochs = 5
batch_size = 10
num_workers = batch_size
learning_rate = 1e-4
optimizer = Adam(EGG_Model.parameters(), lr=learning_rate)
target = torch.tensor([1.0, 0.0])
edit_obs = ad_train_nx_100
criterion = nn.BCELoss()
lambda_1 = 1
lambda_2 = 1
lambda_3 = 1

# %% Training Loop 
def main(): 
    for epoch in range(num_epochs): 
        start = time.time()

        optimizer.zero_grad()

        losses = [] # stored for plotting. 

        dataset = EGG_Loader(EGG_Model, explainee, target, edit_obs)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        for batch in dataloader:
            explainee_preds, edit_dists, logLik_DisNodeFeat, logLik_AdjacencyMatrix = batch

            # Compute losses using the batched results
            pred_loss = criterion(explainee_preds, target) * (1 + logLik_DisNodeFeat + logLik_AdjacencyMatrix)
            edit_loss = edit_dists * (logLik_DisNodeFeat + logLik_AdjacencyMatrix)
        
            edge_pen = torch.norm(EGG_Model.AdjacencyMatrix.edge_logits, p=1)

            loss = (lambda_1 * pred_loss / batch_size
                + lambda_2 * edit_loss / batch_size
                + lambda_3 * edge_pen)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        end = time.time()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.10f}" + 
            f"Time: {end-start:.2f} seconds")

if __name__ == '__main__':
    main()