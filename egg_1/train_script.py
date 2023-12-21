# %% Dependencies
import random 
import importlib

import torch 
from torch import nn 

from model import EGG 
from losses import loss_diff_dense
from utils import clear_iso_nodes

import sys 
sys.path.append("../scripts/ceograph/")
from ceograph import NucleiData, load_model

#importlib.reload(losses) # Unknown bug. 
# %% Training Config

# Reproducibility 
random.seed(0)
torch.manual_seed(0)
device = torch.device(0)

# Model to be explained. 
explainee = load_model(path = "../data/trained/epoch_263.pt",
device=device)

# 
BCELoss = nn.BCELoss()

# %% Runnable Entry Point
if __name__ == "__main__":

    # Test Forward
    EGG_For = EGG(node_size=50, cont_node_feat=11, 
        cell_types=5, cont_edge_feat=2, batch_size=2)

    print(EGG_For().shape)
    #X, C_x, A, edge_features = EGG_For()
    #print(X.shape, C_x.shape, A.shape, edge_features.shape)
    #target = torch.tensor([1.0, 0.0])
    #Example = NucleiData(X, C_x, A, edge_features).to(torch.device(0))
    #Example = clear_iso_nodes(Example, num_nodes=EGG_For.node_size)

    #loss = loss_diff_dense(Example, explainee, BCELoss, target)
    #print(loss)

# %%
