# %% Dependencies
import random 
import importlib

import torch 
from torch import nn 
import model 
from model import EGG 

import sys 
sys.path.append("../scripts/ceograph/")
from ceograph import NucleiData, load_model

#importlib.reload(model) Unknown bug. 

# %% Runnable Entry Point
if __name__ == "__main__":

    # Reproducibility 
    random.seed(0)
    torch.manual_seed(0)
    device = torch.device(0)

    # Load Model 
    explainee = load_model(path = "../data/trained/epoch_263.pt",
    device=device)

    # Test Forward
    EGG_For = EGG(node_size=50, cont_node_feat=11, 
        cell_types=5, cont_edge_feat=2)

    X, C_x, A, edge_features = EGG_For()

    explainee.eval()
    Example = NucleiData(X, C_x, A, edge_features).to(torch.device(0))
    out = explainee(Example) 
    print(Example, out)

# %%
