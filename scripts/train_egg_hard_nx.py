# %%

import torch
import torch.nn as nn 
import torch.optim
torch.autograd.set_detect_anomaly(True)

# Adds the repo directory to the import paths.
import sys
from os.path import dirname, abspath
repo_dir = dirname(dirname(abspath(__file__)))
sys.path.append(repo_dir)
import pickle

from egg_models.egg_hard_nx import EggHardNx
from egg_models.losses import PredLoss
from egg_models.trainer import Trainer
from utils import ceograph

# %%
if __name__ == '__main__':

    MAX_NODES = 50
    CONT_NODE_FEATS = 11
    CELL_TYPES = 5
    CONT_EDGE_FEAT = 2
    BATCH_SIZE = 10

    device = torch.device(0)

    target = torch.tensor([1.0, 0.0]).to(device)
    path = repo_dir + "/" + "data/slides/LUDA/ad_train_nx_100.pkl" 
    with open(path, 'rb') as f:
        obs = pickle.load(f)

    explainee = ceograph.NucleiNet(CONT_NODE_FEATS, CONT_EDGE_FEAT, batch=False)
    explainee.to(device)
    explainee.load_state_dict(torch.load(
            repo_dir + "/data/explainees/ceograph/epoch_263.pt",
            map_location=device
        )
    )
    explainee.eval()

    generator = EggHardNx(
        node_size=MAX_NODES, cont_node_feat=CONT_NODE_FEATS, 
        node_types=CELL_TYPES, cont_edge_feat=CONT_EDGE_FEAT, 
        batch_size=BATCH_SIZE, 
    )
    generator.to(device)

    optimizer = torch.optim.Adam(generator.parameters())    

    criterion = nn.BCELoss()

    trainer = Trainer(generator, explainee, optimizer, criterion, target, obs)

    trainer.train()