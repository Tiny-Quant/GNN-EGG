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
# import argparse
# import json

from egg_models.egg_soft import EggSoft
from egg_models.losses import AffinityScore
# from egg_models.losses import PredLoss
# from egg_models.trainer import Trainer
# from utils import ceograph

# %%
if __name__ == '__main__':
    generator = EggSoft(50, 11, 6, 2, batch_size=100)
    _, soft_list, _, _ = generator()


    obs_path = "data/slides/LUSC/sc_train_data_list_100.pkl"
    with open(obs_path, 'rb') as f:
        obs_list = pickle.load(f)
    
    optimizer = Adam(generator.parameters())

    sim_score_fn = AffinityScore(optimizer=optimizer)
    print(f'cell {generator.DisNodeFeats.probs}')
    print("Scoring...")
    scores = sim_score_fn(soft_list, obs_list)

    print(scores)
    print(f'cell {generator.DisNodeFeats.probs}')
