import torch 
from torch.utils.data import DataLoader

import sys 
sys.path.append("../scripts/ceograph/")
from ceograph import NucleiData, load_model

from model import EGG
from utils import clear_iso_nodes
from edit_distance import nuclei_to_nx, list_edit_distance

class EGG_Loader:
    def __init__(self, EGG_Model, explainee, target, edit_obs):
        self.EGG_Model = EGG_Model
        self.explainee = explainee
        self.target = target
        self.edit_obs = edit_obs

    def __len__(self):
        return 1

    def __getitem__(self, _):
        for _ in range(100):
            try:
                X, C_x, A, E = self.EGG_Model()
                example = NucleiData(X, C_x, A, E)
                example = clear_iso_nodes(example).to(torch.device(0))
                explainee_pred = torch.softmax(self.explainee(example), dim=0).cpu()

                break

            except Exception as e:
                print({e})

        with torch.no_grad():
            example_nx = nuclei_to_nx(example.detach().cpu())
            edit_dist = list_edit_distance(example_nx, self.edit_obs)

        logLik_DisNodeFeat = self.EGG_Model.DisNodeFeat.logLik
        logLik_AdjacencyMatrix = self.EGG_Model.AdjacencyMatrix.logLik

        return explainee_pred, edit_dist, logLik_DisNodeFeat, logLik_AdjacencyMatrix