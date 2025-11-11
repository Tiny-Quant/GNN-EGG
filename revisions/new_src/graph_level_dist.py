import random 

import torch 
from torch import nn
from torch_geometric.data import Batch

from .utils import convert_hard_to_soft_edges

class dummyDist(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cont_data):
        print(cont_data)

        return 1

class neural_approx_ged_dist(nn.Module): 

    def __init__(self, data, model):
        super().__init__()
        self.data = data
        self.model = model.eval()
       
    def _sample_obs_data(self, gen_graph):
        target_n = gen_graph.num_graphs

        indices = random.choices(range(len(self.data)), k=target_n)
        chosen_graphs = [
            convert_hard_to_soft_edges(self.data[i]) 
            for i in indices
        ]

        return Batch.from_data_list(chosen_graphs)

    def forward(self, cont_data): 
        obs_data = self._sample_obs_data(cont_data)

        dist = self.model(cont_data, obs_data) 
        
        return dist.mean()

 