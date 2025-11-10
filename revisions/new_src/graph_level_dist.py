import torch 
from torch import nn

class dummyDist(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cont_data):
        print(cont_data)

        return 1