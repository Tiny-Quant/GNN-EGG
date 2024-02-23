# %%
from typing import Tuple

import torch 

# %%
def concat_possible_none_tensors(tensor1: torch.tensor, tensor2: torch.tensor, 
                                 dim = 0): 
    if tensor1 is None and tensor2 is None: 
        return None 

    elif tensor2 is None: 
        return tensor1
    
    elif tensor1 is None: 
        return tensor2

    else: 
        return torch.cat((tensor1, tensor2), dim=dim)

# %%
def concat_one_hot_to_labels(one_hot: torch.tensor, indices: Tuple[int]):

    ret = []
    start = 0
    for index in indices:
        ret.append(one_hot[:, :, start:start + index])
        start += index 

    label_tensors = [torch.argmax(t, dim=2, keepdim=True) for t in ret]

    return torch.stack(label_tensors, dim = 2).squeeze(-1)