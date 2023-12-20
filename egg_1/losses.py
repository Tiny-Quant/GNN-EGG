# %% Dependencies
import torch 
from torch import tensor

import sys 
sys.path.append("../scripts/ceograph/")
from ceograph import NucleiData

# %%
def loss_diff_dense(example: NucleiData, explainee: callable, 
                    criterion: callable, target: tensor) -> tensor: # with grad. 
    """
    This loss function is differentiable w.r.t the continuous node features
    created by a standard dense layer.
    """
    explainee.eval()
    explainee_out = explainee(example)
    explainee_pred = torch.softmax(explainee_out, dim=0).cpu()

    return criterion(explainee_pred, target)
