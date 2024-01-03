# %%

import torch

# Adds the repo directory to the import paths.
import sys
from os.path import dirname, abspath
repo_dir = dirname(dirname(abspath(__file__)))
sys.path.append(repo_dir)
print(sys.path)

from egg_models.egg_hard_nx import EggHardNx
from egg_models.losses import PredLoss
from utils import ceograph

# %%
if __name__ == '__main__':

    MAX_NODES = 50
    CONT_NODE_FEATS = 11
    CELL_TYPES = 5
    CONT_EDGE_FEAT = 2
    BATCH_SIZE = 1

    device = torch.device(0)

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
        batch_size=BATCH_SIZE
    )
    generator.to(device)

    X, C_x, A, E, C_x_logLik, A_logLik = generator()

    graph_list = [
        ceograph.NucleiData(X, C_x, A, E) for (X, C_x, A, E) in 
        zip(X.unbind(), C_x.unbind(), A.unbind(), E.unbind())
    ]

    explainee(graph_list[0])

