
# %% Dependencies

# base 
import sys

# torch 
import torch 
from torch.multiprocessing import Pool  

# within 
sys.path.append("../ceograph")
from ceograph import get_edge_type

# %%

def get_edge_type_app(A, C_x):
    edges = list(zip(A.long()[0], A.long()[1]))
    e_c = list(map(lambda x: get_edge_type(x, C_x.int()), edges))
    e_c = torch.tensor(e_c).unsqueeze(-1)

    return e_c

def assign_edge_type_par(A, C_x):
    '''
    Assigns edge types to batches in parallel. 
    A - Batched sparse adjacency matrix : tensor [batch, 2, #edges]
    C_x - Batched node cell types : tensor [batch, #nodes]
    '''
    with torch.no_grad():
        with Pool() as pool: 
            e_c = torch.stack(
                pool.starmap(get_edge_type_app, 
                            zip(A.unbind(), C_x.unbind()))
            ) 