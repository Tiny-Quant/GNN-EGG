
# %% Dependencies

# torch 
import torch 
import torch.multiprocessing as mp 
from torch.multiprocessing import Pool

# %%

# Note: from ceograph. 
def get_edge_type(edge, cell_type):
    """
    Args: 
        edge: (in_cell_index, out_cell_index, 1/edge_length)
        
    Returns:
        edge type index.
        0: 1-1; 1: 1-2; 2: 1-3; 3: 1-4; 4: 1-5; 5: 1-6
        6: 2-1; 7: 2-2; 8: 2-3; 9: 2-4; 10: 2-5; 11: 2-6
        12: 3-1; 13: 3-2; 14: 3-3; 15: 3-4; 16: 3-5; 17: 3-6
        18: 4-1; 19: 4-2; 20: 4-3; 21: 4-4; 22: 4-5; 23: 4-6
        24: 5-1; 25: 5-2; 26: 5-3; 27: 5-4; 28: 5-5; 29: 5-6
        30: 6-1; 31: 6-2; 32: 6-3; 33: 6-4; 34: 6-5; 35: 6-6
    """
    mapping = {"1-1": 0, "1-2": 1, "1-3": 2, "1-4": 3, "1-5": 4, "1-6": 5,
                   "2-1": 6, "2-2": 7, "2-3": 8, "2-4": 9, "2-5": 10, "2-6": 11,
                   "3-1": 12, "3-2": 13, "3-3": 14, "3-4": 15, "3-5": 16, "3-6": 17,
                   "4-1": 18, "4-2": 19, "4-3": 20, "4-4": 21, "4-5": 22, "4-6": 23,
                   "5-1": 24, "5-2": 25, "5-3": 26, "5-4": 27, "5-5": 28, "5-6": 29,
                   "6-1": 30, "6-2": 31, "6-3": 32, "6-4": 33, "6-5": 34, "6-6": 35}
    return mapping['{}-{}'.format(cell_type[edge[0]], cell_type[edge[1]])]

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
        mp.set_start_method('spawn', force=True)
        with Pool() as pool: 
            e_c = torch.stack(
                pool.starmap(get_edge_type_app, 
                            zip(A.unbind(), C_x.unbind()))
            ) 
            
    return e_c

def assign_edge_type(A, C_x):
    with torch.no_grad():
        e_c = torch.stack(
            [get_edge_type_app(A, C_x) for (A, C_x) in 
             zip(A.unbind(), C_x.unbind())]
        )
    
    return e_c
