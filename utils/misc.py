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
# TODO: add_possible_none_tensor helper function.

# %%
def concat_one_hot_to_labels(one_hot: torch.tensor, indices: Tuple[int]):

    ret = []
    start = 0
    for index in indices:
        ret.append(one_hot[:, :, start:start + index])
        start += index 

    label_tensors = [torch.argmax(t, dim=2, keepdim=True) for t in ret]

    return torch.stack(label_tensors, dim = 2).squeeze(-1)


# %%
def subset_tensor(tensor: torch.tensor, indices: Tuple[any], 
                  dim: int) -> torch.tensor:
    # Convert single indices to tuples for uniform handling
    if isinstance(indices, int):
        indices = (indices,)
    
    # Initialize slices for all dimensions
    sliced_indices = [slice(None)] * tensor.dim()
    
    # Update indices along the specified dimension
    selected_indices = []
    for idx in indices:
        if isinstance(idx, int):
            selected_indices.append(idx)
        elif isinstance(idx, slice):
            selected_indices.extend(range(*idx.indices(tensor.size(dim))))
        else:
            raise ValueError("Invalid index type")
    
    # Construct the sliced indices for the specified dimension
    sliced_indices[dim] = selected_indices
    
    return tensor[tuple(sliced_indices)]

# %%
