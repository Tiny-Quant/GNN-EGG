# %% Dependencies
from typing import Optional
import copy 
from torch_geometric.utils import remove_isolated_nodes

import sys 
sys.path.append("../scripts/ceograph/")
from ceograph import NucleiData

# %%
def clear_iso_nodes(example: NucleiData, 
                    num_nodes: Optional[int] = None) -> NucleiData: 
    edge_index, _, mask = (
        remove_isolated_nodes(example.edge_index, num_nodes=num_nodes)
    )
    example_masked = NucleiData(
        x = example.x[mask], 
        edge_index = edge_index, 
        cell_type = example.cell_type[mask], 
        edge_attr = example.edge_attr,
    )

    return example_masked