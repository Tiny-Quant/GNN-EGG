# %%

from egg_hard_nx import EggHardNx

# %%
if __name__ == '__main__':

    MAX_NODES = 50
    CONT_NODE_FEATS = 11
    CELL_TYPES = 5
    CONT_EDGE_FEAT = 2
    BATCH_SIZE = 1

    generator = EggHardNx(
        node_size=MAX_NODES, cont_node_feat=CONT_NODE_FEATS, 
        node_types=CELL_TYPES, cont_edge_feat=CONT_EDGE_FEAT, 
        batch_size=BATCH_SIZE
    )

    generator()