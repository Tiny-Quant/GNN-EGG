import pytest

import sys
from os.path import dirname, abspath
repo_dir = dirname(dirname(abspath(__file__)))
sys.path.append(repo_dir)

from egg_models import egg_generic

@pytest.fixture
def generator():
    mock_generator = egg_generic.EggGeneric(
        max_node_size=10, 
        cont_node_feats=10, 
        cont_edge_feats=10, 
        dis_node_feats=(1, 2, 3), 
        dis_edge_feats=(1, 2), 
        batch_size=2
    )

    return mock_generator

@pytest.fixture
def gen_output(generator):
    return generator()

def test_gen_shapes(gen_output):
    assert gen_output['dis_node_feats'].shape == (2, 10, 1 + 2 + 3)
    assert gen_output['dis_edge_feats'].shape == (2, 100, 1 + 2)
    assert gen_output['full_edge_indices'].shape == (2, 2, 10**2)
    assert gen_output['adjacency_matrix'].shape == (2, 10, 10)
    assert gen_output['C_x_logLik'].shape == (2, 3)
    assert gen_output['C_e_logLik'].shape == (2, 2)
    assert gen_output['A_logLik'].shape == (2,)
    assert gen_output['cont_node_feats'].shape == (2, 10, 10)