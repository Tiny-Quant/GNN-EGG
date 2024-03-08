# %% Dependencies
import pytest

import sys
from os.path import dirname, abspath
repo_dir = dirname(dirname(abspath(__file__)))
sys.path.append(repo_dir)

import torch

from egg_models import egg_generic

# %% Fixture/Mocks/Data Test Cases
@pytest.fixture
def generator():
    """
    Create test case I for an EggGeneric generator model. 
    """
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
def generator2():
    """
    Create test case II for an EggGeneric generator model. 
    """
    mock_generator = egg_generic.EggGeneric(
        max_node_size=10, 
        cont_node_feats=10, 
        cont_edge_feats=10, 
        dis_node_feats=(1, 2, 3), 
        # dis_edge_feats=(1, 2), 
        batch_size=2
    )

    return mock_generator

@pytest.fixture
def gen_output(generator):
    """
    Output test case I for an EggGeneric generator model. 
    """
    return generator()

@pytest.fixture
def gen_output2(generator2):
    """
    Output test case II for an EggGeneric generator model. 
    """
    return generator2()

# %% Helper Functions
def check_grads_exist(loss: torch.tensor, model: torch.nn.Module, retain=False): 
    """
    Asserts that the gradients of all model parameters (except "device_param")
    are not None. 
    """
    loss.sum().backward(retain_graph=retain) # Sum ensures scalar loss. 

    for name, param in model.named_parameters():
        if name != "device_param": 
            assert(
                param.grad is not None, 
                f"Parameter '{name}' does not have gradients"
            ) 

def compare_grads(loss_1, loss_2, model, retain=False): 
    """
    Checks if to losses produce the same parameter gradients. 
    """
    model.zero_grad()
    loss_1.backward(retain_graph=retain)
    grad_1 = []
    for name, param in model.named_parameters():
        if param.grad is None: 
            print(name)
        else: 
            grad_1.append(param.grad.clone().detach())

    model.zero_grad()
    loss_2.backward(retain_graph=retain)
    grad_2 = []
    for name, param in model.named_parameters():
        if param.grad is None: 
            print(name)
        else: 
            grad_2.append(param.grad.clone().detach())

    for grad1, grad2 in zip(grad_1, grad_2):
        assert torch.allclose(grad1, grad2)

# %% Tests
def test_gen_shapes(gen_output):
    """
    Tests if the output create by test case I matches the expected and 
    documented tensor shapes.
    """
    assert gen_output['dis_node_feats'].shape == (2, 10, 1 + 2 + 3)
    assert gen_output['dis_edge_feats'].shape == (2, 100, 1 + 2)
    assert gen_output['full_edge_indices'].shape == (2, 2, 10**2)
    assert gen_output['adjacency_matrix'].shape == (2, 10, 10)
    assert gen_output['C_x_logLik'].shape == (2, 3)
    assert gen_output['C_e_logLik'].shape == (2, 2)
    assert gen_output['A_logLik'].shape == (2,)
    assert gen_output['cont_node_feats'].shape == (2, 10, 10)

def test_none_logLik(generator2, gen_output2):
    """
    Tests the functionality of EggGeneric when discrete edge features are not 
    requested. 
    """
    loss_1 = (torch.tensor([1.0, 1.0]) @ gen_output2['C_x_logLik']).sum()
    loss_2 = (torch.tensor([1.0, 1.0]) @ gen_output2['C_e_logLik']).sum()

    loss_3 = loss_1.clone()
    loss_4 = loss_1.clone() + loss_2

    assert (
        torch.allclose(
            loss_1, 
            gen_output2['C_x_logLik'].sum()
        )
    ) 

    # Loss contribution of an un-requested feature is expected to be zero.
    assert(
        torch.equal(
            loss_2, 
            torch.tensor(0.0)
        ) 
    )

    check_grads_exist(loss_1, generator2, retain=True)

    # Double checks that adding 0 to the loss does change the gradients. 
    compare_grads(loss_3, loss_4, generator2, retain=True) 
