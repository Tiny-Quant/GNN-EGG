# %% Dependencies
import pytest

import sys
from os.path import dirname, abspath
repo_dir = dirname(dirname(abspath(__file__)))
sys.path.append(repo_dir)

import torch
from torch_geometric.utils import contains_isolated_nodes, contains_self_loops

from egg_models import egg_generic

from utils import oral_ceograph

# %% Fixture/Mocks/Data Test Cases
@pytest.fixture
def device():
    """Fixture to set the device to CUDA if available, otherwise CPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA is available! Using GPU.')
    else:
        device = torch.device('cpu')
        print('CUDA is not available. Using CPU.')
    return device

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
def generator_oral_ceograph(device):
    """
    Create test case oral ceograph for an EggGeneric generator model. 
    """
    mock_generator = egg_generic.EggGeneric(
        max_node_size=10, 
        cont_node_feats=11, 
        cont_edge_feats=2, 
        dis_node_feats=(4,), 
        # dis_edge_feats=(1, 2), 
        batch_size=2,
        allow_self_loops=False
    )

    mock_generator.to(device)

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

@pytest.fixture
def gen_output_oral_ceograph(generator_oral_ceograph, device):
    """
    Output test case oral ceograph an EggGeneric generator model. 
    """
    #generator_oral_ceograph.to(device)
    return generator_oral_ceograph()

@pytest.fixture
def explainee_oral_ceograph(device):
    """
    Returns a mock explainee for an oral ceograph model. 
    """
    mock_explainee = oral_ceograph.NucleiNet(11, 2, batch=True)
    mock_explainee.to(device)
    mock_explainee.load_state_dict(
        torch.load("data/explainees/HN/epoch_15.pt", map_location=device), 
        strict=False
    )
    mock_explainee.eval()

    return mock_explainee

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

def check_graph_grad_fns(graph: dict):
    """
    check if all the tensors in a dict have grad_fns.
    """
    for key, value in graph:
        if isinstance(value, torch.Tensor):
            assert(
                hasattr(value, 'grad_fn'), 
                f"Tensor {key} does not have grad_fn attribute."
            )

# %% Tests
def test_gen_shapes(generator, gen_output):
    """
    Tests if the output create by test case I matches the expected and 
    documented tensor shapes.
    """
    b = generator.batch_size
    n = generator.max_node_size
    f1 = generator.cont_node_feats
    f2 = generator.dis_node_feats 
    f3 = generator.cont_edge_feats
    f4 = generator.dis_edge_feats

    assert gen_output['dis_node_feats'].shape == (b, n, 1 + 2 + 3)
    assert gen_output['dis_edge_feats'].shape == (b, n**2, 1 + 2)
    assert gen_output['full_edge_indices'].shape == (b, 2, n**2)
    assert gen_output['adjacency_matrix'].shape == (b, n, n)
    assert gen_output['C_x_logLik'].shape == (b, len(f2))
    assert gen_output['C_e_logLik'].shape == (b, len(f4))
    assert gen_output['A_logLik'].shape == (b,)
    assert gen_output['cont_node_feats'].shape == (b, n, n)

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

def test_oral_ceograph_egg_to_ex(gen_output_oral_ceograph, 
                                 explainee_oral_ceograph):
    """
    Tests that the egg_to_ex function returns a tensor and that the forward 
    pass through the explainee returns the expected shape.
    """
    egg_formatted_to_ex = oral_ceograph.egg_to_ex(gen_output_oral_ceograph)

    assert egg_formatted_to_ex is not None

    assert contains_self_loops(egg_formatted_to_ex.edge_index) == False

    assert contains_isolated_nodes(egg_formatted_to_ex.edge_index) == False

    check_graph_grad_fns(egg_formatted_to_ex)

    assert isinstance(explainee_oral_ceograph(egg_formatted_to_ex), 
                                              torch.Tensor)
    
    assert explainee_oral_ceograph(egg_formatted_to_ex).shape == (2, 2)
 