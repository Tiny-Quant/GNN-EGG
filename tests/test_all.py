# %% Dependencies
import pytest

from functools import partial
from typing import Dict

import sys
from os.path import dirname, abspath
repo_dir = dirname(dirname(abspath(__file__)))
sys.path.append(repo_dir)

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.utils import contains_isolated_nodes, contains_self_loops

from egg_models import egg_generic
from egg_models import egg_generic_losses

from utils import oral_ceograph

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True 
#np.random.seed(SEED)

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
def gen_output_oral_ceograph(generator_oral_ceograph):
    """
    Output test case oral ceograph an EggGeneric generator model. 
    """
    return generator_oral_ceograph()

@pytest.fixture
def explainee2(): 
    """
    Returns a basic GCN model that works with generator2. 
    """
    mock_explainee = GCN(hidden_channels=64)
    mock_explainee.eval()

    return mock_explainee

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

@pytest.fixture
def data_list2(generator2, explainee2, gen_output2):
    """
    Returns a mock data_list for generator2. 
    """
    dummy_trainer = egg_generic.EggGenericTrainer(
        model=generator2, 
        explainee=explainee2, 
        target=None, 
        obs_data_list=None, 
        optimizer=None, 
        loss_term_weights=None, 
        tensorboard_path=None, 
        checkpoint_path=None,
    )
    mock_data_list = dummy_trainer.egg_to_ex(gen_output2) 
    #mock_data_list.to(torch.device('cpu'))

    return mock_data_list.to_data_list()

@pytest.fixture
def data_list_oral_ceograph(gen_output_oral_ceograph):
    """
    Returns a mock data_list of oral ceograph training data. 
    """
    mock_data_list = oral_ceograph.egg_to_ex(gen_output_oral_ceograph)
    mock_data_list.to(torch.device('cpu'))

    return mock_data_list.to_data_list()

@pytest.fixture
def EggGenericTrainer2(generator2, explainee2, data_list2):
    mock_trainer = egg_generic.EggGenericTrainer(
        model=generator2, 
        explainee=explainee2, 
        target=None, 
        obs_data_list=data_list2, 
        optimizer=None, 
        loss_term_weights=None, 
        tensorboard_path=None, 
        checkpoint_path=None,
    )

    return mock_trainer

@pytest.fixture
def EggGenericTrainer_oral_ceograph(generator_oral_ceograph, 
                                    explainee_oral_ceograph, 
                                    data_list_oral_ceograph):

    mock_trainer = egg_generic.EggGenericTrainer(
       model=generator_oral_ceograph, 
       explainee=explainee_oral_ceograph, 
       target=None, 
       obs_data_list=data_list_oral_ceograph, 
       optimizer=None, 
       loss_term_weights=None, 
       tensorboard_path=None, 
       checkpoint_path=None, 
    )

    return mock_trainer

@pytest.fixture
def target2():
    return torch.tensor([1.0, 0.0, 0.0])

@pytest.fixture
def target_oral_ceograph():
    target = torch.tensor([1.0, 0.0])
    return target

@pytest.fixture
def avg_embed_targets_oral_ceograph(device):
    mock_embeds = {
       'conv2': torch.randn((1, 20)), 
       'conv3': torch.randn((1, 20))
    }

    return mock_embeds

# %% Helper Functions
def check_grads_exist(loss: torch.tensor, model: torch.nn.Module, retain=False): 
    """
    Asserts that the gradients of all model parameters (except "device_param")
    are not None. 
    """
    loss.sum().backward(retain_graph=retain) # Sum ensures scalar loss. 

    for name, param in model.named_parameters():
        if name != "device_param": 
            assert param.grad is not None, (
                f"Parameter '{name}' does not have gradients"
            )

def check_at_least_1_grad(loss: torch.tensor, model: torch.nn.Module, 
                          retain=False):
    """
    Asserts that a least one parameter has a gradient.  
    """
    loss.sum().backward(retain_graph=retain)

    num_grads = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f'{name} has gradient.')
            num_grads += 1
    
    assert num_grads > 0, "There are no gradients."

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
            assert hasattr(value, 'grad_fn'), (
                f"Tensor {key} does not have grad_fn attribute."
            ) 
# %% Mock Standard Explainee GCN Model 
# Reference: https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing#scrollTo=CN3sRVuaQ88l
class GCN(torch.nn.Module):
    """
    Stock standard GCN model.  
    """
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = pyg.nn.GCNConv(13, hidden_channels)
        self.conv2 = pyg.nn.GCNConv(hidden_channels, hidden_channels)
        self.conv3 = pyg.nn.GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 3)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = pyg.nn.global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        x = F.softmax(x, dim=-1)
        
        return x

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

    # Double checks that adding 0 to the loss does change the gradients. 
    compare_grads(loss_3, loss_4, generator2, retain=True) 

def test_default_egg_to_ex(gen_output2, explainee2, EggGenericTrainer2): 
    """
    Tests that the egg_to_ex function returns a tensor and that the forward 
    pass through the explainee returns the expected shape.
    """
    egg_formatted_to_ex = EggGenericTrainer2.egg_to_ex(gen_output2)

    assert egg_formatted_to_ex is not None

    check_graph_grad_fns(egg_formatted_to_ex)

    assert isinstance(explainee2(egg_formatted_to_ex), torch.Tensor)
    
    assert explainee2(egg_formatted_to_ex).shape == (2, 3)

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

@pytest.mark.parametrize("sub_sampler", [
    (None), 
    ("default"), 
    (partial(pyg.loader.GraphSAINTRandomWalkSampler, 
        batch_size=1, 
        walk_length=10, 
        num_steps=2, 
        sample_coverage=1, 
        log=False
    )), 
    ("else")
])
def test_create_dataloader2(EggGenericTrainer2, sub_sampler): 
    """
    Tests is a dataloader is returned, that iterating return a batch, and 
    the forward pass through the explainee produces at least 1 gradient. 
    """
    EggGenericTrainer2.sub_sampler = sub_sampler

    if sub_sampler == "else": 
        with pytest.raises(ValueError):
            EggGenericTrainer2.create_data_loader()

    else: 
        EggGenericTrainer2.create_data_loader()
        assert isinstance(EggGenericTrainer2.obs_data_loader, 
                        pyg.loader.DataLoader)

        for _, obs_batch in enumerate(
            EggGenericTrainer2.obs_data_loader): 
            
            assert isinstance(obs_batch, pyg.data.Batch)

            obs_batch.to(EggGenericTrainer2.model.device_param.device)
            generated = EggGenericTrainer2.model()

            gen_ex = EggGenericTrainer2.egg_to_ex(generated)

            pred = EggGenericTrainer2.explainee(gen_ex)

            assert isinstance(
                pred, 
                torch.Tensor
            )

            assert pred.shape == (2, 3) 

            loss = pred.sum()

            check_at_least_1_grad(loss, EggGenericTrainer2.model)

@pytest.mark.parametrize("sub_sampler", [
    (None), 
    ("default"), 
    (partial(pyg.loader.GraphSAINTRandomWalkSampler, 
        batch_size=1, 
        walk_length=10, 
        num_steps=2, 
        sample_coverage=1, 
        log=False
    )), 
    ("else")
])
def test_create_dataloader_oral_ceograph(EggGenericTrainer_oral_ceograph, 
                                         sub_sampler): 
    """
    Tests is a dataloader is returned, that iterating return a batch, and 
    the forward pass through the explainee produces at least 1 gradient. 
    """
    EggGenericTrainer_oral_ceograph.sub_sampler = sub_sampler

    if sub_sampler == "else": 
        with pytest.raises(ValueError):
            EggGenericTrainer_oral_ceograph.create_data_loader()

    else: 
        EggGenericTrainer_oral_ceograph.create_data_loader()
        assert isinstance(EggGenericTrainer_oral_ceograph.obs_data_loader, 
                        pyg.loader.DataLoader)

        for _, obs_batch in enumerate(
            EggGenericTrainer_oral_ceograph.obs_data_loader): 
            
            assert isinstance(obs_batch, pyg.data.Batch)

            obs_batch.to(EggGenericTrainer_oral_ceograph.model.device_param.device)
            generated = EggGenericTrainer_oral_ceograph.model()

            gen_ex = oral_ceograph.egg_to_ex(generated)

            pred = EggGenericTrainer_oral_ceograph.explainee(gen_ex)

            assert isinstance(
                pred, 
                torch.Tensor
            )

            assert pred.shape == (2, 2) 

            loss = pred.sum()

            check_at_least_1_grad(loss, EggGenericTrainer_oral_ceograph.model)


@pytest.mark.parametrize(
    "dict1, dict2, act_pool_func, agg_func, expected", 
    [
        ({"layer_1": torch.ones((2, 10)), "layer_2": torch.zeros((2, 25))}, 
         {"layer_1": -1 * torch.ones((2, 10)), "layer_2": torch.ones((2, 25))}, 
         lambda x, batch: x, torch.mean, torch.tensor([1.5, 1.5])), 
        ({"layer_1": torch.ones((2, 10)), "layer_2": torch.zeros((2, 25))}, 
         {"layer_1": -1 * torch.ones((2, 10)), "layer_2": torch.ones((2, 25))}, 
         lambda x, batch: x, torch.sum, torch.tensor([3., 3.])), 
        ({"layer_1": torch.ones((2, 10)), "layer_2": torch.zeros((2, 25))}, 
         {"layer_1": -1 * torch.ones((1, 10)), "layer_2": torch.ones((1, 25))}, 
         lambda x, batch: x, torch.mean, torch.tensor([1.5, 1.5])), 
        ({"layer_1": torch.ones((2, 10)), "layer_2": torch.zeros((2, 25))}, 
         {"layer_1": -1 * torch.ones((1, 10)), "layer_2": torch.ones((1, 25))}, 
         lambda x, batch: x, torch.sum, torch.tensor([3., 3.])), 
        ({"layer_1": torch.cat([torch.ones((1, 10)), torch.zeros((1, 10))]), 
          "layer_2": torch.cat([-1 * torch.ones((1, 25)), torch.ones((1, 25))])}, 
         {"layer_1": -1 * torch.ones((1, 10)), "layer_2": torch.ones((1, 25))}, 
         lambda x, batch: x, torch.sum, torch.tensor([4., 1.])), 
        ({"layer_1": torch.cat([torch.ones((1, 10)), torch.zeros((1, 10))]), 
          "layer_2": torch.cat([-1 * torch.ones((1, 25)), torch.ones((1, 25))])}, 
         {"layer_1": -1 * torch.ones((1, 10)), "layer_2": torch.ones((1, 25))}, 
         lambda x, batch: x, torch.mean, torch.tensor([2., 0.5]))
    ]
)
def test_dict_cos_dist(dict1, dict2, act_pool_func, agg_func, expected):
    """
    General test cases for embedding cosine distance. 
    """
    loss = egg_generic_losses.dict_cos_dist(dict1, dict2, 
                                            batch_indices=None, 
                                            act_pool_func=act_pool_func, 
                                            agg_func=agg_func) 

    assert torch.equal(loss, expected)

def test_PredLossBatched_oral_ceograph(target_oral_ceograph, 
                                       explainee_oral_ceograph, 
                                       avg_embed_targets_oral_ceograph, 
                                       generator_oral_ceograph, 
                                       gen_output_oral_ceograph):
    """
    Test the output shape and gradients for the oral ceograph test case. 
    """
    loss_func = egg_generic_losses.PredLossBatched(
        target=target_oral_ceograph, explainee=explainee_oral_ceograph, 
        avg_embed_targets=avg_embed_targets_oral_ceograph
    )
    egg_formatted_to_ex = oral_ceograph.egg_to_ex(gen_output_oral_ceograph)

    loss, activations = loss_func(egg_formatted_to_ex)

    assert loss.shape == (2,) 

    check_at_least_1_grad(loss, generator_oral_ceograph)

def test_Edge_Penalty(generator, generator2, generator_oral_ceograph): 
    """
    Test that the EdgePenalty function returns and scalar and can produce a 
    gradient. 
    """
    loss_fn_1 = egg_generic_losses.EdgePenalty()
    loss_fn_2 = egg_generic_losses.EdgePenalty(edge_budget=10)

    loss_1 = loss_fn_1(generator.AdjacencyMatrix.probs)
    loss_2 = loss_fn_1(generator2.AdjacencyMatrix.probs)
    loss_3 = loss_fn_2(generator_oral_ceograph.AdjacencyMatrix.probs)
    loss_4 = loss_fn_2(generator.AdjacencyMatrix.probs)
    loss_5 = loss_fn_2(generator2.AdjacencyMatrix.probs)
    loss_6 = loss_fn_2(generator_oral_ceograph.AdjacencyMatrix.probs)

    assert loss_1.shape == torch.Size([])
    assert loss_2.shape == torch.Size([])
    assert loss_3.shape == torch.Size([])
    assert loss_4.shape == torch.Size([])
    assert loss_5.shape == torch.Size([])
    assert loss_6.shape == torch.Size([])

    check_at_least_1_grad(loss_1, generator)
    check_at_least_1_grad(loss_2, generator2)
    check_at_least_1_grad(loss_3, generator_oral_ceograph)
    check_at_least_1_grad(loss_4, generator)
    check_at_least_1_grad(loss_5, generator2)
    check_at_least_1_grad(loss_6, generator_oral_ceograph)
