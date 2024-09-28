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
from torch_geometric.data import Batch
import pygmtools as pygm

from egg_models import egg_generic
from egg_models import egg_generic_losses

from utils import oral_ceograph, ceograph

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
def generator_oral_ceograph_no_nodes(device):
    """
    Create test case oral ceograph for an EggGeneric generator model in the 
    case where all nodes get cleared.  
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

    mock_generator.AdjacencyMatrix.probs = torch.nn.Parameter(
        torch.zeros_like(mock_generator.AdjacencyMatrix.probs)
    )

    mock_generator.to(device)

    return mock_generator

@pytest.fixture
def generator_ceograph(device):
    """
    Testable EggGeneric generator model for lung ceograph data.
    """
    mock_generator = egg_generic.EggGeneric(
        max_node_size=10, 
        cont_node_feats=11,
        cont_edge_feats=2, 
        dis_node_feats=(6,),
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
def gen_output_oral_ceograph_no_nodes(generator_oral_ceograph_no_nodes):
    """
    Output test case oral ceograph an EggGeneric generator model for the case
    where all nodes get cleared. 
    """
    return generator_oral_ceograph_no_nodes()

@pytest.fixture
def gen_output_ceograph(generator_ceograph):
    """
    Output test case lung ceograph an EggGeneric generator model. 
    """
    return generator_ceograph()

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
def explainee_ceograph(device):
    """
    Returns a mock explainee for the lung ceograph model.
    """
    mock_explainee = ceograph.NucleiNet(11, 2, batch=True)
    mock_explainee.to(device)
    mock_explainee.load_state_dict(
        torch.load("data/explainees/ceograph/epoch_263.pt", 
                    map_location=device)
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
        target=torch.tensor([1.0, 0.0, 0.0]), 
        uninfo_target=torch.tensor([0.33, 0.33, 0.33]), 
        obs_data_list=None, 
        optimizer=None, 
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
def data_list_ceograph(gen_output_ceograph):
    """
    Returns a mock data list for lung ceograph training data.
    """
    mock_data_list = ceograph.egg_to_ex(gen_output_ceograph) 
    mock_data_list.to(torch.device('cpu'))

    return mock_data_list.to_data_list()

@pytest.fixture
def EggGenericTrainer2(generator2, explainee2, data_list2):
    mock_trainer = egg_generic.EggGenericTrainer(
        model=generator2, 
        explainee=explainee2, 
        target=torch.tensor([1.0, 0.0, 0.0]), 
        uninfo_target=torch.tensor([0.33, 0.33, 0.33]), 
        obs_data_list=data_list2, 
        optimizer=None, 
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
       target=torch.tensor([1.0, 0.0]), 
       uninfo_target=torch.tensor([0.5, 0.5]),
       obs_data_list=data_list_oral_ceograph, 
       optimizer=None, 
       tensorboard_path=None, 
       checkpoint_path=None, 
    )

    return mock_trainer

@pytest.fixture
def EggGenericTrainer_ceograph(generator_ceograph, 
                               explainee_ceograph, 
                               data_list_ceograph):

    mock_trainer = egg_generic.EggGenericTrainer(
       model=generator_ceograph, 
       explainee=explainee_ceograph, 
       target=torch.tensor([1.0, 0.0]), 
       uninfo_target=torch.tensor([0.5, 0.5]),
       obs_data_list=data_list_ceograph, 
       optimizer=None, 
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

@pytest.fixture
def avg_embed_targets_ceograph(device):
    mock_embeds = {
       'conv1': torch.randn((1, 10)), 
       'conv2': torch.randn((1, 10))
    }

    return mock_embeds

@pytest.fixture
def gamma_oral_ceograph(target_oral_ceograph): 
    uninfo = torch.tensor([0.5, 0.5])
    gamma = F.cross_entropy(target_oral_ceograph, uninfo)
    return gamma

@pytest.fixture
def EggGenericTrainerFull_ceograph(
    generator_ceograph, 
    explainee_ceograph, 
    data_list_ceograph, 
    avg_embed_targets_ceograph):
    mock_trainer = egg_generic.EggGenericTrainer(
        model=generator_ceograph, 
        explainee=explainee_ceograph, 
        target=torch.tensor([1.0, 0.0]), 
        uninfo_target=torch.tensor([0.5, 0.5]), 
        avg_embed_targets=avg_embed_targets_ceograph, 
        avg_embed_other_class=avg_embed_targets_ceograph, 
        cont_node_indices=(slice(0, 11), ), 
        dis_node_indices=(slice(11, 18), ), 
        cont_edge_indices=(slice(1, 3), ), 
        dis_edge_indices=(3, ), 
        obs_data_list=data_list_ceograph, 
        optimizer=torch.optim.RMSprop(generator_ceograph.parameters()), 
        tensorboard_path=None, 
        checkpoint_path=None, 
        retain_comp_graph=True
    )

    # Overload data formatting.
    # def ex_to_egg(self, obs_batch): 
    #     """
    #     """
    #     return ceograph.ex_to_egg(obs_batch, 
    #                                 self.model.dis_node_feats[0])

    mock_trainer.egg_to_ex = ceograph.egg_to_ex
    mock_trainer.ex_to_egg = partial(ceograph.ex_to_egg, num_cell_types=6)
    mock_trainer.egg_to_egg = ceograph.egg_to_egg

    return mock_trainer

@pytest.fixture
def basic_graph():

    X_base = torch.tensor(
        [
            [1.0, 0.0, 0.0], 
            [1.0, 0.0, 0.0], 
            [1.0, 0.0, 0.0], 
            [0.0, 1.0, 0.0], 
            [0.0, 0.0, 1.0]
        ]
    )

    A_base = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 2, 3, 4, 4], 
            [1, 4, 0, 2, 1, 3, 4, 2, 0, 2]
        ]
    )

    E_base = torch.tensor(
        [
            [1.0, 0.0, 0.0], 
            [0.0, 1.0, 0.0], 
            [1.0, 0.0, 0.0], 
            [1.0, 0.0, 0.0], 
            [1.0, 0.0, 0.0], 
            [0.0, 0.0, 1.0], 
            [0.0, 0.0, 1.0], 
            [0.0, 0.0, 1.0], 
            [0.0, 1.0, 0.0], 
            [0.0, 0.0, 1.0], 
        ]
    )

    return pyg.data.Batch(x=X_base, edge_index=A_base, edge_attr=E_base)

@pytest.fixture
def basic_graph_extended():

    X_base = torch.tensor(
        [
            [1.0, 0.0, 0.0], 
            [1.0, 0.0, 0.0], 
            [1.0, 0.0, 0.0], 
            [0.0, 1.0, 0.0], 
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0], 
            [0.0, 0.0, 1.0], 
            [1.0, 0.0 ,0.0], 
        ]
    )

    A_base = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6, 6, 7, 7], 
            [1, 4, 0, 2, 1, 3, 4, 2, 4, 5, 0, 2, 3, 3, 0, 4, 0, 2]
        ]
    )

    E_base = torch.tensor(
        [
            [1.0, 0.0, 0.0], 
            [0.0, 1.0, 0.0], 
            [1.0, 0.0, 0.0], 
            [1.0, 0.0, 0.0], 
            [1.0, 0.0, 0.0], 
            [0.0, 0.0, 1.0], 
            [0.0, 0.0, 1.0], 
            [0.0, 0.0, 1.0], 
            [0.0, 0.0, 1.0], 
            [0.0, 0.0, 1.0], 
            [0.0, 1.0, 0.0], 
            [0.0, 0.0, 1.0], 
            [1.0, 0.0, 0.0], 
            [0.0, 1.0, 0.0], 
            [1.0, 0.0, 0.0], 
            [1.0, 0.0, 0.0], 
            [1.0, 0.0, 0.0], 
            [0.0, 0.0, 1.0], 
        ]
    )
    return pyg.data.Batch(x=X_base, edge_index=A_base, edge_attr=E_base)

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

def affinity_fn(feat1: torch.tensor, feat2: torch.tensor) -> torch.tensor:
        feat1_norm = F.normalize(feat1, p=2, dim=-1)
        feat2_norm = F.normalize(feat2, p=2, dim=-1)

        cos_sim_mat = torch.einsum('bij, bkj -> bik', 
                                   feat1_norm, feat2_norm)

        return -1 * (1 - cos_sim_mat)

def create_aff_mat(G1: pyg.data.Batch, G2: pyg.data.Batch) -> torch.tensor:

    n1 = G1.x.size(0) 
    ne1 = G1.edge_index.size(1)
    n2 = G2.x.size(0) 
    ne2 = G2.edge_index.size(1)

    aff_mat = pygm.utils.build_aff_mat(
        node_feat1=G1.x, 
        edge_feat1=G1.edge_attr, 
        connectivity1=G1.edge_index.transpose(0, 1), 
        node_feat2=G2.x, 
        edge_feat2=G2.edge_attr, 
        connectivity2=G2.edge_index.transpose(0, 1), 
        node_aff_fn=affinity_fn, 
        edge_aff_fn=affinity_fn,
        n1=n1, 
        ne1=ne1, 
        n2=n2,
        ne2=ne2
    ) 

    return aff_mat

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


    only_self_loops = gen_output_oral_ceograph
    only_self_loops['adjacency_matrix'] = torch.zeros_like(
        only_self_loops['adjacency_matrix']
    )
    diag = torch.arange(only_self_loops['adjacency_matrix'].shape[1])
    only_self_loops['adjacency_matrix'][:, diag, diag] = 1.

    only_self_loops_cleaned = oral_ceograph.egg_to_ex(only_self_loops)

    assert contains_self_loops(only_self_loops_cleaned.edge_index) == False
    assert contains_isolated_nodes(only_self_loops_cleaned.edge_index) == False
    assert only_self_loops_cleaned.x.sum() == 0
    assert only_self_loops_cleaned.edge_index.shape[1] == 0
    assert only_self_loops_cleaned.edge_attr.shape[0] == 0

def test_ceograph_egg_to_ex(gen_output_ceograph, 
                            explainee_ceograph):
    """
    Tests that the egg_to_ex function returns a tensor and that the forward 
    pass through the explainee returns the expected shape.
    """
    egg_formatted_to_ex = ceograph.egg_to_ex(gen_output_ceograph)

    assert egg_formatted_to_ex is not None

    assert contains_self_loops(egg_formatted_to_ex.edge_index) == False

    assert contains_isolated_nodes(egg_formatted_to_ex.edge_index) == False

    check_graph_grad_fns(egg_formatted_to_ex)

    assert isinstance(explainee_ceograph(egg_formatted_to_ex), torch.Tensor)
    
    assert explainee_ceograph(egg_formatted_to_ex).shape == (2, 2)


    only_self_loops = gen_output_ceograph
    only_self_loops['adjacency_matrix'] = torch.zeros_like(
        only_self_loops['adjacency_matrix']
    )
    diag = torch.arange(only_self_loops['adjacency_matrix'].shape[1])
    only_self_loops['adjacency_matrix'][:, diag, diag] = 1.

    only_self_loops_cleaned = ceograph.egg_to_ex(only_self_loops)

    assert contains_self_loops(only_self_loops_cleaned.edge_index) == False
    assert contains_isolated_nodes(only_self_loops_cleaned.edge_index) == False
    assert only_self_loops_cleaned.x.sum() == 0
    assert only_self_loops_cleaned.edge_index.shape[1] == 0
    assert only_self_loops_cleaned.edge_attr.shape[0] == 0

def test_oral_ceograph_egg_to_egg(gen_output_oral_ceograph):
    gen_X, gen_A, gen_E = oral_ceograph.egg_to_egg(gen_output_oral_ceograph) 

    assert isinstance(gen_X, torch.Tensor)
    assert isinstance(gen_A, torch.Tensor)
    assert isinstance(gen_E, torch.Tensor)

    assert hasattr(gen_X, 'grad_fn')
    assert hasattr(gen_A, 'grad_fn')
    assert hasattr(gen_E, 'grad_fn') 

def test_ceograph_egg_to_egg(gen_output_ceograph):
    gen_X, gen_A, gen_E = ceograph.egg_to_egg(gen_output_ceograph) 

    assert isinstance(gen_X, torch.Tensor)
    assert isinstance(gen_A, torch.Tensor)
    assert isinstance(gen_E, torch.Tensor)

    assert hasattr(gen_X, 'grad_fn')
    assert hasattr(gen_A, 'grad_fn')
    assert hasattr(gen_E, 'grad_fn') 

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

@pytest.mark.parametrize("sub_sampler", [
    (None), 
    ("default"), 
    ("else")
])
def test_create_dataloader_ceograph(EggGenericTrainer_ceograph, 
                                         sub_sampler): 
    """
    Tests is a dataloader is returned, that iterating return a batch, and 
    the forward pass through the explainee produces at least 1 gradient. 
    """
    EggGenericTrainer_ceograph.sub_sampler = sub_sampler

    if sub_sampler == "else": 
        with pytest.raises(ValueError):
            EggGenericTrainer_ceograph.create_data_loader()

    else: 
        EggGenericTrainer_ceograph.create_data_loader()
        assert isinstance(EggGenericTrainer_ceograph.obs_data_loader, 
                        pyg.loader.DataLoader)

        for _, obs_batch in enumerate(
            EggGenericTrainer_ceograph.obs_data_loader): 
            
            assert isinstance(obs_batch, pyg.data.Batch)

            obs_batch.to(EggGenericTrainer_ceograph.model.device_param.device)
            generated = EggGenericTrainer_ceograph.model()

            gen_ex = ceograph.egg_to_ex(generated)

            pred = EggGenericTrainer_ceograph.explainee(gen_ex)

            assert isinstance(
                pred, 
                torch.Tensor
            )

            assert pred.shape == (2, 2) 

            loss = pred.sum()

            check_at_least_1_grad(loss, EggGenericTrainer_ceograph.model)

@pytest.mark.parametrize(
    "dict1, dict2, act_pool_func, agg_func, expected", 
    [
        ({"layer_1": torch.ones((2, 10)), "layer_2": torch.zeros((2, 25))}, 
         {"layer_1": -1 * torch.ones((2, 10)), "layer_2": torch.ones((2, 25))}, 
         lambda x, batch, size: x, torch.mean, torch.tensor([1.5, 1.5])), 
        ({"layer_1": torch.ones((2, 10)), "layer_2": torch.zeros((2, 25))}, 
         {"layer_1": -1 * torch.ones((2, 10)), "layer_2": torch.ones((2, 25))}, 
         lambda x, batch, size: x, torch.sum, torch.tensor([3., 3.])), 
        ({"layer_1": torch.ones((2, 10)), "layer_2": torch.zeros((2, 25))}, 
         {"layer_1": -1 * torch.ones((1, 10)), "layer_2": torch.ones((1, 25))}, 
         lambda x, batch, size: x, torch.mean, torch.tensor([1.5, 1.5])), 
        ({"layer_1": torch.ones((2, 10)), "layer_2": torch.zeros((2, 25))}, 
         {"layer_1": -1 * torch.ones((1, 10)), "layer_2": torch.ones((1, 25))}, 
         lambda x, batch, size: x, torch.sum, torch.tensor([3., 3.])), 
        ({"layer_1": torch.cat([torch.ones((1, 10)), torch.zeros((1, 10))]), 
          "layer_2": torch.cat([-1 * torch.ones((1, 25)), torch.ones((1, 25))])}, 
         {"layer_1": -1 * torch.ones((1, 10)), "layer_2": torch.ones((1, 25))}, 
         lambda x, batch, size: x, torch.sum, torch.tensor([4., 1.])), 
        ({"layer_1": torch.cat([torch.ones((1, 10)), torch.zeros((1, 10))]), 
          "layer_2": torch.cat([-1 * torch.ones((1, 25)), torch.ones((1, 25))])}, 
         {"layer_1": -1 * torch.ones((1, 10)), "layer_2": torch.ones((1, 25))}, 
         lambda x, batch, size: x, torch.mean, torch.tensor([2., 0.5]))
    ]
)
def test_dict_cos_dist(dict1, dict2, act_pool_func, agg_func, expected):
    """
    General test cases for embedding cosine distance. 
    """
    loss = egg_generic_losses.dict_cos_dist(dict1, dict2, 
                                            batch_indices1=None, 
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

    loss, activations, batch_indices = loss_func(egg_formatted_to_ex)

    assert loss.shape == (2,) 

    check_at_least_1_grad(loss, generator_oral_ceograph)

def test_PredLossBatched_oral_ceograph_no_nodes(target_oral_ceograph, 
                                       explainee_oral_ceograph, 
                                       avg_embed_targets_oral_ceograph, 
                                       generator_oral_ceograph_no_nodes, 
                                       gen_output_oral_ceograph_no_nodes):
    """
    Test the output shape and gradients for the oral ceograph test case. 
    """
    loss_func = egg_generic_losses.PredLossBatched(
        target=target_oral_ceograph, explainee=explainee_oral_ceograph, 
        avg_embed_targets=avg_embed_targets_oral_ceograph
    )
    egg_formatted_to_ex = oral_ceograph.egg_to_ex(gen_output_oral_ceograph_no_nodes)

    loss, activations, batch_indices = loss_func(egg_formatted_to_ex)
    assert loss.shape == (2,) 
    assert activations is not None
    assert batch_indices is not None

    check_at_least_1_grad(loss, generator_oral_ceograph_no_nodes)

def test_PredLossBatched_ceograph(target_oral_ceograph, 
                                       explainee_ceograph, 
                                       avg_embed_targets_ceograph, 
                                       generator_ceograph, 
                                       gen_output_ceograph):
    """
    Test the output shape and gradients for the oral ceograph test case. 
    """
    loss_func = egg_generic_losses.PredLossBatched(
        target=target_oral_ceograph, explainee=explainee_ceograph, 
        avg_embed_targets=avg_embed_targets_ceograph
    )
    egg_formatted_to_ex = ceograph.egg_to_ex(gen_output_ceograph)

    loss, activations, batch_indices = loss_func(egg_formatted_to_ex)

    assert loss.shape == (2,) 

    check_at_least_1_grad(loss, generator_ceograph)

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

def test_GEDasMatchLoss_oral_ceograph(gen_output_oral_ceograph, data_list_oral_ceograph, 
                        generator_oral_ceograph, 
                        device):
    """
    Test the output shape, range, and all model gradients for the GEDasMatchLoss 
    function for the oral ceograph test case. 
    """
    loss_fn = egg_generic_losses.GEDasMatchLoss(
        node_size=10, 
        cont_node_indices=(slice(0, 11), ), 
        dis_node_indices=(slice(11, 16), ),
        cont_edge_indices=(slice(1, 3), ), 
        dis_edge_indices=(3, ), 
    )

    gen = oral_ceograph.egg_to_egg(gen_output_oral_ceograph)

    ex_batch = Batch.from_data_list(data_list_oral_ceograph).to(device)

    obs = oral_ceograph.ex_to_egg(ex_batch, 4)

    loss = loss_fn(*gen, *obs)

    assert loss.shape == (2,)

    # assert (aff_mat <= 0.0).all()
    # assert (aff_mat >= -1.0).all()

    assert (loss >= -1e-5).all()
    assert (loss <= gen[0].shape[1] + gen[2].shape[1]).all()

    check_grads_exist(loss, generator_oral_ceograph)

def test_GEDasMatchLoss_ceograph(gen_output_ceograph, data_list_ceograph, 
                        generator_ceograph, 
                        device):
    """
    Test the output shape, range, and all model gradients for the GEDasMatchLoss 
    function for the oral ceograph test case. 
    """
    loss_fn = egg_generic_losses.GEDasMatchLoss(
        node_size=10, 
        cont_node_indices=(slice(0, 11), ), 
        dis_node_indices=(slice(11, 18), ),
        cont_edge_indices=(slice(1, 3), ), 
        dis_edge_indices=(3, ), 
    )

    gen = ceograph.egg_to_egg(gen_output_ceograph)

    ex_batch = Batch.from_data_list(data_list_ceograph).to(device)

    obs = ceograph.ex_to_egg(ex_batch, 6)

    loss = loss_fn(*gen, *obs)

    assert loss.shape == (2,)

    # assert (aff_mat <= 0.0).all()
    # assert (aff_mat >= -1.0).all()

    assert (loss >= -1e-5).all()
    assert (loss <= gen[0].shape[1] + gen[2].shape[1]).all()

    check_grads_exist(loss, generator_ceograph)

def test_GEDasMatchLoss_identity_oral_ceograph(
    gen_output_oral_ceograph, data_list_oral_ceograph, 
    generator_oral_ceograph, device):
    """
    Test the output shape, range, and all model gradients for the GEDasMatchLoss 
    function using the identity solver for the oral ceograph test case. 
    """
    loss_fn = egg_generic_losses.GEDasMatchLoss(
        node_size=10, 
        cont_node_indices=(slice(0, 11), ), 
        dis_node_indices=(slice(11, 16), ),
        cont_edge_indices=(slice(1, 3), ), 
        dis_edge_indices=(3, ), 
        QAP_solver="identity"
    )

    gen = oral_ceograph.egg_to_egg(gen_output_oral_ceograph)

    ex_batch = Batch.from_data_list(data_list_oral_ceograph).to(device)

    obs = oral_ceograph.ex_to_egg(ex_batch, 4)

    loss = loss_fn(*gen, *obs)

    assert loss.shape == (2,)

    # assert (aff_mat <= 0.0).all()
    # assert (aff_mat >= -1.0).all()

    assert (loss >= -1e-5).all()
    assert (loss <= gen[0].shape[1] + gen[2].shape[1]).all()

    check_grads_exist(loss, generator_oral_ceograph)

    loss_0 = loss_fn(*obs, *obs)

    assert loss_0.shape == (2,)
    assert torch.allclose(loss_0, torch.zeros_like(loss_0), atol=1e-5)
    assert (loss_0 <= obs[0].shape[1] + obs[2].shape[1]).all()

def test_StructuralLoss_oral_ceograph(explainee_oral_ceograph, 
                                      gamma_oral_ceograph, 
                                      target_oral_ceograph, 
                                      gen_output_oral_ceograph, 
                                      data_list_oral_ceograph,
                                      generator_oral_ceograph, 
                                      avg_embed_targets_oral_ceograph, 
                                      device): 
    """
    Tests that the structural loss that integrates GED, prediction, and 
    embedding losses returns the correct shape and creates all gradients.  
    """
    GED_fn = egg_generic_losses.GEDasMatchLoss(
        node_size=10, 
        cont_node_indices=(slice(0, 11), ), 
        dis_node_indices=(slice(11, 16), ),
        cont_edge_indices=(slice(1, 3), ), 
        dis_edge_indices=(3, ), 
    )

    loss_fn = egg_generic_losses.StructuralLoss(
        GED_fn, explainee_oral_ceograph, 
        gamma_oral_ceograph, target_oral_ceograph
    )

    pred_loss = egg_generic_losses.PredLossBatched(
        target=target_oral_ceograph, explainee=explainee_oral_ceograph, 
        avg_embed_targets=avg_embed_targets_oral_ceograph
    )

    gen_ex = oral_ceograph.egg_to_ex(gen_output_oral_ceograph)

    gen_egg = oral_ceograph.egg_to_egg(gen_output_oral_ceograph)

    obs_ex = Batch.from_data_list(data_list_oral_ceograph).to(device)

    obs_egg = oral_ceograph.ex_to_egg(obs_ex, 4)

    _, activations, batch_indices = pred_loss(gen_ex)

    loss = loss_fn(gen_egg, obs_egg, 
                   gen_ex, obs_ex, 
                   gen_acts=activations, 
                   gen_acts_batch=batch_indices)

    assert loss.shape == (2, )

    check_grads_exist(loss, generator_oral_ceograph)

def test_StructuralLoss_oral_ceograph_no_nodes(explainee_oral_ceograph, 
                                      gamma_oral_ceograph, 
                                      target_oral_ceograph, 
                                      gen_output_oral_ceograph_no_nodes, 
                                      data_list_oral_ceograph,
                                      generator_oral_ceograph_no_nodes, 
                                      avg_embed_targets_oral_ceograph, 
                                      device): 
    """
    Tests that the structural loss that integrates GED, prediction, and 
    embedding losses returns the correct shape and creates all gradients.  
    """
    GED_fn = egg_generic_losses.GEDasMatchLoss(
        node_size=10, 
        cont_node_indices=(slice(0, 11), ), 
        dis_node_indices=(slice(11, 16), ),
        cont_edge_indices=(slice(1, 3), ), 
        dis_edge_indices=(3, ), 
    )

    loss_fn = egg_generic_losses.StructuralLoss(
        GED_fn, explainee_oral_ceograph, 
        gamma_oral_ceograph, target_oral_ceograph
    )

    pred_loss = egg_generic_losses.PredLossBatched(
        target=target_oral_ceograph, explainee=explainee_oral_ceograph, 
        avg_embed_targets=avg_embed_targets_oral_ceograph
    )

    gen_ex = oral_ceograph.egg_to_ex(gen_output_oral_ceograph_no_nodes)

    gen_egg = oral_ceograph.egg_to_egg(gen_output_oral_ceograph_no_nodes)

    obs_ex = Batch.from_data_list(data_list_oral_ceograph).to(device)

    obs_egg = oral_ceograph.ex_to_egg(obs_ex, 4)

    _, activations, batch_indices = pred_loss(gen_ex)

    loss = loss_fn(gen_egg, obs_egg, 
                   gen_ex, obs_ex, 
                   gen_acts=activations, 
                   gen_acts_batch=batch_indices)

    assert loss.shape == (2, )

    check_grads_exist(loss, generator_oral_ceograph_no_nodes)

def test_StructuralLoss_ceograph(explainee_ceograph, 
                                      gamma_oral_ceograph, 
                                      target_oral_ceograph, 
                                      gen_output_ceograph, 
                                      data_list_ceograph,
                                      generator_ceograph, 
                                      avg_embed_targets_ceograph, 
                                      device): 
    """
    Tests that the structural loss that integrates GED, prediction, and 
    embedding losses returns the correct shape and creates all gradients.  
    """
    GED_fn = egg_generic_losses.GEDasMatchLoss(
        node_size=10, 
        cont_node_indices=(slice(0, 11), ), 
        dis_node_indices=(slice(11, 16), ),
        cont_edge_indices=(slice(1, 3), ), 
        dis_edge_indices=(3, ), 
    )

    loss_fn = egg_generic_losses.StructuralLoss(
        GED_fn, explainee_ceograph, 
        gamma_oral_ceograph, target_oral_ceograph
    )

    pred_loss = egg_generic_losses.PredLossBatched(
        target=target_oral_ceograph, explainee=explainee_ceograph, 
        avg_embed_targets=avg_embed_targets_ceograph
    )

    gen_ex = ceograph.egg_to_ex(gen_output_ceograph)

    gen_egg = ceograph.egg_to_egg(gen_output_ceograph)

    obs_ex = Batch.from_data_list(data_list_ceograph).to(device)

    obs_egg = ceograph.ex_to_egg(obs_ex, 6)

    _, activations, batch_indices = pred_loss(gen_ex)

    loss = loss_fn(gen_egg, obs_egg, 
                   gen_ex, obs_ex, 
                   gen_acts=activations, 
                   gen_acts_batch=batch_indices)

    assert loss.shape == (2, )

    check_grads_exist(loss, generator_ceograph)

def test_ceograph_trainer_full(EggGenericTrainerFull_ceograph):

    results = EggGenericTrainerFull_ceograph.train_one_epoch()

    model = EggGenericTrainerFull_ceograph.model

    for name, param in model.named_parameters():
        if name != "device_param": 
            assert param.grad is not None, (
                f"Parameter '{name}' does not have gradients"
            )

# def test_GED_reversible(basic_graph, basic_graph_extended):
    
#     dist_2_1 = 0.
#     dist_1_2 = 0.
#     for i in range(100):
#         n1 = basic_graph.x.size(0)
#         ne1 = basic_graph.edge_index.size(0)
#         n2 = basic_graph_extended.x.size(0)
#         ne2 = basic_graph_extended.edge_index.size(0)

#         aff_mat_G2_to_G1 = create_aff_mat(basic_graph, basic_graph_extended)
#         aff_mat_G1_to_G2 = create_aff_mat(basic_graph_extended, basic_graph)

#         assert not torch.allclose(aff_mat_G2_to_G1, aff_mat_G1_to_G2), f"On index {i}"

#         GED_2_to_1 = (
#             -1 * pygm.utils.compute_affinity_score(
#                 pygm.hungarian(pygm.ngm(aff_mat_G2_to_G1, n1, n2)), 
#                 aff_mat_G2_to_G1
#             )
#         )
#         dist_2_1 += GED_2_to_1

#         GED_1_to_2 = (
#             -1 * pygm.utils.compute_affinity_score(
#                 pygm.hungarian(pygm.ngm(aff_mat_G2_to_G1, n2, n1)), 
#                 aff_mat_G2_to_G1
#             )
#         )
#         dist_1_2 += GED_1_to_2

#     assert torch.allclose(dist_2_1 / 100, dist_1_2 / 100, rtol=1e-2)

def test_GED_del_zero(basic_graph, basic_graph_extended):

    aff_mat_G2_to_G1 = create_aff_mat(basic_graph, basic_graph_extended)
    aff_mat_G1_to_G2 = create_aff_mat(basic_graph_extended, basic_graph)

    true_match_2_to_1 = torch.tensor(
        [
            [1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0.],
        ]
    )

    GED_2_to_1 = (
        -1 * pygm.utils.compute_affinity_score(
            true_match_2_to_1, aff_mat_G2_to_G1
        )
    )

    true_match_1_to_2 = torch.tensor(
        [
            [1., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
        ]
    )

    GED_1_to_2 = (
        -1 * pygm.utils.compute_affinity_score(
            true_match_1_to_2, aff_mat_G1_to_G2
        )
    )

    assert torch.allclose(GED_2_to_1, GED_1_to_2)
    