import pytest


torch = pytest.importorskip('torch')
pytest.importorskip('torch_geometric')
from torch_geometric.data import Data

from new_src.config import DatasetConfig, ExperimentConfig, GeneratorConfig
from new_src.data import DatasetSplits
from new_src.workflow import GNNEggExperiment


def _make_splits(num_nodes: int) -> DatasetSplits:
    edge_index = torch.stack(
        [torch.arange(num_nodes - 1, dtype=torch.long), torch.arange(1, num_nodes, dtype=torch.long)]
    )
    data = Data(
        x=torch.ones((num_nodes, 1), dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor([0], dtype=torch.long),
        num_nodes=num_nodes,
    )
    return DatasetSplits(train=[data], val=[data], test=[data])


def _make_experiment(max_node_size=None) -> GNNEggExperiment:
    config = ExperimentConfig(
        dataset=DatasetConfig(name='Dummy'),
        generator=GeneratorConfig(max_node_size=max_node_size),
    )
    return GNNEggExperiment(config)


def test_build_generator_auto_expands_to_dataset():
    splits = _make_splits(num_nodes=6)
    experiment = _make_experiment(max_node_size=None)

    generator = experiment.build_generator(splits, node_feature_dim=1, edge_feature_dim=0)

    assert generator.max_node_size == 6


def test_build_generator_rejects_too_small_max_node_size():
    splits = _make_splits(num_nodes=6)
    experiment = _make_experiment(max_node_size=4)

    with pytest.raises(ValueError, match=r'max_node_size \(4\) is smaller'):
        experiment.build_generator(splits, node_feature_dim=1, edge_feature_dim=0)