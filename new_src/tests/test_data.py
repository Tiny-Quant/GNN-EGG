import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.data import Data

from new_src.data import GeneratorAdapter, infer_feature_dimensions


def _sample_graph() -> Data:
    x = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.1, 0.2],
            [0.0, 1.0, 0.0, 0.3, 0.4],
            [0.0, 0.0, 1.0, 0.2, 0.5],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 0]],
        dtype=torch.long,
    )
    edge_attr = torch.tensor(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.4],
            [1.0, 0.0, 0.3],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([0], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def test_generator_adapter_detects_mixed_features():
    data = _sample_graph()
    adapter = GeneratorAdapter([data])
    spec = adapter.spec

    assert spec.max_nodes == data.num_nodes
    assert spec.num_cont_node_feats == 2
    assert spec.dis_node_blocks == (3,)
    assert spec.num_cont_edge_feats == 1
    assert spec.dis_edge_blocks == (2,)

    node_dim, edge_dim = infer_feature_dimensions([data])
    assert node_dim == data.x.size(-1)
    assert edge_dim == data.edge_attr.size(-1)
