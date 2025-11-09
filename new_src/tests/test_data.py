import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.data import Data

from new_src.data import GeneratorAdapter, infer_feature_dimensions


def _sample_graph(
    include_discrete_nodes: bool = True,
    include_continuous_nodes: bool = True,
    include_discrete_edges: bool = True,
    include_continuous_edges: bool = True,
) -> Data:
    node_parts = []
    if include_discrete_nodes:
        node_parts.append(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            )
        )
    if include_continuous_nodes:
        node_parts.append(
            torch.tensor(
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                    [0.2, 0.5],
                ],
                dtype=torch.float32,
            )
        )

    edge_parts = []
    if include_discrete_edges:
        edge_parts.append(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                ],
                dtype=torch.float32,
            )
        )
    if include_continuous_edges:
        edge_parts.append(
            torch.tensor(
                [[0.5], [0.4], [0.3]],
                dtype=torch.float32,
            )
        )

    x = torch.cat(node_parts, dim=-1) if node_parts else None
    edge_attr = torch.cat(edge_parts, dim=-1) if edge_parts else None
    edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 0]],
        dtype=torch.long,
    )
    y = torch.tensor([0], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def _mutag_graph() -> Data:
    x = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 0]],
        dtype=torch.long,
    )
    edge_attr = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([1], dtype=torch.long)
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


def test_generator_adapter_handles_missing_continuous_features():
    data = _sample_graph(include_continuous_nodes=False, include_continuous_edges=False)
    adapter = GeneratorAdapter([data])
    spec = adapter.spec

    assert spec.num_cont_node_feats == 0
    assert spec.dis_node_blocks == (3,)
    assert spec.num_cont_edge_feats == 0
    assert spec.dis_edge_blocks == (2,)


def test_generator_adapter_handles_missing_discrete_features():
    data = _sample_graph(include_discrete_nodes=False, include_discrete_edges=False)
    adapter = GeneratorAdapter([data])
    spec = adapter.spec

    assert spec.num_cont_node_feats == 2
    assert spec.dis_node_blocks == ()
    assert spec.num_cont_edge_feats == 1
    assert spec.dis_edge_blocks == ()


def test_generator_adapter_mutag_discrete_counts():
    data = _mutag_graph()
    adapter = GeneratorAdapter([data])
    spec = adapter.spec

    assert spec.num_cont_node_feats == 0
    assert spec.dis_node_blocks == (7,)
    assert spec.num_cont_edge_feats == 0
    assert spec.dis_edge_blocks == (4,)
