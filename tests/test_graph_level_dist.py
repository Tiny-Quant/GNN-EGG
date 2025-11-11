import unittest

try:
    import torch
    from torch_geometric.data import Batch, Data
except ModuleNotFoundError:  # pragma: no cover - handled via test skip
    torch = None  # type: ignore
    Batch = Data = None  # type: ignore

try:
    from revisions.new_src.graph_level_dist import mcs_soft_graph_dist
except ModuleNotFoundError:  # pragma: no cover - handled via test skip
    mcs_soft_graph_dist = None  # type: ignore


def dense_graph(adj, features=None) -> Data:
    if torch is None:
        raise RuntimeError("PyTorch is required to construct dense graphs")
    num_nodes = adj.size(0)
    row = torch.arange(num_nodes).repeat_interleave(num_nodes)
    col = torch.arange(num_nodes).repeat(num_nodes)
    mask = row != col
    edge_index = torch.stack([row[mask], col[mask]], dim=0)
    if features is None:
        features = torch.eye(num_nodes, dtype=adj.dtype)
    edge_weight = adj[mask]
    return Data(
        x=features,
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=num_nodes,
    )


@unittest.skipUnless(torch is not None and mcs_soft_graph_dist is not None, "PyTorch is required for this test")
class TestSoftMCSDistance(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_identical_graphs_have_zero_distance(self):
        adj = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        graph = dense_graph(adj)

        module = mcs_soft_graph_dist([graph])
        cont_batch = Batch.from_data_list([graph])

        distance = module(cont_batch)

        self.assertTrue(torch.is_tensor(distance))
        self.assertAlmostEqual(distance.item(), 0.0, places=5)

    def test_distance_positive_for_different_graphs(self):
        adj_obs = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        adj_gen = torch.zeros_like(adj_obs)
        obs_graph = dense_graph(adj_obs)
        gen_graph = dense_graph(adj_gen)

        module = mcs_soft_graph_dist([obs_graph])
        cont_batch = Batch.from_data_list([gen_graph])

        distance = module(cont_batch)

        self.assertGreater(distance.item(), 0.1)

    def test_backward_pass_produces_gradients(self):
        adj = torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
        graph = dense_graph(adj.detach())
        module = mcs_soft_graph_dist([graph])

        cont_batch = Batch.from_data_list([graph])
        cont_batch.edge_weight = cont_batch.edge_weight.clone().detach().requires_grad_(True)

        distance = module(cont_batch)
        distance.backward()

        self.assertIsNotNone(cont_batch.edge_weight.grad)
        self.assertTrue(torch.all(torch.isfinite(cont_batch.edge_weight.grad)))
        self.assertGreater(cont_batch.edge_weight.grad.abs().sum().item(), 0.0)

    def test_mismatched_sizes_are_supported(self):
        adj_obs = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        adj_gen = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

        obs_graph = dense_graph(adj_obs)
        gen_graph = dense_graph(adj_gen)

        module = mcs_soft_graph_dist([obs_graph])
        cont_batch = Batch.from_data_list([gen_graph])

        distance = module(cont_batch)

        self.assertGreaterEqual(distance.item(), 0.0)
        self.assertLessEqual(distance.item(), 1.0)

    def test_regression_for_missing_self_loops(self):
        num_nodes = 20
        adj_obs = torch.rand(num_nodes, num_nodes)
        adj_obs.fill_diagonal_(0.0)
        adj_gen = adj_obs.clone()

        obs_graph = dense_graph(adj_obs)
        gen_graph = dense_graph(adj_gen)

        module = mcs_soft_graph_dist([obs_graph])
        cont_batch = Batch.from_data_list([gen_graph])

        # Ensure the flattened edge_weight matches the no-self-loop layout
        self.assertEqual(
            cont_batch.edge_weight.numel(), num_nodes * (num_nodes - 1)
        )

        distance = module(cont_batch)

        self.assertTrue(torch.isfinite(distance))


if __name__ == "__main__":
    unittest.main()
