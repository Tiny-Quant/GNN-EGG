import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.data import Batch, Data

from egg_models.egg_generic import EggGeneric

from new_src.data import GeneratorAdapter
from new_src.explainee import GeneralGCN
from new_src.trainer import GenericGNNEggTrainer


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


def _flat_index(src: int, dst: int, max_nodes: int) -> int:
    return src * max_nodes + dst


def _dummy_generated(spec, data: Data):
    batch_size = 1
    max_nodes = spec.max_nodes

    adjacency = torch.zeros(batch_size, max_nodes, max_nodes)
    adjacency[0, data.edge_index[0], data.edge_index[1]] = 1.0

    cont_node_feats = None
    if spec.num_cont_node_feats:
        cont_node_feats = torch.zeros(
            batch_size, max_nodes, spec.num_cont_node_feats
        )
        cont_slice = data.x[:, -spec.num_cont_node_feats :]
        cont_node_feats[0, : cont_slice.size(0), :] = cont_slice

    dis_node_feats = None
    total_dis_node = sum(spec.dis_node_blocks)
    if total_dis_node > 0:
        dis_node_feats = torch.zeros(batch_size, max_nodes, total_dis_node)
        dis_slice = data.x[:, :total_dis_node]
        dis_node_feats[0, : dis_slice.size(0), :] = dis_slice

    cont_edge_feats = None
    if spec.num_cont_edge_feats:
        cont_edge_feats = torch.zeros(
            batch_size, max_nodes ** 2, spec.num_cont_edge_feats
        )
        cont_slice = data.edge_attr[:, -spec.num_cont_edge_feats :]
        for idx, (src, dst) in enumerate(zip(data.edge_index[0], data.edge_index[1])):
            cont_edge_feats[0, _flat_index(int(src), int(dst), max_nodes), :] = cont_slice[idx]

    dis_edge_feats = None
    total_dis_edge = sum(spec.dis_edge_blocks)
    if total_dis_edge > 0:
        dis_edge_feats = torch.zeros(batch_size, max_nodes ** 2, total_dis_edge)
        dis_slice = data.edge_attr[:, :total_dis_edge]
        for idx, (src, dst) in enumerate(zip(data.edge_index[0], data.edge_index[1])):
            dis_edge_feats[0, _flat_index(int(src), int(dst), max_nodes), :] = dis_slice[idx]

    return {
        "cont_node_feats": cont_node_feats,
        "dis_node_feats": dis_node_feats,
        "cont_edge_feats": cont_edge_feats,
        "dis_edge_feats": dis_edge_feats,
        "adjacency_matrix": adjacency,
    }


@pytest.fixture
def trainer_bundle():
    data = _sample_graph()
    adapter = GeneratorAdapter([data])
    spec = adapter.spec

    generator = EggGeneric(
        max_node_size=spec.max_nodes,
        cont_node_feats=spec.num_cont_node_feats or None,
        dis_node_feats=spec.dis_node_blocks if spec.dis_node_blocks else None,
        cont_edge_feats=spec.num_cont_edge_feats or None,
        dis_edge_feats=spec.dis_edge_blocks if spec.dis_edge_blocks else None,
        batch_size=1,
        allow_self_loops=False,
    )
    generator.train()

    explainee = GeneralGCN(
        node_features=data.x.size(-1),
        hidden_channels=8,
        num_classes=2,
        num_layers=2,
        dropout=0.0,
    )
    explainee.train()

    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)

    hidden = explainee.readout.out_features
    avg_embed_targets = {"readout": torch.zeros(1, hidden)}
    avg_embed_other = {"readout": torch.zeros(1, hidden)}

    trainer = GenericGNNEggTrainer(
        model=generator,
        explainee=explainee,
        target=torch.tensor([1.0, 0.0]),
        uninfo_target=torch.tensor([0.5, 0.5]),
        obs_data_list=[data],
        optimizer=optimizer,
        loss_term_weights=torch.ones(4),
        node_feature_dim=data.x.size(-1),
        edge_feature_dim=data.edge_attr.size(-1),
        avg_embed_targets=avg_embed_targets,
        avg_embed_other_class=avg_embed_other,
    )

    return data, spec, generator, trainer, explainee


def test_egg_to_ex_matches_explainee_expectations(trainer_bundle):
    data, spec, _, trainer, explainee = trainer_bundle
    generated = _dummy_generated(spec, data)
    batch = trainer.egg_to_ex(generated)

    assert batch.num_graphs == 1
    assert batch.x.size(-1) == data.x.size(-1)
    assert batch.edge_attr is not None
    assert batch.edge_attr.size(-1) == data.edge_attr.size(-1)

    logits = explainee(batch)
    assert logits.shape == (1, 2)


def test_egg_to_ex_preserves_gradients(trainer_bundle):
    _, _, generator, trainer, _ = trainer_bundle
    generator.zero_grad()

    generated = generator()
    batch = trainer.egg_to_ex(generated)

    total = batch.x.sum()
    if batch.edge_attr is not None:
        total = total + batch.edge_attr.sum()

    total.backward()

    assert any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in generator.parameters() if p.requires_grad
    ), "egg_to_ex should preserve gradients to generator parameters"

    generator.zero_grad()


def test_egg_to_egg_preserves_gradients(trainer_bundle):
    _, _, generator, trainer, _ = trainer_bundle
    generator.zero_grad()

    generated = generator()
    egg_tensors = trainer.egg_to_egg(generated)

    total = 0.0
    for tensor in egg_tensors:
        if tensor is not None and tensor.requires_grad:
            total = total + tensor.sum()

    params = [p for p in generator.parameters() if p.requires_grad]
    grads = torch.autograd.grad(total, params, allow_unused=True)

    assert any(
        g is not None and torch.any(g != 0)
        for g in grads
    ), "egg_to_egg should preserve gradients to generator parameters"


def test_ex_to_egg_preserves_gradients(trainer_bundle):
    data, _, _, trainer, _ = trainer_bundle
    data = data.clone()
    data.x = data.x.clone().detach().requires_grad_()
    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr.clone().detach().requires_grad_()

    batch = Batch.from_data_list([data])
    obs_tensors = trainer.ex_to_egg(batch)

    obs_X, _, obs_E = obs_tensors
    total = obs_X.sum()
    if obs_E is not None:
        total = total + obs_E.sum()

    total.backward()

    assert data.x.grad is not None and torch.any(data.x.grad != 0)
    if data.edge_attr is not None:
        assert data.edge_attr.grad is not None and torch.any(data.edge_attr.grad != 0)


def test_loss_terms_preserve_gradients(trainer_bundle):
    data, _, generator, trainer, _ = trainer_bundle
    generator.zero_grad()

    obs_batch = Batch.from_data_list([data])
    obs_batch = obs_batch.to(trainer.model.device_param.device)

    generated = generator()
    gen_ex = trainer.egg_to_ex(generated)
    gen_egg = trainer.egg_to_egg(generated)
    obs_egg = trainer.ex_to_egg(obs_batch)

    loss_terms = trainer.compute_loss_terms(
        generated,
        obs_batch,
        gen_ex,
        gen_egg,
        obs_egg,
    )

    params = [p for p in generator.parameters() if p.requires_grad]
    for idx in range(loss_terms.size(0)):
        grads = torch.autograd.grad(
            loss_terms[idx],
            params,
            retain_graph=True,
            allow_unused=True,
        )
        assert any(
            g is not None and torch.any(g != 0)
            for g in grads
        ), f"Loss term {idx} should retain gradient flow"


def test_structural_loss_preserves_gradients(trainer_bundle):
    data, _, generator, trainer, _ = trainer_bundle
    generator.zero_grad()

    obs_batch = Batch.from_data_list([data])
    obs_batch = obs_batch.to(trainer.model.device_param.device)

    generated = generator()
    gen_ex = trainer.egg_to_ex(generated)
    gen_egg = trainer.egg_to_egg(generated)
    obs_egg = trainer.ex_to_egg(obs_batch)

    struct_loss = trainer.struct_loss_fn(
        gen_egg,
        obs_egg,
        gen_ex,
        obs_batch,
        None,
        None,
    )

    assert torch.all(torch.isfinite(struct_loss)), "Structural loss produced NaNs"

    struct_loss.mean().backward()

    grads = [p.grad for p in generator.parameters() if p.requires_grad]
    assert any(
        g is not None and torch.any(g != 0)
        for g in grads
    ), "Structural loss should propagate gradients to the generator"

    generator.zero_grad()
