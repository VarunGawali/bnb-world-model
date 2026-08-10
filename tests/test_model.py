"""
test_model.py — Smoke tests for the BnB World Model.

Run with: pytest tests/test_model.py -v
"""

import torch
import pytest
from torch_geometric.data import Data, Batch
from bnb_wm.model import BnBWorldModel, BipartiteGNN, PolicyHead, ValueHead


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def tiny_batch(device):
    """Minimal bipartite graph: 4 variables, 3 constraints, 6 edges."""
    n_vars, n_cons = 4, 3

    x = torch.randn(n_vars + n_cons, 19)

    node_type = torch.cat([
        torch.zeros(n_vars, dtype=torch.long),
        torch.ones(n_cons,  dtype=torch.long),
    ])

    # Bidirectional edges (P0.2): constraint<->variable, with edge_attr, matching
    # build_pyg_data so the encoder's v2c pass sees edges.
    con = torch.tensor([4, 4, 5, 5, 6, 6], dtype=torch.long)   # constraint nodes
    var = torch.tensor([0, 1, 1, 2, 2, 3], dtype=torch.long)   # variable nodes
    edge_index = torch.cat([
        torch.stack([con, var]), torch.stack([var, con]),
    ], dim=1)
    edge_attr = torch.randn(edge_index.size(1), 3)

    data = Data(x=x, edge_index=edge_index, node_type=node_type,
                edge_attr=edge_attr)
    return Batch.from_data_list([data])


def test_encoder_output_shapes(tiny_batch, device):
    model = BnBWorldModel(hidden_dim=32, n_gnn_layers=2).to(device)
    h_vars, z = model.encode(tiny_batch.to(device))

    assert h_vars.shape == (4, 32), f"Expected (4, 32), got {h_vars.shape}"
    assert z.shape      == (1, 32), f"Expected (1, 32), got {z.shape}"


def test_policy_head(tiny_batch, device):
    model = BnBWorldModel(hidden_dim=32, n_gnn_layers=2).to(device)
    scores, z = model(tiny_batch.to(device))

    assert scores.shape == (4,), f"Expected (4,), got {scores.shape}"
    assert z.shape      == (1, 32)


def test_value_head(tiny_batch, device):
    model = BnBWorldModel(hidden_dim=32, n_gnn_layers=2).to(device)
    batch = tiny_batch.to(device)
    h_vars, z = model.encode(batch)
    # value_pred needs (z, h_vars, batch_vec); single graph -> all vars in batch 0.
    bvec = torch.zeros(h_vars.size(0), dtype=torch.long, device=device)
    v = model.value_pred(z, h_vars, bvec)

    assert v.shape == (1,), f"Expected (1,), got {v.shape}"


def test_integrality_head(tiny_batch, device):
    model = BnBWorldModel(hidden_dim=32, n_gnn_layers=2).to(device)
    _, z = model(tiny_batch.to(device))
    logit = model.integrality_logit(z)

    assert logit.shape == (1,), f"Expected (1,), got {logit.shape}"


def test_dynamics_step(device):
    model = BnBWorldModel(hidden_dim=32, n_gnn_layers=2).to(device)
    z_t    = torch.randn(1, 32, device=device)
    a_emb  = torch.randn(1, 32, device=device)
    # dynamics_step returns (z_next [B,H], token_buffer [B,t+1,H]).
    z_next, tokens = model.dynamics_step(z_t, a_emb)

    assert z_next.shape == (1, 32)
    assert tokens.shape == (1, 1, 32)      # one token in the fresh buffer


def test_no_nan_in_forward(tiny_batch, device):
    model = BnBWorldModel(hidden_dim=32, n_gnn_layers=2).to(device)
    scores, z = model(tiny_batch.to(device))

    assert not torch.isnan(scores).any(), "NaN in policy scores"
    assert not torch.isnan(z).any(),      "NaN in graph embedding"
