"""
Latent-rollout correctness (Eq. 17). CPU-only, no checkpoint needed.

1. The batched rollout used by the solver equals the recursive reference
   implementation across depth, branch factor, scoring mode and masks.
2. With every head patched to a constant, the score equals the closed-form
   tree expectation: rewards summed over both children, averaged over the
   policy's top-b continuations, discounted gamma^(j-1) at transition j, and
   the leaf bootstrapped with gamma^K (gamma^j where candidates run out).
"""

import itertools

import pytest
import torch
import torch.nn as nn

from bnb_wm.model.world_model import BnBWorldModel

V, H = 16, 32


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return BnBWorldModel(hidden_dim=H, n_gnn_layers=1, n_gnn_heads=4,
                         n_dyn_layers=2, n_dyn_heads=4).eval()


@pytest.fixture(scope="module")
def state():
    g = torch.Generator().manual_seed(1)
    z = torch.randn(1, H, generator=g)
    h = torch.randn(V, H, generator=g)
    past = torch.randn(1, 3, H, generator=g)
    return z, h, past


def _mask(n_valid):
    m = torch.zeros(V, dtype=torch.bool)
    m[:n_valid] = True
    return m


def _both(model, z, h, **kw):
    with torch.no_grad():
        rec = model.rollout_candidate(z, h, 0, batched=False, **kw)
        bat = model.rollout_candidate(z, h, 0, batched=True, **kw)
    return rec, bat


@pytest.mark.parametrize(
    "depth,b,reward_return,ctg,size,n_valid,use_past",
    list(itertools.product(
        [1, 2, 3], [1, 2, 3], [True, False], [0.0, 1.0], [0.0, 1.0],
        [None, 12, 2], [False, True],
    )),
)
def test_batched_matches_recursive(model, state, depth, b, reward_return,
                                   ctg, size, n_valid, use_past):
    z, h, past = state
    rec, bat = _both(
        model, z, h, depth=depth, gamma=0.9,
        valid_mask=None if n_valid is None else _mask(n_valid),
        past_tokens=past if use_past else None,
        size_weight=size, ctg_weight=ctg, branch_factor=b,
        use_reward_return=reward_return,
    )
    assert bat == pytest.approx(rec, rel=1e-4, abs=1e-4)


class _Const(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = float(c)

    def forward(self, z, *args, **kwargs):
        return torch.full((z.size(0),), self.c)


def _constant_heads(reward, value, ctg):
    torch.manual_seed(0)
    m = BnBWorldModel(hidden_dim=H, n_gnn_layers=1, n_gnn_heads=4,
                      n_dyn_layers=1, n_dyn_heads=4).eval()
    m.value = _Const(value)
    m.cost_to_go = _Const(ctg)
    m.dynamics_reward_pred = lambda z: torch.full((z.size(0),), float(reward))
    return m


@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize(
    "reward,value,ctg,n_valid,expected",
    [
        # Full depth K=3, b=2, gamma=0.5: 2^j nodes at transition j.
        (1, 0, 0, None, 2 * 1 + 4 * 0.5 + 8 * 0.25),   # rewards      = 6
        (0, 1, 0, None, 8 * 0.5 ** 3),                  # leaf gamma^K = 1
        (0, 0, 1, None, -(2 * 1 + 4 * 0.5 + 8 * 0.25)), # cost-to-go   = -6
        # Only the candidate and one other variable valid: depth-1 children
        # expand with k=1, depth-2 children have no candidates left and are
        # bootstrapped early with gamma^2.
        (1, 0, 0, 2, 2 * 1 + 4 * 0.5),                  # rewards      = 4
        (0, 1, 0, 2, 4 * 0.5 ** 2),                     # leaf gamma^2 = 1
    ],
)
def test_closed_form_weights(state, batched, reward, value, ctg, n_valid,
                             expected):
    z, h, _ = state
    m = _constant_heads(reward, value, ctg)
    with torch.no_grad():
        got = m.rollout_candidate(
            z, h, 0, depth=3, gamma=0.5,
            valid_mask=None if n_valid is None else _mask(n_valid),
            size_weight=0.0, ctg_weight=1.0, branch_factor=2,
            use_reward_return=True, batched=batched,
        )
    assert got == pytest.approx(expected, abs=1e-5)
