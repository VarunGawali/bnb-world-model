"""
End-to-end smoke test of NeuralBnBSolver with a tiny random model (CPU).

Checks that the model-driven branching modes run to completion on a small Set
Cover instance, reach the same optimum as most-fractional branching (learned
decisions never change correctness), give each child the history token of its
own branch direction, and keep the token history inside the trained window.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("highspy")

from bnb_wm.model.world_model import BnBWorldModel
from bnb_wm.solver.config import SolverConfig
from bnb_wm.solver.neural_bnb import NeuralBnBSolver


def _set_cover(m=40, n=60, density=0.1, seed=2):
    # Unit-range costs keep the LP fractional: ~15 nodes with most-fractional.
    rng = np.random.default_rng(seed)
    A = (rng.random((m, n)) < density).astype(np.float64)
    for i in range(m):                      # every row coverable
        if A[i].sum() == 0:
            A[i, rng.integers(n)] = 1.0
    c = rng.integers(1, 2, n).astype(np.float64)
    return A, np.ones(m), c


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return BnBWorldModel(hidden_dim=32, n_gnn_layers=1, n_gnn_heads=4,
                         n_dyn_layers=1, n_dyn_heads=4).eval()


def _solve(model, **kw):
    cfg = SolverConfig(cut_mode="none", primal_heuristic=True,
                       time_limit=120, node_limit=400, **kw)
    solver = NeuralBnBSolver(model, torch.device("cpu"), cfg)
    return solver, solver.solve(*_set_cover())


def test_model_branching_modes_match_classical_optimum(model):
    _, ref = _solve(model, branch_mode="most_fractional")
    assert ref.status == "optimal"
    assert ref.n_nodes > 5          # the instance must actually branch
    for mode in ("policy", "rollout"):
        _, res = _solve(model, branch_mode=mode, dyn_history_window=4)
        assert res.status == "optimal", mode
        assert res.objective == pytest.approx(ref.objective), mode


def test_children_get_direction_specific_tokens(model):
    solver, _ = _solve(model, branch_mode="policy")
    z = torch.randn(1, 32)
    h = torch.randn(10, 32)
    past = torch.randn(1, 2, 32)
    toks = solver._child_tokens(z, h, 3, past)
    assert toks[1.0].shape == toks[-1.0].shape == (1, 3, 32)
    torch.testing.assert_close(toks[1.0][:, :2], past)    # history preserved
    assert not torch.allclose(toks[1.0][:, -1], toks[-1.0][:, -1])


def test_history_window_caps_token_buffer(model):
    solver, _ = _solve(model, branch_mode="policy", dyn_history_window=4)
    assert model.dynamics.history_window == 4
    past = torch.randn(1, 4, 32)
    toks = solver._child_tokens(torch.randn(1, 32), torch.randn(10, 32), 0, past)
    assert toks[1.0].size(1) == 4                           # slid, not grown
    torch.testing.assert_close(toks[1.0][:, :3], past[:, 1:])
