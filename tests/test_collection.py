"""
test_collection.py — Collector/cut correctness tests (P2.4).

- Cut validity: generated root Gomory cuts must NOT remove any integer-feasible
  point of the original set-cover system (guards against the P0.4 invalid-cut
  regression). Requires highspy; skipped where unavailable.
- Schema: the field-level validator accepts a well-formed trajectory and rejects
  the classic P0.11 failure (tree depth stored as the node id). Pure NumPy.
"""

import importlib.util
from itertools import product
from pathlib import Path

import numpy as np
import pytest


def _load_validator():
    p = Path(__file__).resolve().parent.parent / "scripts" / "validate_collection.py"
    spec = importlib.util.spec_from_file_location("validate_collection", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_root_gomory_cuts_are_valid():
    highspy = pytest.importorskip("highspy")
    from bnb_wm.solver.gomory import generate_root_gomory_cuts

    # Small set-cover: min 1^T x  s.t.  A x >= 1,  x in {0,1}^4.
    A = np.array([[1, 1, 0, 0],
                  [0, 1, 1, 0],
                  [0, 0, 1, 1]], dtype=np.float64)
    b = np.ones(3)
    c = np.ones(4)

    cuts = generate_root_gomory_cuts(A, b, c, highspy, max_cuts=20)

    # Every integer-feasible point must satisfy every generated cut (alpha·x>=beta).
    feasible = [np.array(x, float) for x in product([0, 1], repeat=4)
                if np.all(A @ np.array(x, float) >= b - 1e-9)]
    assert feasible, "no feasible integer points — bad test setup"
    for alpha, beta in cuts:
        alpha = np.asarray(alpha, float)
        for x in feasible:
            assert float(alpha @ x) >= float(beta) - 1e-6, (
                f"cut removes a feasible integer point: {alpha}·{x} < {beta}")


def _good_npz(tmp, **over):
    T = 3
    d = dict(
        n_steps=T,
        var_features=np.array([np.random.randn(5, 19).astype(np.float32)
                               for _ in range(T)], dtype=object),
        con_features=np.array([np.random.randn(4, 5).astype(np.float32)
                               for _ in range(T)], dtype=object),
        edge_indices=np.array([np.zeros((2, 3), np.int64) for _ in range(T)],
                              dtype=object),
        edge_values=np.array([np.zeros(3, np.float32) for _ in range(T)],
                             dtype=object),
        action_sets=np.array([np.array([0, 1, 2]) for _ in range(T)], dtype=object),
        branching_vars=np.array([0, 0, 0]),
        local_branching_label=np.array([0, 0, 0]),
        dual_bounds=np.array([10., 12., 15.], np.float32),
        depths=np.array([0, 1, 2], np.int32),
        node_ids=np.array([1, 2, 3], np.int64),
        parent_ids=np.array([0, 1, 2], np.int64),
        branch_dirs=np.array([0, 1, -1], np.int8),
        next_is_leaf=np.array([0, 0, 1], np.float32),
        root_bound=np.float32(10.), primal_bound=np.float32(20.),
        optimal_valid=np.asarray(True),
        n_cuts=np.array([50, 0, 0], np.int32),
        cut_features=np.array([np.zeros((50, 6), np.float32),
                               np.zeros((0, 6), np.float32),
                               np.zeros((0, 6), np.float32)], dtype=object),
        cut_labels=np.array([np.zeros(50, np.float32)] * 3, dtype=object),
    )
    d.update(over)
    p = Path(tmp) / "traj_x.npz"
    np.savez_compressed(p, **d)
    return p


def test_schema_validator_accepts_good_and_rejects_depth_as_nodeid(tmp_path):
    v = _load_validator()
    errs, _ = v.validate_file(str(_good_npz(tmp_path)))
    assert errs == [], f"good file flagged: {errs}"

    # P0.11 regression: depth stored as node id (huge integers).
    bad = _good_npz(tmp_path, depths=np.array([1, 2, 300000], np.int32))
    errs, _ = v.validate_file(str(bad))
    assert any("node id" in e for e in errs), errs
