import numpy as np
import pytest

hs = pytest.importorskip("highspy")

from bnb_wm.solver.lp_backend import LPBackend, final_status


def _set_cover(m=150, n=300, seed=0):
    rng = np.random.default_rng(seed)
    A = (rng.random((m, n)) < 0.05).astype(np.float64)
    for i in range(m):
        if A[i].sum() == 0:
            A[i, rng.integers(n)] = 1.0
    return A, np.ones(m), rng.uniform(1.0, 10.0, n)


def test_per_solve_time_limit_is_not_cumulative():
    # HiGHS's time_limit is checked against a clock that accumulates over
    # every run() on the model; a per-solve budget must not expire just
    # because earlier solves used that much time in total.
    A, b, c = _set_cover()
    lp = LPBackend(hs, A, b, c)
    rng = np.random.default_rng(1)
    limit = 0.05
    for _ in range(5000):
        vlb, vub = np.zeros(lp.n), np.ones(lp.n)
        vub[rng.choice(lp.n, 3, replace=False)] = 0.0
        lp._model.clearSolver()
        res = lp.solve(vlb, vub, time_limit=limit)
        assert res.status in ("optimal", "infeasible")
        if lp._model.getRunTime() > 3 * limit:
            break
    else:
        pytest.skip("solves too fast to accumulate past the limit")
    assert lp.status_counts.get("time_limit", 0) == 0


def test_infeasible_status_is_certified():
    A, b, c = _set_cover()
    lp = LPBackend(hs, A, b, c)
    vub = np.ones(lp.n)
    vub[np.flatnonzero(A[0])] = 0.0  # row 0 cannot be covered
    res = lp.solve(np.zeros(lp.n), vub)
    assert not res.feasible and res.status == "infeasible"


class _N:
    def __init__(self, lb):
        self.lb = lb


def test_final_status_never_optimal_with_open_or_unresolved_nodes():
    diag = {"timeout": False, "node_limit_hit": False}
    sol = np.ones(3)
    assert final_status([], [], sol, 10.0, diag, 1e-4) == ("optimal", 0.0)
    st, gap = final_status([], [_N(8.0)], sol, 10.0, diag, 1e-4)
    assert st == "incomplete" and gap == pytest.approx(0.2)
    st, gap = final_status([_N(9.0)], [], sol, 10.0, {"timeout": True}, 1e-4)
    assert st == "timeout" and gap == pytest.approx(0.1)
