"""
lp_backend.py — Shared LP layer for NeuralBnBSolver and ClassicalBnBSolver.

One persistent HiGHS model per instance.  Only changed column bounds are
pushed per node (changeColBounds diff), and the parent basis is warm-started
so each re-solve is a dual simplex pivot from a feasible point rather than a
cold start.

Both solvers import LPBackend and hold it as self.lp.  This is the fairness
requirement: every LP timing, warm-start, and tolerance is identical across
the ablation ladder.

Public API
----------
    lp = LPBackend(highspy, A, b, c)
    lp_result = lp.solve(vlb, vub, warm_basis)   -> LPResult
    lp.commit_cuts(cuts)                          -> None  (adds rows)
    lp.augmented_A_b()                            -> (A_aug, b_aug)
    lp.n_rows          current row count (original + committed cuts)
    lp.lp_count        total LP solves so far
    lp.lp_time         cumulative LP solve time (seconds)
    lp.struct_version  incremented on every cut commit (cache invalidation)

LPResult fields
---------------
    feasible : bool
    obj      : float | None
    x        : np.ndarray | None   (primal, length n)
    sol      : dict | None         (full sol_from_highs dict)
    basis    : tuple | None        (col_status, row_status)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from bnb_wm.features import sol_from_highs


# ---------------------------------------------------------------------------
# Cut record  (shared with classical_bnb via import)
# ---------------------------------------------------------------------------

@dataclass
class CutRecord:
    lhs: np.ndarray
    rhs: float
    cut_type: str = "gomory"


# ---------------------------------------------------------------------------
# LP result
# ---------------------------------------------------------------------------

@dataclass
class LPResult:
    feasible: bool
    obj: Optional[float] = None
    x: Optional[np.ndarray] = None
    sol: Optional[dict] = None
    basis: Optional[tuple] = None
    # "optimal" | "infeasible" (certified, safe to prune) | "time_limit" | "error"
    status: str = "optimal"


# ---------------------------------------------------------------------------
# Search status helpers (shared by both solvers)
# ---------------------------------------------------------------------------

def open_lb(heap, unresolved, global_ub) -> float:
    lbs = [nd.lb for nd in heap] + [nd.lb for nd in unresolved]
    return min(lbs) if lbs else global_ub


def final_status(heap, unresolved, best_sol, global_ub, diag, gap_tol):
    """Status and relative gap. "optimal" only when every open node was
    certified closed, or the remaining bound gap is within tolerance."""
    if best_sol is None:
        if diag.get("timeout"):
            return "timeout", np.inf
        if diag.get("node_limit_hit"):
            return "node_limit", np.inf
        return ("incomplete" if (heap or unresolved) else "infeasible"), np.inf
    global_lb = open_lb(heap, unresolved, global_ub)
    gap = max(0.0, (global_ub - global_lb) / (abs(global_ub) + 1e-10))
    if not heap and not unresolved:
        return "optimal", 0.0
    if gap < gap_tol:
        return "optimal", gap
    if diag.get("timeout"):
        return "timeout", gap
    if diag.get("node_limit_hit"):
        return "node_limit", gap
    return "incomplete", gap


def incumbent_check(A: np.ndarray, b: np.ndarray, sol) -> dict:
    """Independent feasibility check of a returned solution against Ax >= b, x binary."""
    if sol is None:
        return {"incumbent_checked": False}
    x = np.asarray(sol, dtype=np.float64)
    viol = float(np.max(np.maximum(b - A @ x, 0.0))) if len(b) else 0.0
    nonbin = float(np.max(np.abs(x - np.round(x)))) if len(x) else 0.0
    return {"incumbent_checked": True,
            "incumbent_max_row_violation": viol,
            "incumbent_max_nonbinary": nonbin,
            "incumbent_feasible": bool(viol <= 1e-6 and nonbin <= 1e-6)}


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class LPBackend:
    """
    Persistent HiGHS LP model for one MIP instance.

    Parameters
    ----------
    highspy : module
        The `highspy` module (passed in so the caller controls the import).
    A : [m, n] float64
        Constraint matrix.
    b : [m] float64
        Right-hand sides (Ax >= b).
    c : [n] float64
        Objective coefficients (minimise c^T x).
    """

    def __init__(
        self,
        highspy,
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
    ):
        self._hs = highspy
        self._A = A
        self._b = b
        self._c = c
        self._m0, self._n = A.shape

        self._committed_cuts: list[CutRecord] = []
        self._n_rows: int = self._m0
        self.struct_version: int = 0

        self.lp_count: int = 0
        self.lp_time: float = 0.0
        self.status_counts: dict[str, int] = {}
        self.cold_retries: int = 0

        self._cur_lb: np.ndarray = np.zeros(self._n, dtype=np.float64)
        self._cur_ub: np.ndarray = np.ones(self._n, dtype=np.float64)

        self._model = self._build()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        return self._n

    @property
    def m0(self) -> int:
        return self._m0

    @property
    def n_rows(self) -> int:
        return self._n_rows

    @property
    def A(self) -> np.ndarray:
        return self._A

    @property
    def b(self) -> np.ndarray:
        return self._b

    @property
    def c(self) -> np.ndarray:
        return self._c

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self):
        hs = self._hs
        h = hs.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("presolve", "off")
        h.setOptionValue("time_limit", 1e30)

        n, m = self._n, self._m0
        lb = np.zeros(n, dtype=np.float64)
        ub = np.ones(n, dtype=np.float64)
        h.addVars(n, lb, ub)
        col_idx = np.arange(n, dtype=np.int32)
        h.changeColsCost(n, col_idx, self._c)

        inf = hs.kHighsInf
        for i in range(m):
            idx = np.where(np.abs(self._A[i]) > 1e-12)[0]
            if len(idx) == 0:
                continue
            h.addRow(
                float(self._b[i]), inf, len(idx),
                idx.astype(np.int32),
                self._A[i, idx].astype(np.float64),
            )
        return h

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def _apply_bounds(self, vlb: np.ndarray, vub: np.ndarray):
        h = self._model
        changed = np.where((self._cur_lb != vlb) | (self._cur_ub != vub))[0]
        for j in changed:
            h.changeColBounds(int(j), float(vlb[j]), float(vub[j]))
        if len(changed):
            self._cur_lb = vlb.copy()
            self._cur_ub = vub.copy()

    def solve(
        self,
        vlb: np.ndarray,
        vub: np.ndarray,
        warm_basis: Optional[tuple] = None,
        time_limit: float = 1e30,
    ) -> LPResult:
        """
        Solve the LP for the given variable bounds.

        Parameters
        ----------
        vlb, vub : [n] float64
            Per-variable bounds for this node.
        warm_basis : (col_status, row_status) | None
            Basis from the parent node.  Row-status is padded / trimmed to
            match the current row count (original rows + committed cuts).

        Returns
        -------
        LPResult
        """
        t0 = time.perf_counter()
        hs = self._hs
        h = self._model

        self._apply_bounds(vlb, vub)

        if warm_basis is not None:
            col_status, row_status = warm_basis
            n_new = self._n_rows - len(row_status)
            if n_new > 0:
                row_status = list(row_status) + [1] * n_new
            elif n_new < 0:
                row_status = list(row_status)[: self._n_rows]
            try:
                h.setBasis(list(col_status), list(row_status))
            except Exception:
                pass

        status = self._run(time_limit)
        if status == "error":
            # Warm start or numerics failed; one cold re-solve before giving up.
            self.cold_retries += 1
            h.clearSolver()
            status = self._run(time_limit)
        self.status_counts[status] = self.status_counts.get(status, 0) + 1
        self.lp_time += time.perf_counter() - t0

        if status != "optimal":
            return LPResult(feasible=False, status=status)

        sol = sol_from_highs(hs, h, self._n, self._n_rows)
        return LPResult(
            feasible=True,
            obj=sol["obj"],
            x=sol["x"],
            sol=sol,
            basis=(sol["col_status"], sol["row_status"]),
        )

    def _run(self, time_limit: float) -> str:
        hs = self._hs
        h = self._model
        # HiGHS measures time_limit against its run clock, which accumulates
        # over every run() on this object, so the per-solve budget must be
        # offset by the time already spent.
        if np.isfinite(time_limit) and time_limit < 1e29:
            h.setOptionValue("time_limit", float(h.getRunTime() + time_limit))
        else:
            h.setOptionValue("time_limit", 1e30)
        h.run()
        self.lp_count += 1
        try:
            ms = h.getModelStatus()
        except Exception:
            return "error"
        M = hs.HighsModelStatus
        if ms == M.kOptimal:
            return "optimal"
        # Variables are bounded in [0, 1], so "unbounded or infeasible" is infeasible.
        if ms in (M.kInfeasible, M.kUnboundedOrInfeasible):
            return "infeasible"
        if ms == M.kTimeLimit:
            return "time_limit"
        return "error"

    # ------------------------------------------------------------------
    # Cuts
    # ------------------------------------------------------------------

    def commit_cuts(self, cuts: list[CutRecord]):
        """Add cuts as new rows and increment struct_version."""
        h = self._model
        inf = self._hs.kHighsInf
        for cut in cuts:
            idx = np.where(np.abs(cut.lhs) > 1e-12)[0]
            if len(idx) == 0:
                continue
            h.addRow(
                float(cut.rhs), inf, len(idx),
                idx.astype(np.int32),
                cut.lhs[idx].astype(np.float64),
            )
            self._committed_cuts.append(cut)
            self._n_rows += 1
        self.struct_version += 1

    @property
    def committed_cuts(self) -> list[CutRecord]:
        return self._committed_cuts

    def augmented_A_b(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (A, b) with all committed cuts appended as extra rows."""
        if not self._committed_cuts:
            return self._A, self._b
        cut_A = np.vstack([
            cut.lhs.reshape(1, self._n) for cut in self._committed_cuts
        ])
        cut_b = np.array([cut.rhs for cut in self._committed_cuts])
        return (
            np.vstack([self._A, cut_A]),
            np.concatenate([self._b, cut_b]),
        )
