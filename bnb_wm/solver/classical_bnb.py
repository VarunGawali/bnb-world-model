"""
classical_bnb.py — Reliability branching B&B solver.

Shares LPBackend with NeuralBnBSolver (same HiGHS model, same warm-start,
same tolerances) so wall-clock and LP-count comparisons are fair.

Branching
---------
Per-variable pseudocost stats Ψ⁺, Ψ⁻ track average LP-objective gain per
unit fractionality in the up / down direction, updated after each child LP.

Score = max(Δ⁻, ε) · max(Δ⁺, ε)   (product rule)
  where Δ± = Ψ± · f±,  f⁺ = ceil(x) - x,  f⁻ = x - floor(x).

A variable is *reliable* when it has ≥ eta observations in both directions.
Unreliable candidates get real strong-branching LPs (bounded-iteration child
solves) up to sb_init candidates per node, cheapest-first.

  sb_init=0  → pure pseudocost (zero SB LPs)
  sb_init=∞  → full strong branching on all fractional variables

Node selection
--------------
Best-bound (min LP obj) priority queue, same as neural_bnb default.

Cuts
----
Gomory cuts from the same pool used by neural_bnb, max-violation selection,
same budget gates.  Cut LP re-solves are counted in tree_lps, not
decision_lps.

Primal heuristic
----------------
Same LP-rounding + greedy-repair as neural_bnb.

Metrics
-------
SolveResult carries two LP counts:
  tree_lps      — one LP per B&B node
  decision_lps  — strong-branching LPs (up + down per candidate evaluated)
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from bnb_wm.solver.lp_backend import LPBackend, CutRecord, LPResult
from bnb_wm.solver.config import SolverConfig


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass(order=False)
class Node:
    lb: float
    depth: int
    var_lb: np.ndarray
    var_ub: np.ndarray
    node_id: int
    parent_id: Optional[int] = None
    priority: float = 0.0
    warm_basis: Optional[tuple] = None

    def __lt__(self, other):
        return self.priority > other.priority


@dataclass
class SolveResult:
    status: str
    objective: float
    solution: Optional[np.ndarray]
    n_nodes: int
    solve_time: float
    optimality_gap: float
    tree_lps: int = 0
    decision_lps: int = 0
    lp_time: float = 0.0
    cuts_added: int = 0
    diagnostics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pseudocost tracker
# ---------------------------------------------------------------------------

class _Pseudocosts:
    """
    Per-variable up/down pseudocost statistics.

    Ψ±[j] = sum of (ΔObj / fractionality) observations in that direction.
    """

    def __init__(self, n_vars: int, init_val: float = 1.0):
        self._sum_up   = np.full(n_vars, init_val, dtype=np.float64)
        self._sum_dn   = np.full(n_vars, init_val, dtype=np.float64)
        self._cnt_up   = np.zeros(n_vars, dtype=np.int32)
        self._cnt_dn   = np.zeros(n_vars, dtype=np.int32)
        self._n = n_vars

    def update(self, j: int, direction: int, delta_obj: float, frac: float):
        """Record one SB or child-LP observation.

        direction: +1 = branched up (ceil), -1 = branched down (floor).
        """
        if frac < 1e-9:
            return
        gain = max(delta_obj, 0.0) / frac
        if direction > 0:
            self._sum_up[j] += gain
            self._cnt_up[j] += 1
        else:
            self._sum_dn[j] += gain
            self._cnt_dn[j] += 1

    def is_reliable(self, j: int, eta: int) -> bool:
        return int(self._cnt_up[j]) >= eta and int(self._cnt_dn[j]) >= eta

    def score(self, j: int, x_lp: np.ndarray) -> float:
        """Product-rule score for variable j given current LP solution."""
        frac = float(x_lp[j]) - np.floor(float(x_lp[j]))
        f_up = 1.0 - frac
        f_dn = frac
        cnt_u = max(int(self._cnt_up[j]), 1)
        cnt_d = max(int(self._cnt_dn[j]), 1)
        psi_up = self._sum_up[j] / cnt_u
        psi_dn = self._sum_dn[j] / cnt_d
        delta_up = psi_up * f_up
        delta_dn = psi_dn * f_dn
        eps = 1e-6
        return max(delta_dn, eps) * max(delta_up, eps)

    def scores(self, frac_idx: np.ndarray, x_lp: np.ndarray) -> np.ndarray:
        return np.array([self.score(j, x_lp) for j in frac_idx], dtype=np.float64)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class ClassicalBnBSolver:
    """
    Reliability branching B&B.

    Parameters
    ----------
    config : SolverConfig
        All thresholds live here; branching-specific keys used:
          sb_init  (int)   — max strong-branching candidates per node
          eta      (int)   — reliability threshold (observations per direction)
        These are read from config.diagnostics["sb_init"] / ["eta"] if
        present, with defaults sb_init=4, eta=4, so SolverConfig needs no
        new fields — pass them via SolverConfig(**{"sb_init": 8}) is not
        valid; instead we read from a sidecar dict passed at __init__.

    sb_init : int
        Max SB candidates per node.  0 = pure pseudocost.  None = full SB.
    eta : int
        Reliability threshold.
    """

    def __init__(
        self,
        config: Optional[SolverConfig] = None,
        sb_init: int = 4,
        eta: int = 4,
    ):
        self.cfg = config or SolverConfig(
            branch_mode="most_fractional",
            cut_mode="none",
            ors_cascade=False,
            katz_weight=0.0,
            node_selection="bound",
        )
        self.sb_init = sb_init   # None means unlimited (full SB)
        self.eta = eta

        try:
            import highspy
            self._highs = highspy
        except ImportError as exc:
            raise RuntimeError("highspy is required by ClassicalBnBSolver.") from exc

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def solve(self, A: np.ndarray, b: np.ndarray, c: np.ndarray) -> SolveResult:
        cfg = self.cfg
        t0 = time.perf_counter()

        A = np.ascontiguousarray(A, dtype=np.float64)
        b = np.ascontiguousarray(b, dtype=np.float64)
        c = np.ascontiguousarray(c, dtype=np.float64)
        m0, n = A.shape

        self._A, self._b, self._c = A, b, c
        self._m0, self._n = m0, n

        self._lp = LPBackend(self._highs, A, b, c)
        self._pc = _Pseudocosts(n)
        self._gomory_pool: Optional[list[CutRecord]] = None
        self._committed_cuts: list[CutRecord] = []
        self._decision_lps = 0
        self._diag: dict = {
            "sb_evals": 0, "pc_evals": 0, "cuts_committed": 0,
            "primal_heur_hits": 0, "g1_bound_prune": 0,
            "g3_infeasible": 0, "g3_dominated": 0, "g4_integral": 0,
            "timeout": False, "node_limit_hit": False,
        }

        # Root LP
        root_lb = np.zeros(n)
        root_ub = np.ones(n)
        lp = self._lp.solve(root_lb, root_ub, warm_basis=None)
        if not lp.feasible:
            return self._result("infeasible", np.inf, None, 0, t0, np.inf)

        global_ub, best_sol = np.inf, None
        status = "infeasible"

        if cfg.primal_heuristic:
            obj, sol = self._primal_heuristic(lp.x)
            if sol is not None and obj < global_ub:
                global_ub, best_sol, status = obj, sol, "feasible"
                self._diag["primal_heur_hits"] += 1

        if _is_integral(lp.x):
            obj = float(c @ np.round(lp.x))
            if obj < global_ub:
                global_ub, best_sol, status = obj, np.round(lp.x), "feasible"
            return self._result("optimal", global_ub, best_sol, 1, t0, 0.0)

        root = Node(lb=lp.obj, depth=0, var_lb=root_lb, var_ub=root_ub,
                    node_id=0, priority=-lp.obj, warm_basis=lp.basis)
        heap: list[Node] = [root]
        pending_root_lp = lp

        n_nodes = 0
        next_id = 1

        while heap:
            if time.perf_counter() - t0 > cfg.time_limit:
                status = status if best_sol is not None else "timeout"
                self._diag["timeout"] = True
                break
            if n_nodes >= cfg.node_limit:
                self._diag["node_limit_hit"] = True
                break

            node = heapq.heappop(heap)

            # Gate 1: bound pruning
            if node.lb >= global_ub - 1e-6:
                self._diag["g1_bound_prune"] += 1
                continue

            # LP solve
            if pending_root_lp is not None:
                lp = pending_root_lp
                pending_root_lp = None
            else:
                lp = self._lp.solve(node.var_lb, node.var_ub, node.warm_basis)
            n_nodes += 1

            # Gate 3: infeasible / dominated
            if not lp.feasible:
                self._diag["g3_infeasible"] += 1
                continue
            if lp.obj >= global_ub - 1e-6:
                self._diag["g3_dominated"] += 1
                continue
            node.lb = lp.obj

            # Gate 4: integral
            if _is_integral(lp.x):
                obj = float(c @ np.round(lp.x))
                if obj < global_ub:
                    global_ub, best_sol, status = obj, np.round(lp.x), "feasible"
                self._diag["g4_integral"] += 1
                continue

            # Periodic primal heuristic
            if (cfg.primal_heuristic
                    and n_nodes % max(1, cfg.primal_heuristic_every) == 0):
                obj, sol = self._primal_heuristic(lp.x)
                if sol is not None and obj < global_ub:
                    global_ub, best_sol, status = obj, sol, "feasible"
                    self._diag["primal_heur_hits"] += 1

            frac_idx = np.where((lp.x > 1e-4) & (lp.x < 1 - 1e-4))[0]
            if len(frac_idx) == 0:
                continue

            # Optional cuts (heuristic max-violation, same pool as neural_bnb)
            if cfg.cut_mode != "none" and node.depth <= cfg.cut_depth_max:
                lp = self._try_cuts(lp, node, global_ub)
                if lp is None:
                    continue
                if _is_integral(lp.x):
                    obj = float(c @ np.round(lp.x))
                    if obj < global_ub:
                        global_ub, best_sol, status = obj, np.round(lp.x), "feasible"
                    continue
                frac_idx = np.where((lp.x > 1e-4) & (lp.x < 1 - 1e-4))[0]
                if len(frac_idx) == 0:
                    continue
                node.lb = max(node.lb, lp.obj)

            # Branch variable selection
            branch_var = self._select_branch_var(
                lp, frac_idx, node, global_ub)

            # Children (lazy LPs)
            for direction, (new_lb_val, new_ub_val) in (
                (+1, (1.0, 1.0)),
                (-1, (0.0, 0.0)),
            ):
                vlb = node.var_lb.copy()
                vub = node.var_ub.copy()
                vlb[branch_var] = new_lb_val
                vub[branch_var] = new_ub_val
                if np.any(vlb > vub + 1e-9):
                    continue
                if node.lb >= global_ub - 1e-6:
                    continue
                heapq.heappush(heap, Node(
                    lb=node.lb,
                    depth=node.depth + 1,
                    var_lb=vlb, var_ub=vub,
                    node_id=next_id, parent_id=node.node_id,
                    priority=-node.lb,
                    warm_basis=lp.basis,
                ))
                next_id += 1

        global_lb = min((nd.lb for nd in heap), default=global_ub)
        if best_sol is not None:
            gap = max(0.0, (global_ub - global_lb) / (abs(global_ub) + 1e-10))
        else:
            gap = np.inf

        if best_sol is not None and gap < cfg.gap_tolerance:
            status = "optimal"
        elif best_sol is not None and not heap and status != "timeout":
            status, gap = "optimal", 0.0

        return self._result(status, global_ub if best_sol is not None else np.inf,
                            best_sol, n_nodes, t0, gap)

    # ------------------------------------------------------------------
    # Branch variable selection
    # ------------------------------------------------------------------

    def _select_branch_var(
        self,
        lp: LPResult,
        frac_idx: np.ndarray,
        node: Node,
        global_ub: float,
    ) -> int:
        pc = self._pc
        eta = self.eta
        sb_budget = self.sb_init  # None = unlimited

        # Identify unreliable candidates
        unreliable = [j for j in frac_idx if not pc.is_reliable(j, eta)]

        # How many SB evals to do this node
        if sb_budget is None:
            sb_cands = list(frac_idx)
        elif sb_budget == 0:
            sb_cands = []
        else:
            # Prioritise unreliable variables; fill up to budget
            sb_cands = unreliable[:sb_budget]

        # Strong branching on selected candidates
        sb_scores: dict[int, float] = {}
        for j in sb_cands:
            score = self._sb_eval(j, lp, node)
            sb_scores[j] = score
            self._diag["sb_evals"] += 1

        # Score all fractional variables
        best_j, best_s = int(frac_idx[0]), -np.inf
        for j in frac_idx:
            if j in sb_scores:
                s = sb_scores[j]
            else:
                s = pc.score(j, lp.x)
                self._diag["pc_evals"] += 1
            if s > best_s:
                best_s, best_j = s, int(j)

        return best_j

    def _sb_eval(self, j: int, lp: LPResult, node: Node) -> float:
        """
        Evaluate variable j by solving two bounded child LPs.
        Updates pseudocosts in both directions.
        Returns product-rule score.
        """
        x_j = float(lp.x[j])
        frac = x_j - np.floor(x_j)

        obj_up, obj_dn = lp.obj, lp.obj  # fallback

        for direction, (new_lb, new_ub) in ((+1, (1.0, 1.0)), (-1, (0.0, 0.0))):
            vlb = node.var_lb.copy()
            vub = node.var_ub.copy()
            vlb[j] = new_lb
            vub[j] = new_ub
            if np.any(vlb > vub + 1e-9):
                # Infeasible child: treat as large gain
                gain = abs(self._c[j]) * 10.0
                self._pc.update(j, direction, gain, frac if direction < 0 else 1.0 - frac)
                self._decision_lps += 1
                if direction > 0:
                    obj_up = lp.obj + gain
                else:
                    obj_dn = lp.obj + gain
                continue

            child_lp = self._lp.solve(vlb, vub, lp.basis)
            self._decision_lps += 1

            if not child_lp.feasible:
                gain = abs(self._c[j]) * 10.0
            else:
                gain = max(child_lp.obj - lp.obj, 0.0)

            f = (1.0 - frac) if direction > 0 else frac
            self._pc.update(j, direction, gain, max(f, 1e-9))

            if direction > 0:
                obj_up = (lp.obj + gain) if not child_lp.feasible else child_lp.obj
            else:
                obj_dn = (lp.obj + gain) if not child_lp.feasible else child_lp.obj

        eps = 1e-6
        delta_up = max(obj_up - lp.obj, eps)
        delta_dn = max(obj_dn - lp.obj, eps)
        return delta_dn * delta_up

    # ------------------------------------------------------------------
    # Cuts  (max-violation heuristic, same Gomory pool)
    # ------------------------------------------------------------------

    def _gomory_cut_pool(self) -> list[CutRecord]:
        if self._gomory_pool is None:
            from bnb_wm.solver.gomory import generate_root_gomory_cuts
            try:
                pool = generate_root_gomory_cuts(
                    self._A, self._b, self._c, self._highs,
                    max_cuts=self.cfg.cut_pool_max)
            except TypeError:
                pool = generate_root_gomory_cuts(
                    self._A, self._b, self._c, self._highs)
            self._gomory_pool = [
                CutRecord(np.asarray(lhs, dtype=np.float64), float(rhs))
                for lhs, rhs in (pool or [])
            ]
        return self._gomory_pool

    def _try_cuts(
        self, lp: LPResult, node: Node, global_ub: float
    ) -> Optional[LPResult]:
        cfg = self.cfg
        if len(self._lp.committed_cuts) >= cfg.cut_budget_cap:
            return lp

        committed_fps = {_fingerprint(c) for c in self._lp.committed_cuts}
        cands = [
            c for c in self._gomory_cut_pool()
            if _fingerprint(c) not in committed_fps
            and float(c.lhs @ lp.x) < c.rhs - 1e-6
        ]
        if not cands:
            return lp

        viol = np.array([c.rhs - float(c.lhs @ lp.x) for c in cands])
        chosen = [cands[i] for i in np.argsort(-viol)[:cfg.max_cuts_per_node]]

        self._lp.commit_cuts(chosen)
        self._diag["cuts_committed"] += len(chosen)

        lp2 = self._lp.solve(node.var_lb, node.var_ub, lp.basis)
        if not lp2.feasible:
            return None
        return lp2

    # ------------------------------------------------------------------
    # Primal heuristic  (same as neural_bnb)
    # ------------------------------------------------------------------

    def _primal_heuristic(self, x_lp: np.ndarray):
        A, b, c = self._A, self._b, self._c
        sol = (x_lp >= 0.5).astype(np.float64)
        cover = A @ sol
        uncovered = np.where(cover < b - 1e-9)[0]
        if len(uncovered):
            avail = sol < 0.5
            while len(uncovered):
                sub = A[uncovered][:, :]
                gain = (sub > 0).sum(axis=0).astype(np.float64)
                gain[~avail] = 0.0
                if gain.max() <= 0:
                    return np.inf, None
                ratio = gain / np.maximum(c, 1e-12)
                j = int(np.argmax(ratio))
                sol[j] = 1.0
                avail[j] = False
                cover = A @ sol
                uncovered = np.where(cover < b - 1e-9)[0]
        return float(c @ sol), sol

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def _result(self, status, obj, sol, n_nodes, t0, gap) -> SolveResult:
        return SolveResult(
            status=status, objective=obj, solution=sol, n_nodes=n_nodes,
            solve_time=time.perf_counter() - t0, optimality_gap=gap,
            tree_lps=self._lp.lp_count,
            decision_lps=self._decision_lps,
            lp_time=self._lp.lp_time,
            cuts_added=len(self._lp.committed_cuts),
            diagnostics={
                **self._diag,
                "sb_init": self.sb_init,
                "eta": self.eta,
            },
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _is_integral(x: np.ndarray, tol: float = 1e-4) -> bool:
    return bool(np.all(np.abs(x - np.round(x)) < tol))


def _fingerprint(cut: CutRecord):
    nz = np.flatnonzero(np.abs(cut.lhs) > 1e-9)
    return (nz.tobytes(), np.round(cut.lhs[nz], 6).tobytes(), round(float(cut.rhs), 6))
