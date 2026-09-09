"""
collect_highs.py — Trajectory collector for the HiGHS-based B&B solver.

Produces the EXACT same .npz schema as collect_with_cuts_v2.py so all
existing datasets, loaders, and trainers work without modification.

Key design choices
------------------
* HiGHS LP backend (no SCIP, no Ecole) — data is on-distribution with
  the deployed solver, closing the Theorem 3.2 joint-training requirement.
* Strong branching (SB) labels — re-solves the LP with each candidate's
  bound tightened, measures the LP-bound gain, takes the argmax. This is
  the same signal as Ecole's StrongBranchingScores (minus SCIP internals).
* Ecole-compatible 19-dim variable features — manually assembled from
  HiGHS LP solution data to match the exact feature layout expected by
  the pre-trained GNN encoder.
* node_ids / parent_ids / branch_dirs — tracked through our own B&B tree
  so SequenceDataset can build true root→leaf paths (mandatory for P0.3).
* Root Gomory cuts — same generator as deploy (gomory.py) for zero
  train/serve mismatch on cut features (P0.6).

Variable feature layout (19 dims) — mirrors Ecole NodeBipartite:
    0  obj_coef_norm       objective coefficient / max(|obj|)
    1  has_lb              1 if variable has a finite lower bound
    2  has_ub              1 if variable has a finite upper bound
    3  sol_is_at_lb        1 if LP sol value == lower bound (within tol)
    4  sol_is_at_ub        1 if LP sol value == upper bound (within tol)
    5  basis_status        encoded: 0=lower, 1=basic, 2=upper, 3=superbasic
    6  reduced_cost_norm   reduced cost / max(|rc|) + 1e-8
    7  avg_incumbent       0 (no incumbent tracking per-var in this collector)
    8  n_rows_norm         number of constraints variable appears in / n_cons
    9  obj_sense           0=minimize (always for set-cover), 1=maximize
   10  col_age             0 (static instances: no column aging)
   11  incumbent_value     0 (no per-variable incumbent stored here)
   12  avg_incumbent       0 (duplicate — kept for layout compatibility)
   13  sol_val             LP solution value x_j ∈ [0, 1]
   14  sol_frac            |sol_val - round(sol_val)|   ← CRITICAL: used as
                           fractional mask by encoder and Phase 4 integrality head
   15  lp_obj_norm         LP objective value / |LP obj| + 1e-8
   16  n_rows_tight_norm   fraction of rows where constraint is (near-)tight
   17  lb                  lower bound value (0 for binary)
   18  ub                  upper bound value (1 for binary)

Constraint feature layout (5 dims) — mirrors Ecole NodeBipartite:
    0  obj_cos             cosine between constraint coefficients and objective
    1  rhs                 right-hand side value (1 for set-cover)
    2  is_tight            1 if the constraint is (near-)tight at LP solution
    3  dual_value_norm     dual variable / max(|dual|) + 1e-8
    4  n_vars_norm         number of variables in constraint / n_vars

Usage
-----
    # Easy tier (curriculum warmup, fast SB):
    python scripts/collect_highs.py \\
        --n_instances 200 --n_rows 100 --n_cols 200 --max_steps 150 \\
        --out_dir data/highs_trajectories/easy --seed 0

    # Medium tier (primary training distribution):
    python scripts/collect_highs.py \\
        --n_instances 400 --n_rows 500 --n_cols 1000 --max_steps 300 \\
        --out_dir data/highs_trajectories/medium --seed 1000

    # Hard tier (generalisation target):
    python scripts/collect_highs.py \\
        --n_instances 200 --n_rows 1000 --n_cols 2000 --max_steps 500 \\
        --out_dir data/highs_trajectories/hard --seed 2000

    # Smoke-test:
    python scripts/collect_highs.py --n_instances 3 --n_rows 100 --n_cols 200 \\
        --max_steps 50 --debug
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

try:
    import highspy
    _HIGHS_OK = True
except ImportError:
    highspy = None
    _HIGHS_OK = False

from bnb_wm.solver.gomory import generate_root_gomory_cuts

# ── tolerances ────────────────────────────────────────────────────────────────
_FRAC_TOL   = 1e-6   # LP value is "integer" if within this of 0 or 1
_TIGHT_TOL  = 1e-4   # constraint is "tight" if slack < this


# ── instance generation ───────────────────────────────────────────────────────

def _gen_instance(rng, n_rows, n_cols, density):
    """Random set-cover instance: A x >= 1, x ∈ {0,1}^n, min c'x."""
    A = (rng.random((n_rows, n_cols)) < density).astype(np.float64)
    for i in range(n_rows):
        if A[i].sum() == 0:
            A[i, rng.integers(n_cols)] = 1.0
    b = np.ones(n_rows, dtype=np.float64)
    c = rng.uniform(1.0, 10.0, size=n_cols)
    return A, b, c


# ── HiGHS LP solver wrapper ───────────────────────────────────────────────────

class _HiGHSLP:
    """Thin wrapper: build and solve an LP with HiGHS, expose solution data."""

    def __init__(self, A, b, c, var_lb=None, var_ub=None):
        self.m, self.n = A.shape
        self.A = A
        self.b = b
        self.c = c
        self.var_lb = var_lb if var_lb is not None else np.zeros(self.n)
        self.var_ub = var_ub if var_ub is not None else np.ones(self.n)
        self.h = None
        self._sol = None

    def solve(self):
        """Solve the LP; returns True if optimal."""
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("presolve", "off")

        inf = highspy.kHighsInf
        lb = np.asarray(self.var_lb, dtype=np.float64)
        ub = np.asarray(self.var_ub, dtype=np.float64)
        h.addVars(self.n, lb, ub)
        col_idx = np.arange(self.n, dtype=np.int32)
        h.changeColsCost(self.n, col_idx, np.asarray(self.c, dtype=np.float64))
        for i in range(self.m):
            idx = np.where(np.abs(self.A[i]) > 1e-12)[0]
            if len(idx) == 0:
                continue
            h.addRow(float(self.b[i]), inf,
                     len(idx), idx.astype(np.int32),
                     self.A[i, idx].astype(np.float64))
        h.run()
        # HighsModelStatus.kOptimal is the correct enum member for getModelStatus().
        # kSolutionStatusOptimal does not exist; using it would always mismatch.
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            self.h = h
            return False
        self.h = h
        sol = h.getSolution()
        self._sol = {
            "x":    np.array(sol.col_value, dtype=np.float64),
            "rc":   np.array(sol.col_dual,  dtype=np.float64),
            "y":    np.array(sol.row_dual,  dtype=np.float64),
            "slack": np.array(sol.row_value, dtype=np.float64),
            "obj":  h.getInfoValue("objective_function_value")[1],
        }
        basis = h.getBasis()
        kBasic = highspy.HighsBasisStatus.kBasic
        kLower = highspy.HighsBasisStatus.kLower
        kUpper = highspy.HighsBasisStatus.kUpper
        col_status = list(basis.col_status)
        self._sol["basis"] = np.array(
            [1 if col_status[j] == kBasic else 0 for j in range(self.n)], dtype=np.int8)
        self._sol["at_lb"] = np.array(
            [1 if col_status[j] == kLower else 0 for j in range(self.n)], dtype=np.int8)
        self._sol["at_ub"] = np.array(
            [1 if col_status[j] == kUpper else 0 for j in range(self.n)], dtype=np.int8)
        # encode basis_status: lower=0, basic=1, upper=2
        self._sol["basis_status"] = np.array(
            [1 if col_status[j] == kBasic else
             (2 if col_status[j] == kUpper else 0)
             for j in range(self.n)], dtype=np.int8)
        return True

    @property
    def x(self):
        return self._sol["x"] if self._sol else None

    @property
    def obj(self):
        return self._sol["obj"] if self._sol else float("nan")

    def tighten_and_solve(self, j, direction):
        """Re-solve with x_j ∈ {ceil(x_j), 1} (+1) or {0, floor(x_j)} (-1).
        Returns LP objective or +inf if infeasible."""
        xj = float(self._sol["x"][j])
        lb = self.var_lb.copy()
        ub = self.var_ub.copy()
        if direction == +1:   # up branch: x_j >= ceil
            lb[j] = float(np.ceil(xj - _FRAC_TOL))
        else:                 # down branch: x_j <= floor
            ub[j] = float(np.floor(xj + _FRAC_TOL))
        child = _HiGHSLP(self.A, self.b, self.c, lb, ub)
        ok = child.solve()
        if not ok:
            return float("inf")
        return child.obj


# ── feature extraction ────────────────────────────────────────────────────────

def _var_features(A, b, c, sol, n_rows_per_var):
    """Build [n_vars, 19] variable features (Ecole-compatible layout)."""
    n_vars = len(c)
    x    = sol["x"]
    rc   = sol["rc"]
    lp_obj = sol["obj"]

    obj_max = max(np.abs(c).max(), 1e-8)
    rc_max  = max(np.abs(rc).max(), 1e-8)

    sol_frac = np.abs(x - np.round(np.clip(x, 0.0, 1.0)))

    # Fraction of rows tight per variable: for each j, count rows where
    # the constraint is tight AND A[i,j] != 0.
    slack  = sol["slack"] - b   # row_value - b; tight when slack ≈ 0
    tight_rows = (np.abs(slack) < _TIGHT_TOL)   # [m]
    n_tight_per_var = np.array(
        [int((tight_rows & (A[:, j] != 0)).sum()) for j in range(n_vars)],
        dtype=np.float32)
    n_rows_f = max(A.shape[0], 1)

    vf = np.stack([
        c / obj_max,                               # 0  obj_coef_norm
        np.ones(n_vars, dtype=np.float32),         # 1  has_lb  (always 0 for binary)
        np.ones(n_vars, dtype=np.float32),         # 2  has_ub  (always 1 for binary)
        sol["at_lb"].astype(np.float32),           # 3  sol_is_at_lb
        sol["at_ub"].astype(np.float32),           # 4  sol_is_at_ub
        sol["basis_status"].astype(np.float32),    # 5  basis_status (0/1/2)
        rc / rc_max,                               # 6  reduced_cost_norm
        np.zeros(n_vars, dtype=np.float32),        # 7  avg_incumbent (not tracked)
        n_rows_per_var / n_rows_f,                 # 8  n_rows_norm
        np.zeros(n_vars, dtype=np.float32),        # 9  obj_sense (min=0)
        np.zeros(n_vars, dtype=np.float32),        # 10 col_age
        np.zeros(n_vars, dtype=np.float32),        # 11 incumbent_value
        np.zeros(n_vars, dtype=np.float32),        # 12 avg_incumbent (dup)
        x.astype(np.float32),                      # 13 sol_val  ← index 13
        sol_frac.astype(np.float32),               # 14 sol_frac ← index 14 CRITICAL
        np.full(n_vars, lp_obj / (abs(lp_obj) + 1e-8), dtype=np.float32),  # 15
        n_tight_per_var / n_rows_f,                # 16 n_rows_tight_norm
        np.zeros(n_vars, dtype=np.float32),        # 17 lb  (0 for binary)
        np.ones(n_vars, dtype=np.float32),         # 18 ub  (1 for binary)
    ], axis=1).astype(np.float32)

    return vf


def _con_features(A, b, c, sol):
    """Build [n_cons, 5] constraint features (Ecole-compatible layout)."""
    n_cons, n_vars = A.shape
    x     = sol["x"]
    y     = sol["y"]  # dual variables
    slack = sol["slack"] - b   # A_i x - b_i; tight when ≈ 0

    y_max = max(np.abs(y).max(), 1e-8)
    obj_norm = c / (np.linalg.norm(c) + 1e-8)

    # Cosine between row coefficients and objective
    row_norms = np.linalg.norm(A, axis=1) + 1e-8
    obj_cos = (A @ obj_norm) / row_norms

    tight  = (np.abs(slack) < _TIGHT_TOL).astype(np.float32)
    n_vars_f = max(n_vars, 1)

    cf = np.stack([
        obj_cos.astype(np.float32),                 # 0 obj_cos
        b.astype(np.float32),                        # 1 rhs
        tight,                                       # 2 is_tight
        (y / y_max).astype(np.float32),             # 3 dual_value_norm
        (A != 0).sum(axis=1).astype(np.float32) / n_vars_f,  # 4 n_vars_norm
    ], axis=1).astype(np.float32)

    return cf


def _bipartite_edges(A):
    """Return (edge_indices [2, E], edge_values [E]) for the bipartite graph."""
    rows, cols = np.where(np.abs(A) > 1e-12)
    ei = np.stack([rows.astype(np.int64), cols.astype(np.int64)], axis=0)
    ev = A[rows, cols].astype(np.float32)
    return ei, ev


# ── strong branching ──────────────────────────────────────────────────────────

def _strong_branching_scores_cached(lp: _HiGHSLP, action_set: np.ndarray,
                                    cache: dict) -> np.ndarray:
    """SB with LP-solve caching to avoid re-solving the same child twice."""
    scores = np.zeros(len(action_set), dtype=np.float64)
    lp_obj = lp.obj
    for k, j in enumerate(action_set):
        if (j, +1) not in cache:
            cache[(j, +1)] = lp.tighten_and_solve(int(j), +1)
        if (j, -1) not in cache:
            cache[(j, -1)] = lp.tighten_and_solve(int(j), -1)
        gain_up   = max(0.0, cache[(j, +1)] - lp_obj)
        gain_down = max(0.0, cache[(j, -1)] - lp_obj)
        scores[k] = max(gain_up, gain_down)
    return scores.astype(np.float32)


# ── cut features (identical to collect_with_cuts_v2) ─────────────────────────

def _cut_features(lhs, rhs, x_lp, obj, n_vars):
    viol = max(0.0, rhs - float(lhs @ x_lp))
    norm = float(np.linalg.norm(lhs)) + 1e-8
    obj_u = obj / (np.linalg.norm(obj) + 1e-8)
    frac = np.abs(x_lp - np.round(np.clip(x_lp, 0.0, 1.0)))
    sup  = np.abs(lhs) > 1e-9
    sup_frac = float((frac[sup] > 0.05).mean()) if sup.any() else 0.0
    return [viol, viol / norm, float(np.count_nonzero(lhs)) / n_vars,
            float((lhs / norm) @ obj_u),
            viol * abs(float(obj @ lhs) / (norm ** 2 + 1e-8)), sup_frac]


def _collect_root_cuts(A, b, c, x_lp, max_cuts=50):
    """Generate root Gomory cuts + 6-dim features. Returns (feats, lhs, rhs)."""
    n_vars = A.shape[1]
    empty_f = np.zeros((0, 6), np.float32)
    empty_l = np.zeros((0, n_vars), np.float32)
    empty_r = np.zeros(0, np.float32)
    if not _HIGHS_OK:
        return empty_f, empty_l, empty_r
    cuts = generate_root_gomory_cuts(A, b, c, highspy, max_cuts=max_cuts,
                                     x_lp=x_lp)
    if not cuts:
        return empty_f, empty_l, empty_r
    lhs = np.stack([np.asarray(a, dtype=np.float32) for a, _ in cuts])
    rhs = np.array([float(be) for _, be in cuts], dtype=np.float32)
    feats = np.array(
        [_cut_features(lhs[i], float(rhs[i]), x_lp, c, n_vars)
         for i in range(len(rhs))], dtype=np.float32)
    return feats, lhs, rhs


# ── main B&B collection loop ──────────────────────────────────────────────────

class _NodeInfo:
    __slots__ = ("node_id", "parent_id", "branch_dir", "depth",
                 "var_lb", "var_ub", "lp")
    def __init__(self, node_id, parent_id, branch_dir, depth, var_lb, var_ub):
        self.node_id    = node_id
        self.parent_id  = parent_id
        self.branch_dir = branch_dir
        self.depth      = depth
        self.var_lb     = var_lb
        self.var_ub     = var_ub
        self.lp         = None   # filled after LP solve


def _record_trajectory(A, b, c, args, rng):
    """
    Run one strong-branching B&B trajectory on (A, b, c).

    The tree is a simple best-bound FIFO queue (priority = -lp_obj).
    We stop after `args.max_steps` RECORDED nodes (nodes where SB fires).

    Returns the trajectory dict (same schema as collect_with_cuts_v2.py)
    or None if fewer than `args.min_steps` nodes were recorded.
    """
    import heapq

    n_rows, n_vars = A.shape
    n_rows_per_var = (A != 0).sum(axis=0).astype(np.float32)

    # ── solve root LP ───────────────────────────────────────────────────────
    root_lp = _HiGHSLP(A, b, c)
    if not root_lp.solve():
        return None   # infeasible root LP
    root_obj = root_lp.obj

    # ── root cuts (once, same as deploy) ───────────────────────────────────
    x_root = root_lp.x
    cut_feats_root, cut_lhs_root, cut_rhs_root = _collect_root_cuts(
        A, b, c, x_root, max_cuts=args.max_cut_evals)
    n_root_cuts = len(cut_rhs_root)

    # ── collect root-node observation ───────────────────────────────────────
    root_vf = _var_features(A, b, c, root_lp._sol, n_rows_per_var)
    root_cf = _con_features(A, b, c, root_lp._sol)
    root_ei, root_ev = _bipartite_edges(A)

    frac_mask = (root_vf[:, 14] > 0.05)  # sol_frac > 0.05
    action_set_root = np.where(frac_mask)[0].astype(np.int32)
    if len(action_set_root) == 0:
        return None  # root LP already integer

    # SB at root
    sb_cache = {}
    sb_root = _strong_branching_scores_cached(root_lp, action_set_root, sb_cache)
    best_local_root = int(np.argmax(sb_root))
    chosen_root = int(action_set_root[best_local_root])
    dual_bound_root = root_obj

    # ── buffer ──────────────────────────────────────────────────────────────
    keys = ("var_features", "con_features", "edge_indices", "edge_values",
            "action_sets", "branching_vars", "local_branching_label",
            "sb_scores", "dual_bounds", "depths", "node_ids", "parent_ids",
            "branch_dirs")
    buf = {k: [] for k in keys}

    def _record_node(vf, cf, ei, ev, aset, chosen, best_local, sb_scores,
                     lp_obj, depth, nid, pid, bdir):
        buf["var_features"].append(vf)
        buf["con_features"].append(cf)
        buf["edge_indices"].append(ei)
        buf["edge_values"].append(ev)
        buf["action_sets"].append(aset)
        buf["branching_vars"].append(int(chosen))
        buf["local_branching_label"].append(int(best_local))
        buf["sb_scores"].append(sb_scores)
        buf["dual_bounds"].append(float(lp_obj))
        buf["depths"].append(int(depth))
        buf["node_ids"].append(int(nid))
        buf["parent_ids"].append(int(pid))
        buf["branch_dirs"].append(int(bdir))

    _record_node(root_vf, root_cf, root_ei, root_ev,
                 action_set_root, chosen_root, best_local_root,
                 sb_root, dual_bound_root, 0, 1, -1, 0)

    # ── B&B tree (best-bound priority queue) ────────────────────────────────
    # Each item: (priority, counter, node_info)
    # priority = -lp_obj so that best-bound (lowest lp_obj for minimisation)
    # is explored first. Counter breaks ties deterministically.
    _ctr = [2]  # node id counter (root = 1)
    heap = []

    def _push_children(parent_lp, parent_id, parent_depth, branch_var):
        for bdir in (+1, -1):
            nid = _ctr[0]; _ctr[0] += 1
            lb  = parent_lp.var_lb.copy()
            ub  = parent_lp.var_ub.copy()
            xj  = float(parent_lp.x[branch_var])
            if bdir == +1:
                lb[branch_var] = float(np.ceil(xj - _FRAC_TOL))
            else:
                ub[branch_var] = float(np.floor(xj + _FRAC_TOL))
            ni  = _NodeInfo(nid, parent_id, bdir, parent_depth + 1, lb, ub)
            heapq.heappush(heap, (-parent_lp.obj, nid, ni))

    _push_children(root_lp, 1, 0, chosen_root)

    best_int = float("inf")
    n_recorded = 1

    while heap and n_recorded < args.max_steps:
        _, _, ni = heapq.heappop(heap)

        # LP solve for this node
        lp = _HiGHSLP(A, b, c, ni.var_lb, ni.var_ub)
        if not lp.solve():
            continue  # infeasible: prune
        lp_obj = lp.obj
        if lp_obj >= best_int - 1e-6:
            continue  # pruned by bound

        x = lp.x
        frac = (x > _FRAC_TOL) & (x < 1.0 - _FRAC_TOL)
        aset = np.where(frac)[0].astype(np.int32)

        if len(aset) == 0:
            # Integer feasible
            obj_val = float(c @ np.round(x))
            if obj_val < best_int:
                best_int = obj_val
            continue  # leaf: do not record (no branching decision)

        # SB scores for this node (no cache: different LP state each node)
        sb_scores = _strong_branching_scores_cached(lp, aset, {})
        best_local = int(np.argmax(sb_scores))
        chosen = int(aset[best_local])

        vf = _var_features(A, b, c, lp._sol, n_rows_per_var)
        cf = _con_features(A, b, c, lp._sol)
        ei, ev = _bipartite_edges(A)

        _record_node(vf, cf, ei, ev, aset, chosen, best_local, sb_scores,
                     lp_obj, ni.depth, ni.node_id, ni.parent_id, ni.branch_dir)
        n_recorded += 1

        _push_children(lp, ni.node_id, ni.depth, chosen)

    n = len(buf["branching_vars"])
    if n < args.min_steps:
        return None

    db         = np.asarray(buf["dual_bounds"], dtype=np.float64)
    node_ids   = np.asarray(buf["node_ids"],    dtype=np.int64)
    parent_ids = np.asarray(buf["parent_ids"],  dtype=np.int64)

    # leaf labels: a recorded node with no recorded child is a leaf
    child_of    = set(int(p) for p in parent_ids.tolist())
    next_is_leaf = np.array(
        [0.0 if int(nid) in child_of else 1.0 for nid in node_ids],
        dtype=np.float32)

    # subtree_size: number of recorded descendants (inclusive of self) for each
    # node, computed bottom-up over the recorded tree.
    # Required by SubtreeSizeHead training and solver's size_weight scoring.
    nid_to_idx = {int(nid): t for t, nid in enumerate(node_ids.tolist())}
    children: dict[int, list[int]] = {int(nid): [] for nid in node_ids.tolist()}
    for t, (nid, pid) in enumerate(zip(node_ids.tolist(), parent_ids.tolist())):
        if int(pid) in children:
            children[int(pid)].append(int(nid))
    subtree_size_arr = np.ones(n, dtype=np.float32)
    # Process nodes in reverse-recorded order (children tend to appear after
    # parents in best-bound order, so reverse is a reasonable bottom-up pass).
    for t in range(n - 1, -1, -1):
        nid = int(node_ids[t])
        pid = int(parent_ids[t])
        if pid in nid_to_idx:
            subtree_size_arr[nid_to_idx[pid]] += subtree_size_arr[t]

    # norm_dual_bounds: per-traj fallback (real anchors stored separately)
    ptp = float(np.ptp(db))
    norm_db = ((db - db.min()) / (ptp + 1e-8)).astype(np.float32)

    # optimal_valid: True if we found any integer solution in the partial B&B
    primal = float(best_int) if best_int < float("inf") else float(db[-1])
    optimal_valid = best_int < float("inf")

    # ── build cut arrays (root-only, broadcast) ──────────────────────────────
    cut_features = np.empty(n, dtype=object)
    cut_labels   = np.empty(n, dtype=object)
    cut_scores   = np.empty(n, dtype=object)
    cut_lhs      = np.empty(n, dtype=object)
    cut_rhs      = np.empty(n, dtype=object)
    n_cuts       = np.zeros(n, dtype=np.int32)

    for t in range(n):
        if t == 0 and n_root_cuts > 0:
            cut_features[t] = cut_feats_root
            cut_lhs[t]      = cut_lhs_root
            cut_rhs[t]      = cut_rhs_root
            # No dive scoring here: use violation as proxy label (top-k by viol)
            viol = cut_feats_root[:, 0]   # col 0 = violation
            scores_t = viol.astype(np.float32)
            labels_t = np.zeros(n_root_cuts, dtype=np.float32)
            topk = min(5, n_root_cuts)
            if topk > 0:
                labels_t[np.argsort(-viol)[:topk]] = 1.0
            cut_scores[t]  = scores_t
            cut_labels[t]  = labels_t
            n_cuts[t]      = n_root_cuts
        else:
            cut_features[t] = np.zeros((0, 6), np.float32)
            cut_labels[t]   = np.zeros(0, np.float32)
            cut_scores[t]   = np.zeros(0, np.float32)
            cut_lhs[t]      = np.zeros((0, n_vars), np.float32)
            cut_rhs[t]      = np.zeros(0, np.float32)

    return {
        "n_steps":              np.asarray(n),
        "var_features":         np.asarray(buf["var_features"], dtype=object),
        "con_features":         np.asarray(buf["con_features"], dtype=object),
        "edge_indices":         np.asarray(buf["edge_indices"], dtype=object),
        "edge_values":          np.asarray(buf["edge_values"], dtype=object),
        "action_sets":          np.asarray(buf["action_sets"], dtype=object),
        "branching_vars":       np.asarray(buf["branching_vars"], dtype=np.int32),
        "local_branching_label":np.asarray(buf["local_branching_label"], dtype=np.int32),
        "sb_scores":            np.asarray(buf["sb_scores"], dtype=object),
        "dual_bounds":          db.astype(np.float32),
        "norm_dual_bounds":     norm_db,
        "root_bound":           np.asarray(float(db[0]), dtype=np.float32),
        "primal_bound":         np.asarray(primal, dtype=np.float32),
        "optimal_valid":        np.asarray(optimal_valid),
        "next_is_leaf":         next_is_leaf,
        "subtree_size":         subtree_size_arr,
        "depths":               np.asarray(buf["depths"], dtype=np.int32),
        "node_ids":             node_ids.astype(np.int64),
        "parent_ids":           parent_ids.astype(np.int64),
        "branch_dirs":          np.asarray(buf["branch_dirs"], dtype=np.int8),
        "cut_features":         cut_features,
        "cut_labels":           cut_labels,
        "cut_scores":           cut_scores,
        "cut_lhs":              cut_lhs,
        "cut_rhs":              cut_rhs,
        "n_cuts":               n_cuts,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_instances",    type=int,   default=500)
    p.add_argument("--n_rows",         type=int,   default=200)
    p.add_argument("--n_cols",         type=int,   default=400)
    p.add_argument("--density",        type=float, default=0.05)
    p.add_argument("--max_steps",      type=int,   default=300,
                   help="max recorded (branchable) nodes per trajectory")
    p.add_argument("--min_steps",      type=int,   default=5)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--out_dir",        type=Path,  default=Path("data/highs_trajectories/train"))
    p.add_argument("--max_cut_evals",  type=int,   default=50)
    p.add_argument("--overwrite",      action="store_true")
    p.add_argument("--debug",          action="store_true",
                   help="collect 3 instances and print schema summary")
    args = p.parse_args()

    if not _HIGHS_OK:
        raise SystemExit("ERROR: highspy is not importable. Install with: pip install highspy")

    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    n_inst = 3 if args.debug else args.n_instances

    manifest = []
    t0_global = time.perf_counter()

    for i in range(n_inst):
        out_path = args.out_dir / f"traj_{i:05d}.npz"
        if out_path.exists() and not args.overwrite:
            print(f"  [{i+1}/{n_inst}] skip (exists) {out_path.name}")
            continue

        t0 = time.perf_counter()
        A, b, c = _gen_instance(rng, args.n_rows, args.n_cols, args.density)
        traj = _record_trajectory(A, b, c, args, rng)
        elapsed = time.perf_counter() - t0

        if traj is None:
            print(f"  [{i+1}/{n_inst}] skip (too few steps or infeasible)")
            continue

        np.savez_compressed(out_path, **traj)
        n = int(traj["n_steps"])
        n_cuts = int(traj["n_cuts"][0])
        manifest.append({"file": str(out_path), "n_steps": n,
                          "root_cuts": n_cuts})

        if args.debug or i == 0 or (i + 1) % 50 == 0 or i + 1 == n_inst:
            print(f"  [{i+1}/{n_inst}] steps={n} root_cuts={n_cuts} "
                  f"optimal_valid={bool(traj['optimal_valid'])} "
                  f"time={elapsed:.1f}s")

        if args.debug and i == 0:
            # Schema validation printout
            print("\n  ── Schema check ──")
            print(f"  var_features[0] shape: {traj['var_features'][0].shape}  "
                  f"(expected [{args.n_cols}, 19])")
            print(f"  con_features[0] shape: {traj['con_features'][0].shape}  "
                  f"(expected [{args.n_rows}, 5])")
            print(f"  sol_val  (feat 13) range: "
                  f"[{traj['var_features'][0][:,13].min():.3f}, "
                  f"{traj['var_features'][0][:,13].max():.3f}]")
            print(f"  sol_frac (feat 14) range: "
                  f"[{traj['var_features'][0][:,14].min():.3f}, "
                  f"{traj['var_features'][0][:,14].max():.3f}]")
            print(f"  node_ids:   {traj['node_ids'][:5]}")
            print(f"  parent_ids: {traj['parent_ids'][:5]}")
            print(f"  branch_dirs:{traj['branch_dirs'][:5]}")
            print(f"  depths:     {traj['depths'][:5]}")
            frac_nonzero = (traj['var_features'][0][:, 14] > 0.05).sum()
            print(f"  fractional vars at root: {frac_nonzero}/{args.n_cols}")
            print(f"  subtree_size[:5]:  {traj['subtree_size'][:5]}")
            print(f"  next_is_leaf[:5]:  {traj['next_is_leaf'][:5]}")
            print()

    total = time.perf_counter() - t0_global
    (args.out_dir / "manifest.json").write_text(
        json.dumps({"n_instances": n_inst, "args": vars(args),
                    "trajectories": manifest}, indent=2, default=str))
    print(f"\nDone: {len(manifest)}/{n_inst} trajectories saved to {args.out_dir}")
    print(f"Total time: {total/60:.1f} min  "
          f"({total/max(len(manifest),1):.1f}s/instance)")


if __name__ == "__main__":
    main()
