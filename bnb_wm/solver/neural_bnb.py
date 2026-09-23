"""
neural_bnb.py — Neural branch-and-cut solver, gate-by-gate.

One code path, every component behind a flag on SolverConfig, so the ablation
ladder is a config sweep rather than a set of forked solvers.

Node pipeline
-------------
    pop node
      Gate 1  bound pruning          node.lb >= global_ub          -> prune
      Gate 2  ORS                    low significance              -> skip   [UNSAFE]
    LP solve (persistent HiGHS, warm-started from the parent basis)
      Gate 3  infeasible / dominated                               -> prune
      Gate 4  integral solution      -> record incumbent           -> prune
      Gate 5  primal heuristic       every N nodes + root          -> tighten UB
    GNN encode -> z, h_vars
      Gate 6  neural pruning         near-UB + bad predictions     -> prune  [UNSAFE]
    policy_scores (+ optional Katz blend) -> top-K candidates
      Gate 7  cut gate (7 sub-checks)
        NO  -> rollout over top-K -> branch variable
        YES -> Gomory cuts -> cut embeddings from h_vars
                 Phase A: latent cut beam (cut_rounds x cut_beam, d=0)
                 Phase B: rollout from the best cut state
               commit cuts -> re-solve LP -> re-encode -> branch
    score children, push

Design notes that differ from the earlier prototype
---------------------------------------------------
1. LAZY CHILD LPs. Children are pushed without solving; LP is solved once on
   pop. Children inherit the parent's LP bound as their priority.

2. ONE PERSISTENT HiGHS MODEL per instance. Built once; only changed bounds
   are pushed per node.

3. GLOBAL CUT POOL. Root Gomory cuts are globally valid; no per-node
   inherited_cuts bookkeeping.

4. CACHED GRAPH STRUCTURE. edge_index / edge_attr rebuilt only on cut commit.

5. SHARED FEATURES. Node features from bnb_wm.features.var_features —
   the same function the collector used.

Exactness: with exact=True (default) Gates 2 and 6 are off and every prune
is bound-justified.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from bnb_wm.features import (
    var_features, con_features, edge_arrays,
)
from bnb_wm.solver.config import SolverConfig
from bnb_wm.solver.lp_backend import LPBackend, CutRecord


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

# Alias so existing internal code (e.g. _cut_pool) still works unchanged.
CutData = CutRecord


@dataclass(order=False)
class Node:
    lb: float
    depth: int
    var_lb: np.ndarray
    var_ub: np.ndarray
    node_id: int
    parent_id: Optional[int] = None
    priority: float = 0.0
    past_tokens: Optional[torch.Tensor] = None
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
    lp_solves: int = 0
    lp_time: float = 0.0
    model_time: float = 0.0
    cuts_added: int = 0
    is_exact: bool = True
    diagnostics: dict = field(default_factory=dict)


class _LP:
    __slots__ = ("feasible", "obj", "x", "sol", "basis")

    def __init__(self, feasible, obj=None, x=None, sol=None, basis=None):
        self.feasible = feasible
        self.obj = obj
        self.x = x
        self.sol = sol
        self.basis = basis


# ---------------------------------------------------------------------------
# Katz centrality
# ---------------------------------------------------------------------------

def katz_frac_scores(A: np.ndarray, x_lp: np.ndarray,
                     alpha: float = 0.1, iters: int = 3) -> np.ndarray:
    """Fractionality-weighted Katz-style centrality.

    s <- frac + alpha * A^T (A s), repeated `iters` times.
    A variable scores highly when it is fractional AND shares constraints
    with other fractional variables.
    """
    frac = np.abs(x_lp - np.round(np.clip(x_lp, 0.0, 1.0))).astype(np.float64)
    s = frac.copy()
    for _ in range(max(0, int(iters))):
        s = frac + alpha * (A.T @ (A @ s))
    return s


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class NeuralBnBSolver:

    def __init__(self, model, device, config: SolverConfig | None = None):
        self.model = model
        self.device = device
        self.cfg = config or SolverConfig()

        try:
            import highspy
            self._highs = highspy
        except ImportError as exc:
            raise RuntimeError(
                "highspy is required by NeuralBnBSolver."
            ) from exc

        self._cut_head_untrained = (
            self.cfg.cut_mode == "latent"
            and not bool(getattr(model, "cut_head_trained", torch.tensor(False)))
        )

    # ==================================================================
    # Public
    # ==================================================================

    def solve(self, A: np.ndarray, b: np.ndarray, c: np.ndarray) -> SolveResult:
        cfg = self.cfg
        t0 = time.perf_counter()

        A = np.ascontiguousarray(A, dtype=np.float64)
        b = np.ascontiguousarray(b, dtype=np.float64)
        c = np.ascontiguousarray(c, dtype=np.float64)
        self._A, self._b, self._c = A, b, c
        self._m0, self._n = A.shape

        self._reset_instance_state()
        self._lp_init()  # creates self._lp

        root_lb = np.zeros(self._n)
        root_ub = np.ones(self._n)

        lp = self._lp_solve(root_lb, root_ub, warm_basis=None)
        if not lp.feasible:
            return self._result("infeasible", np.inf, None, 0, t0, np.inf)

        global_ub, best_sol = np.inf, None
        status = "infeasible"

        if cfg.primal_heuristic:
            obj, sol = self._primal_heuristic(lp.x)
            if sol is not None and obj < global_ub:
                global_ub, best_sol, status = obj, sol, "feasible"
                self._diag["primal_heur_hits"] += 1

        if self._is_integral(lp.x):
            obj = float(c @ np.round(lp.x))
            if obj < global_ub:
                global_ub, best_sol, status = obj, np.round(lp.x), "feasible"
            return self._result("optimal", global_ub, best_sol, 1, t0, 0.0)

        root = Node(lb=lp.obj, depth=0, var_lb=root_lb, var_ub=root_ub,
                    node_id=0, priority=-lp.obj, warm_basis=lp.basis)
        heap: list[Node] = [root]
        self._pending_root_lp = lp

        n_nodes = 0
        next_id = 1

        while heap:
            if time.perf_counter() - t0 > cfg.time_limit:
                status = "timeout" if best_sol is None else status
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

            # Gate 2: node significance (UNSAFE)
            if cfg.node_significance_prune and self._gate_node_significance(
                    node, global_ub):
                self._diag["g2_node_sig_skip"] += 1
                continue

            # LP solve
            if self._pending_root_lp is not None:
                lp = self._pending_root_lp
                self._pending_root_lp = None
            else:
                lp = self._lp_solve(node.var_lb, node.var_ub, node.warm_basis)
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
            if self._is_integral(lp.x):
                obj = float(c @ np.round(lp.x))
                if obj < global_ub:
                    global_ub, best_sol, status = obj, np.round(lp.x), "feasible"
                self._diag["g4_integral"] += 1
                continue

            # Gate 5: periodic primal heuristic
            if (cfg.primal_heuristic
                    and n_nodes % max(1, cfg.primal_heuristic_every) == 0):
                obj, sol = self._primal_heuristic(lp.x)
                if sol is not None and obj < global_ub:
                    global_ub, best_sol, status = obj, sol, "feasible"
                    self._diag["primal_heur_hits"] += 1

            # encode
            t_enc = time.perf_counter()
            h_vars, z = self._encode(lp.sol, node.var_lb, node.var_ub)
            self._timing["encode"] += time.perf_counter() - t_enc

            frac_idx = np.where((lp.x > 1e-4) & (lp.x < 1 - 1e-4))[0]
            if len(frac_idx) == 0:
                continue
            n_frac = int(len(frac_idx))

            # Gate 6: neural prune (UNSAFE)
            if cfg.neural_prune and self._gate_neural_prune(
                    z, h_vars, lp.obj, global_ub):
                self._diag["g6_neural_prune"] += 1
                continue

            leaf_prob = self._leaf_prob(z, node.depth, n_frac)

            # Gate 7: cut gate
            fire_cuts, reason = self._cut_gate(
                z, h_vars, node, leaf_prob, n_frac, lp, global_ub)
            self._diag["cut_gate_reason"][reason] = (
                self._diag["cut_gate_reason"].get(reason, 0) + 1)

            if fire_cuts:
                lp, h_vars, z, frac_idx, committed = self._do_cuts(
                    lp, h_vars, z, node, frac_idx)
                if lp is None:
                    continue
                if committed and self._is_integral(lp.x):
                    obj = float(c @ np.round(lp.x))
                    if obj < global_ub:
                        global_ub, best_sol, status = obj, np.round(lp.x), "feasible"
                    continue
                if len(frac_idx) == 0:
                    continue
                node.lb = max(node.lb, lp.obj)

            # branch variable
            t_br = time.perf_counter()
            branch_var, child_tokens = self._select_branch_var(
                h_vars, z, lp.x, frac_idx, node, leaf_prob)
            self._timing["branch"] += time.perf_counter() - t_br

            # children (lazy LPs)
            for direction, fix in ((1.0, 1.0), (-1.0, 0.0)):
                vlb = node.var_lb.copy()
                vub = node.var_ub.copy()
                if fix == 1.0:
                    vlb[branch_var] = 1.0
                else:
                    vub[branch_var] = 0.0
                if np.any(vlb > vub + 1e-9):
                    continue
                if node.lb >= global_ub - 1e-6:
                    continue

                priority = self._child_priority(
                    z, h_vars, branch_var, direction, node, lp.obj)

                heapq.heappush(heap, Node(
                    lb=node.lb,
                    depth=node.depth + 1,
                    var_lb=vlb, var_ub=vub,
                    node_id=next_id, parent_id=node.node_id,
                    priority=priority,
                    past_tokens=child_tokens,
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

        return self._result(
            status,
            global_ub if best_sol is not None else np.inf,
            best_sol, n_nodes, t0, gap,
        )

    # ==================================================================
    # Instance state
    # ==================================================================

    def _reset_instance_state(self):
        self._lp: Optional[LPBackend] = None
        self._gomory_pool: Optional[list[CutData]] = None
        self._struct_cache = None
        self._pending_root_lp = None
        self._last_cut_gain = float("inf")
        self._timing = {"encode": 0.0, "branch": 0.0, "rollout": 0.0,
                        "cutbeam": 0.0, "lp": 0.0}
        self._diag = {
            "g1_bound_prune": 0, "g2_node_sig_skip": 0, "g3_infeasible": 0,
            "g3_dominated": 0, "g4_integral": 0, "g6_neural_prune": 0,
            "primal_heur_hits": 0, "cuts_committed": 0, "cut_rounds_run": 0,
            "cut_gate_reason": {}, "leaf_gate_skips": 0,
            "confident_skips": 0, "timeout": False, "node_limit_hit": False,
            "cut_head_untrained": self._cut_head_untrained,
            "ors_nodes": 0, "ors_cands_seen": 0, "ors_cands_survived": 0,
            "ors_single_survivor": 0, "ors_choice_changed": 0,
        }

    def _result(self, status, obj, sol, n_nodes, t0, gap) -> SolveResult:
        lp = self._lp
        return SolveResult(
            status=status, objective=obj, solution=sol, n_nodes=n_nodes,
            solve_time=time.perf_counter() - t0, optimality_gap=gap,
            lp_solves=lp.lp_count if lp else 0,
            lp_time=lp.lp_time if lp else 0.0,
            model_time=sum(v for k, v in self._timing.items() if k != "lp"),
            cuts_added=len(lp.committed_cuts) if lp else 0,
            is_exact=self.cfg.is_exact,
            diagnostics={**self._diag, "timing": dict(self._timing),
                         "config": self.cfg.to_dict()},
        )

    # ==================================================================
    # LP layer  (delegated to LPBackend)
    # ==================================================================

    def _lp_init(self):
        self._lp = LPBackend(self._highs, self._A, self._b, self._c)

    def _lp_solve(self, vlb, vub, warm_basis) -> _LP:
        res = self._lp.solve(vlb, vub, warm_basis)
        self._timing["lp"] = self._lp.lp_time
        if not res.feasible:
            return _LP(False)
        return _LP(True, obj=res.obj, x=res.x, sol=res.sol, basis=res.basis)

    def _lp_commit_cuts(self, cuts: list[CutData]):
        self._lp.commit_cuts(cuts)
        self._diag["cuts_committed"] = len(self._lp.committed_cuts)

    # ==================================================================
    # Encoding
    # ==================================================================

    def _structure(self):
        sv = self._lp.struct_version
        if self._struct_cache is not None and self._struct_cache[0] == sv:
            return self._struct_cache[1]

        A, b = self._lp.augmented_A_b()
        m, n = A.shape
        ei, ev = edge_arrays(A)

        rhs_src = b[ei[0]].astype(np.float32)
        edge_attr_np = np.stack(
            [ev, ev / (np.abs(rhs_src) + 1e-8), np.sign(ev)], axis=1
        ).astype(np.float32)

        con_to_var = np.vstack([ei[0] + n, ei[1]])
        var_to_con = np.vstack([ei[1], ei[0] + n])
        edge_index_np = np.hstack([con_to_var, var_to_con]).astype(np.int64)
        edge_attr_np = np.concatenate([edge_attr_np, edge_attr_np], axis=0)

        dev = self.device
        struct = dict(
            edge_index=torch.as_tensor(edge_index_np, dtype=torch.long, device=dev),
            edge_attr=torch.as_tensor(edge_attr_np, dtype=torch.float32, device=dev),
            node_type=torch.cat([
                torch.zeros(n, dtype=torch.long, device=dev),
                torch.ones(m, dtype=torch.long, device=dev)]),
            batch=torch.zeros(n + m, dtype=torch.long, device=dev),
            n=n, m=m,
            n_rows_per_var=(A != 0).sum(axis=0).astype(np.float32),
            A=A, b=b,
        )
        self._struct_cache = (sv, struct)
        return struct

    def _encode(self, sol, vlb, vub):
        st = self._structure()
        A, b, n, m = st["A"], st["b"], st["n"], st["m"]

        vf = var_features(A, b, self._c, sol, st["n_rows_per_var"])
        cf = con_features(A, b, self._c, sol)

        x_np = np.zeros((n + m, 19), dtype=np.float32)
        x_np[:n] = vf
        x_np[n:, :5] = cf
        x = torch.as_tensor(x_np, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            h_vars, z = self.model.encoder(
                x, st["edge_index"], st["node_type"], st["batch"],
                edge_attr=st["edge_attr"],
            )
        return h_vars, z

    # ==================================================================
    # Gates
    # ==================================================================

    def _gate_node_significance(self, node: Node, global_ub: float) -> bool:
        cfg = self.cfg
        score = float(cfg.significance_fn(self, node, {"global_ub": global_ub}))
        if score >= cfg.ors_sig_thresh:
            return False
        return np.random.random() >= cfg.ors_p_explore

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

    def _leaf_prob(self, z, depth, n_frac) -> float:
        dev = self.device
        with torch.no_grad():
            logit = self.model.integrality_logit(
                z,
                torch.tensor([float(depth)], device=dev),
                torch.tensor([float(n_frac)], device=dev),
            )
            return float(torch.sigmoid(logit).item())

    def _gate_neural_prune(self, z, h_vars, lp_obj, global_ub) -> bool:
        cfg = self.cfg
        if not np.isfinite(global_ub):
            return False
        if lp_obj < global_ub * (1.0 - cfg.neural_prune_margin):
            return False
        bvec = torch.zeros(h_vars.size(0), dtype=torch.long, device=self.device)
        with torch.no_grad():
            s = float(self.model.subtree_size_pred(z, h_vars, bvec).item())
            v = float(self.model.value_pred(z, h_vars, bvec).item())
        return (s > cfg.neural_prune_s_thresh) and (v < cfg.neural_prune_v_thresh)

    def _cut_gate(self, z, h_vars, node, leaf_prob, n_frac, lp, global_ub):
        cfg = self.cfg
        if cfg.cut_mode == "none":
            return False, "mode_none"
        if node.depth == 0 and cfg.force_root_cuts:
            if len(self._committed_cuts) >= cfg.cut_budget_cap:
                return False, "budget"
            return True, "root_forced"
        if leaf_prob >= cfg.cut_integrality_thresh:
            return False, "near_leaf"
        if n_frac < cfg.cut_min_nfrac:
            return False, "few_frac"
        if node.depth > cfg.cut_depth_max:
            return False, "too_deep"
        if len(self._committed_cuts) >= cfg.cut_budget_cap:
            return False, "budget"
        if self._last_cut_gain < cfg.cut_min_gain:
            return False, "gain_vanished"
        if np.isfinite(global_ub):
            gap = (global_ub - lp.obj) / (abs(global_ub) + 1e-10)
            if gap < cfg.gap_tolerance:
                return False, "already_tight"
        if cfg.cut_subtree_thresh > 0.0:
            bvec = torch.zeros(h_vars.size(0), dtype=torch.long, device=self.device)
            with torch.no_grad():
                s = float(self.model.subtree_size_pred(z, h_vars, bvec).item())
            if s < cfg.cut_subtree_thresh:
                return False, "small_subtree"
        return True, "fire"

    # ==================================================================
    # Cuts
    # ==================================================================

    def _cut_pool(self) -> list[CutData]:
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
                CutData(np.asarray(lhs, dtype=np.float64), float(rhs))
                for lhs, rhs in (pool or [])
            ]
        return self._gomory_pool

    def _violated_cuts(self, x_lp: np.ndarray) -> list[CutData]:
        committed = {self._fingerprint(cut) for cut in self._lp.committed_cuts}
        out = []
        for cut in self._cut_pool():
            if self._fingerprint(cut) in committed:
                continue
            if float(cut.lhs @ x_lp) < cut.rhs - 1e-6:
                out.append(cut)
        return out

    @staticmethod
    def _fingerprint(cut: CutData):
        nz = np.flatnonzero(np.abs(cut.lhs) > 1e-9)
        return (nz.tobytes(), np.round(cut.lhs[nz], 6).tobytes(),
                round(float(cut.rhs), 6))

    def _cut_embeds(self, cuts: list[CutData], h_vars: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        H = h_vars.size(1)
        embeds = torch.zeros(len(cuts), H, device=self.device, dtype=h_vars.dtype)
        for i, cut in enumerate(cuts):
            nz = np.flatnonzero(np.abs(cut.lhs) > 1e-9)
            if len(nz) == 0:
                continue
            idx = torch.as_tensor(nz, dtype=torch.long, device=self.device)
            w = torch.as_tensor(cut.lhs[nz], dtype=h_vars.dtype, device=self.device)
            rows = h_vars.index_select(0, idx)
            e = (w.unsqueeze(1) * rows).sum(dim=0)

            if cfg.cut_embed_norm == "mean":
                e = e / float(len(nz))
            elif cfg.cut_embed_norm == "l2match":
                target = rows.norm(dim=1).mean()
                cur = e.norm()
                if float(cur) > 1e-8:
                    e = e * (target / cur)
            embeds[i] = e
        return embeds

    def _cut_beam(self, z, h_vars, cut_embeds, cuts, past_tokens):
        cfg = self.cfg
        C = cut_embeds.size(0)
        if C == 0:
            return z, past_tokens, []

        bvec = torch.zeros(h_vars.size(0), dtype=torch.long, device=self.device)

        with torch.no_grad():
            base_score = float(self.model.value_pred(z, h_vars, bvec).item())
        beams = [(base_score, z, past_tokens, [])]
        best = (base_score, z, past_tokens, [])

        with torch.no_grad():
            for _ in range(max(1, cfg.cut_rounds)):
                self._diag["cut_rounds_run"] += 1
                cands = []
                for score, z_cur, tok_cur, chosen in beams:
                    remaining = [i for i in range(C) if i not in chosen]
                    if not remaining:
                        continue
                    rem = torch.as_tensor(remaining, dtype=torch.long,
                                          device=self.device)
                    B = len(remaining)
                    z_b = z_cur.expand(B, -1)
                    a_b = cut_embeds.index_select(0, rem)
                    tok_b = (tok_cur.expand(B, -1, -1)
                             if tok_cur is not None else None)
                    d_b = torch.zeros(B, device=self.device, dtype=z.dtype)

                    z_next, tok_next = self.model.dynamics_step(
                        z_b, a_b, tok_b, d_b)
                    h_flat = h_vars.unsqueeze(0).expand(B, -1, -1).reshape(
                        B * h_vars.size(0), -1)
                    bv = torch.arange(B, device=self.device).repeat_interleave(
                        h_vars.size(0))
                    vals = self.model.value_pred(z_next, h_flat, bv)

                    for bi in range(B):
                        cands.append((
                            float(vals[bi].item()),
                            z_next[bi:bi + 1],
                            tok_next[bi:bi + 1],
                            chosen + [remaining[bi]],
                        ))

                if not cands:
                    break
                cands.sort(key=lambda t: -t[0])
                beams = cands[:max(1, cfg.cut_beam)]
                if beams[0][0] > best[0]:
                    best = beams[0]

        chosen_cuts = [cuts[i] for i in best[3][:cfg.max_cuts_per_node]]
        return best[1], best[2], chosen_cuts

    def _do_cuts(self, lp, h_vars, z, node, frac_idx):
        cfg = self.cfg
        cands = self._violated_cuts(lp.x)
        if not cands:
            self._last_cut_gain = 0.0
            return lp, h_vars, z, frac_idx, False

        if cfg.cut_mode == "heuristic":
            viol = np.array([cut.rhs - float(cut.lhs @ lp.x) for cut in cands])
            chosen = [cands[i] for i in np.argsort(-viol)[:cfg.max_cuts_per_node]]
        else:
            t0 = time.perf_counter()
            embeds = self._cut_embeds(cands, h_vars)
            _z_best, _tok_best, chosen = self._cut_beam(
                z, h_vars, embeds, cands, node.past_tokens)
            self._timing["cutbeam"] += time.perf_counter() - t0

        if not chosen:
            self._last_cut_gain = 0.0
            return lp, h_vars, z, frac_idx, False

        obj_before = lp.obj
        self._lp_commit_cuts(chosen)
        lp2 = self._lp_solve(node.var_lb, node.var_ub, lp.basis)
        if not lp2.feasible:
            return None, None, None, None, True

        self._last_cut_gain = float(lp2.obj - obj_before)
        t_enc = time.perf_counter()
        h2, z2 = self._encode(lp2.sol, node.var_lb, node.var_ub)
        self._timing["encode"] += time.perf_counter() - t_enc
        frac2 = np.where((lp2.x > 1e-4) & (lp2.x < 1 - 1e-4))[0]
        return lp2, h2, z2, frac2, True

    # ==================================================================
    # Branching
    # ==================================================================

    def _policy_scores(self, h_vars, z, x_lp):
        bvec = torch.zeros(h_vars.size(0), dtype=torch.long, device=self.device)
        with torch.no_grad():
            scores = self.model.policy_scores(h_vars, z, bvec)
        if self.cfg.katz_weight != 0.0:
            st = self._structure()
            katz = katz_frac_scores(st["A"][:self._m0], x_lp,
                                    self.cfg.katz_alpha, self.cfg.katz_iters)
            scores = scores + self.cfg.katz_weight * torch.as_tensor(
                np.log1p(katz), dtype=scores.dtype, device=self.device)
        return scores

    def _select_branch_var(self, h_vars, z, x_lp, frac_idx, node, leaf_prob):
        cfg = self.cfg

        if cfg.branch_mode == "most_fractional":
            fr = np.abs(x_lp - np.round(x_lp))
            return int(frac_idx[np.argmax(fr[frac_idx])]), node.past_tokens
        if cfg.branch_mode == "random":
            return int(np.random.choice(frac_idx)), node.past_tokens

        scores = self._policy_scores(h_vars, z, x_lp)
        frac_t = torch.as_tensor(frac_idx, dtype=torch.long, device=self.device)
        masked = torch.full_like(scores, -1e4)
        masked[frac_t] = scores[frac_t]

        def _advance(var):
            with torch.no_grad():
                a = h_vars[var].unsqueeze(0)
                _z, tok = self.model.dynamics_step(z, a, node.past_tokens, 0.0)
            return tok

        if cfg.branch_mode == "policy":
            v = int(masked.argmax())
            return v, _advance(v)

        if leaf_prob > cfg.leaf_prob_skip:
            self._diag["leaf_gate_skips"] += 1
            v = int(masked.argmax())
            return v, _advance(v)

        if cfg.skip_confident is not None:
            p_top = float(torch.softmax(scores[frac_t], dim=0).max())
            if p_top >= cfg.skip_confident:
                self._diag["confident_skips"] += 1
                v = int(masked.argmax())
                return v, _advance(v)

        k = min(cfg.lookahead_k, len(frac_idx))
        top_k = masked.topk(k).indices
        valid_mask = torch.zeros(h_vars.size(0), dtype=torch.bool,
                                 device=self.device)
        valid_mask[frac_t] = True

        t0 = time.perf_counter()
        cands = [int(v) for v in top_k]

        # ORS cascade
        shallow_best = None
        if cfg.ors_cascade and len(cands) > 1:
            q_shallow, v1mag = self._ors_shallow(
                z, h_vars, cands, node, valid_mask)
            best_q = max(q_shallow)
            gpow = cfg.lookahead_gamma ** max(
                0, cfg.lookahead_depth - cfg.ors_shallow_depth)

            survivors = [
                cands[i] for i in range(len(cands))
                if q_shallow[i] + gpow * abs(v1mag[i]) >= best_q - cfg.ors_margin
            ]
            shallow_best = cands[int(np.argmax(q_shallow))]
            if shallow_best not in survivors:
                survivors.append(shallow_best)

            self._diag["ors_nodes"] += 1
            self._diag["ors_cands_seen"] += len(cands)
            self._diag["ors_cands_survived"] += len(survivors)

            if len(survivors) == 1:
                self._diag["ors_single_survivor"] += 1
                self._timing["rollout"] += time.perf_counter() - t0
                v = survivors[0]
                return v, _advance(v)
            cands = survivors

        # Full-depth rollout on survivors
        best_var, best_score = cands[0], -float("inf")
        with torch.no_grad():
            for cand in cands:
                r = self.model.rollout_candidate_batched(
                    z, h_vars, int(cand),
                    depth=cfg.lookahead_depth, gamma=cfg.lookahead_gamma,
                    valid_mask=valid_mask, past_tokens=node.past_tokens,
                    size_weight=cfg.size_weight, ctg_weight=cfg.ctg_weight,
                    branch_factor=cfg.branch_factor,
                    use_reward_return=cfg.use_reward_return,
                )
                r = float(r) if not torch.is_tensor(r) else float(r.item())
                if r > best_score:
                    best_score, best_var = r, int(cand)
        self._timing["rollout"] += time.perf_counter() - t0

        if shallow_best is not None and best_var != shallow_best:
            self._diag["ors_choice_changed"] += 1

        return best_var, _advance(best_var)

    def _ors_shallow(self, z, h_vars, cands, node, valid_mask):
        """ORS Phase 2: shallow return and depth-1 value magnitude per candidate."""
        cfg = self.cfg
        K, V, H = len(cands), h_vars.size(0), h_vars.size(1)
        dev = self.device

        with torch.no_grad():
            q_shallow = []
            for cand in cands:
                r = self.model.rollout_candidate_batched(
                    z, h_vars, int(cand),
                    depth=cfg.ors_shallow_depth, gamma=cfg.lookahead_gamma,
                    valid_mask=valid_mask, past_tokens=node.past_tokens,
                    size_weight=cfg.size_weight, ctg_weight=cfg.ctg_weight,
                    branch_factor=cfg.branch_factor,
                    use_reward_return=cfg.use_reward_return,
                )
                q_shallow.append(float(r) if not torch.is_tensor(r)
                                 else float(r.item()))

            idx = torch.as_tensor(cands, dtype=torch.long, device=dev)
            a = h_vars.index_select(0, idx).repeat_interleave(2, dim=0)
            B = 2 * K
            z_b = z.expand(B, -1)
            h_b = h_vars.unsqueeze(0).expand(B, -1, -1)
            tok_b = (node.past_tokens.expand(B, -1, -1)
                     if node.past_tokens is not None else None)
            d_b = torch.tensor([1.0, -1.0], device=dev, dtype=z.dtype).repeat(K)

            z1, h1, _tok = self.model.dynamics_step_full_batched(
                z_b, a, h_b, tok_b, d_b)

            h1_flat = h1.reshape(B * V, H)
            bv = torch.arange(B, device=dev).repeat_interleave(V)
            v = self.model.value_pred(z1, h1_flat, bv).reshape(K, 2).abs()
            v1mag = v.max(dim=1).values.tolist()

        return q_shallow, v1mag

    def _child_priority(self, z, h_vars, branch_var, direction, node, lp_obj):
        cfg = self.cfg
        if cfg.node_selection == "bound":
            return -lp_obj

        with torch.no_grad():
            a = h_vars[branch_var].unsqueeze(0)
            z_c, h_c, _tok = self.model.dynamics_step_full(
                z, a, h_vars, node.past_tokens, direction)
            bvec = torch.zeros(h_c.size(0), dtype=torch.long, device=self.device)
            if cfg.node_selection == "cost_to_go":
                return -float(self.model.cost_to_go_pred(z_c, h_c, bvec).item())
            return -float(self.model.subtree_size_pred(z_c, h_c, bvec).item())

    # ==================================================================
    # Utilities
    # ==================================================================

    @staticmethod
    def _is_integral(x: np.ndarray, tol: float = 1e-4) -> bool:
        return bool(np.all(np.abs(x - np.round(x)) < tol))
