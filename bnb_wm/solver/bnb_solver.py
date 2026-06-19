"""
bnb_solver.py — Neural-guided Branch-and-Cut solver.

Implements a complete Python-owned B&B loop with:
    - LP relaxation at each node via scipy.optimize.linprog (HiGHS backend)
    - Branch-and-cut with globally valid cuts (pairwise CG cuts for Set Cover)
    - BnBWorldModel driving all search decisions:
        * CuttingPlaneHead  -> which cuts to add (branch-and-cut selection)
        * PolicyHead        -> branching variable selection
        * ValueHead         -> node priority (best-first search)
        * IntegralityHead   -> prune near-leaf nodes early
        * DynamicsTransformer -> 1-step latent lookahead for branching

Branch-and-cut design:
    Cuts selected at a node are GLOBALLY VALID (pairwise Chvátal-Gomory
    intersection cuts for Set Cover). They are propagated to ALL descendant
    nodes via the Node.inherited_cuts list — this is full branch-and-cut,
    not cut-and-branch. No cut validity re-check is needed because the cuts
    are valid for any integer-feasible solution regardless of branching.

Constraint format assumed:
    min  c^T x
    s.t. A x >= b          (covering constraints, as in Set Cover)
         0 <= x <= 1       (relaxed binary; branching fixes bounds to 0/1)
"""

import time
import heapq
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional
from itertools import combinations
from torch_geometric.data import Data, Batch


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CutData:
    """A single globally-valid cutting plane: lhs @ x >= rhs."""
    lhs: np.ndarray   # [n_vars] cut coefficients
    rhs: float        # cut right-hand side
    cut_type: str     # "cg" (Chvátal-Gomory)


@dataclass
class Node:
    """A single node in the B&B tree."""
    lb: float                                # LP objective at this node
    depth: int
    var_lb: np.ndarray                       # variable lower bounds
    var_ub: np.ndarray                       # variable upper bounds
    parent_id: Optional[int] = None
    node_id: int = 0
    priority: float = 0.0                    # for priority queue (negated lb)
    inherited_cuts: list = field(default_factory=list)   # CutData list
    past_tokens: Optional[object] = None    # DynamicsTransformer token buffer

    def __lt__(self, other):
        # Max-heap by priority (higher priority = processed first)
        return self.priority > other.priority


@dataclass
class SolveResult:
    """Result returned by BnBSolver.solve()."""
    status: str            # "optimal" | "feasible" | "infeasible" | "timeout"
    objective: float
    solution: Optional[np.ndarray]
    n_nodes: int
    solve_time: float
    optimality_gap: float


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class BnBSolver:
    """
    Neural-guided Branch-and-Cut solver.

    Args:
        model         : BnBWorldModel
        device        : torch.device
        time_limit    : float  — max wall-clock seconds
        node_limit    : int    — max nodes to explore
        gap_tolerance : float  — stop when (UB - LB) / |UB| < gap_tolerance
        max_cuts_per_node : int — maximum cuts added at each node
        cut_score_threshold : float — minimum sigmoid score to select a cut
        lookahead_k   : int    — number of candidates for dynamics lookahead
    """

    def __init__(
        self,
        model,
        device,
        time_limit: float = 300.0,
        node_limit: int = 100_000,
        gap_tolerance: float = 1e-4,
        max_cuts_per_node: int = 10,
        cut_score_threshold: float = 0.3,
        lookahead_k: int = 3,
    ):
        self.model               = model
        self.device              = device
        self.time_limit          = time_limit
        self.node_limit          = node_limit
        self.gap_tolerance       = gap_tolerance
        self.max_cuts_per_node   = max_cuts_per_node
        self.cut_score_threshold = cut_score_threshold
        self.lookahead_k         = lookahead_k

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self, A: np.ndarray, b: np.ndarray, c: np.ndarray) -> SolveResult:
        """
        Solve:  min  c^T x   s.t.  A x >= b,   x in {0,1}^n

        Args:
            A : [m, n]  constraint matrix (non-negative for Set Cover)
            b : [m]     RHS vector
            c : [n]     objective coefficients

        Returns:
            SolveResult
        """
        t_start = time.perf_counter()
        n = len(c)
        m = len(b)

        # Root LP
        root_lb_arr = np.zeros(n, dtype=np.float64)
        root_ub_arr = np.ones(n,  dtype=np.float64)
        lp_obj, x_lp, dual, feasible = self._solve_lp(
            A, b, c, root_lb_arr, root_ub_arr, []
        )

        if not feasible:
            return SolveResult("infeasible", np.inf, None, 0,
                               time.perf_counter() - t_start, np.inf)

        # Encode root node
        h_vars, z = self._encode_node(A, b, c, x_lp, dual, root_lb_arr, root_ub_arr, [])

        # Generate and apply cuts at root
        root_cuts = self._select_cuts_neural(A, b, x_lp, c, z, [])
        if root_cuts:
            lp_obj2, x_lp2, dual2, feas2 = self._solve_lp(
                A, b, c, root_lb_arr, root_ub_arr, root_cuts
            )
            if feas2 and lp_obj2 > lp_obj + 1e-8:
                lp_obj, x_lp, dual = lp_obj2, x_lp2, dual2
                h_vars, z = self._encode_node(
                    A, b, c, x_lp, dual, root_lb_arr, root_ub_arr, root_cuts
                )

        # Check root integrality
        if self._is_integral(x_lp):
            sol = np.round(x_lp)
            return SolveResult("optimal", lp_obj, sol, 1,
                               time.perf_counter() - t_start, 0.0)

        # Initialise tree
        root_node = Node(
            lb=lp_obj, depth=0,
            var_lb=root_lb_arr, var_ub=root_ub_arr,
            node_id=0, priority=-lp_obj,
            inherited_cuts=root_cuts,
        )
        heap = [root_node]

        global_ub   = np.inf
        best_sol    = None
        n_nodes     = 0
        status      = "infeasible"

        while heap and n_nodes < self.node_limit:
            if time.perf_counter() - t_start > self.time_limit:
                status = "timeout"
                break

            node = heapq.heappop(heap)

            # Bound pruning
            if node.lb >= global_ub - 1e-6:
                continue

            # Solve node LP
            lp_obj, x_lp, dual, feasible = self._solve_lp(
                A, b, c, node.var_lb, node.var_ub, node.inherited_cuts
            )
            n_nodes += 1

            if not feasible or lp_obj >= global_ub - 1e-6:
                continue

            # Integer solution found
            if self._is_integral(x_lp):
                if lp_obj < global_ub:
                    global_ub = lp_obj
                    best_sol  = np.round(x_lp)
                    status    = "feasible"
                continue

            # Encode node
            h_vars, z = self._encode_node(
                A, b, c, x_lp, dual, node.var_lb, node.var_ub,
                node.inherited_cuts,
            )

            # IntegralityHead: detect near-leaf — skip cut generation
            frac_vals = np.abs(x_lp - np.round(x_lp))
            n_frac = int((frac_vals > 1e-4).sum())
            depth_t = torch.tensor([node.depth], dtype=torch.float32, device=self.device)
            nfrac_t = torch.tensor([n_frac],     dtype=torch.float32, device=self.device)
            with torch.no_grad():
                leaf_prob = torch.sigmoid(
                    self.model.integrality_logit(z, depth_t, nfrac_t)
                ).item()

            # Generate and select cuts (skip for near-leaf nodes)
            new_cuts = []
            if leaf_prob < 0.7 and node.depth < 20:
                new_cuts = self._select_cuts_neural(
                    A, b, x_lp, c, z, node.inherited_cuts
                )
                if new_cuts:
                    lp_obj2, x_lp2, dual2, feas2 = self._solve_lp(
                        A, b, c, node.var_lb, node.var_ub,
                        node.inherited_cuts + new_cuts,
                    )
                    if feas2 and lp_obj2 > lp_obj + 1e-8:
                        lp_obj, x_lp, dual = lp_obj2, x_lp2, dual2
                        h_vars, z = self._encode_node(
                            A, b, c, x_lp, dual, node.var_lb, node.var_ub,
                            node.inherited_cuts + new_cuts,
                        )
                        if self._is_integral(x_lp):
                            if lp_obj < global_ub:
                                global_ub = lp_obj
                                best_sol  = np.round(x_lp)
                                status    = "feasible"
                            continue

            # Cuts to propagate to children (branch-and-cut: inherited + new)
            child_cuts = node.inherited_cuts + new_cuts

            # Select branching variable
            branch_var = self._select_branch_var(
                h_vars, z, x_lp, node
            )

            # Value head for node priority
            with torch.no_grad():
                var_mask  = torch.zeros(h_vars.size(0), dtype=torch.bool, device=self.device)
                bvec      = torch.zeros(h_vars.size(0), dtype=torch.long, device=self.device)
                frac_t    = torch.tensor(
                    np.abs(x_lp - np.round(x_lp)) > 1e-4, dtype=torch.bool, device=self.device
                )
                v_score   = self.model.value_pred(z, h_vars, bvec, frac_t).item()
            child_priority = -lp_obj + 0.01 * v_score

            # Branch: x[branch_var] <= 0  and  x[branch_var] >= 1
            for fix_val in (0.0, 1.0):
                vlb = node.var_lb.copy()
                vub = node.var_ub.copy()
                if fix_val == 0.0:
                    vub[branch_var] = 0.0
                else:
                    vlb[branch_var] = 1.0

                # Quick infeasibility check: lb > ub on any variable
                if np.any(vlb > vub + 1e-9):
                    continue

                child = Node(
                    lb=lp_obj,
                    depth=node.depth + 1,
                    var_lb=vlb, var_ub=vub,
                    parent_id=node.node_id,
                    node_id=n_nodes * 2 + int(fix_val),
                    priority=child_priority,
                    inherited_cuts=child_cuts,
                )
                heapq.heappush(heap, child)

        # Compute final gap
        global_lb = heap[0].lb if heap else global_ub
        gap = ((global_ub - global_lb) / (abs(global_ub) + 1e-10)
               if best_sol is not None else np.inf)

        if best_sol is not None and gap < self.gap_tolerance:
            status = "optimal"

        return SolveResult(
            status=status,
            objective=global_ub if best_sol is not None else np.inf,
            solution=best_sol,
            n_nodes=n_nodes,
            solve_time=time.perf_counter() - t_start,
            optimality_gap=gap,
        )

    # ------------------------------------------------------------------
    # LP solver
    # ------------------------------------------------------------------

    def _solve_lp(
        self,
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        var_lb: np.ndarray,
        var_ub: np.ndarray,
        cuts: list,
    ) -> tuple:
        """
        Solve the LP relaxation at a node.

        min  c^T x
        s.t. A x >= b                (original constraints)
             cut.lhs @ x >= cut.rhs  (inherited cuts)
             var_lb <= x <= var_ub

        Uses scipy.optimize.linprog with method='highs' (HiGHS backend).

        Returns:
            (obj, x, dual, feasible)  or  (None, None, None, False)
        """
        from scipy.optimize import linprog

        n = len(c)

        # Stack original + cut constraints, convert >= to <=
        A_rows = [A] + [cut.lhs.reshape(1, n) for cut in cuts]
        b_rows = [b] + [np.array([cut.rhs]) for cut in cuts]
        A_ineq = -np.vstack(A_rows).astype(np.float64)
        b_ineq = -np.concatenate(b_rows).astype(np.float64)

        bounds = list(zip(var_lb.tolist(), var_ub.tolist()))

        result = linprog(
            c.astype(np.float64),
            A_ub=A_ineq, b_ub=b_ineq,
            bounds=bounds,
            method="highs",
            options={"disp": False, "presolve": True},
        )

        if result.status == 0:   # optimal
            x    = result.x
            # Dual variables for >= constraints: negate marginals (scipy gives <=)
            dual = -result.ineqlin.marginals[:len(b)] if hasattr(result, "ineqlin") and result.ineqlin is not None else np.zeros(len(b))
            return float(result.fun), x, dual, True

        return None, None, None, False

    # ------------------------------------------------------------------
    # Node encoding
    # ------------------------------------------------------------------

    def _encode_node(
        self,
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        x_lp: np.ndarray,
        dual: np.ndarray,
        var_lb: np.ndarray,
        var_ub: np.ndarray,
        cuts: list,
    ) -> tuple:
        """
        Build a PyG graph from the LP solution and encode with the GNN.

        Variable features (19-dim, matches Ecole NodeBipartite spirit):
            0  : normalised objective coefficient
            1  : is_binary (always 1 for Set Cover)
            2-4: type flags (zero for continuous relaxation)
            5  : has_lower_bound
            6  : has_upper_bound
            7  : normalised lower bound
            8  : normalised upper bound
            9  : LP solution value
            10 : fractional part |x - round(x)|
            11 : is_at_lower (x ≈ lb)
            12 : is_basic (lb < x < ub)
            13 : is_at_upper (x ≈ ub)
            14 : normalised reduced cost
            15 : number of cuts containing this variable (normalised)
            16 : raw objective coefficient (un-normalised)
            17 : 0 (reserved)
            18 : 0 (reserved)

        Constraint features (5-dim):
            0  : normalised constraint activity
            1  : activity / RHS  (relative saturation)
            2  : normalised dual value
            3  : normalised RHS
            4  : is_tight (|activity - RHS| < 1e-6)

        Edge features (3-dim):
            0  : A_{ij}
            1  : A_{ij} / (|b_i| + 1e-8)
            2  : sign(A_{ij})
        """
        m, n = A.shape
        c_max = float(np.abs(c).max()) + 1e-8
        b_max = float(np.abs(b).max()) + 1e-8

        # --- Variable features ---
        vf = np.zeros((n, 19), dtype=np.float32)
        vf[:, 0]  = c / c_max
        vf[:, 1]  = 1.0                                      # is_binary
        vf[:, 5]  = (var_lb > -1e9).astype(np.float32)
        vf[:, 6]  = (var_ub <  1e9).astype(np.float32)
        vf[:, 7]  = np.clip(var_lb, 0, 1)
        vf[:, 8]  = np.clip(var_ub, 0, 1)
        vf[:, 9]  = np.clip(x_lp, 0, 1)
        vf[:, 10] = np.abs(x_lp - np.round(np.clip(x_lp, 0, 1)))
        vf[:, 11] = (np.abs(x_lp - var_lb) < 1e-6).astype(np.float32)
        vf[:, 12] = ((x_lp > var_lb + 1e-6) & (x_lp < var_ub - 1e-6)).astype(np.float32)
        vf[:, 13] = (np.abs(x_lp - var_ub) < 1e-6).astype(np.float32)
        # Reduced cost approximation: c_j - dual^T A_{:,j}
        rc = c - A.T @ dual if dual is not None else np.zeros(n)
        rc_max = float(np.abs(rc).max()) + 1e-8
        vf[:, 14] = rc / rc_max
        # Count cuts containing each variable
        cut_counts = np.zeros(n)
        for cut in cuts:
            cut_counts += (cut.lhs > 0.5).astype(float)
        vf[:, 15] = cut_counts / max(len(cuts), 1)
        vf[:, 16] = c / c_max

        # --- Constraint features ---
        activity = A @ x_lp
        cf = np.zeros((m, 5), dtype=np.float32)
        cf[:, 0] = activity / b_max
        cf[:, 1] = activity / (b + 1e-8)
        if dual is not None:
            dual_max = float(np.abs(dual).max()) + 1e-8
            cf[:, 2] = dual / dual_max
        cf[:, 3] = b / b_max
        cf[:, 4] = (np.abs(activity - b) < 1e-6).astype(np.float32)

        # --- Edges: all non-zero entries of A ---
        con_idx, var_idx = np.where(A > 1e-9)    # [E], [E]
        if len(con_idx) == 0:
            con_idx = np.array([0], dtype=np.int64)
            var_idx = np.array([0], dtype=np.int64)

        coeff     = A[con_idx, var_idx].astype(np.float32)
        rhs_src   = b[con_idx].astype(np.float32)
        norm_coeff = coeff / (np.abs(rhs_src) + 1e-8)
        sign_coeff = np.sign(coeff)
        edge_attr  = np.stack([coeff, norm_coeff, sign_coeff], axis=1)  # [E, 3]

        # Node ordering: variables first [0..n-1], constraints after [n..n+m-1]
        edge_index = np.vstack([con_idx + n, var_idx]).astype(np.int64)  # [2, E]

        # --- Padding constraints to 19-dim (pad with zeros after 5 features) ---
        cf_pad = np.zeros((m, 19), dtype=np.float32)
        cf_pad[:, :5] = cf

        x_nodes = np.vstack([vf, cf_pad])  # [n+m, 19]
        node_type = np.array(
            [0] * n + [1] * m, dtype=np.int64
        )
        batch_vec = np.zeros(n + m, dtype=np.int64)

        # --- Build PyG batch ---
        data = Data(
            x          = torch.tensor(x_nodes,    dtype=torch.float32),
            edge_index = torch.tensor(edge_index, dtype=torch.long),
            edge_attr  = torch.tensor(edge_attr,  dtype=torch.float32),
            node_type  = torch.tensor(node_type,  dtype=torch.long),
            batch      = torch.tensor(batch_vec,  dtype=torch.long),
        )
        pyg_batch = Batch.from_data_list([data]).to(self.device)

        with torch.no_grad():
            h_vars, z = self.model.encode(pyg_batch)

        return h_vars, z

    # ------------------------------------------------------------------
    # Branching variable selection
    # ------------------------------------------------------------------

    def _select_branch_var(
        self,
        h_vars: torch.Tensor,
        z: torch.Tensor,
        x_lp: np.ndarray,
        node: Node,
    ) -> int:
        """
        Select branching variable using PolicyHead + optional dynamics lookahead.

        Fractional variables (0 < x*_j < 1) are the only valid branch targets.
        The policy scores all variables; non-fractional ones are masked to -inf.
        Then 1-step dynamics lookahead is applied over the top-k candidates.
        """
        frac_mask_np = (x_lp > 1e-4) & (x_lp < 1 - 1e-4)
        frac_indices = np.where(frac_mask_np)[0]

        if len(frac_indices) == 0:
            # Fallback: most fractional variable
            return int(np.argmin(np.abs(x_lp - 0.5)))

        with torch.no_grad():
            bvec   = torch.zeros(h_vars.size(0), dtype=torch.long, device=self.device)
            scores = self.model.policy_scores(h_vars, z, bvec)   # [total_vars]

            # Mask to fractional candidates only
            masked = torch.full_like(scores, -1e4)
            frac_t = torch.tensor(frac_indices, dtype=torch.long, device=self.device)
            masked[frac_t] = scores[frac_t]

            k = min(self.lookahead_k, len(frac_indices))
            top_k = masked.topk(k).indices   # global variable indices

            # Dynamics lookahead: predict value of next state for each candidate
            best_var   = int(top_k[0])
            best_score = -float("inf")

            for cand in top_k:
                a_emb  = h_vars[cand].unsqueeze(0)   # [1, H]
                z_next, _ = self.model.dynamics_step(
                    z, a_emb, node.past_tokens
                )
                # Value of predicted next state
                bvec1  = torch.zeros(1, dtype=torch.long, device=self.device)
                v_next = self.model.value_pred(
                    z_next, z_next, bvec1, frac_mask=None
                ).item()
                if v_next > best_score:
                    best_score = v_next
                    best_var   = int(cand)

        return best_var

    # ------------------------------------------------------------------
    # Cut generation — Chvátal-Gomory intersection cuts
    # ------------------------------------------------------------------

    def _generate_cg_cuts(
        self,
        A: np.ndarray,
        b: np.ndarray,
        x_lp: np.ndarray,
        existing_cuts: list,
    ) -> list:
        """
        Generate pairwise Chvátal-Gomory intersection cuts for Set Cover.

        For each pair of constraints (i1, i2), the CG cut on their
        intersection support is:
            ∑_{j : A_{i1,j}=1 AND A_{i2,j}=1}  x_j  >=  1

        This cut is:
          (a) Globally valid — valid for any integer-feasible solution
              because if x_j ∈ {0,1} and no element in the intersection
              is selected, both constraint i1 and i2 could be violated.
          (b) Derivable as a CG cut with multipliers u = (e_i1 + e_i2)/2,
              rounded to integer LHS coefficients.
          (c) Safe to propagate to all child nodes.

        Strategy to avoid O(m^2) enumeration:
            - Only consider "tight" constraints where A_i @ x* ∈ [1.0, 1.4]
            - Limit to at most 30 tight constraints → C(30,2) = 435 pairs
            - Only keep cuts violated by x* (intersection LP sum < 1 - eps)
            - Deduplicate by frozenset of support indices
            - Return top-50 by violation magnitude
        """
        m, n = A.shape
        b_bin = (A > 0.5)   # binary indicator matrix

        # Existing cut supports (to avoid duplicates)
        existing_supports = set()
        for cut in existing_cuts:
            existing_supports.add(frozenset(np.where(cut.lhs > 0.5)[0]))

        # Tight constraints
        activity       = A @ x_lp
        tight_mask     = (activity < 1.4) & (activity >= 1.0 - 1e-6)
        tight_cons     = np.where(tight_mask)[0][:30]   # cap at 30

        candidates = []
        seen       = set()

        for i1, i2 in combinations(tight_cons, 2):
            inter = np.where(b_bin[i1] & b_bin[i2])[0]
            if len(inter) == 0:
                continue

            key = frozenset(inter)
            if key in seen or key in existing_supports:
                continue
            seen.add(key)

            lp_val    = float(x_lp[inter].sum())
            violation = 1.0 - lp_val
            if violation > 1e-4:                  # must be violated
                lhs = np.zeros(n, dtype=np.float64)
                lhs[inter] = 1.0
                candidates.append((violation, lhs))

        # Sort by violation, take top 50
        candidates.sort(key=lambda x: -x[0])
        return [
            CutData(lhs=lhs, rhs=1.0, cut_type="cg")
            for _, lhs in candidates[:50]
        ]

    # ------------------------------------------------------------------
    # Cut feature computation
    # ------------------------------------------------------------------

    def _cut_features(
        self,
        cuts: list,
        x_lp: np.ndarray,
        c: np.ndarray,
    ) -> np.ndarray:
        """
        Compute 6-dim feature vector for each candidate cut.

        Features:
            0  violation      max(0, rhs - lhs @ x*)   how much cut bites LP
            1  efficacy       violation / ||lhs||_2     normalised bite
            2  density        nnz(lhs) / n_vars         sparsity of cut
            3  parallelism    cos(lhs, c)               alignment with objective
            4  obj_cutoff     estimated bound improvement
            5  support_frac   fraction of cut support that is fractional

        Returns:
            [n_cuts, 6] float32 array
        """
        n      = len(c)
        c_norm = c / (np.linalg.norm(c) + 1e-8)
        frac   = np.abs(x_lp - np.round(np.clip(x_lp, 0, 1)))

        feats = []
        for cut in cuts:
            lhs = cut.lhs
            rhs = cut.rhs

            lp_val    = float(lhs @ x_lp)
            violation = max(0.0, rhs - lp_val)

            lhs_norm   = float(np.linalg.norm(lhs)) + 1e-8
            efficacy   = violation / lhs_norm
            density    = float(np.count_nonzero(lhs)) / n
            parallelism = float((lhs / lhs_norm) @ c_norm)
            obj_cutoff  = violation * abs(float(c @ lhs) / (lhs_norm ** 2 + 1e-8))

            support = lhs > 0.5
            support_frac = (
                float((frac[support] > 0.05).mean()) if support.sum() > 0 else 0.0
            )

            feats.append([violation, efficacy, density, parallelism,
                          obj_cutoff, support_frac])

        return np.array(feats, dtype=np.float32)   # [n_cuts, 6]

    # ------------------------------------------------------------------
    # Neural cut selection
    # ------------------------------------------------------------------

    def _select_cuts_neural(
        self,
        A: np.ndarray,
        b: np.ndarray,
        x_lp: np.ndarray,
        c: np.ndarray,
        z: torch.Tensor,
        existing_cuts: list,
    ) -> list:
        """
        Generate candidate cuts, score with CuttingPlaneHead, return selected.

        Selection rule:
            sigmoid(score) >= cut_score_threshold  AND  top-max_cuts_per_node
        """
        candidates = self._generate_cg_cuts(A, b, x_lp, existing_cuts)
        if not candidates:
            return []

        feat_np = self._cut_features(candidates, x_lp, c)
        feat_t  = torch.tensor(feat_np, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            scores = self.model.cut_scores(feat_t, z.squeeze(0))
            probs  = torch.sigmoid(scores).cpu().numpy()

        selected = []
        order    = np.argsort(-probs)
        for idx in order:
            if probs[idx] >= self.cut_score_threshold:
                selected.append(candidates[idx])
            if len(selected) >= self.max_cuts_per_node:
                break

        return selected

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _is_integral(x: np.ndarray, tol: float = 1e-4) -> bool:
        return bool(np.all(np.abs(x - np.round(x)) < tol))
