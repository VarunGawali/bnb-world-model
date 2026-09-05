"""
bnb_solver.py — Neural-guided Branch-and-Cut solver.

Implements a complete Python-owned B&B loop with:
    - LP relaxation at each node via scipy.optimize.linprog (HiGHS backend)
    - Branch-and-cut with globally valid Gomory cuts (from bnb_wm/solver/gomory.py,
      the same generator used at data collection and deploy)
    - BnBWorldModel driving all search decisions:
        * BipartiteGNN (GATv2, bidirectional passes) -> node encoder
        * CuttingPlaneHead  -> which cuts to add (branch-and-cut selection)
        * PolicyHead        -> branching variable selection
        * ValueHead / CostToGoHead -> node priority (best-first search)
        * IntegralityHead   -> prune near-leaf nodes early
        * DynamicsTransformer -> direction-conditioned latent lookahead

Branch-and-cut design:
    Cuts selected at a node are GLOBALLY VALID Gomory cuts (same generator as
    collect/train/deploy). They are propagated to ALL descendant nodes via the
    Node.inherited_cuts list — this is full branch-and-cut, not cut-and-branch.
    No cut validity re-check is needed because the cuts are valid for any
    integer-feasible solution regardless of branching.

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
    # HiGHS basis from parent LP for warmstarting (col_status, row_status arrays)
    warm_basis: Optional[tuple] = None

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
        lookahead_depth: int = 3,
        lookahead_gamma: float = 0.95,
        size_weight: float = 1.0,
        ctg_weight: float = 0.0,
        branch_factor: int = 1,
        node_selection: str = "bound",
        use_reward_return: bool = False,
        uncertainty_weight: float = 0.0,
        cut_mode: str = "learned",
        cut_depth_max: int = 3,
        force_root_cuts: bool = True,           # always attempt cuts at depth=0
        cut_entropy_thresh_root: float = 0.2,   # depth=0 fallback (only when force_root_cuts=False)
        cut_ctg_thresh_root: float = 10.0,
        cut_entropy_thresh: float = 0.5,        # depth 1+: require more uncertainty
        cut_ctg_thresh: float = 30.0,           # depth 1+: require larger subtree
        cut_budget_cap: int = 10,
    ):
        self.model               = model
        self.device              = device
        self.time_limit          = time_limit
        self.node_limit          = node_limit
        self.gap_tolerance       = gap_tolerance
        self.max_cuts_per_node   = max_cuts_per_node
        self.cut_score_threshold = cut_score_threshold
        self.lookahead_k         = lookahead_k
        self.lookahead_depth     = lookahead_depth   # steps of latent rollout
        self.lookahead_gamma     = lookahead_gamma   # discount per step
        self.size_weight         = size_weight       # predicted-subtree-size penalty
        self.ctg_weight          = ctg_weight         # cost-to-go penalty (Gap 3)
        self.branch_factor       = branch_factor      # rollout tree width (Gap 4)
        self.node_selection      = node_selection     # "bound" | "cost_to_go" (Gap 5)
        self.use_reward_return   = use_reward_return   # MuZero-style return (Fix 3)
        self.uncertainty_weight  = uncertainty_weight  # direction-spread penalty (0=off)
        # Cut-selection mode for the ablation: "learned" (CuttingPlaneHead),
        # "heuristic" (max-violation over the valid Gomory pool), "none".
        # P0.8: `learned` requires a Phase-5-trained cut head. Running the
        # untrained head produces arbitrary selections that silently corrupt
        # results, so fall back to the verified heuristic and warn instead.
        if cut_mode == "learned" and not bool(
                getattr(model, "cut_head_trained", torch.tensor(False))):
            import warnings
            warnings.warn(
                "cut_mode='learned' but the model's cut head is untrained "
                "(no Phase-5 checkpoint); falling back to cut_mode='heuristic'. "
                "Load a Phase-5 checkpoint to use learned cut selection.",
                RuntimeWarning, stacklevel=2)
            cut_mode = "heuristic"
        self.cut_mode                = cut_mode
        self.cut_depth_max           = cut_depth_max
        self.force_root_cuts         = force_root_cuts
        self.cut_entropy_thresh_root = cut_entropy_thresh_root
        self.cut_ctg_thresh_root     = cut_ctg_thresh_root
        self.cut_entropy_thresh      = cut_entropy_thresh
        self.cut_ctg_thresh          = cut_ctg_thresh
        self.cut_budget_cap          = cut_budget_cap
        self._cuts_added         = 0                   # per-solve cut counter
        # Branching mode for the ablation: "rollout" (latent world-model
        # lookahead), "policy" (argmax policy, no rollout), "most_fractional".
        self.branch_mode         = "rollout"

        # Main LP solves use the proven scipy path. highspy (if present) is used
        # ONLY for basis extraction in Gomory cut generation, via the stable
        # passModel API — the incremental highspy build API (addVars/
        # changeColsCostByRange) varies across versions and is avoided here.
        try:
            import highspy
            self._highs = highspy
        except ImportError:
            self._highs = None
        self._use_highs_direct = False

        # P2.1: cut generation needs HiGHS (for the LP basis). If cuts are
        # requested but HiGHS is missing, fail loud rather than silently solving
        # with zero cuts — a "cut" experiment that never cuts is a silent lie.
        if self.cut_mode != "none" and self._highs is None:
            raise RuntimeError(
                "cut_mode=%r requires highspy (basis access for Gomory cuts), "
                "but it is not importable. Install highspy, or set cut_mode='none' "
                "to run branching-only." % self.cut_mode)

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

        # Reset the per-instance Gomory cut pool: cuts are derived from THIS
        # instance's (A, b, c) and are invalid for any other instance, so the
        # cache must not leak across solve() calls (the benchmark reuses one
        # solver object over many instances).
        self._gomory_pool = None
        self._cuts_added = 0

        # Root LP
        root_lb_arr = np.zeros(n, dtype=np.float64)
        root_ub_arr = np.ones(n,  dtype=np.float64)
        lp_obj, x_lp, dual, feasible, root_basis = self._solve_lp(
            A, b, c, root_lb_arr, root_ub_arr, []
        )

        if not feasible:
            return SolveResult("infeasible", np.inf, None, 0,
                               time.perf_counter() - t_start, np.inf)

        # Encode root node
        h_vars, z, h_cons = self._encode_node(
            A, b, c, x_lp, dual, root_lb_arr, root_ub_arr, []
        )

        # Generate and apply cuts at root (legacy path — uses _select_cuts_neural)
        root_cuts = self._select_cuts_neural(A, b, x_lp, c, z, [])
        if root_cuts:
            lp_obj2, x_lp2, dual2, feas2, root_basis2 = self._solve_lp(
                A, b, c, root_lb_arr, root_ub_arr, root_cuts, warm_basis=root_basis
            )
            if feas2 and lp_obj2 > lp_obj + 1e-8:
                lp_obj, x_lp, dual, root_basis = lp_obj2, x_lp2, dual2, root_basis2
                h_vars, z, h_cons = self._encode_node(
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
            warm_basis=root_basis,
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

            # Solve node LP (warmstart from parent basis)
            lp_obj, x_lp, dual, feasible, node_basis = self._solve_lp(
                A, b, c, node.var_lb, node.var_ub, node.inherited_cuts,
                warm_basis=node.warm_basis,
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
            h_vars, z, h_cons = self._encode_node(
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

            # Neural cut gate: H(π) + CTG + depth (replaces leaf_prob heuristic).
            # On gate fire: generate GMI candidates, score with zero-shot GNN
            # cross-attention, apply best subset, warmstart LP re-solve, then
            # patch policy scores with Δ-fractionality / Δ-reduced-cost adjustment
            # (no full GNN re-encode — h_vars structural info stays valid).
            new_cuts = []
            scores_adj = None
            if self.cut_mode != "none" and leaf_prob < 0.7:
                fire, scores_precut = self._cut_gate(h_vars, z, x_lp, node)
                if fire:
                    candidates = self._generate_cg_cuts(
                        A, b, c, x_lp, node.inherited_cuts
                    )
                    if candidates:
                        cut_scores = self._score_cuts_gnn(
                            candidates, h_vars, h_cons, x_lp, A, b
                        )
                        order = np.argsort(-cut_scores)
                        new_cuts = [candidates[i]
                                    for i in order[:self.max_cuts_per_node]]

                    if new_cuts:
                        lp_obj2, x_lp2, dual2, feas2, node_basis2 = self._solve_lp(
                            A, b, c, node.var_lb, node.var_ub,
                            node.inherited_cuts + new_cuts,
                            warm_basis=node_basis,
                        )
                        if feas2 and lp_obj2 > lp_obj + 1e-8:
                            # Δ-score adjustment — patch policy scores for the
                            # changed LP state without a full GNN re-encode.
                            frac_old   = np.abs(x_lp  - np.round(x_lp))
                            frac_new   = np.abs(x_lp2 - np.round(x_lp2))
                            delta_frac = torch.tensor(
                                frac_new - frac_old, dtype=torch.float32,
                                device=self.device
                            )
                            d_old = dual  if dual  is not None else np.zeros(len(b))
                            d_new = dual2 if dual2 is not None else np.zeros(len(b))
                            rc_old = c - A.T @ d_old
                            rc_new = c - A.T @ d_new
                            rc_scale = float(np.abs(rc_new).max()) + 1e-8
                            delta_rc = torch.tensor(
                                (rc_new - rc_old) / rc_scale, dtype=torch.float32,
                                device=self.device
                            )
                            # More fractional → raise branching priority;
                            # higher reduced cost → lower priority.
                            scores_adj = scores_precut + 1.0 * delta_frac \
                                         - 0.5 * delta_rc

                            lp_obj, x_lp, dual, node_basis = (
                                lp_obj2, x_lp2, dual2, node_basis2
                            )
                            self._cuts_added += len(new_cuts)

                            if self._is_integral(x_lp):
                                if lp_obj < global_ub:
                                    global_ub = lp_obj
                                    best_sol  = np.round(x_lp)
                                    status    = "feasible"
                                continue
                        else:
                            new_cuts = []  # LP did not improve; discard cuts

            # Cuts to propagate to children (branch-and-cut: inherited + new)
            child_cuts = node.inherited_cuts + new_cuts

            # Select branching variable (multi-step lookahead).
            # Pass Δ-adjusted scores when cuts were applied, so branching sees
            # the updated LP state without a full GNN re-encode.
            branch_var = self._select_branch_var(
                h_vars, z, x_lp, node, policy_scores_override=scores_adj
            )

            # Shared context for child scoring (same for both children).
            bvec   = torch.zeros(h_vars.size(0), dtype=torch.long, device=self.device)
            frac_t = torch.tensor(
                np.abs(x_lp - np.round(x_lp)) > 1e-4, dtype=torch.bool, device=self.device
            )

            # Branch: x[branch_var] <= 0  (down)  and  x[branch_var] >= 1  (up).
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

                # P1.6: score EACH child separately (the old code shared one
                # priority for both). The down branch (x<=0) and up branch (x>=1)
                # are distinct states, so we roll the dynamics forward in the
                # matching direction (+1 up / -1 down) — the direction-conditioned
                # transition the model was trained on (P0.12).
                direction = 1.0 if fix_val == 1.0 else -1.0
                with torch.no_grad():
                    # One direction-conditioned dynamics step from parent->child.
                    # Its updated token buffer is written onto the child (P0.9) so
                    # that when the child later branches, its latent lookahead has
                    # the true history that led to it — not an empty buffer.
                    a_emb = h_vars[branch_var].unsqueeze(0)
                    z_child, h_child, child_tokens = self.model.dynamics_step_full(
                        z, a_emb, h_vars, node.past_tokens, direction
                    )
                    # P1.6/P1.1: both selection modes score the PREDICTED child
                    # state (z_child/h_child), which differs per child via the
                    # branch direction — not the shared parent state. The child's
                    # fractional set is unknown (imagined), so frac_mask=None
                    # rather than the stale parent mask.
                    if self.node_selection == "cost_to_go":
                        # Gap 5: learned best-first search. Order the frontier by
                        # this child's predicted cost-to-go. Node order never
                        # affects correctness, only efficiency, so exactness holds.
                        ctg = self.model.cost_to_go_pred(
                            z_child, h_child, bvec, None
                        ).item()
                        # Node.__lt__ is a MAX-heap on priority (higher popped
                        # first), so negate: the SMALLEST predicted remaining
                        # work gets the highest priority and is explored first.
                        child_priority = -ctg
                    else:
                        # Best-bound (default): both children inherit this node's
                        # LP bound, so the value head on the predicted child state
                        # is what distinguishes the up vs down child here.
                        v_score = self.model.value_pred(
                            z_child, h_child, bvec, None).item()
                        child_priority = -lp_obj + 0.01 * v_score

                child = Node(
                    lb=lp_obj,
                    depth=node.depth + 1,
                    var_lb=vlb, var_ub=vub,
                    parent_id=node.node_id,
                    node_id=n_nodes * 2 + int(fix_val),
                    priority=child_priority,
                    inherited_cuts=child_cuts,
                    warm_basis=node_basis,   # child warmstarts from current node's basis
                    past_tokens=child_tokens,   # P0.9: propagate updated history
                )
                heapq.heappush(heap, child)

        # Compute final gap. The global lower bound is the MINIMUM lb over all
        # open nodes, not heap[0] (which under learned node selection is the
        # max-priority node, not the min-bound one). Using heap[0] would
        # misreport the gap and could wrongly declare optimality.
        global_lb = min((nd.lb for nd in heap), default=global_ub)
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
        warm_basis: Optional[tuple] = None,
    ) -> tuple:
        """
        Solve the LP relaxation at a node.

        min  c^T x
        s.t. A x >= b                (original constraints)
             cut.lhs @ x >= cut.rhs  (inherited cuts)
             var_lb <= x <= var_ub

        Warmstarting: if highspy is available and warm_basis is provided
        (col_status, row_status arrays from the parent node), the dual
        simplex is hot-started from that basis, typically reducing the
        solve to O(1-10) pivots instead of a full re-solve.

        Falls back to scipy linprog (cold start) if highspy is absent.

        Returns:
            (obj, x, dual, feasible, basis)
            basis is (col_status, row_status) or None if highspy unavailable.
        """
        n = len(c)
        m_orig = len(b)

        # Build stacked constraint matrix (original >= cuts)
        A_rows = [A] + [cut.lhs.reshape(1, n) for cut in cuts]
        b_rows = [b] + [np.array([cut.rhs]) for cut in cuts]
        A_all  = np.vstack(A_rows).astype(np.float64)
        b_all  = np.concatenate(b_rows).astype(np.float64)
        m_all  = len(b_all)

        if self._use_highs_direct:
            return self._solve_lp_highs(
                c, A_all, b_all, var_lb, var_ub, m_orig, warm_basis
            )
        else:
            return self._solve_lp_scipy(
                c, A_all, b_all, var_lb, var_ub, m_orig
            )

    def _solve_lp_highs(
        self,
        c: np.ndarray,
        A_all: np.ndarray,
        b_all: np.ndarray,
        var_lb: np.ndarray,
        var_ub: np.ndarray,
        m_orig: int,
        warm_basis: Optional[tuple],
    ) -> tuple:
        """HiGHS direct API solve with optional basis warmstart."""
        h = self._highs.Highs()
        h.silent()

        n    = len(c)
        m_all = len(b_all)

        # Add variables
        h.addVars(n, var_lb.tolist(), var_ub.tolist())
        h.changeColsCostByRange(0, n - 1, c.tolist())

        # Add constraints: A_all x >= b_all  →  b_all <= A_all x <= +inf
        inf = self._highs.kHighsInf
        for i in range(m_all):
            row = A_all[i]
            nz_idx = np.where(np.abs(row) > 1e-12)[0]
            h.addRow(
                float(b_all[i]), inf,
                len(nz_idx),
                nz_idx.tolist(),
                row[nz_idx].tolist(),
            )

        # Warmstart: inject parent basis (dual simplex will repair it)
        if warm_basis is not None:
            col_status, row_status = warm_basis
            # Extend basis if cuts were added (new rows default to BASIC)
            n_new_rows = m_all - len(row_status)
            if n_new_rows > 0:
                # kBasic = 1 in HiGHS basis enum
                row_status = list(row_status) + [1] * n_new_rows
            try:
                h.setBasis(list(col_status), list(row_status))
            except Exception:
                pass   # ignore invalid basis; cold start

        h.run()

        info  = h.getInfoValue("primal_solution_status")[1]
        if info != self._highs.kSolutionStatusFeasible:
            return None, None, None, False, None

        sol   = h.getSolution()
        x     = np.array(sol.col_value[:n])
        dual  = np.array(sol.row_dual[:m_orig])

        # Extract basis for children to warmstart from
        basis_obj  = h.getBasis()
        col_status = list(basis_obj.col_status)
        row_status = list(basis_obj.row_status)

        obj = float(h.getInfoValue("objective_function_value")[1])
        return obj, x, dual, True, (col_status, row_status)

    def _solve_lp_scipy(
        self,
        c: np.ndarray,
        A_all: np.ndarray,
        b_all: np.ndarray,
        var_lb: np.ndarray,
        var_ub: np.ndarray,
        m_orig: int,
    ) -> tuple:
        """scipy linprog cold-start fallback (no warmstarting)."""
        from scipy.optimize import linprog

        # Convert >= to <=
        A_ineq = -A_all
        b_ineq = -b_all
        bounds  = list(zip(var_lb.tolist(), var_ub.tolist()))

        result = linprog(
            c.astype(np.float64),
            A_ub=A_ineq, b_ub=b_ineq,
            bounds=bounds,
            method="highs",
            options={"disp": False, "presolve": True},
        )

        if result.status == 0:
            x    = result.x
            dual = (
                -result.ineqlin.marginals[:m_orig]
                if hasattr(result, "ineqlin") and result.ineqlin is not None
                else np.zeros(m_orig)
            )
            return float(result.fun), x, dual, True, None

        return None, None, None, False, None

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

        Variable features (19-dim) — exact Ecole NodeBipartite column layout
        so inference features match the training distribution:
            see inline comments in the implementation block below

        Constraint features (5-dim) — matches Ecole NodeBipartite row layout:
            0  : obj_cosine_similarity (zero; not computable without SCIP context)
            1  : bias / normalised RHS
            2  : is_tight (|activity - RHS| < 1e-6)
            3  : dual_solution_value (normalised)
            4  : age (zero placeholder)

        Variable features (19-dim) — exact Ecole NodeBipartite column layout:
            0  : obj_coef (normalised)
            1  : is_type_binary  (always 1 for Set Cover)
            2  : is_type_integer (0)
            3  : is_type_implicit_integer (0)
            4  : is_type_continuous (0)
            5  : has_lower_bound
            6  : has_upper_bound
            7  : lower_bound (normalised)
            8  : upper_bound (normalised)
            9  : basis_lower  (x ≈ lb)
            10 : basis_basic  (lb < x < ub)
            11 : basis_upper  (x ≈ ub)
            12 : basis_zero_free (0)
            13 : sol_val   — LP solution value  ← key: same index as Ecole
            14 : sol_frac  — |x - round(x)|    ← key: same index as Ecole
            15 : sol_at_lb
            16 : sol_at_ub
            17 : reduced_cost (normalised)
            18 : age proxy (log1p of number of cuts containing this variable)

        Edge features (3-dim):
            0  : A_{ij}
            1  : A_{ij} / (|b_i| + 1e-8)
            2  : sign(A_{ij})
        """
        m, n = A.shape
        c_max = float(np.abs(c).max()) + 1e-8
        b_max = float(np.abs(b).max()) + 1e-8

        # --- Variable features (Ecole-aligned) ---
        at_lb = np.abs(x_lp - var_lb) < 1e-6
        at_ub = np.abs(x_lp - var_ub) < 1e-6
        is_basic = (~at_lb) & (~at_ub)

        rc = (c - A.T @ dual) if dual is not None else np.zeros(n)
        rc_max = float(np.abs(rc).max()) + 1e-8

        cut_counts = np.zeros(n, dtype=np.float32)
        for cut in cuts:
            cut_counts += (cut.lhs > 0.5).astype(np.float32)

        vf = np.zeros((n, 19), dtype=np.float32)
        vf[:, 0]  = c / c_max                                        # obj_coef
        vf[:, 1]  = 1.0                                               # is_type_binary
        # 2,3,4: integer type flags — 0 (binary is handled via bounds)
        vf[:, 5]  = (var_lb > -1e9).astype(np.float32)               # has_lower_bound
        vf[:, 6]  = (var_ub <  1e9).astype(np.float32)               # has_upper_bound
        vf[:, 7]  = np.clip(var_lb, 0, 1)                            # lower_bound
        vf[:, 8]  = np.clip(var_ub, 0, 1)                            # upper_bound
        vf[:, 9]  = at_lb.astype(np.float32)                          # basis_lower
        vf[:, 10] = is_basic.astype(np.float32)                       # basis_basic
        vf[:, 11] = at_ub.astype(np.float32)                          # basis_upper
        # 12: basis_zero_free — 0
        vf[:, 13] = np.clip(x_lp, 0.0, 1.0)                          # sol_val
        vf[:, 14] = np.abs(x_lp - np.round(np.clip(x_lp, 0.0, 1.0))) # sol_frac
        vf[:, 15] = at_lb.astype(np.float32)                          # sol_at_lb
        vf[:, 16] = at_ub.astype(np.float32)                          # sol_at_ub
        vf[:, 17] = rc / rc_max                                        # reduced_cost
        vf[:, 18] = np.log1p(cut_counts)                              # age proxy

        # --- Constraint features (Ecole-aligned, 5-dim) ---
        activity = A @ x_lp
        cf = np.zeros((m, 5), dtype=np.float32)
        # cf[:, 0] = obj_cosine_similarity — skip (0)
        cf[:, 1] = b / b_max                                          # bias / normalised RHS
        cf[:, 2] = (np.abs(activity - b) < 1e-6).astype(np.float32)  # is_tight
        if dual is not None:
            dual_max = float(np.abs(dual).max()) + 1e-8
            cf[:, 3] = dual / dual_max                                 # dual_solution_value
        # cf[:, 4] = age — 0

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
        # P0.2: bidirectional edges (must match build_pyg_data / _format_obs).
        # Without the variable->constraint direction the encoder's v2c pass sees
        # no edges. Duplicate edge_attr for the reverse edges.
        con_to_var = np.vstack([con_idx + n, var_idx]).astype(np.int64)   # [2, E]
        var_to_con = np.vstack([var_idx, con_idx + n]).astype(np.int64)   # [2, E]
        edge_index = np.hstack([con_to_var, var_to_con])                 # [2, 2E]
        edge_attr  = np.concatenate([edge_attr, edge_attr], axis=0)      # [2E, 3]

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
            h_vars, z, h_cons = self.model.encode_with_cons(pyg_batch)

        return h_vars, z, h_cons

    # ------------------------------------------------------------------
    # Branching variable selection
    # ------------------------------------------------------------------

    def _select_branch_var(
        self,
        h_vars: torch.Tensor,
        z: torch.Tensor,
        x_lp: np.ndarray,
        node: Node,
        policy_scores_override: "torch.Tensor | None" = None,
    ) -> int:
        """
        Select branching variable using PolicyHead + a real multi-step
        latent rollout (learned world-model lookahead).

        For each of the top-k candidates by policy score, the model rolls the
        learned dynamics forward `lookahead_depth` steps in latent space (no
        LP solves). At every rollout step it predicts both the next graph
        latent z_{t+1} AND the next per-variable embeddings h_vars_{t+1}, then
        re-runs the policy on the predicted state to choose the *next*
        branching action — so the rollout simulates a genuine branching
        sequence rather than replaying the same variable. The value head
        scores each predicted future state; discounted returns are compared
        and the best candidate is branched on.
        """
        frac_mask_np = (x_lp > 1e-4) & (x_lp < 1 - 1e-4)
        frac_indices = np.where(frac_mask_np)[0]

        if len(frac_indices) == 0:
            return int(np.argmin(np.abs(x_lp - 0.5)))

        # Classical baseline: branch on the most-fractional variable (no model).
        if self.branch_mode == "most_fractional":
            fr = np.abs(x_lp - np.round(x_lp))
            return int(frac_indices[np.argmax(fr[frac_indices])])

        with torch.no_grad():
            bvec = torch.zeros(h_vars.size(0), dtype=torch.long, device=self.device)
            # Use Δ-adjusted scores when provided (post-cut state without re-encode),
            # otherwise compute fresh from the policy head.
            if policy_scores_override is not None:
                scores = policy_scores_override
            else:
                scores = self.model.policy_scores(h_vars, z, bvec)

            # Policy-only: take the top policy score, skip the latent rollout.
            if self.branch_mode == "policy":
                masked0 = torch.full_like(scores, -1e4)
                masked0[torch.tensor(frac_indices, dtype=torch.long,
                                     device=self.device)] = \
                    scores[torch.tensor(frac_indices, dtype=torch.long,
                                        device=self.device)]
                return int(masked0.argmax())

            frac_t = torch.tensor(frac_indices, dtype=torch.long, device=self.device)
            valid_mask = torch.zeros(
                h_vars.size(0), dtype=torch.bool, device=self.device
            )
            valid_mask[frac_t] = True

            masked = torch.full_like(scores, -1e4)
            masked[frac_t] = scores[frac_t]

            k     = min(self.lookahead_k, len(frac_indices))
            top_k = masked.topk(k).indices   # [k] LongTensor

            # Evaluate all k candidates in one batched rollout pass.
            # rollout_top_k_batched amortises the shared dynamics prefix
            # across all candidates — k× cheaper than a sequential loop
            # for branch_factor > 1 or depth > 1.
            cand_scores = self.model.rollout_top_k_batched(
                z, h_vars, top_k,
                depth=self.lookahead_depth,
                gamma=self.lookahead_gamma,
                valid_mask=valid_mask,
                past_tokens=node.past_tokens,
                size_weight=self.size_weight,
                ctg_weight=self.ctg_weight,
                branch_factor=self.branch_factor,
                use_reward_return=self.use_reward_return,
                uncertainty_weight=self.uncertainty_weight,
            )   # [k] FloatTensor

            best_var = int(top_k[cand_scores.argmax()].item())

        return best_var

    # ------------------------------------------------------------------
    # Cut generation — valid Gomory fractional cuts (solver/gomory.py)
    # ------------------------------------------------------------------

    def _generate_cg_cuts(
        self,
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        x_lp: np.ndarray,
        existing_cuts: list,
    ) -> list:
        """
        Generate globally valid Gomory fractional cuts (see solver/gomory.py).

        The previous pairwise "intersection >= 1" cut was mathematically INVALID
        for a covering system — it rounded a >= constraint's coefficients down,
        which can remove feasible integer points, and made the solver return
        wrong optima. Gomory fractional cuts are valid for every integer-feasible
        point by construction.

        The cut pool is derived ONCE at the root (globally valid, so it can be
        propagated to any node) and cached. At a given node we return the cached
        cuts that are (i) not already present and (ii) violated by the current
        LP point x_lp, so only useful cuts are offered to the selection head.
        If highspy is unavailable or nothing valid is derivable, the pool is
        empty and no cuts are added — which keeps the solver correct.
        """
        n = A.shape[1]

        # Build the global Gomory cut pool once (root), then reuse everywhere.
        if getattr(self, "_gomory_pool", None) is None:
            from .gomory import generate_root_gomory_cuts
            pool = generate_root_gomory_cuts(A, b, c, self._highs)
            self._gomory_pool = [
                CutData(lhs=np.asarray(lhs, dtype=np.float64),
                        rhs=float(rhs), cut_type="gomory")
                for lhs, rhs in pool
            ]

        # Existing cut fingerprints (avoid re-adding an inherited cut).
        existing = set()
        for cut in existing_cuts:
            existing.add((tuple(np.round(cut.lhs, 6)), round(cut.rhs, 6)))

        out = []
        for cut in self._gomory_pool:
            key = (tuple(np.round(cut.lhs, 6)), round(cut.rhs, 6))
            if key in existing:
                continue
            # Keep only cuts violated by the current LP point: lhs @ x < rhs.
            if float(cut.lhs @ x_lp) < cut.rhs - 1e-6:
                out.append(cut)
        return out[:50]

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

            # P0.6: support = NONZERO coefficients (must match the collector's
            # `abs(lhs) > 1e-9`). The old `lhs > 0.5` dropped negative and small
            # coefficients, so the deployed support_frac feature came from a
            # different distribution than the one the cut head was trained on.
            support = np.abs(lhs) > 1e-9
            support_frac = (
                float((frac[support] > 0.05).mean()) if support.sum() > 0 else 0.0
            )

            feats.append([violation, efficacy, density, parallelism,
                          obj_cutoff, support_frac])

        return np.array(feats, dtype=np.float32)   # [n_cuts, 6]

    # ------------------------------------------------------------------
    # Neural cut gate and zero-shot GNN scoring
    # ------------------------------------------------------------------

    def _cut_gate(
        self,
        h_vars: torch.Tensor,
        z: torch.Tensor,
        x_lp: np.ndarray,
        node: Node,
    ) -> "tuple[bool, torch.Tensor | None]":
        """
        Three-part cut gate: depth cap, policy entropy, and CTG.

        Returns (fire, policy_scores):
          fire         : bool — whether to attempt cut generation
          policy_scores: the pre-cut policy logits (reused for Δ-adjustment)

        Entropy measures genuine branching uncertainty over the fractional
        variables; CTG ensures the remaining tree is large enough that saving
        nodes is worth the LP re-solve overhead.
        """
        if node.depth > self.cut_depth_max:
            return False, None
        if self._cuts_added >= self.cut_budget_cap:
            return False, None

        frac_mask_np = (x_lp > 1e-4) & (x_lp < 1 - 1e-4)
        if not frac_mask_np.any():
            return False, None

        # Root node: skip entropy gate (noisy at root) but keep CTG guard so we
        # don't waste an LP re-solve on instances the model predicts as trivial
        # (e.g. already-integral or single-node solves).
        if node.depth == 0 and self.force_root_cuts:
            with torch.no_grad():
                bvec   = torch.zeros(h_vars.size(0), dtype=torch.long, device=self.device)
                scores = self.model.policy_scores(h_vars, z, bvec)
                ctg    = self.model.cost_to_go_pred(z, h_vars, bvec, frac_mask=None).item()
            if ctg < self.cut_ctg_thresh_root:
                return False, scores
            return True, scores

        with torch.no_grad():
            bvec   = torch.zeros(h_vars.size(0), dtype=torch.long, device=self.device)
            scores = self.model.policy_scores(h_vars, z, bvec)

            frac_t  = torch.tensor(frac_mask_np, dtype=torch.bool, device=self.device)
            probs   = torch.softmax(scores[frac_t], dim=0)
            entropy = float(-(probs * torch.log(probs + 1e-12)).sum())

            eth = self.cut_entropy_thresh_root if node.depth == 0 else self.cut_entropy_thresh
            if entropy < eth:
                return False, scores

            cth = self.cut_ctg_thresh_root if node.depth == 0 else self.cut_ctg_thresh
            ctg = self.model.cost_to_go_pred(z, h_vars, bvec, frac_mask=None).item()
            if ctg < cth:
                return False, scores

        return True, scores

    def _score_cuts_gnn(
        self,
        candidates: list,
        h_vars: torch.Tensor,
        h_cons: torch.Tensor,
        x_lp: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        """
        Zero-shot GNN cut scoring — no new parameters, no retraining.

        For each candidate cut with coefficient vector lhs:
          h_cut  = normalize( lhs @ h_vars )          # coeff-weighted var embedding [H]
          attn   = softmax( h_cut @ h_cons.T / √H )   # cross-attention over constraints
          loose  = max(0, b - A @ x_lp)               # constraint looseness [m]
          score  = attn · loose + violation            # geometry + LP bite

        Cross-attention over h_cons is the proper bipartite analog — h_cut queries
        which constraints are most geometrically related to the cut, weighted by how
        loose those constraints currently are. Combined with LP violation this scores
        cuts by both structural relevance and immediate LP improvement.
        """
        if not candidates:
            return np.array([], dtype=np.float32)

        H   = h_vars.size(1)
        loose = np.maximum(0.0, b - A @ x_lp).astype(np.float32)  # [m]
        loose_t = torch.tensor(loose, dtype=torch.float32, device=self.device)

        scores = []
        with torch.no_grad():
            for cut in candidates:
                lhs_t = torch.tensor(cut.lhs, dtype=torch.float32, device=self.device)
                h_cut = F.normalize((lhs_t @ h_vars).unsqueeze(0), dim=1).squeeze(0)
                # Cross-attention: h_cut queries h_cons [m, H]
                attn  = torch.softmax(h_cons @ h_cut / (H ** 0.5), dim=0)  # [m]
                geom  = float((attn * loose_t).sum())
                viol  = float(max(0.0, cut.rhs - float(cut.lhs @ x_lp)))
                scores.append(geom + viol)

        return np.array(scores, dtype=np.float32)

    # ------------------------------------------------------------------
    # Neural cut selection (legacy CuttingPlaneHead path — kept for ablation)
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
        if self.cut_mode == "none":
            return []

        candidates = self._generate_cg_cuts(A, b, c, x_lp, existing_cuts)
        if not candidates:
            return []

        if self.cut_mode == "heuristic":
            # Max-violation selection over the same valid Gomory pool (classical
            # baseline): rank by how much each cut is violated by x_lp.
            viol = np.array([cut.rhs - float(cut.lhs @ x_lp) for cut in candidates])
            order = np.argsort(-viol)
            selected = [candidates[i] for i in order[: self.max_cuts_per_node]]
            self._cuts_added += len(selected)
            return selected

        # learned: score the valid candidates with the CuttingPlaneHead.
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

        self._cuts_added += len(selected)
        return selected

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _is_integral(x: np.ndarray, tol: float = 1e-4) -> bool:
        return bool(np.all(np.abs(x - np.round(x)) < tol))
