"""
benchmark_solvers.py — Full ablation benchmark.

Runs every method in the ablation ladder on collector-distribution instances
and emits one JSON with all metrics.  Two budgets per method run:
  - wall: equal wall-clock (--time_limit, default 60s)
  - nodes: equal node budget (--node_limit, default 2000)

Usage
-----
    # Dev run (seed 1, n=20, tune thresholds here)
    PYTHONPATH=. python tools/benchmark_solvers.py \
        --checkpoint checkpoints/phase4_best.pt \
        --n_rows 100 --n_cols 200 --n_instances 20 --seed 1 \
        --optima results/optima_100x200.json \
        --out results/benchmark_dev.json

    # Final run (seeds 0,2,3 — never tuned on these)
    PYTHONPATH=. python tools/benchmark_solvers.py \
        --checkpoint checkpoints/phase4_best.pt \
        --n_rows 100 --n_cols 200 --n_instances 50 --seed 0 \
        --optima results/optima_100x200_final.json \
        --out results/benchmark_seed0.json

    # Specific methods only
    PYTHONPATH=. python tools/benchmark_solvers.py ... \
        --methods mf classical_sb4 rollout neural_full

    # Skip SCIP group (no pyscipopt needed)
    PYTHONPATH=. python tools/benchmark_solvers.py ... --no_scip

Method matrix
-------------
Group A (our harness — wall-clock and nodes both comparable):
  mf                    most-fractional branching
  classical_sb1         reliability branching, eta=1
  classical_sb4         reliability branching, eta=4  ← primary baseline
  classical_sb8         reliability branching, eta=8
  policy                GNN policy argmax (no rollout)
  mf_blend_20           rank-blend α=0.20  ← cheap headline
  mf_blend_20_cuts_attn mf_blend_20 + attention-scored cuts
  rollout_d1            rollout depth=1, size_weight=0, ctg_weight=0
  rollout_d1_size1      rollout depth=1, size_weight=1.0, ctg_weight=0
  rollout_d2_size1      rollout depth=2, size_weight=1.0, ctg_weight=0  ← ablation axis
  blend_rollout_d1_size1_cuts_attn   mf_blend_20 + rollout_d1 + size1 + attn cuts  ← candidate headline

Group B (SCIP harness — node counts only, never wall-clock vs Group A):
  scip_default      SCIP defaults
  scip_pseudocost   SCIP, presolve off, cuts off, pseudocost
  scip_fullstrong   SCIP, presolve off, cuts off, fullstrong

Metrics (all emitted unconditionally)
--------------------------------------
  nodes, tree_lps, decision_lps, wall_time, lp_time, model_time,
  encode_time, policy_time, rollout_time, cutbeam_time,
  time_minus_model, gap_at_end, gap_closed, status,
  cuts_added, obj, dist_from_optimum, solved
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Instance generator
# ---------------------------------------------------------------------------

def gen_instance(n_rows, n_cols, rng, density=0.05):
    A = (rng.random((n_rows, n_cols)) < density).astype(np.float64)
    for i in range(n_rows):
        if A[i].sum() == 0:
            A[i, rng.integers(n_cols)] = 1.0
    c = rng.uniform(1.0, 10.0, n_cols)
    b = np.ones(n_rows, dtype=np.float64)
    return A, b, c


# ---------------------------------------------------------------------------
# SGM
# ---------------------------------------------------------------------------

def sgm(values, shift=10.0):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(np.exp(np.mean(np.log(np.maximum(vals, 0) + shift))) - shift)


def sgm_time(values, shift=1.0):
    return sgm(values, shift=shift)


# ---------------------------------------------------------------------------
# Metric row builder
# ---------------------------------------------------------------------------

def _row(
    status: str,
    obj: Optional[float],
    nodes: int,
    wall_time: float,
    lp_time: float,
    model_time: float,
    timing: dict,
    tree_lps: int,
    decision_lps: int,
    cuts_added: int,
    opt: Optional[float],
):
    gap = float("inf")
    gap_closed = float("nan")
    dist = float("inf")
    solved = status in ("optimal",)

    if obj is not None and np.isfinite(obj):
        if opt is not None and np.isfinite(opt) and opt > 0:
            dist = (obj - opt) / opt
            gap_closed = 1.0 - dist if dist <= 1.0 else 0.0
        gap = 0.0 if solved else float("inf")

    return {
        "status": status,
        "obj": obj,
        "nodes": nodes,
        "tree_lps": tree_lps,
        "decision_lps": decision_lps,
        "wall_time": wall_time,
        "lp_time": lp_time,
        "model_time": model_time,
        "encode_time": timing.get("encode", 0.0),
        "policy_time": timing.get("branch", 0.0),
        "rollout_time": timing.get("rollout", 0.0),
        "cutbeam_time": timing.get("cutbeam", 0.0),
        "time_minus_model": wall_time - model_time,
        "gap_at_end": gap,
        "gap_closed": gap_closed,
        "cuts_added": cuts_added,
        "dist_from_optimum": dist,
        "solved": solved,
    }


# ---------------------------------------------------------------------------
# Group A: our harness
# ---------------------------------------------------------------------------

def build_classical(sb_init, eta, time_limit, node_limit):
    from bnb_wm.solver.classical_bnb import ClassicalBnBSolver
    from bnb_wm.solver.config import SolverConfig
    cfg = SolverConfig(
        branch_mode="most_fractional",
        cut_mode="none",
        ors_cascade=False,
        katz_weight=0.0,
        node_selection="bound",
        primal_heuristic=True,
        time_limit=time_limit,
        node_limit=node_limit,
        exact=True,
    )
    return ClassicalBnBSolver(config=cfg, sb_init=sb_init, eta=eta)


def run_classical(solver, A, b, c, opt):
    t0 = time.perf_counter()
    r = solver.solve(A, b, c)
    wall = time.perf_counter() - t0
    timing = r.diagnostics.get("timing", {})
    return _row(
        status=r.status, obj=r.objective, nodes=r.n_nodes,
        wall_time=wall, lp_time=r.lp_time, model_time=0.0,
        timing=timing,
        tree_lps=r.tree_lps, decision_lps=r.decision_lps,
        cuts_added=r.cuts_added, opt=opt,
    )


def build_neural(model, device, branch_mode, cut_mode, ors_cascade,
                 katz_weight, node_selection, time_limit, node_limit,
                 size_weight=0.0, ctg_weight=0.0, cut_pool_max=200, **kw):
    from bnb_wm.solver.neural_bnb import NeuralBnBSolver
    from bnb_wm.solver.config import SolverConfig
    cfg = SolverConfig(
        branch_mode=branch_mode,
        cut_mode=cut_mode,
        ors_cascade=ors_cascade,
        katz_weight=katz_weight,
        node_selection=node_selection,
        size_weight=size_weight,
        ctg_weight=ctg_weight,
        cut_pool_max=cut_pool_max,
        cut_budget_cap=cut_pool_max,
        primal_heuristic=True,
        time_limit=time_limit,
        node_limit=node_limit,
        exact=True,
        **kw,
    )
    return NeuralBnBSolver(model, device, cfg)


def _build_blend_rollout(model, device, alpha, time_limit, node_limit):
    """Rollout lookahead with blended (policy + MF) candidate scoring."""
    from bnb_wm.solver.neural_bnb import NeuralBnBSolver
    from bnb_wm.solver.config import SolverConfig
    cfg = SolverConfig(
        branch_mode="rollout",
        cut_mode="none",
        ors_cascade=False,
        katz_weight=0.0,
        mf_blend_alpha=alpha,
        node_selection="bound",
        primal_heuristic=True,
        time_limit=time_limit,
        node_limit=node_limit,
        exact=True,
    )
    return NeuralBnBSolver(model, device, cfg)


def _build_dyn_pseudo(model, device, time_limit, node_limit, mf_blend_alpha=None):
    """Dynamics-as-pseudocost branching with optional MF blend fallback."""
    from bnb_wm.solver.neural_bnb import NeuralBnBSolver
    from bnb_wm.solver.config import SolverConfig
    cfg = SolverConfig(
        branch_mode="dyn_pseudo",
        cut_mode="none",
        ors_cascade=False,
        katz_weight=0.0,
        mf_blend_alpha=mf_blend_alpha,
        node_selection="bound",
        primal_heuristic=True,
        time_limit=time_limit,
        node_limit=node_limit,
        exact=True,
    )
    return NeuralBnBSolver(model, device, cfg)


def _build_mf_blend_cuts(model, device, alpha, time_limit, node_limit):
    """Blended branching + heuristic cuts, no rollout."""
    from bnb_wm.solver.neural_bnb import NeuralBnBSolver
    from bnb_wm.solver.config import SolverConfig
    cfg = SolverConfig(
        branch_mode="policy",
        cut_mode="heuristic",
        ors_cascade=False,
        katz_weight=0.0,
        mf_blend_alpha=alpha,
        node_selection="bound",
        primal_heuristic=True,
        time_limit=time_limit,
        node_limit=node_limit,
        exact=True,
    )
    return NeuralBnBSolver(model, device, cfg)


def _build_blend_rollout_cuts(model, device, alpha, time_limit, node_limit):
    """Blended rollout + heuristic cuts."""
    from bnb_wm.solver.neural_bnb import NeuralBnBSolver
    from bnb_wm.solver.config import SolverConfig
    cfg = SolverConfig(
        branch_mode="rollout",
        cut_mode="heuristic",
        ors_cascade=False,
        katz_weight=0.0,
        mf_blend_alpha=alpha,
        node_selection="bound",
        primal_heuristic=True,
        time_limit=time_limit,
        node_limit=node_limit,
        exact=True,
    )
    return NeuralBnBSolver(model, device, cfg)


def _build_mf_blend(model, device, alpha, time_limit, node_limit):
    from bnb_wm.solver.neural_bnb import NeuralBnBSolver
    from bnb_wm.solver.config import SolverConfig
    cfg = SolverConfig(
        branch_mode="policy",
        cut_mode="none",
        ors_cascade=False,
        katz_weight=0.0,
        mf_blend_alpha=alpha,
        node_selection="bound",
        primal_heuristic=True,
        time_limit=time_limit,
        node_limit=node_limit,
        exact=True,
    )
    return NeuralBnBSolver(model, device, cfg)


def run_neural(solver, A, b, c, opt):
    t0 = time.perf_counter()
    r = solver.solve(A, b, c)
    wall = time.perf_counter() - t0
    diag = r.diagnostics or {}
    timing = diag.get("timing", {})
    return _row(
        status=r.status, obj=r.objective, nodes=r.n_nodes,
        wall_time=wall, lp_time=r.lp_time, model_time=r.model_time,
        timing=timing,
        tree_lps=r.lp_solves, decision_lps=0,
        cuts_added=r.cuts_added, opt=opt,
    )


# ---------------------------------------------------------------------------
# Group B: SCIP harness
# ---------------------------------------------------------------------------

def run_scip(A, b, c, opt, time_limit, variant="default"):
    try:
        from pyscipopt import Model
    except ImportError:
        return None

    m_rows, n_cols = A.shape
    m = Model()
    m.hideOutput(True)
    m.setParam("limits/time", time_limit)

    if variant in ("pseudocost", "fullstrong"):
        m.setParam("presolving/maxrounds", 0)
        m.setParam("separating/maxrounds", 0)
        m.setParam("separating/maxroundsroot", 0)

    if variant == "pseudocost":
        m.setParam("branching/pscost/priority", 536870911)
    elif variant == "fullstrong":
        m.setParam("branching/fullstrong/priority", 536870911)

    xs = [m.addVar(f"x{j}", vtype="B", obj=float(c[j])) for j in range(n_cols)]
    for i in range(m_rows):
        nz = np.where(A[i] > 0)[0]
        m.addCons(sum(xs[j] for j in nz) >= 1.0, name=f"c{i}")
    m.setMinimize()

    t0 = time.perf_counter()
    m.optimize()
    wall = time.perf_counter() - t0

    status = m.getStatus()
    nodes = int(m.getNNodes())
    try:
        obj = float(m.getObjVal())
    except Exception:
        obj = None

    return _row(
        status=status, obj=obj, nodes=nodes,
        wall_time=wall, lp_time=0.0, model_time=0.0,
        timing={}, tree_lps=nodes, decision_lps=0,
        cuts_added=0, opt=opt,
    )


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

ALL_METHODS_A = [
    # Classical baselines
    "mf",
    "classical_sb1",
    "classical_sb4",
    "classical_sb8",
    # Neural: cheap / no-rollout
    "policy",
    "mf_blend_20",
    "mf_blend_20_cuts_attn",
    # Neural: rollout ladder (two depth × two size_weight rows)
    "rollout_d1",
    "rollout_d1_size1",
    "rollout_d2_size1",
    # Candidate headline: blend + rollout + size + cuts
    "blend_rollout_d1_size1_cuts_attn",
]

ALL_METHODS_B = [
    "scip_default",
    "scip_pseudocost",
    "scip_fullstrong",
]


def method_needs_model(name):
    return name not in ("mf", "classical_sb1", "classical_sb4", "classical_sb8",
                        "scip_default", "scip_pseudocost", "scip_fullstrong")


def method_is_scip(name):
    return name.startswith("scip_")


# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------

def print_table(results: dict, methods: list, opt_col="nodes"):
    """results[method][instance_idx] -> row dict"""
    # Find the common-solved set across all methods in this table
    all_inst = set()
    for m in methods:
        if m in results:
            all_inst.update(results[m].keys())
    common_solved = all_inst.copy()
    for m in methods:
        if m not in results:
            continue
        solved_inst = {i for i, r in results[m].items() if r["solved"]}
        common_solved &= solved_inst
    n_common = len(common_solved)

    print(f"\n{'Method':<34}  {'SGM nodes':>10}  {'SGM time':>9}  "
          f"{'solved':>6}  {'dec_lps':>8}  {'gap@end':>8}  {'SGM dist':>9}")
    print("-" * 92)

    for m in methods:
        if m not in results:
            continue
        rows = [results[m][i] for i in sorted(results[m])]
        if not rows:
            continue
        n_solved = sum(1 for r in rows if r["solved"])
        node_sgm = sgm([r["nodes"] for r in rows])
        time_sgm = sgm_time([r["wall_time"] for r in rows])
        dist_sgm = sgm([r["dist_from_optimum"] for r in rows
                        if np.isfinite(r["dist_from_optimum"])], shift=0.01)
        dec_sgm  = sgm([r["decision_lps"] for r in rows], shift=1)
        gaps = [r["gap_at_end"] for r in rows
                if r["gap_at_end"] is not None and np.isfinite(r["gap_at_end"])]
        gap_mean = np.mean(gaps) if gaps else float("nan")
        print(f"{m:<34}  {node_sgm:>10.1f}  {time_sgm:>9.2f}  "
              f"{n_solved:>6}/{len(rows)}  {dec_sgm:>8.1f}  "
              f"{gap_mean:>8.4f}  {dist_sgm:>9.4f}")

    if n_common < len(all_inst):
        print(f"  [common-solved subset: {n_common}/{len(all_inst)} instances]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",   default=None)
    ap.add_argument("--n_rows",       type=int,   default=100)
    ap.add_argument("--n_cols",       type=int,   default=200)
    ap.add_argument("--n_instances",  type=int,   default=20)
    ap.add_argument("--seed",         type=int,   default=1)
    ap.add_argument("--density",      type=float, default=0.05)
    ap.add_argument("--time_limit",   type=float, default=60.0)
    ap.add_argument("--node_limit",   type=int,   default=2000)
    ap.add_argument("--optima",       default=None,
                    help="Path to precomputed optima JSON.")
    ap.add_argument("--methods",      nargs="+",  default=None,
                    help="Subset of methods to run (default: all).")
    ap.add_argument("--no_scip",      action="store_true")
    ap.add_argument("--device",       default="cuda")
    ap.add_argument("--out",          required=True)
    ap.add_argument("--cut_pool_max", type=int, default=200,
                    help="Cap cut pool size (use ~20 for hard/large instances).")
    args = ap.parse_args()

    # ---- load optima ----
    optima: dict = {}
    if args.optima and Path(args.optima).exists():
        with open(args.optima) as f:
            opt_data = json.load(f)
        optima = opt_data.get("instances", {})
        print(f"Loaded {len(optima)} precomputed optima from {args.optima}")
    else:
        print("Warning: no optima file — dist_from_optimum will be inf.")

    def get_opt(i):
        key = f"seed={args.seed}/{i}"
        entry = optima.get(key)
        return entry["obj"] if entry and entry["obj"] is not None else None

    # ---- select methods ----
    methods_a = ALL_METHODS_A
    methods_b = [] if args.no_scip else ALL_METHODS_B
    all_methods = methods_a + methods_b
    if args.methods:
        all_methods = [m for m in all_methods if m in args.methods]
        methods_a = [m for m in methods_a if m in all_methods]
        methods_b = [m for m in methods_b if m in all_methods]

    need_model = any(method_needs_model(m) for m in all_methods)

    # ---- load model ----
    model = device = None
    if need_model:
        if args.checkpoint is None:
            raise ValueError("--checkpoint required for neural methods.")
        import torch
        from bnb_wm.model.world_model import BnBWorldModel
        from bnb_wm.training.checkpoint import load_weights_only
        device = torch.device(
            args.device if torch.cuda.is_available() else "cpu")
        model = BnBWorldModel().to(device)
        load_weights_only(model, args.checkpoint, device=device)
        model.eval()
        print(f"Model loaded on {device}.")

    # ---- build solver instances ----
    tl = args.time_limit
    nl = args.node_limit
    cpm = args.cut_pool_max

    solvers: dict = {}
    for m in methods_a:
        # ---- Classical ----
        if m == "mf":
            solvers[m] = ("classical", build_classical(0, 99, tl, nl))
        elif m == "classical_sb1":
            solvers[m] = ("classical", build_classical(1, 1, tl, nl))
        elif m == "classical_sb4":
            solvers[m] = ("classical", build_classical(4, 4, tl, nl))
        elif m == "classical_sb8":
            solvers[m] = ("classical", build_classical(8, 4, tl, nl))
        # ---- Neural: cheap ----
        elif m == "policy":
            solvers[m] = ("neural", build_neural(
                model, device, "policy", "none", False, 0.0, "bound", tl, nl))
        elif m == "mf_blend_20":
            solvers[m] = ("neural", _build_mf_blend(model, device, 0.20, tl, nl))
        elif m == "mf_blend_20_cuts_attn":
            from bnb_wm.solver.neural_bnb import NeuralBnBSolver
            from bnb_wm.solver.config import SolverConfig
            cfg = SolverConfig(
                branch_mode="policy", cut_mode="attention",
                ors_cascade=False, katz_weight=0.0,
                mf_blend_alpha=0.20, node_selection="bound",
                size_weight=0.0, ctg_weight=0.0,
                cut_pool_max=cpm, cut_budget_cap=cpm,
                primal_heuristic=True, time_limit=tl, node_limit=nl, exact=True,
            )
            solvers[m] = ("neural", NeuralBnBSolver(model, device, cfg))
        # ---- Neural: rollout ladder ----
        elif m == "rollout_d1":
            solvers[m] = ("neural", build_neural(
                model, device, "rollout", "none", False, 0.0, "bound", tl, nl,
                size_weight=0.0, ctg_weight=0.0))
        elif m == "rollout_d1_size1":
            from bnb_wm.solver.neural_bnb import NeuralBnBSolver
            from bnb_wm.solver.config import SolverConfig
            cfg = SolverConfig(
                branch_mode="rollout", cut_mode="none",
                ors_cascade=False, katz_weight=0.0, node_selection="bound",
                lookahead_depth=1, size_weight=1.0, ctg_weight=0.0,
                cut_pool_max=cpm, cut_budget_cap=cpm,
                primal_heuristic=True, time_limit=tl, node_limit=nl, exact=True,
            )
            solvers[m] = ("neural", NeuralBnBSolver(model, device, cfg))
        elif m == "rollout_d2_size1":
            from bnb_wm.solver.neural_bnb import NeuralBnBSolver
            from bnb_wm.solver.config import SolverConfig
            cfg = SolverConfig(
                branch_mode="rollout", cut_mode="none",
                ors_cascade=False, katz_weight=0.0, node_selection="bound",
                lookahead_depth=2, size_weight=1.0, ctg_weight=0.0,
                cut_pool_max=cpm, cut_budget_cap=cpm,
                primal_heuristic=True, time_limit=tl, node_limit=nl, exact=True,
            )
            solvers[m] = ("neural", NeuralBnBSolver(model, device, cfg))
        # ---- Neural: candidate headline ----
        elif m == "blend_rollout_d1_size1_cuts_attn":
            from bnb_wm.solver.neural_bnb import NeuralBnBSolver
            from bnb_wm.solver.config import SolverConfig
            cfg = SolverConfig(
                branch_mode="rollout", cut_mode="attention",
                ors_cascade=False, katz_weight=0.0, mf_blend_alpha=0.20,
                node_selection="bound", lookahead_depth=1,
                size_weight=1.0, ctg_weight=0.0,
                cut_pool_max=cpm, cut_budget_cap=cpm,
                primal_heuristic=True, time_limit=tl, node_limit=nl, exact=True,
            )
            solvers[m] = ("neural", NeuralBnBSolver(model, device, cfg))

    # ---- generate instances ----
    rng = np.random.default_rng(args.seed)
    instances = []
    for i in range(args.n_instances):
        A, b, c = gen_instance(args.n_rows, args.n_cols, rng, args.density)
        instances.append((A, b, c, get_opt(i)))

    print(f"\nRunning {len(all_methods)} methods × {args.n_instances} instances "
          f"({args.n_rows}×{args.n_cols}, seed={args.seed})\n")

    # ---- run ----
    results: dict = {m: {} for m in all_methods}

    for i, (A, b, c, opt) in enumerate(instances):
        for m in methods_a:
            kind, solver = solvers[m]
            if kind == "classical":
                row = run_classical(solver, A, b, c, opt)
            else:
                row = run_neural(solver, A, b, c, opt)
            results[m][i] = row
            obj_s    = f"obj={row['obj']:.3f}" if row['obj'] is not None else "obj=   N/A"
            gap_s    = f"gap={row['gap_at_end']*100:5.1f}%" if row['gap_at_end'] is not None else "gap=   N/A"
            tlp_s    = f"tLPs={row['tree_lps']:>5}"
            dlp_s    = f"dLPs={row['decision_lps']:>5}"
            wt_s     = f"t={row['wall_time']:>6.2f}s"
            cuts_s   = f"cuts={row['cuts_added']:>3}" if row['cuts_added'] else ""
            dist_s   = (f"dist={row['dist_from_optimum']:.3f}"
                        if row['dist_from_optimum'] is not None else "")
            extras = "  ".join(x for x in [tlp_s, dlp_s, wt_s, cuts_s, dist_s] if x)
            print(f"  [{i+1:>3}/{args.n_instances}] {m:<22}  "
                  f"nodes={row['nodes']:>6}  {gap_s}  {obj_s}  {extras}")

        if not args.no_scip:
            for m in methods_b:
                variant = m.replace("scip_", "")
                row = run_scip(A, b, c, opt, tl, variant)
                if row is None:
                    row = _row("scip_unavailable", None, 0, 0, 0, 0, {}, 0, 0, 0, opt)
                results[m][i] = row
                print(f"  [{i+1:>3}/{args.n_instances}] {m:<22}  "
                      f"nodes={row['nodes']:>6}  status={row['status']:>8}  "
                      f"t={row['wall_time']:>6.2f}s")

    # ---- summary table ----
    print(f"\n{'='*75}")
    print(f"GROUP A — our harness  ({args.n_rows}×{args.n_cols}, "
          f"seed={args.seed}, n={args.n_instances})")
    print(f"  time_limit={tl}s  node_limit={nl}")
    print_table(results, [m for m in methods_a if m in results], "nodes")

    if not args.no_scip and methods_b:
        print(f"\nGROUP B — SCIP  (node counts only)")
        print_table(results, [m for m in methods_b if m in results], "nodes")

    # ---- write JSON ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert int keys to strings for JSON
    json_results = {
        m: {str(i): row for i, row in rows.items()}
        for m, rows in results.items()
    }

    with open(out_path, "w") as f:
        json.dump({
            "config": vars(args),
            "results": json_results,
            "sgm": {
                m: {
                    "nodes": sgm([results[m][i]["nodes"]
                                  for i in results[m]]),
                    "wall_time": sgm_time([results[m][i]["wall_time"]
                                           for i in results[m]]),
                    "decision_lps": sgm([results[m][i]["decision_lps"]
                                         for i in results[m]], shift=1),
                }
                for m in all_methods if m in results
            },
        }, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
