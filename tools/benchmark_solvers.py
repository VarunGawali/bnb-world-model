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
  mf                most-fractional branching
  classical_sb0     pure pseudocost (0 SB LPs)
  classical_sb1     reliability branching, eta=1
  classical_sb4     reliability branching, eta=4  ← primary baseline
  classical_sb8     reliability branching, eta=8
  classical_sbfull  full strong branching
  policy            GNN policy argmax
  rollout           + latent lookahead
  rollout_ors       + ORS cascade
  rollout_katz      + Katz blend
  rollout_cuts_heur + max-violation cuts
  rollout_cuts_lat  + latent cut beam
  rollout_cuts_attn + attention-scored cuts (parameter-free)
  neural_full       best assembled config: rollout + heuristic cuts + best-bound  ← headline

Leave-one-out ablation against neural_full:
  neural_loo_no_rollout  policy + cuts (no lookahead)
  neural_loo_no_cuts     rollout, no cuts
  neural_loo_ors         neural_full + ORS cascade
  neural_loo_katz        neural_full + Katz blend
  neural_loo_ctg         neural_full + cost-to-go node selection

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
                 katz_weight, node_selection, time_limit, node_limit):
    from bnb_wm.solver.neural_bnb import NeuralBnBSolver
    from bnb_wm.solver.config import SolverConfig
    cfg = SolverConfig(
        branch_mode=branch_mode,
        cut_mode=cut_mode,
        ors_cascade=ors_cascade,
        katz_weight=katz_weight,
        node_selection=node_selection,
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
    "mf",
    "classical_sb0", "classical_sb1", "classical_sb4",
    "classical_sb8", "classical_sbfull",
    "policy",
    "rollout",
    "rollout_ors",
    "rollout_katz",
    "rollout_cuts_heur",
    "rollout_cuts_lat",
    "rollout_cuts_attn",
    "neural_full",
    # leave-one-out rows against neural_full
    "neural_loo_no_rollout",
    "neural_loo_no_cuts",
    "neural_loo_ors",
    "neural_loo_katz",
    "neural_loo_ctg",
]

ALL_METHODS_B = [
    "scip_default",
    "scip_pseudocost",
    "scip_fullstrong",
]


def method_needs_model(name):
    return name in ("policy", "rollout", "rollout_ors", "rollout_katz",
                    "rollout_cuts_heur", "rollout_cuts_lat", "rollout_cuts_attn",
                    "neural_full",
                    "neural_loo_no_rollout", "neural_loo_no_cuts",
                    "neural_loo_ors", "neural_loo_katz", "neural_loo_ctg")


def method_is_scip(name):
    return name.startswith("scip_")


# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------

def print_table(results: dict, methods: list, opt_col="nodes"):
    """results[method][instance_idx] -> row dict"""
    print(f"\n{'Method':<22}  {'SGM nodes':>10}  {'SGM time':>9}  "
          f"{'solved':>6}  {'dec_lps':>8}  {'SGM dist':>9}")
    print("-" * 75)

    for m in methods:
        rows = [results[m][i] for i in sorted(results[m])]
        if not rows:
            continue
        n_solved = sum(1 for r in rows if r["solved"])
        node_sgm = sgm([r["nodes"] for r in rows])
        time_sgm = sgm_time([r["wall_time"] for r in rows])
        dist_sgm = sgm([r["dist_from_optimum"] for r in rows
                        if np.isfinite(r["dist_from_optimum"])], shift=0.01)
        dec_sgm  = sgm([r["decision_lps"] for r in rows], shift=1)
        print(f"{m:<22}  {node_sgm:>10.1f}  {time_sgm:>9.2f}  "
              f"{n_solved:>6}/{len(rows)}  {dec_sgm:>8.1f}  {dist_sgm:>9.4f}")


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

    solvers: dict = {}
    for m in methods_a:
        if m == "mf":
            solvers[m] = ("classical", build_classical(0, 99, tl, nl))
        elif m == "classical_sb0":
            solvers[m] = ("classical", build_classical(0, 4, tl, nl))
        elif m == "classical_sb1":
            solvers[m] = ("classical", build_classical(1, 1, tl, nl))
        elif m == "classical_sb4":
            solvers[m] = ("classical", build_classical(4, 4, tl, nl))
        elif m == "classical_sb8":
            solvers[m] = ("classical", build_classical(8, 4, tl, nl))
        elif m == "classical_sbfull":
            solvers[m] = ("classical", build_classical(None, 4, tl, nl))
        elif m == "policy":
            solvers[m] = ("neural", build_neural(
                model, device, "policy", "none", False, 0.0, "bound", tl, nl))
        elif m == "rollout":
            solvers[m] = ("neural", build_neural(
                model, device, "rollout", "none", False, 0.0, "bound", tl, nl))
        elif m == "rollout_ors":
            solvers[m] = ("neural", build_neural(
                model, device, "rollout", "none", True, 0.0, "bound", tl, nl))
        elif m == "rollout_katz":
            solvers[m] = ("neural", build_neural(
                model, device, "rollout", "none", True, 0.3, "bound", tl, nl))
        elif m == "rollout_cuts_heur":
            solvers[m] = ("neural", build_neural(
                model, device, "rollout", "heuristic", True, 0.3, "bound", tl, nl))
        elif m == "rollout_cuts_lat":
            solvers[m] = ("neural", build_neural(
                model, device, "rollout", "latent", True, 0.3, "bound", tl, nl))
        elif m == "rollout_cuts_attn":
            solvers[m] = ("neural", build_neural(
                model, device, "rollout", "attention", False, 0.0, "bound", tl, nl))
        elif m == "neural_full":
            # best assembled config: rollout + heuristic cuts + best-bound node sel
            solvers[m] = ("neural", build_neural(
                model, device, "rollout", "heuristic", False, 0.0, "bound", tl, nl))
        # ---- leave-one-out ablation against neural_full ----
        elif m == "neural_loo_no_rollout":
            # policy argmax + heuristic cuts (no rollout lookahead)
            solvers[m] = ("neural", build_neural(
                model, device, "policy", "heuristic", False, 0.0, "bound", tl, nl))
        elif m == "neural_loo_no_cuts":
            # rollout, no cuts
            solvers[m] = ("neural", build_neural(
                model, device, "rollout", "none", False, 0.0, "bound", tl, nl))
        elif m == "neural_loo_ors":
            # neural_full + ORS cascade
            solvers[m] = ("neural", build_neural(
                model, device, "rollout", "heuristic", True, 0.0, "bound", tl, nl))
        elif m == "neural_loo_katz":
            # neural_full + Katz blend
            solvers[m] = ("neural", build_neural(
                model, device, "rollout", "heuristic", False, 0.3, "bound", tl, nl))
        elif m == "neural_loo_ctg":
            # neural_full + cost-to-go node selection
            solvers[m] = ("neural", build_neural(
                model, device, "rollout", "heuristic", False, 0.0, "cost_to_go", tl, nl))

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
    LOO_METHODS = {"neural_loo_no_rollout", "neural_loo_no_cuts",
                   "neural_loo_ors", "neural_loo_katz", "neural_loo_ctg"}
    additive_methods = [m for m in methods_a if m in results and m not in LOO_METHODS]
    loo_methods      = [m for m in methods_a if m in results and m in LOO_METHODS]

    print(f"\n{'='*75}")
    print(f"GROUP A — our harness  ({args.n_rows}×{args.n_cols}, "
          f"seed={args.seed}, n={args.n_instances})")
    print(f"  time_limit={tl}s  node_limit={nl}")
    print("\n--- additive ablation ladder ---")
    print_table(results, additive_methods, "nodes")

    if loo_methods:
        print("\n--- leave-one-out ablation (vs neural_full) ---")
        print_table(results, ["neural_full"] + loo_methods, "nodes")

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
