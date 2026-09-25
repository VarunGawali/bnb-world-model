"""
wall_clock_benchmark.py — Time-to-optimality benchmark.

Runs every method on the same held-out instances with a generous time limit
and records the exact second each method proves optimality (or marks DNF).

Methods:
    - SCIP (default settings, via pyscipopt — optional)
    - HiGHS + MF          (most-fractional, sb_init=0)
    - HiGHS + SB-1/4/8    (strong branching depths)
    - CutWorld Neural-Best-d1 (loaded from checkpoint)

Usage:
    # Full run (20 instances, 300x600, 3600s limit):
    python scripts/wall_clock_benchmark.py \\
        --checkpoint checkpoints/phase4_best.pt \\
        --n-instances 20 --n-vars 300 --n-cons 600 \\
        --time-limit 3600 --out results/wall_clock.json

    # Pilot run (3 small instances, 2 min limit, no SB to be fast):
    python scripts/wall_clock_benchmark.py \\
        --checkpoint checkpoints/phase4_best.pt \\
        --n-instances 3 --n-vars 150 --n-cons 300 \\
        --time-limit 120 --skip-sb \\
        --out /tmp/wc_pilot.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from bnb_wm.model import BnBWorldModel
from bnb_wm.solver.config import SolverConfig
from bnb_wm.solver.classical_bnb import ClassicalBnBSolver
from bnb_wm.solver.neural_bnb import NeuralBnBSolver
from bnb_wm.training.checkpoint import load_weights_only


# ---------------------------------------------------------------------------
# Extract (A, b, c) from an ecole instance via pyscipopt
# (same logic as ablation.py:_extract_Abc)
# ---------------------------------------------------------------------------

def extract_Abc(instance) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (A, b, c) for min c^T x s.t. Ax >= b, x in {0,1}^n."""
    try:
        scip = instance.copy_orig().as_pyscipopt()
        scip.hideOutput()

        vars_ = scip.getVars(transformed=False)
        n = len(vars_)
        c = np.array([v.getObj() for v in vars_], dtype=np.float64)
        var_idx = {v.name: i for i, v in enumerate(vars_)}

        rows_data = []
        for con in scip.getConss():
            try:
                lhs = scip.getLhs(con)
                rhs = scip.getRhs(con)
                coef_map = scip.getValsLinear(con)
                rows_data.append((lhs, rhs, coef_map))
            except Exception:
                pass

        if not rows_data:
            return None

        m = len(rows_data)
        A = np.zeros((m, n), dtype=np.float64)
        b = np.zeros(m, dtype=np.float64)

        for ri, (lhs, rhs, coef_map) in enumerate(rows_data):
            if lhs > -1e19:
                b[ri] = lhs
                for v, coef in coef_map.items():
                    j = var_idx.get(v.name, -1)
                    if j >= 0:
                        A[ri, j] = coef
            else:
                b[ri] = -rhs
                for v, coef in coef_map.items():
                    j = var_idx.get(v.name, -1)
                    if j >= 0:
                        A[ri, j] = -coef
        return A, b, c
    except Exception as e:
        print(f"    [extract_Abc error: {e}]")
        return None


# ---------------------------------------------------------------------------
# SCIP wrapper
# ---------------------------------------------------------------------------

def solve_with_scip(instance, time_limit: float) -> dict:
    try:
        scip = instance.copy_orig().as_pyscipopt()
        scip.hideOutput()
        scip.setRealParam("limits/time", time_limit)
        t0 = time.perf_counter()
        scip.optimize()
        elapsed = time.perf_counter() - t0

        status = scip.getStatus()
        solved = status == "optimal"
        gap = scip.getGap() if not solved else 0.0
        nodes = scip.getNNodes()

        return {
            "method": "SCIP",
            "solved": solved,
            "solve_time": elapsed if solved else None,
            "wall_time": elapsed,
            "nodes": int(nodes),
            "gap": float(gap),
            "status": status,
        }
    except Exception as e:
        return {"method": "SCIP", "solved": False, "solve_time": None,
                "wall_time": None, "nodes": None, "gap": None,
                "error": str(e)}


# ---------------------------------------------------------------------------
# HiGHS classical solver wrapper
# ---------------------------------------------------------------------------

def solve_classical(A, b, c, method: str, time_limit: float) -> dict:
    """method: one of 'mf', 'sb1', 'sb4', 'sb8'"""
    sb_map = {"mf": 0, "sb1": 1, "sb4": 4, "sb8": 8}

    cfg = SolverConfig(
        time_limit=time_limit,
        sb_init=sb_map[method],
        use_cuts=False,
        use_neural=False,
        exact=True,
    )

    solver = ClassicalBnBSolver(cfg)
    t0 = time.perf_counter()
    result = solver.solve(A, b, c)
    elapsed = time.perf_counter() - t0

    return {
        "method": method.upper(),
        "solved": result.solved,
        "solve_time": elapsed if result.solved else None,
        "wall_time": elapsed,
        "nodes": result.nodes,
        "gap": float(result.optimality_gap),
    }


# ---------------------------------------------------------------------------
# Neural solver wrapper
# ---------------------------------------------------------------------------

def solve_neural(A, b, c, model, time_limit: float,
                 device: torch.device) -> dict:
    cfg = SolverConfig(
        time_limit=time_limit,
        use_neural=True,
        use_cuts=True,
        rollout_depth=1,
        top_k=5,
        exact=True,
        ctg_weight=0.0,   # Neural-Best config: CostToGo zeroed out
        size_weight=0.3,
    )

    solver = NeuralBnBSolver(model, device, config=cfg)
    t0 = time.perf_counter()
    result = solver.solve(A, b, c)
    elapsed = time.perf_counter() - t0

    return {
        "method": "CutWorld",
        "solved": result.solved,
        "solve_time": elapsed if result.solved else None,
        "wall_time": elapsed,
        "nodes": result.nodes,
        "gap": float(result.optimality_gap),
    }


# ---------------------------------------------------------------------------
# SGM helper
# ---------------------------------------------------------------------------

def sgm(values: list, shift: float = 1.0) -> float:
    vals = [v for v in values if v is not None]
    if not vals:
        return float("nan")
    return float(np.exp(np.mean(np.log([v + shift for v in vals]))) - shift)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(all_results: list[list[dict]], methods: list[str],
                  n_instances: int, time_limit: float):
    print("\n" + "=" * 72)
    print(f"TIME-TO-OPTIMALITY BENCHMARK  "
          f"({n_instances} instances, limit={time_limit}s)")
    print("=" * 72)
    hdr = (f"{'Method':<18} {'Solved':>9}  {'SGM Time(s)':>12}  "
           f"{'Median(s)':>10}  {'SGM Nodes':>10}  {'Mean Gap':>9}")
    print(hdr)
    print("-" * 72)

    for method in methods:
        rows = [r for inst in all_results
                for r in inst if r["method"] == method]
        n = len(rows)
        n_solved = sum(1 for r in rows if r["solved"])
        solve_times = [r["solve_time"] for r in rows if r["solved"]]
        nodes = [r["nodes"] for r in rows
                 if r["nodes"] is not None and r["solved"]]
        gaps = [r["gap"] for r in rows if r["gap"] is not None]

        sgm_t   = sgm(solve_times, shift=1.0)
        med_t   = float(np.median(solve_times)) if solve_times else float("nan")
        sgm_n   = sgm(nodes, shift=10.0)
        mean_g  = float(np.mean(gaps) * 100) if gaps else float("nan")

        dnf = f"  ({n - n_solved} DNF)" if n_solved < n else ""
        print(f"{method:<18} {n_solved:>4}/{n}{dnf:<10}"
              f"{sgm_t:>12.1f}  {med_t:>10.1f}  "
              f"{sgm_n:>10.0f}  {mean_g:>8.1f}%")

    print("=" * 72)
    print("SGM Time / Nodes computed over *solved* instances only (shift 1 / 10).")
    print("DNF = did not finish (hit time limit without proving optimality).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Wall-clock time-to-optimality benchmark")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to trained CutWorld checkpoint (.pt)")
    ap.add_argument("--n-instances", type=int, default=20,
                    help="Number of fresh held-out instances")
    ap.add_argument("--n-vars", type=int, default=300,
                    help="Variables per instance (Set Cover columns)")
    ap.add_argument("--n-cons", type=int, default=600,
                    help="Constraints per instance (Set Cover rows)")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="Per-instance wall-clock limit in seconds. "
                         "Omit to run with no limit (methods run until they "
                         "prove optimality). Use a limit only if you need to "
                         "cap runaway classical solvers.")
    ap.add_argument("--seed-offset", type=int, default=99999,
                    help="RNG seed base (keep high to avoid training overlap)")
    ap.add_argument("--skip-scip", action="store_true",
                    help="Skip SCIP (use if pyscipopt unavailable)")
    ap.add_argument("--skip-sb", action="store_true",
                    help="Skip SB-1/4/8 (faster pilot runs)")
    ap.add_argument("--out", default="results/wall_clock.json",
                    help="Output JSON path")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    # ---- device ----
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ---- model ----
    print(f"Loading checkpoint: {args.checkpoint}")
    model = BnBWorldModel(hidden_dim=128, n_gnn_layers=3).to(device)
    load_weights_only(model, args.checkpoint, device)
    model.eval()
    print("Model ready.\n")

    # ---- instance generator ----
    try:
        import ecole
    except ImportError:
        raise SystemExit("ecole required for instance generation: pip install ecole")

    gen = ecole.instance.SetCoverGenerator(
        n_rows=args.n_cons, n_cols=args.n_vars, density=0.05
    )
    gen.seed(args.seed_offset)

    # ---- method list ----
    classical_methods = ["mf"]
    if not args.skip_sb:
        classical_methods += ["sb1", "sb4", "sb8"]
    all_method_labels = (
        [m.upper() for m in classical_methods]
        + (["SCIP"] if not args.skip_scip else [])
        + ["CutWorld"]
    )

    # Resolve effective time limit (None = no limit = use a very large float)
    time_limit = args.time_limit if args.time_limit is not None else float("inf")
    limit_str = f"{time_limit}s" if args.time_limit else "no limit"
    print(f"Time limit per instance: {limit_str}\n")

    # ---- run ----
    all_results: list[list[dict]] = []
    instance_meta: list[dict] = []

    for i in range(args.n_instances):
        ecole_inst = next(gen)
        print(f"--- Instance {i+1}/{args.n_instances} "
              f"({args.n_vars}×{args.n_cons}) ---")

        abc = extract_Abc(ecole_inst)
        if abc is None:
            print("  [SKIP] Could not extract A, b, c from instance.")
            continue
        A, b, c = abc
        instance_meta.append({"index": i, "n_vars": args.n_vars,
                               "n_cons": args.n_cons})
        inst_results = []

        # classical HiGHS
        for method in classical_methods:

            print(f"  {method.upper():<8}", end=" ", flush=True)
            r = solve_classical(A, b, c, method, args.time_limit)
            tag = f"{r['wall_time']:.1f}s" + (" ✓" if r["solved"] else " DNF")
            print(tag)
            inst_results.append(r)

        # SCIP
        if not args.skip_scip:
            print(f"  SCIP    ", end=" ", flush=True)
            r = solve_with_scip(ecole_inst, args.time_limit)
            tag = (f"{r['wall_time']:.1f}s" if r["wall_time"] else "error")
            tag += " ✓" if r["solved"] else " DNF"
            print(tag)
            inst_results.append(r)

        # CutWorld
        print(f"  CutWorld", end=" ", flush=True)
        r = solve_neural(A, b, c, model, args.time_limit, device)
        tag = f"{r['wall_time']:.1f}s" + (" ✓" if r["solved"] else " DNF")
        print(tag)
        inst_results.append(r)

        all_results.append(inst_results)

    # ---- summary ----
    print_summary(all_results, all_method_labels,
                  len(all_results), time_limit)

    # ---- save ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "config": vars(args),
        "instances": instance_meta,
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
