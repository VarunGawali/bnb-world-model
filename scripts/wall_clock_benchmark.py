"""
wall_clock_benchmark.py — Time-to-optimality benchmark with dual cutoffs.

Runs every method on the same hard-tier held-out instances. Two time cutoffs
are evaluated from a single run:
    - short_cutoff  (default 900s)  : practically useful methods
    - long_cutoff   (default 3600s) : generous limit, last chance for classical

If a method hits the long_cutoff the instance is marked DNF and we move on.

Methods:
    - SCIP default             (via pyscipopt, optional)
    - HiGHS + MF               (most-fractional)
    - HiGHS + SB-1 / SB-4 / SB-8
    - CutWorld Neural-Best-d1  (checkpoint)

Usage:
    # Full hard-tier run:
    python scripts/wall_clock_benchmark.py \\
        --checkpoint checkpoints/phase4_best.pt \\
        --n-instances 20 --n-vars 1000 --n-cons 2000 \\
        --short-cutoff 600 --long-cutoff 3600 \\
        --out results/wall_clock_hard.json

    # Pilot (3 instances, smaller size, quick cutoffs):
    python scripts/wall_clock_benchmark.py \\
        --checkpoint checkpoints/phase4_best.pt \\
        --n-instances 3 --n-vars 300 --n-cons 600 \\
        --short-cutoff 120 --long-cutoff 300 \\
        --skip-sb --out /tmp/pilot.json
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
# Extract (A, b, c) from ecole instance via pyscipopt
# ---------------------------------------------------------------------------

def extract_Abc(instance) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (A, b, c) for min c^T x  s.t.  Ax >= b,  x in {0,1}^n."""
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
# Solver wrappers
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
        return {
            "method": "SCIP",
            "solved": solved,
            "solve_time": elapsed if solved else None,
            "wall_time": elapsed,
            "nodes": int(scip.getNNodes()),
            "gap": float(scip.getGap() if not solved else 0.0),
        }
    except Exception as e:
        return {"method": "SCIP", "solved": False, "solve_time": None,
                "wall_time": None, "nodes": None, "gap": None, "error": str(e)}


def solve_classical(A, b, c, method: str, time_limit: float) -> dict:
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


def solve_neural(A, b, c, model, time_limit: float,
                 device: torch.device) -> dict:
    cfg = SolverConfig(
        time_limit=time_limit,
        use_neural=True,
        use_cuts=True,
        rollout_depth=1,
        top_k=5,
        exact=True,
        ctg_weight=0.0,
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
# SGM
# ---------------------------------------------------------------------------

def sgm(values: list, shift: float) -> float:
    vals = [v for v in values if v is not None]
    if not vals:
        return float("nan")
    return float(np.exp(np.mean(np.log([v + shift for v in vals]))) - shift)


# ---------------------------------------------------------------------------
# Summary table — printed for each cutoff
# ---------------------------------------------------------------------------

def print_summary(all_results: list[list[dict]], methods: list[str],
                  cutoff: float, label: str):
    n_inst = len(all_results)
    print(f"\n{'='*72}")
    print(f"CUTOFF = {cutoff}s  [{label}]   ({n_inst} instances)")
    print(f"{'='*72}")
    hdr = (f"{'Method':<18} {'Solved':>9}  {'SGM Time(s)':>12}  "
           f"{'Median(s)':>10}  {'SGM Nodes':>10}  {'Mean Gap':>9}")
    print(hdr)
    print("-" * 72)

    for method in methods:
        # For this cutoff: a result counts as solved only if solve_time <= cutoff
        rows = [r for inst in all_results for r in inst
                if r["method"] == method]
        n = len(rows)
        solved_rows = [r for r in rows
                       if r["solved"] and r["solve_time"] is not None
                       and r["solve_time"] <= cutoff]
        n_solved = len(solved_rows)

        solve_times = [r["solve_time"] for r in solved_rows]
        nodes       = [r["nodes"] for r in solved_rows
                       if r["nodes"] is not None]
        gaps        = [r["gap"] for r in rows if r["gap"] is not None]

        sgm_t  = sgm(solve_times, shift=1.0)
        med_t  = float(np.median(solve_times)) if solve_times else float("nan")
        sgm_n  = sgm(nodes, shift=10.0)
        mean_g = float(np.mean(gaps) * 100) if gaps else float("nan")

        dnf = f"  ({n - n_solved} DNF)" if n_solved < n else ""
        print(f"{method:<18} {n_solved:>4}/{n}{dnf:<10}"
              f"{sgm_t:>12.1f}  {med_t:>10.1f}  "
              f"{sgm_n:>10.0f}  {mean_g:>8.1f}%")

    print("=" * 72)
    print("SGM over solved-within-cutoff instances only. Shift: time=1, nodes=10.")
    print("DNF = did not prove optimality within this cutoff.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Time-to-optimality benchmark with dual cutoffs")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-instances", type=int, default=20)
    ap.add_argument("--n-vars",  type=int, default=1000,
                    help="Variables per instance (default: hard tier)")
    ap.add_argument("--n-cons",  type=int, default=2000,
                    help="Constraints per instance (default: hard tier)")
    ap.add_argument("--short-cutoff", type=float, default=900.0,
                    help="Short cutoff in seconds (default 900s / 15 min)")
    ap.add_argument("--long-cutoff",  type=float, default=3600.0,
                    help="Long cutoff in seconds — hard ceiling, "
                         "instance skipped if any method exceeds this "
                         "(default 3600s / 1 hour)")
    ap.add_argument("--seed-offset", type=int, default=99999,
                    help="RNG seed base — keep high to avoid training overlap")
    ap.add_argument("--skip-scip", action="store_true")
    ap.add_argument("--skip-sb",   action="store_true",
                    help="Skip SB-1/4/8 (faster pilot)")
    ap.add_argument("--out", default="results/wall_clock.json")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    assert args.short_cutoff < args.long_cutoff, \
        "--short-cutoff must be less than --long-cutoff"

    # ---- device & model ----
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    print(f"Device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = BnBWorldModel(hidden_dim=128, n_gnn_layers=3).to(device)
    load_weights_only(model, args.checkpoint, device)
    model.eval()
    print("Model ready.")

    print(f"\nInstance size : {args.n_vars}×{args.n_cons} Set Cover")
    print(f"Short cutoff  : {args.short_cutoff}s")
    print(f"Long cutoff   : {args.long_cutoff}s  (hard ceiling — skip instance if hit)")
    print(f"N instances   : {args.n_instances}\n")

    # ---- ecole generator ----
    try:
        import ecole
    except ImportError:
        raise SystemExit("ecole required: pip install ecole")

    gen = ecole.instance.SetCoverGenerator(
        n_rows=args.n_cons, n_cols=args.n_vars, density=0.05)
    gen.seed(args.seed_offset)

    # ---- method list ----
    classical_methods = ["mf"] + ([] if args.skip_sb else ["sb1", "sb4", "sb8"])
    all_labels = ([m.upper() for m in classical_methods]
                  + (["SCIP"] if not args.skip_scip else [])
                  + ["CutWorld"])

    # ---- run ----
    all_results:   list[list[dict]] = []
    instance_meta: list[dict]       = []
    n_skipped = 0

    for i in range(args.n_instances):
        ecole_inst = next(gen)
        print(f"--- Instance {i+1}/{args.n_instances} "
              f"({args.n_vars}×{args.n_cons}) ---")

        abc = extract_Abc(ecole_inst)
        if abc is None:
            print("  [SKIP] Could not extract A,b,c.")
            n_skipped += 1
            continue
        A, b, c = abc
        inst_results = []
        skip_instance = False

        # Classical HiGHS
        for method in classical_methods:
            print(f"  {method.upper():<8}", end=" ", flush=True)
            r = solve_classical(A, b, c, method, args.long_cutoff)
            tag = f"{r['wall_time']:.1f}s" + (" ✓" if r["solved"] else " DNF")
            print(tag)
            inst_results.append(r)
            # If any method blows through the long cutoff we still continue —
            # we just mark it DNF. We never skip the instance mid-loop.

        # SCIP
        if not args.skip_scip:
            print(f"  SCIP    ", end=" ", flush=True)
            r = solve_with_scip(ecole_inst, args.long_cutoff)
            wt = r.get("wall_time")
            tag = (f"{wt:.1f}s" if wt is not None else "error")
            tag += " ✓" if r["solved"] else " DNF"
            print(tag)
            inst_results.append(r)

        # CutWorld
        print(f"  CutWorld", end=" ", flush=True)
        r = solve_neural(A, b, c, model, args.long_cutoff, device)
        tag = f"{r['wall_time']:.1f}s" + (" ✓" if r["solved"] else " DNF")
        print(tag)
        inst_results.append(r)

        all_results.append(inst_results)
        instance_meta.append({"index": i, "n_vars": args.n_vars,
                               "n_cons": args.n_cons})

    if n_skipped:
        print(f"\n[{n_skipped} instances skipped due to extraction errors]")

    # ---- summary at both cutoffs ----
    print_summary(all_results, all_labels,
                  args.short_cutoff, "SHORT CUTOFF")
    print_summary(all_results, all_labels,
                  args.long_cutoff,  "LONG CUTOFF")

    # ---- save ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "config": vars(args),
            "instances": instance_meta,
            "results": all_results,
        }, f, indent=2)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
