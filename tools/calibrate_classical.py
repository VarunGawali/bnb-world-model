"""
calibrate_classical.py — Gate 4: verify classical_sb4 is within ~2x of SCIP.

Runs classical_sb4 and SCIP (presolve off, cuts off, pseudocost branching)
on the same collector-distribution instances and prints per-instance node
counts plus the shifted geometric mean ratio.

If classical_sb4 / scip_nodes SGM > 4, the baseline is too weak — stop and
investigate before running the full ablation.

Usage
-----
    PYTHONPATH=. python tools/calibrate_classical.py \
        --n_rows 100 --n_cols 200 --n_instances 20 --seed 1 \
        --out results/calibration.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bnb_wm.solver.classical_bnb import ClassicalBnBSolver
from bnb_wm.solver.config import SolverConfig


# ---------------------------------------------------------------------------
# Instance generator (training distribution)
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
# SCIP solve via PySCIPOpt
# ---------------------------------------------------------------------------

def scip_solve(A, b, c, time_limit=120.0):
    """
    Solve with SCIP, presolve off, cuts off, pseudocost branching.
    Returns (nodes, obj, status, time).
    """
    try:
        from pyscipopt import Model
    except ImportError:
        return None, None, "scip_unavailable", 0.0

    m_rows, n_cols = A.shape
    m = Model()
    m.hideOutput(True)
    m.setParam("limits/time", time_limit)
    m.setParam("presolving/maxrounds", 0)
    m.setParam("separating/maxrounds", 0)
    m.setParam("separating/maxroundsroot", 0)
    m.setParam("branching/pscost/priority", 536870911)  # make pseudocost default

    xs = [m.addVar(f"x{j}", vtype="B", obj=float(c[j])) for j in range(n_cols)]
    for i in range(m_rows):
        nz = np.where(A[i] > 0)[0]
        m.addCons(sum(xs[j] for j in nz) >= 1.0, name=f"c{i}")
    m.setMinimize()

    t0 = time.perf_counter()
    m.optimize()
    elapsed = time.perf_counter() - t0

    status = m.getStatus()
    nodes = int(m.getNNodes())
    try:
        obj = float(m.getObjVal())
    except Exception:
        obj = None

    return nodes, obj, status, elapsed


# ---------------------------------------------------------------------------
# Shifted geometric mean
# ---------------------------------------------------------------------------

def sgm(values, shift=10.0):
    vals = np.array([v for v in values if v is not None and np.isfinite(v)],
                    dtype=np.float64)
    if len(vals) == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(np.maximum(vals, 0) + shift))) - shift)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_rows", type=int, default=100)
    ap.add_argument("--n_cols", type=int, default=200)
    ap.add_argument("--n_instances", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--density", type=float, default=0.05)
    ap.add_argument("--time_limit", type=float, default=120.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    cfg = SolverConfig(
        branch_mode="most_fractional",
        cut_mode="none",
        ors_cascade=False,
        katz_weight=0.0,
        node_selection="bound",
        primal_heuristic=True,
        time_limit=args.time_limit,
        node_limit=500_000,
        exact=True,
    )
    classical = ClassicalBnBSolver(config=cfg, sb_init=4, eta=4)

    rows = []
    print(f"\nCalibration: {args.n_rows}×{args.n_cols}, "
          f"n={args.n_instances}, seed={args.seed}")
    print(f"{'#':>3}  {'classical_sb4':>14}  {'scip_pscost':>12}  "
          f"{'ratio':>6}  {'cl_status':>10}  {'sc_status':>10}")
    print("-" * 70)

    for i in range(args.n_instances):
        A, b, c = gen_instance(args.n_rows, args.n_cols, rng, args.density)

        # Classical sb4
        t0 = time.perf_counter()
        cr = classical.solve(A, b, c)
        cl_time = time.perf_counter() - t0
        cl_nodes = cr.n_nodes
        cl_status = cr.status

        # SCIP pseudocost
        sc_nodes, sc_obj, sc_status, sc_time = scip_solve(A, b, c, args.time_limit)

        ratio = (cl_nodes / max(sc_nodes, 1)) if sc_nodes else None
        ratio_str = f"{ratio:.2f}" if ratio is not None else "  N/A"

        print(f"{i+1:>3}  {cl_nodes:>14}  "
              f"{str(sc_nodes) if sc_nodes is not None else 'N/A':>12}  "
              f"{ratio_str:>6}  {cl_status:>10}  {sc_status:>10}")

        rows.append({
            "instance": i,
            "classical_nodes": cl_nodes,
            "classical_status": cl_status,
            "classical_time": cl_time,
            "classical_tree_lps": cr.tree_lps,
            "classical_decision_lps": cr.decision_lps,
            "scip_nodes": sc_nodes,
            "scip_status": sc_status,
            "scip_time": sc_time,
            "ratio": ratio,
        })

    # Aggregate
    cl_sgm = sgm([r["classical_nodes"] for r in rows])
    sc_sgm = sgm([r["scip_nodes"] for r in rows if r["scip_nodes"] is not None])
    ratio_sgm = (cl_sgm / max(sc_sgm, 1)) if sc_sgm else None

    print("-" * 70)
    print(f"SGM (shift=10):  classical_sb4={cl_sgm:.1f}  "
          f"scip_pscost={sc_sgm:.1f}  ratio={ratio_sgm:.2f}" if ratio_sgm else
          f"SGM:  classical_sb4={cl_sgm:.1f}  scip_pscost=N/A")

    if ratio_sgm is not None:
        if ratio_sgm <= 4.0:
            print(f"\n[GATE 4 PASS] ratio {ratio_sgm:.2f} <= 4.0 — baseline is calibrated.")
        else:
            print(f"\n[GATE 4 FAIL] ratio {ratio_sgm:.2f} > 4.0 — classical is too weak; "
                  f"investigate before running ablation.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "config": vars(args),
                "instances": rows,
                "sgm": {"classical_sb4": cl_sgm, "scip_pscost": sc_sgm,
                        "ratio": ratio_sgm},
            }, f, indent=2)
        print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
