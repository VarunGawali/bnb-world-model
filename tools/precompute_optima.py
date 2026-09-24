"""
precompute_optima.py — Solve instances to proven optimality and cache results.

Uses SCIP (via pyscipopt) with generous time/node limits to get reference
optima for benchmark scoring.  Timed-out instances are stored with obj=None
so the benchmark can still score gap@end on them.

Usage
-----
    # Easy (100×200) — finishes quickly
    PYTHONPATH=. python tools/precompute_optima.py \
        --n_rows 100 --n_cols 200 --n_instances 20 --seed 1 \
        --time_limit 600 --out results/optima_100x200_s1.json

    # Medium (500×1000)
    PYTHONPATH=. python tools/precompute_optima.py \
        --n_rows 500 --n_cols 1000 --n_instances 15 --seed 1 \
        --time_limit 600 --out results/optima_500x1000_s1.json

    # Hard (1000×2000)
    PYTHONPATH=. python tools/precompute_optima.py \
        --n_rows 1000 --n_cols 2000 --n_instances 10 --seed 1 \
        --time_limit 900 --out results/optima_1000x2000_s1.json

Output format
-------------
{
  "config": {...},
  "instances": {
    "seed=1/0": {"obj": 61.588, "status": "optimal", "nodes": 5, "time": 0.2},
    "seed=1/1": {"obj": null,   "status": "timelimit", ...},
    ...
  }
}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def gen_instance(n_rows, n_cols, rng, density=0.05):
    A = (rng.random((n_rows, n_cols)) < density).astype(np.float64)
    for i in range(n_rows):
        if A[i].sum() == 0:
            A[i, rng.integers(n_cols)] = 1.0
    c = rng.uniform(1.0, 10.0, n_cols)
    b = np.ones(n_rows, dtype=np.float64)
    return A, b, c


def solve_scip(A, b, c, time_limit):
    """Returns (obj_or_None, status, n_nodes, wall_time, lb)."""
    try:
        from pyscipopt import Model
    except ImportError:
        raise RuntimeError("pyscipopt not installed; run: pip install pyscipopt")

    m_rows, n_cols = A.shape
    m = Model()
    m.hideOutput(True)
    m.setParam("limits/time", time_limit)
    m.setParam("limits/nodes", 10_000_000)

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

    obj = None
    try:
        if status in ("optimal", "infeasible"):
            pass
        obj = float(m.getObjVal())
    except Exception:
        pass

    lb = None
    try:
        lb = float(m.getDualbound())
    except Exception:
        pass

    return obj, status, nodes, wall, lb


def solve_highs(A, b, c, time_limit):
    """Fallback: use our own ClassicalBnBSolver with full strong branching."""
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
        node_limit=10_000_000,
        exact=True,
    )
    solver = ClassicalBnBSolver(config=cfg, sb_init=None, eta=1)
    t0 = time.perf_counter()
    r = solver.solve(A, b, c)
    wall = time.perf_counter() - t0
    obj = r.objective if r.objective < 1e18 else None
    return obj, r.status, r.n_nodes, wall, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_rows",      type=int,   default=100)
    ap.add_argument("--n_cols",      type=int,   default=200)
    ap.add_argument("--n_instances", type=int,   default=20)
    ap.add_argument("--seed",        type=int,   default=1)
    ap.add_argument("--density",     type=float, default=0.05)
    ap.add_argument("--time_limit",  type=float, default=600.0,
                    help="Per-instance time limit in seconds.")
    ap.add_argument("--out",         required=True)
    ap.add_argument("--solver",      default="scip",
                    choices=["scip", "highs"],
                    help="'scip' uses pyscipopt; 'highs' uses ClassicalBnBSolver "
                         "(full SB, slower but no extra dependency).")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    instances = []
    for _ in range(args.n_instances):
        instances.append(gen_instance(args.n_rows, args.n_cols, rng, args.density))

    print(f"Solving {args.n_instances} instances "
          f"({args.n_rows}×{args.n_cols}, seed={args.seed}) "
          f"with {args.solver}, tl={args.time_limit}s each\n")

    records: dict = {}
    n_optimal = 0

    for i, (A, b, c) in enumerate(instances):
        key = f"seed={args.seed}/{i}"
        t0 = time.perf_counter()

        try:
            if args.solver == "scip":
                obj, status, nodes, wall, lb = solve_scip(A, b, c, args.time_limit)
            else:
                obj, status, nodes, wall, lb = solve_highs(A, b, c, args.time_limit)
        except Exception as e:
            print(f"  [{i+1}/{args.n_instances}] ERROR: {e}")
            records[key] = {"obj": None, "status": "error", "nodes": 0,
                            "time": 0.0, "lb": None}
            continue

        is_opt = status in ("optimal",) or (
            obj is not None and lb is not None
            and abs(obj - lb) < 1e-4 * max(abs(obj), 1.0)
        )
        if is_opt:
            n_optimal += 1

        records[key] = {
            "obj": obj,
            "status": status,
            "nodes": nodes,
            "time": wall,
            "lb": lb,
        }

        obj_s = f"{obj:.4f}" if obj is not None else "N/A"
        lb_s  = f"{lb:.4f}" if lb is not None else "N/A"
        flag  = "✓" if is_opt else "~"
        print(f"  {flag} [{i+1:>3}/{args.n_instances}]  "
              f"obj={obj_s}  lb={lb_s}  status={status:>10}  "
              f"nodes={nodes:>7}  t={wall:.1f}s")

    print(f"\nOptimal: {n_optimal}/{args.n_instances}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "config": vars(args),
            "n_optimal": n_optimal,
            "instances": records,
        }, f, indent=2)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
