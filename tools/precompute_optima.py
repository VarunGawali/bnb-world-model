"""
precompute_optima.py — Solve instances to proven optimality offline.

Runs SCIP with no time limit (presolve on, all cuts on) on every instance
in the specified tier/seed combinations and caches results to JSON.
The benchmark script loads this cache to compute dist_from_optimum and
verify that exact solvers return the correct objective.

Usage
-----
    # Dev set (seed 1, n=20)
    PYTHONPATH=. python tools/precompute_optima.py \
        --n_rows 100 --n_cols 200 --n_instances 20 --seeds 1 \
        --out results/optima_100x200.json

    # Final set (seeds 0,2,3, n=50)
    PYTHONPATH=. python tools/precompute_optima.py \
        --n_rows 100 --n_cols 200 --n_instances 50 --seeds 0 2 3 \
        --out results/optima_100x200.json

    # Secondary tier
    PYTHONPATH=. python tools/precompute_optima.py \
        --n_rows 200 --n_cols 400 --n_instances 50 --seeds 0 2 3 \
        --out results/optima_200x400.json

Output format
-------------
{
  "config": {...},
  "instances": {
    "seed=1/0": {"obj": 12.34, "status": "optimal", "nodes": 5, "time": 0.3},
    "seed=1/1": {...},
    ...
  }
}

Keys are "seed=<s>/<i>" so multiple seeds live in one file and the
benchmark script can look up any (seed, index) pair.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Instance generator (training distribution — must match benchmark)
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
# SCIP exact solve
# ---------------------------------------------------------------------------

def scip_exact(A, b, c):
    """Solve to proven optimality with SCIP (all presolve and cuts enabled)."""
    from pyscipopt import Model

    m_rows, n_cols = A.shape
    m = Model()
    m.hideOutput(True)
    # No time limit — this is the offline oracle.
    # All presolve and separation enabled (SCIP defaults).

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

    sol = None
    if obj is not None:
        try:
            sol = [float(m.getVal(xs[j])) for j in range(n_cols)]
        except Exception:
            pass

    return {
        "obj": obj,
        "status": status,
        "nodes": nodes,
        "time": elapsed,
        "sol": sol,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_rows",      type=int,   default=100)
    ap.add_argument("--n_cols",      type=int,   default=200)
    ap.add_argument("--n_instances", type=int,   default=20)
    ap.add_argument("--seeds",       type=int,   nargs="+", default=[1])
    ap.add_argument("--density",     type=float, default=0.05)
    ap.add_argument("--out",         required=True)
    ap.add_argument("--resume",      action="store_true",
                    help="Load existing output and skip already-solved instances.")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results if resuming
    existing: dict = {}
    if args.resume and out_path.exists():
        with open(out_path) as f:
            data = json.load(f)
        existing = data.get("instances", {})
        print(f"Resuming: {len(existing)} instances already solved.")

    results: dict = dict(existing)

    total = len(args.seeds) * args.n_instances
    done = 0

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        for i in range(args.n_instances):
            key = f"seed={seed}/{i}"
            A, b, c = gen_instance(args.n_rows, args.n_cols, rng, args.density)

            if key in results:
                done += 1
                continue

            t0 = time.perf_counter()
            result = scip_exact(A, b, c)
            elapsed = time.perf_counter() - t0

            results[key] = result
            done += 1

            obj_str = f"{result['obj']:.4f}" if result["obj"] is not None else "None"
            print(f"[{done:>4}/{total}] {key:>14}  "
                  f"status={result['status']:>8}  "
                  f"obj={obj_str}  nodes={result['nodes']:>6}  "
                  f"time={elapsed:.1f}s")

            # Save incrementally so progress survives interruption
            with open(out_path, "w") as f:
                json.dump({
                    "config": vars(args),
                    "instances": results,
                }, f, indent=2)

    # Summary
    solved = [r for r in results.values() if r["status"] == "optimal"]
    objs = [r["obj"] for r in solved if r["obj"] is not None]
    times = [r["time"] for r in solved]
    print(f"\nDone. {len(solved)}/{total} optimal. "
          f"Obj: min={min(objs):.2f} max={max(objs):.2f} mean={np.mean(objs):.2f}. "
          f"Time: mean={np.mean(times):.1f}s max={max(times):.1f}s")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
