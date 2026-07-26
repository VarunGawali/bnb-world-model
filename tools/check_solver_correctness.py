#!/usr/bin/env python
"""
check_solver_correctness.py — Does bnb_solver.py reach the TRUE optimum?

This is the go/no-go gate for evaluating learned cut selection inside our own
neural Branch-and-Cut solver (Option A). Node counts from bnb_solver.py are only
meaningful if the solver actually solves each instance to optimality; otherwise
any cut-ablation numbers are noise.

Design
------
Correctness of the B&B is INDEPENDENT of the model weights: the model only guides
which variable to branch on and which (globally valid) cuts to add — search order
and valid cuts — never the bound-pruning or integrality tests that guarantee the
optimum. So we deliberately use a FRESH, UNTRAINED model. This isolates
"is the solver correct?" from "is the model any good?". If the optimum is wrong
with random guidance, it is a solver bug, not a training problem.

Baseline: SCIP via pyscipopt solves the same instance exactly.

We generate small random Set Cover instances (min c^T x s.t. A x >= 1, x binary),
solve each with both, and compare objectives.

Run (on a machine with torch + torch_geometric + pyscipopt + scipy; the `varun`
env after the pip installs has all four):

    python tools/check_solver_correctness.py --n_instances 20 --n_sets 24 --n_elems 16

A clean result is "MATCH" on every feasible instance (objectives equal within
tolerance). Any "MISMATCH" means the solver does not find the true optimum and
Option A is blocked until it is fixed.
"""

import argparse
import numpy as np
import torch

from bnb_wm.model.world_model import BnBWorldModel
from bnb_wm.solver.bnb_solver import BnBSolver


def make_set_cover(rng, n_sets, n_elems, density=0.25):
    """Random feasible Set Cover: A [n_elems, n_sets] 0/1, b=1, c>0.

    min c^T x  s.t.  A x >= 1,  x in {0,1}^n_sets
    Each element must be coverable, so we guarantee >=1 set covers each element.
    """
    A = (rng.random((n_elems, n_sets)) < density).astype(np.float64)
    # Guarantee feasibility: every element covered by at least one set.
    for i in range(n_elems):
        if A[i].sum() == 0:
            A[i, rng.integers(n_sets)] = 1.0
    # Guarantee every set is non-empty-ish is not required; costs are positive.
    c = rng.uniform(1.0, 10.0, size=n_sets)
    b = np.ones(n_elems, dtype=np.float64)
    return A, b, c


def solve_scip(A, b, c):
    """Exact Set Cover objective via SCIP (pyscipopt)."""
    from pyscipopt import Model, quicksum
    m = Model()
    m.hideOutput()
    n = len(c)
    x = [m.addVar(vtype="B", name=f"x{j}") for j in range(n)]
    for i in range(len(b)):
        cover = [x[j] for j in range(n) if A[i, j] > 0.5]
        m.addCons(quicksum(cover) >= b[i])
    m.setObjective(quicksum(c[j] * x[j] for j in range(n)), "minimize")
    m.optimize()
    status = m.getStatus()
    obj = m.getObjVal() if status == "optimal" else float("inf")
    return status, obj, int(m.getNNodes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_instances", type=int, default=20)
    ap.add_argument("--n_sets", type=int, default=24)
    ap.add_argument("--n_elems", type=int, default=16)
    ap.add_argument("--density", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time_limit", type=float, default=60.0)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Fresh, UNTRAINED model — correctness must not depend on weights.
    model = BnBWorldModel().to(device).eval()

    solver = BnBSolver(
        model, device,
        time_limit=args.time_limit,
        # Neutral guidance settings; correctness is independent of these.
        lookahead_k=3, lookahead_depth=2, branch_factor=1,
        node_selection="bound",
    )

    rng = np.random.default_rng(args.seed)
    n_match = n_mismatch = n_skip = 0
    print(f"{'inst':<6}{'scip_obj':<12}{'bnb_obj':<12}{'bnb_status':<12}"
          f"{'bnb_nodes':<10}{'verdict':<10}")
    print("-" * 62)

    for i in range(args.n_instances):
        A, b, c = make_set_cover(rng, args.n_sets, args.n_elems, args.density)
        s_status, s_obj, s_nodes = solve_scip(A, b, c)
        if s_status != "optimal":
            print(f"{i:<6}{'--':<12}{'--':<12}{'(scip ' + s_status + ')':<12}"
                  f"{'--':<10}{'SKIP':<10}")
            n_skip += 1
            continue

        res = solver.solve(A, b, c)
        b_obj = res.objective
        match = (res.status in ("optimal", "feasible")
                 and abs(b_obj - s_obj) <= args.tol * max(1.0, abs(s_obj)))
        verdict = "MATCH" if match else "MISMATCH"
        if match:
            n_match += 1
        else:
            n_mismatch += 1
        print(f"{i:<6}{s_obj:<12.4f}{b_obj:<12.4f}{res.status:<12}"
              f"{res.n_nodes:<10}{verdict:<10}")

    print("-" * 62)
    print(f"MATCH={n_match}  MISMATCH={n_mismatch}  SKIP={n_skip}  "
          f"(of {args.n_instances})")
    if n_mismatch == 0 and n_match > 0:
        print("\nGREEN LIGHT: solver reaches the true optimum. Option A (cut "
              "ablation inside bnb_solver.py) is viable.")
    else:
        print("\nBLOCKED: solver does not match SCIP's optimum on every "
              "instance. Fix the solver before trusting any node/cut numbers.")


if __name__ == "__main__":
    main()
