"""
check_solvers.py — Gate 2 + Gate 3 correctness checks.

Runs NeuralBnBSolver and ClassicalBnBSolver on small random set-cover
instances and verifies each returns the brute-force optimal.

Usage
-----
    # Gate 3 only (no checkpoint needed)
    python tools/check_solvers.py --classical_only

    # Gates 2 + 3
    python tools/check_solvers.py --checkpoint checkpoints/phase4_best.pt

Options
-------
    --checkpoint PATH   model checkpoint (required for neural check)
    --classical_only    skip neural solver
    --n_instances N     instances per size (default 20)
    --seed S            RNG seed (default 0)
    --sizes WxH         comma-separated "rows x cols" (default 8x15,10x20,12x25)
    --sb_variants       also check sb0 / sb1 / sbfull classical variants
    --device DEVICE     cuda / cpu (default cpu)
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Brute-force optimal via exhaustive search (tiny instances only)
# ---------------------------------------------------------------------------

def brute_force_optimal(A, b, c) -> float:
    """Enumerate all 2^n binary vectors, return minimum feasible cost."""
    m, n = A.shape
    assert n <= 25, "brute_force only for n<=25"
    best = np.inf
    for mask in range(1 << n):
        x = np.array([(mask >> j) & 1 for j in range(n)], dtype=np.float64)
        if np.all(A @ x >= b - 1e-9):
            obj = float(c @ x)
            if obj < best:
                best = obj
    return best


# ---------------------------------------------------------------------------
# Instance generator (training distribution)
# ---------------------------------------------------------------------------

def gen_instance(n_rows, n_cols, rng, density=0.5):
    """
    Small dense instances so brute-force is tractable.
    density=0.4 ensures most are feasible with a few variables.
    """
    A = (rng.random((n_rows, n_cols)) < density).astype(np.float64)
    # Ensure each row has at least one nonzero
    for i in range(n_rows):
        if A[i].sum() == 0:
            A[i, rng.integers(n_cols)] = 1.0
    c = rng.uniform(1.0, 10.0, n_cols)
    b = np.ones(n_rows, dtype=np.float64)
    return A, b, c


# ---------------------------------------------------------------------------
# Load neural solver
# ---------------------------------------------------------------------------

def load_neural_solver(checkpoint_path, device_str):
    import torch
    from bnb_wm.model.bnb_world_model import BnBWorldModel
    from bnb_wm.solver.neural_bnb import NeuralBnBSolver
    from bnb_wm.solver.config import SolverConfig
    from bnb_wm.training.checkpoint import load_weights_only

    device = torch.device(device_str)
    model = BnBWorldModel().to(device)
    load_weights_only(model, checkpoint_path, device=device)
    model.eval()

    cfg = SolverConfig(
        branch_mode="rollout",
        cut_mode="none",
        ors_cascade=False,
        katz_weight=0.0,
        node_selection="bound",
        time_limit=60.0,
        node_limit=50_000,
        exact=True,
    )
    return NeuralBnBSolver(model, device, cfg)


# ---------------------------------------------------------------------------
# Check one solver on N instances
# ---------------------------------------------------------------------------

def check_solver(name, solver_fn, instances, tol=1e-4):
    """
    solver_fn(A, b, c) -> result with .objective and .status
    Returns (n_pass, n_fail, failures[])
    """
    n_pass = n_fail = 0
    failures = []

    for idx, (A, b, c, opt) in enumerate(instances):
        result = solver_fn(A, b, c)

        obj = result.objective
        status = result.status

        gap = abs(obj - opt) / (abs(opt) + 1e-10) if np.isfinite(opt) else np.inf

        ok = (
            status in ("optimal", "feasible")
            and np.isfinite(obj)
            and gap < tol
        )

        if ok:
            n_pass += 1
        else:
            n_fail += 1
            failures.append({
                "idx": idx,
                "status": status,
                "obj": obj,
                "opt": opt,
                "gap": gap,
                "shape": A.shape,
            })

    return n_pass, n_fail, failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_sizes(s):
    out = []
    for part in s.split(","):
        r, c = part.strip().split("x")
        out.append((int(r), int(c)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--classical_only", action="store_true")
    ap.add_argument("--n_instances", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sizes", default="6x12,7x14,8x15")
    ap.add_argument("--sb_variants", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    sizes = parse_sizes(args.sizes)
    rng = np.random.default_rng(args.seed)

    from bnb_wm.solver.classical_bnb import ClassicalBnBSolver
    from bnb_wm.solver.config import SolverConfig

    all_pass = True

    for n_rows, n_cols in sizes:
        print(f"\n{'='*60}")
        print(f"Size {n_rows}×{n_cols}  ({args.n_instances} instances)")
        print(f"{'='*60}")

        # Generate instances + precompute optima
        instances = []
        t_bf = time.perf_counter()
        for _ in range(args.n_instances):
            A, b, c = gen_instance(n_rows, n_cols, rng)
            opt = brute_force_optimal(A, b, c)
            instances.append((A, b, c, opt))
        print(f"  Brute-force optima computed in {time.perf_counter()-t_bf:.1f}s")

        # ---- Classical variants ----
        base_cfg = SolverConfig(
            branch_mode="most_fractional",
            cut_mode="none",
            ors_cascade=False,
            katz_weight=0.0,
            node_selection="bound",
            primal_heuristic=True,
            time_limit=30.0,
            node_limit=200_000,
            exact=True,
        )

        variants = [("classical_sb4", 4)]
        if args.sb_variants:
            variants = [
                ("classical_sb0",    0),
                ("classical_sb1",    1),
                ("classical_sb4",    4),
                ("classical_sb8",    8),
                ("classical_sbfull", None),
            ]

        for vname, sb_init in variants:
            solver = ClassicalBnBSolver(config=base_cfg, sb_init=sb_init, eta=4)
            t0 = time.perf_counter()
            n_pass, n_fail, failures = check_solver(
                vname,
                lambda A, b, c, s=solver: s.solve(A, b, c),
                instances,
            )
            elapsed = time.perf_counter() - t0
            tag = "PASS" if n_fail == 0 else "FAIL"
            print(f"  [{tag}] {vname:22s}  pass={n_pass:3d}  fail={n_fail:3d}  "
                  f"time={elapsed:.1f}s")
            if failures:
                all_pass = False
                for f in failures[:3]:
                    print(f"         instance {f['idx']} shape={f['shape']} "
                          f"status={f['status']} obj={f['obj']:.4f} "
                          f"opt={f['opt']:.4f} gap={f['gap']:.2e}")

        # ---- Neural solver ----
        if not args.classical_only:
            if args.checkpoint is None:
                print("  [SKIP] neural: no --checkpoint provided")
            else:
                try:
                    neural = load_neural_solver(args.checkpoint, args.device)
                    t0 = time.perf_counter()
                    n_pass, n_fail, failures = check_solver(
                        "neural_rollout",
                        lambda A, b, c, s=neural: s.solve(A, b, c),
                        instances,
                    )
                    elapsed = time.perf_counter() - t0
                    tag = "PASS" if n_fail == 0 else "FAIL"
                    print(f"  [{tag}] {'neural_rollout':22s}  pass={n_pass:3d}  "
                          f"fail={n_fail:3d}  time={elapsed:.1f}s")
                    if failures:
                        all_pass = False
                        for f in failures[:3]:
                            print(f"         instance {f['idx']} shape={f['shape']} "
                                  f"status={f['status']} obj={f['obj']:.4f} "
                                  f"opt={f['opt']:.4f} gap={f['gap']:.2e}")
                except Exception as e:
                    print(f"  [ERR]  neural_rollout: {e}")
                    all_pass = False

    print(f"\n{'='*60}")
    if all_pass:
        print("ALL GATES PASSED")
        sys.exit(0)
    else:
        print("SOME GATES FAILED — see details above")
        sys.exit(1)


if __name__ == "__main__":
    main()
