#!/usr/bin/env python
"""
cut_ablation.py — Learned cut selection vs. heuristic vs. none, inside our own
neural Branch-and-Cut solver (Option A).

Now that solver/gomory.py generates provably valid cuts (verified by
check_solver_correctness.py), a cut ablation inside bnb_solver.py is a sound
experiment. We solve a common set of Set Cover instances three ways and report
nodes-to-optimality, solve time, and cuts added:

    none       no cuts (pure branch-and-bound)
    heuristic  max-violation selection over the valid Gomory pool
    learned    CuttingPlaneHead selection over the same pool

Because every mode uses the SAME valid cut pool, the comparison isolates the
value of the LEARNED selection rule. Correctness is guaranteed by the generator,
and we additionally assert each solve matches SCIP's optimum.

Run (varun env: torch + torch_geometric + pyscipopt + scipy + highspy):

    python tools/cut_ablation.py --checkpoint checkpoints_warm/model_final.pt \
        --n_instances 30 --n_sets 60 --n_elems 40

Use modest instance sizes: bnb_solver.py is a pure-Python LP-based solver, so
keep n_sets/n_elems small enough that every instance closes to optimality.
"""

import argparse
import numpy as np
import torch

from bnb_wm.model.world_model import BnBWorldModel
from bnb_wm.solver.bnb_solver import BnBSolver


MODES = ["none", "heuristic", "learned"]


def make_set_cover(rng, n_sets, n_elems, density=0.15):
    """Random feasible Set Cover: min c^T x s.t. A x >= 1, x in {0,1}^n_sets."""
    A = (rng.random((n_elems, n_sets)) < density).astype(np.float64)
    for i in range(n_elems):
        if A[i].sum() == 0:
            A[i, rng.integers(n_sets)] = 1.0
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
    ap.add_argument("--checkpoint", default=None,
                    help="Phase-5 checkpoint for the learned cut head; if omitted "
                         "an untrained head is used (heuristic/none still valid)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n_instances", type=int, default=30)
    ap.add_argument("--n_sets", type=int, default=60)
    ap.add_argument("--n_elems", type=int, default=40)
    ap.add_argument("--density", type=float, default=0.15)
    ap.add_argument("--time_limit", type=float, default=60.0)
    ap.add_argument("--max_cuts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import yaml
    mcfg = yaml.safe_load(open(args.config))["model"]
    model = BnBWorldModel(
        hidden_dim=mcfg["hidden_dim"], n_gnn_layers=mcfg["n_gnn_layers"],
        n_gnn_heads=mcfg["n_gnn_heads"], n_dyn_layers=mcfg["n_dyn_layers"],
        n_dyn_heads=mcfg["n_dyn_heads"], max_seq=mcfg["max_seq"],
    ).to(device).eval()
    if args.checkpoint:
        from bnb_wm.training.checkpoint import load_weights_only
        load_weights_only(model, args.checkpoint, device=device)
        print(f"Loaded {args.checkpoint}")
    else:
        print("No checkpoint: learned head is UNTRAINED (baseline sanity only)")

    def make_solver(mode):
        return BnBSolver(
            model, device, time_limit=args.time_limit,
            max_cuts_per_node=args.max_cuts, cut_score_threshold=0.3,
            node_selection="bound", cut_mode=mode,
        )

    solvers = {m: make_solver(m) for m in MODES}
    rng = np.random.default_rng(args.seed)

    nodes = {m: [] for m in MODES}
    times = {m: [] for m in MODES}
    cuts  = {m: [] for m in MODES}
    n_bad = 0

    print(f"\n{'inst':<6}" + "".join(f"{m+'_nd':<10}" for m in MODES)
          + f"{'scip_obj':<10}{'ok':<5}")
    print("-" * 60)
    for i in range(args.n_instances):
        A, b, c = make_set_cover(rng, args.n_sets, args.n_elems, args.density)
        s_status, s_obj, _ = solve_scip(A, b, c)
        if s_status != "optimal":
            continue

        row_ok = True
        for m in MODES:
            res = solvers[m].solve(A, b, c)
            # correctness guard: every mode must still reach the true optimum
            ok = (res.solution is not None
                  and abs(res.objective - s_obj) <= 1e-4 * max(1.0, abs(s_obj)))
            row_ok = row_ok and ok
            nodes[m].append(res.n_nodes)
            times[m].append(res.solve_time)
            cuts[m].append(solvers[m]._cuts_added)
        if not row_ok:
            n_bad += 1
        print(f"{i:<6}" + "".join(f"{nodes[m][-1]:<10}" for m in MODES)
              + f"{s_obj:<10.3f}{'Y' if row_ok else 'N!':<5}")

    print("-" * 60)
    print(f"{'MODE':<12}{'nodes(mean)':<14}{'nodes(med)':<12}"
          f"{'time(s)':<10}{'cuts':<8}")
    for m in MODES:
        nd = np.array(nodes[m], dtype=float)
        print(f"{m:<12}{nd.mean():<14.2f}{np.median(nd):<12.1f}"
              f"{np.mean(times[m]):<10.3f}{np.mean(cuts[m]):<8.2f}")

    if n_bad:
        print(f"\nWARNING: {n_bad} instance(s) did NOT match SCIP's optimum. "
              f"Investigate before trusting these numbers.")
    else:
        print(f"\nAll {len(nodes['none'])} instances solved to the true optimum "
              f"in every mode. Comparison is valid.")
        base = np.array(nodes["none"], dtype=float).mean()
        for m in ("heuristic", "learned"):
            mm = np.array(nodes[m], dtype=float).mean()
            red = 100.0 * (base - mm) / max(base, 1e-9)
            print(f"  {m:<10} vs none: {red:+.1f}% nodes "
                  f"({'fewer=better' if red > 0 else 'MORE nodes'})")


if __name__ == "__main__":
    main()
