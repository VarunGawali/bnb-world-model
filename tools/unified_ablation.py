#!/usr/bin/env python
"""
unified_ablation.py — Full CutWorld model (branching + cuts) in one harness.

This is the only experiment where the model's BRANCHING and CUT SELECTION run
together in the same solver, directly supporting the "unified control" claim.
It is complementary to bnb_wm.evaluate.ablation (branching inside SCIP at scale):
here everything runs in our own neural Branch-and-Cut solver on small instances,
so we report NODE COUNTS, OPTIMALITY, and CUTS -- not wall-clock vs SCIP, which
would be unfair to a pure-Python prototype against an industrial C solver.

Every configuration draws branching and cuts from the same trained model and is
verified to reach the true optimum (SCIP oracle). The 2x2 core isolates each
component and their composition:

    branch \\ cut     none            learned
    policy           baseline        cuts only
    rollout          branching only  full CutWorld

Two classical branching baselines (most-fractional, no cuts) are included for
reference.

Run (varun env):
    python tools/unified_ablation.py --checkpoint checkpoints/model_final.pt \
        --n_instances 30 --n_sets 60 --n_elems 40 --max_cuts 1
"""

import argparse
import numpy as np
import torch

from bnb_wm.model.world_model import BnBWorldModel
from bnb_wm.solver.bnb_solver import BnBSolver


# (name, branch_mode, cut_mode)
CONFIGS = [
    ("most_frac / none",   "most_fractional", "none"),
    ("policy   / none",    "policy",          "none"),
    ("policy   / learned", "policy",          "learned"),
    ("rollout  / none",    "rollout",         "none"),
    ("rollout  / learned", "rollout",         "learned"),
]


def make_set_cover(rng, n_sets, n_elems, density=0.15):
    A = (rng.random((n_elems, n_sets)) < density).astype(np.float64)
    for i in range(n_elems):
        if A[i].sum() == 0:
            A[i, rng.integers(n_sets)] = 1.0
    c = rng.uniform(1.0, 10.0, size=n_sets)
    b = np.ones(n_elems, dtype=np.float64)
    return A, b, c


def solve_scip(A, b, c):
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
    return m.getStatus(), (m.getObjVal() if m.getStatus() == "optimal" else np.inf), \
        int(m.getNNodes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n_instances", type=int, default=30)
    ap.add_argument("--n_sets", type=int, default=60)
    ap.add_argument("--n_elems", type=int, default=40)
    ap.add_argument("--density", type=float, default=0.15)
    ap.add_argument("--time_limit", type=float, default=60.0)
    ap.add_argument("--max_cuts", type=int, default=1)
    ap.add_argument("--cut_threshold", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import yaml
    mc = yaml.safe_load(open(args.config))["model"]
    model = BnBWorldModel(
        hidden_dim=mc["hidden_dim"], n_gnn_layers=mc["n_gnn_layers"],
        n_gnn_heads=mc["n_gnn_heads"], n_dyn_layers=mc["n_dyn_layers"],
        n_dyn_heads=mc["n_dyn_heads"], max_seq=mc["max_seq"],
    ).to(device).eval()
    if args.checkpoint:
        from bnb_wm.training.checkpoint import load_weights_only
        load_weights_only(model, args.checkpoint, device=device)
        print(f"Loaded {args.checkpoint}")

    def make_solver(branch_mode, cut_mode):
        s = BnBSolver(
            model, device, time_limit=args.time_limit,
            max_cuts_per_node=args.max_cuts, cut_score_threshold=args.cut_threshold,
            node_selection="bound", cut_mode=cut_mode,
        )
        s.branch_mode = branch_mode
        return s

    solvers = [(name, make_solver(bm, cm)) for name, bm, cm in CONFIGS]
    rng = np.random.default_rng(args.seed)

    nodes = {name: [] for name, _, _ in CONFIGS}
    opt   = {name: [] for name, _, _ in CONFIGS}
    cutn  = {name: [] for name, _, _ in CONFIGS}
    scip_nodes = []
    scip_opt = []

    for i in range(args.n_instances):
        A, b, c = make_set_cover(rng, args.n_sets, args.n_elems, args.density)
        s_status, s_obj, s_nd = solve_scip(A, b, c)
        if s_status != "optimal":
            continue
        scip_nodes.append(s_nd)
        scip_opt.append(True)
        for name, solver in solvers:
            res = solver.solve(A, b, c)
            ok = (res.solution is not None
                  and abs(res.objective - s_obj) <= 1e-4 * max(1.0, abs(s_obj)))
            nodes[name].append(res.n_nodes)
            opt[name].append(ok)
            cutn[name].append(solver._cuts_added)

    n = len(scip_nodes)
    print(f"\n{n} instances solved to optimality by SCIP.\n")
    print(f"{'config':<20}{'nodes(mean)':<13}{'nodes(med)':<12}"
          f"{'%opt':<8}{'cuts':<8}")
    print("-" * 61)
    print(f"{'SCIP (reference)':<20}{np.mean(scip_nodes):<13.2f}"
          f"{np.median(scip_nodes):<12.1f}{'100':<8}{'--':<8}")
    for name, _, _ in CONFIGS:
        nd = np.array(nodes[name], dtype=float)
        pct = 100.0 * float(np.mean(opt[name]))
        print(f"{name:<20}{nd.mean():<13.2f}{np.median(nd):<12.1f}"
              f"{pct:<8.0f}{np.mean(cutn[name]):<8.2f}")
    print("-" * 61)
    print("Node counts are within OUR solver; SCIP row is a reference scale only "
          "(wall-clock is not comparable to a Python prototype).")
    bad = sum(1 for name, _, _ in CONFIGS if any(not o for o in opt[name]))
    if bad:
        print(f"WARNING: {bad} config(s) missed the optimum on some instance.")
    else:
        print("All configs reached the true optimum on every instance.")


if __name__ == "__main__":
    main()
