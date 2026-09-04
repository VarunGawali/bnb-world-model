"""
validate_cuts.py — does the learned cut head actually reduce B&B nodes?

Runs the standalone BnBSolver on the same instances under three cut modes and
compares NODE COUNTS (not wall-clock — the standalone solver's Python LP is not
comparable to SCIP's C LP, so time here is only a within-solver relative signal):

    none      : branching only (no cuts)
    learned   : the Phase-5 cut head selects cuts
    heuristic : rule-based cut selection (violation/efficacy)

If learned < none on nodes, the cut head helps and a full B&C-in-SCIP time test
is worth building. If not, that phase needs upgrading (the goal is fewer nodes ->
less total time). A correctness gate checks all modes reach the SAME optimum, so
we never trust a node count from a solve that cut off the optimal solution.

Usage
-----
    PYTHONPATH=. python scripts/validate_cuts.py \
        --checkpoint checkpoints/model_dagger_r1.pt \
        --n_instances 20 --n_rows 500 --n_cols 1000 --time_limit 120
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import ecole
import yaml
from bnb_wm.model.world_model import BnBWorldModel
from bnb_wm.training.checkpoint import load_weights_only
from bnb_wm.solver.bnb_solver import BnBSolver


def _extract_Abc(pyscip_model):
    """Set-cover (A x >= b, min c^T x) matrices from a pyscipopt Model."""
    vars_ = pyscip_model.getVars()
    n = len(vars_)
    name2col = {v.getName(): j for j, v in enumerate(vars_)}
    c = np.array([v.getObj() for v in vars_], dtype=np.float64)
    conss = pyscip_model.getConss()
    rows = []
    b = []
    for cons in conss:
        try:
            vals = pyscip_model.getValsLinear(cons)
        except Exception:
            continue
        row = np.zeros(n, dtype=np.float64)
        for vname, coeff in vals.items():
            if vname in name2col:
                row[name2col[vname]] = coeff
        lhs = pyscip_model.getLhs(cons)
        rows.append(row)
        b.append(lhs if np.isfinite(lhs) else 1.0)
    A = np.vstack(rows) if rows else np.zeros((0, n))
    b = np.array(b, dtype=np.float64)
    # normalise to cover form (A x >= b) if stored as <=
    nz = A[A != 0]
    if nz.size and float(np.nanmean(nz)) < 0:
        A, b = -A, -b
    return A, b, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n_instances", type=int, default=20)
    ap.add_argument("--n_rows", type=int, default=500)
    ap.add_argument("--n_cols", type=int, default=1000)
    ap.add_argument("--density", type=float, default=0.05)
    ap.add_argument("--time_limit", type=int, default=120)
    ap.add_argument("--node_limit", type=int, default=50000)
    ap.add_argument("--modes", default="none,learned,heuristic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/cut_validation.json")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(args.config))["model"]
    model = BnBWorldModel(
        hidden_dim=cfg["hidden_dim"], n_gnn_layers=cfg["n_gnn_layers"],
        n_gnn_heads=cfg["n_gnn_heads"], n_dyn_layers=cfg["n_dyn_layers"],
        n_dyn_heads=cfg["n_dyn_heads"], max_seq=cfg["max_seq"],
        dyn_residual=cfg.get("dyn_residual", True),
        dyn_heteroscedastic=cfg.get("dyn_heteroscedastic", False),
    ).to(device)
    load_weights_only(model, args.checkpoint, device=device)
    model.eval()
    trained = bool(model.cut_head_trained)
    print(f"Loaded {args.checkpoint} on {device} | cut_head_trained={trained}",
          flush=True)
    if not trained:
        print("WARNING: cut head not marked trained; 'learned' will fall back.",
              flush=True)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    gen = ecole.instance.SetCoverGenerator(
        n_rows=args.n_rows, n_cols=args.n_cols, density=args.density)
    gen.seed(args.seed)

    def make_solver(mode):
        return BnBSolver(model, device, time_limit=args.time_limit,
                         node_limit=args.node_limit, cut_mode=mode)

    records = []          # per instance: {mode: (nodes, time, status, obj)}
    for i in range(args.n_instances):
        inst = next(gen)
        try:
            A, b, c = _extract_Abc(inst.copy_orig().as_pyscipopt())
        except Exception as e:
            print(f"[{i+1}] extract failed: {e}", flush=True)
            continue
        rec = {}
        for mode in modes:
            try:
                res = make_solver(mode).solve(A, b, c)
                rec[mode] = dict(nodes=res.n_nodes, time=res.solve_time,
                                 status=res.status, obj=float(res.objective))
            except Exception as e:
                rec[mode] = dict(nodes=-1, time=-1, status=f"error:{e}", obj=np.nan)
        records.append(rec)
        line = " | ".join(
            f"{m}:{rec[m]['nodes']}n/{rec[m]['status'][:4]}" for m in modes)
        print(f"[{i+1}/{args.n_instances}] {line}", flush=True)

    # ---- correctness gate + summary over instances solved-optimal by ALL modes ----
    def solved_all(rec):
        return all(rec.get(m, {}).get("status") == "optimal" for m in modes)

    def obj_consistent(rec):
        objs = [rec[m]["obj"] for m in modes if m in rec]
        return max(objs) - min(objs) < 1e-4 if objs else False

    valid = [r for r in records if solved_all(r) and obj_consistent(r)]
    mismatched = [r for r in records if solved_all(r) and not obj_consistent(r)]
    print("\n" + "=" * 60, flush=True)
    print(f"{len(valid)}/{len(records)} instances solved-optimal by all modes "
          f"with matching objective", flush=True)
    if mismatched:
        print(f"WARNING: {len(mismatched)} solves reached different objectives "
              f"(a cut may be invalid) -- excluded from the node comparison.",
              flush=True)

    print(f"\n{'mode':<12}{'mean nodes':>12}{'median':>10}{'%solved':>10}", flush=True)
    for mode in modes:
        allrec = [r[mode] for r in records if mode in r]
        sv = [r[mode]["nodes"] for r in valid]
        solved = sum(1 for r in allrec if r["status"] == "optimal")
        mean = np.mean(sv) if sv else float("nan")
        med = np.median(sv) if sv else float("nan")
        print(f"{mode:<12}{mean:>12.1f}{med:>10.1f}"
              f"{100*solved/max(1,len(allrec)):>9.0f}%", flush=True)

    if "none" in modes and "learned" in modes and valid:
        n_none = np.mean([r["none"]["nodes"] for r in valid])
        n_learn = np.mean([r["learned"]["nodes"] for r in valid])
        red = 100.0 * (n_none - n_learn) / max(1e-9, n_none)
        verdict = ("HELPS (fewer nodes)" if red > 2 else
                   "NEUTRAL" if red > -2 else "HURTS (more nodes)")
        print(f"\nlearned vs none: {red:+.1f}% node change  ->  {verdict}",
              flush=True)
        print("If HELPS -> build the full B&C-in-SCIP time harness.\n"
              "If NEUTRAL/HURTS -> upgrade the cut phase (labels/selection) "
              "before the time test.", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    import json
    json.dump(records, open(args.out, "w"), indent=2, default=str)
    print(f"\nSaved per-instance records to {args.out}", flush=True)


if __name__ == "__main__":
    main()
