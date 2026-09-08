"""
ablation.py — Component ablation for the BnB World Model solver.

Runs 7 configurations on the same instances (fixed seed) and prints a
unified comparison table attributing performance to each component.

Configs (in order):
  0. SCIP+HiGHS          — reference solver
  1. Custom baseline      — HiGHS B&B, most_fractional, no cuts, no model
  2. Policy only          — GNN + PolicyHead, no rollout, no cuts
  3. Policy + Dynamics    — GNN + PolicyHead + rollout, no cuts
  4. Cuts only            — most_fractional + Gomory cuts (max_violation)
  5. Policy + Cuts        — GNN policy + Gomory cuts, no rollout
  6. Full model           — policy + rollout + cuts (latent beam)

Usage:
    PYTHONPATH=. python scripts/ablation.py \\
        --checkpoint checkpoints/phase3_best.pt \\
        --n_instances 3 \\
        --time_limit 30 \\
        --out results/ablation.json
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",  default="checkpoints/phase3_best.pt")
parser.add_argument("--config",      default="configs/default.yaml")
parser.add_argument("--n_instances", type=int,   default=3)
parser.add_argument("--time_limit",  type=float, default=30.0)
parser.add_argument("--node_limit",  type=int,   default=50_000)
parser.add_argument("--n_rows",      type=int,   default=200)
parser.add_argument("--n_cols",      type=int,   default=400)
parser.add_argument("--density",     type=float, default=0.05)
parser.add_argument("--seed",        type=int,   default=42)
parser.add_argument("--out",         default="results/ablation.json")
args = parser.parse_args()

import yaml
from bnb_wm.model.world_model import BnBWorldModel
from bnb_wm.solver.bnb_solver import BnBSolver

try:
    import pyscipopt
    _SCIP_OK = True
except ImportError:
    _SCIP_OK = False
    print("[WARN] pyscipopt not available — SCIP config will be skipped.")

# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------

def gen_instances(n, rng):
    out = []
    for _ in range(n):
        A = (rng.random((args.n_rows, args.n_cols)) < args.density).astype(np.float64)
        for i in range(args.n_rows):
            if A[i].sum() == 0:
                A[i, rng.integers(args.n_cols)] = 1.0
        b = np.ones(args.n_rows, dtype=np.float64)
        c = rng.uniform(1.0, 10.0, size=args.n_cols)
        out.append((A, b, c))
    return out

# ---------------------------------------------------------------------------
# SCIP runner
# ---------------------------------------------------------------------------

def run_scip(A, b, c):
    if not _SCIP_OK:
        return None
    m = pyscipopt.Model()
    m.setMinimize()
    vars_ = [m.addVar(vtype="B", name=f"x{j}", obj=float(c[j])) for j in range(len(c))]
    for i in range(len(b)):
        m.addCons(pyscipopt.quicksum(A[i, j] * vars_[j] for j in range(len(c))) >= b[i])
    m.hideOutput()
    m.setParam("limits/time", args.time_limit)
    try:
        m.setParam("lp/solver", "highs")
    except Exception:
        pass
    t0 = time.perf_counter()
    m.optimize()
    elapsed = time.perf_counter() - t0
    status = m.getStatus()
    primal = m.getObjVal() if status in ("optimal", "timelimit") else np.inf
    try:
        dual = m.getDualbound()
    except Exception:
        dual = primal
    gap = abs(primal - dual) / (1.0 + abs(primal))
    n_cuts = int(m.getNCutsApplied()) if hasattr(m, "getNCutsApplied") else 0
    return {
        "status": status,
        "nodes": int(m.getNNodes()),
        "wall_time": elapsed,
        "primal": float(primal),
        "dual_gap": float(gap),
        "n_cuts": n_cuts,
        "lp_time": None,   # not exposed by pyscipopt
        "overhead": None,
    }

# ---------------------------------------------------------------------------
# Our solver runner
# ---------------------------------------------------------------------------

def run_solver(A, b, c, solver):
    t0 = time.perf_counter()
    try:
        result = solver.solve(A, b, c)
    except Exception as e:
        traceback.print_exc()
        return None
    elapsed = time.perf_counter() - t0
    primal = float(result.objective)
    dual   = float(getattr(result, "dual_bound", result.objective))
    gap    = abs(primal - dual) / (1.0 + abs(primal))
    n_cuts = len(result.cut_latent_errors) if result.cut_latent_errors else 0
    # cut_committed from diag
    committed = result.cut_diag.get("cut_committed", 0)
    selected  = result.cut_diag.get("cut_selected", 0)
    return {
        "status":    result.status,
        "nodes":     result.n_nodes,
        "wall_time": elapsed,
        "primal":    primal,
        "dual_gap":  float(result.optimality_gap),
        "n_cuts":    committed,
        "cut_selected": selected,
        "lp_time":   None,
        "overhead":  None,
        "cut_diag":  result.cut_diag,
    }

# ---------------------------------------------------------------------------
# Solver configurations
# ---------------------------------------------------------------------------

def make_solver(model, device, **overrides):
    base = dict(
        model=model,
        device=device,
        time_limit=args.time_limit,
        node_limit=args.node_limit,
        diag_mode=True,   # always on: clean node counts, no ORS/neural-prune distortion
    )
    base.update(overrides)
    return BnBSolver(**base)


CONFIGS = [
    # name, description, solver_kwargs or None (None = SCIP)
    ("SCIP+HiGHS",       "Reference solver",                         None),
    ("Custom-Baseline",  "HiGHS B&B, most_fractional, no model",     dict(
        branch_mode="most_fractional",
        cut_mode="none",
    )),
    ("Policy",           "GNN policy head, no rollout, no cuts",      dict(
        branch_mode="policy",
        cut_mode="none",
    )),
    ("Policy+Dynamics",  "GNN policy + rollout, no cuts",             dict(
        branch_mode="rollout",
        cut_mode="none",
        lookahead_depth=3,
    )),
    ("Cuts-Only",        "most_fractional + Gomory cuts (max_viol)",  dict(
        branch_mode="most_fractional",
        cut_mode="latent",
        cut_selection="max_violation",
        cut_beam=3,
        cut_rounds=2,
    )),
    ("Policy+Cuts",      "GNN policy + Gomory cuts, no rollout",      dict(
        branch_mode="policy",
        cut_mode="latent",
        cut_selection="max_violation",
        cut_beam=3,
        cut_rounds=2,
    )),
    ("Full-Model",       "policy + rollout + cuts (latent beam)",     dict(
        branch_mode="rollout",
        cut_mode="latent",
        cut_selection="model",
        cut_beam=3,
        cut_rounds=2,
        lookahead_depth=3,
    )),
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg   = yaml.safe_load(open(args.config))
    model = BnBWorldModel(**cfg["model"]).to(device).eval()
    sd    = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt  = sd["model"] if "model" in sd else sd
    model.load_state_dict(ckpt, strict=False)
    print(f"Checkpoint: {args.checkpoint}\n")

    instances = []
    for _ in range(args.n_instances):
        A = (rng.random((args.n_rows, args.n_cols)) < args.density).astype(np.float64)
        for i in range(args.n_rows):
            if A[i].sum() == 0:
                A[i, rng.integers(args.n_cols)] = 1.0
        b = np.ones(args.n_rows, dtype=np.float64)
        c = rng.uniform(1.0, 10.0, size=args.n_cols)
        instances.append((A, b, c))

    print(f"Instances: {args.n_instances} × ({args.n_rows}×{args.n_cols}), "
          f"density={args.density}, seed={args.seed}\n")

    all_results = {}   # config_name → list of metric dicts

    col_w = [18, 9, 7, 8, 10, 8, 7]
    header = (f"{'Config':<{col_w[0]}}  {'Status':<{col_w[1]}}  {'Nodes':>{col_w[2]}}"
              f"  {'Time':>{col_w[3]}}  {'Primal':>{col_w[4]}}  {'Gap':>{col_w[5]}}"
              f"  {'Cuts':>{col_w[6]}}")

    for cfg_name, cfg_desc, solver_kwargs in CONFIGS:
        print("=" * len(header))
        print(f"  {cfg_name}  —  {cfg_desc}")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        results = []
        if solver_kwargs is None:
            # SCIP
            for i, (A, b, c) in enumerate(instances):
                r = run_scip(A, b, c)
                if r is None:
                    print(f"  {'[SCIP unavailable]'}")
                    continue
                results.append(r)
                print(f"  {'inst'+str(i+1):<{col_w[0]}}  {r['status']:<{col_w[1]}}  "
                      f"{r['nodes']:>{col_w[2]}d}  {r['wall_time']:>{col_w[3]}.2f}  "
                      f"{r['primal']:>{col_w[4]}.4f}  {r['dual_gap']:>{col_w[5]}.4f}  "
                      f"{r['n_cuts']:>{col_w[6]}d}")
        else:
            solver = make_solver(model, device, **solver_kwargs)
            for i, (A, b, c) in enumerate(instances):
                r = run_solver(A, b, c, solver)
                if r is None:
                    print(f"  inst{i+1}  ERROR")
                    continue
                results.append(r)
                print(f"  {'inst'+str(i+1):<{col_w[0]}}  {r['status']:<{col_w[1]}}  "
                      f"{r['nodes']:>{col_w[2]}d}  {r['wall_time']:>{col_w[3]}.2f}  "
                      f"{r['primal']:>{col_w[4]}.4f}  {r['dual_gap']:>{col_w[5]}.4f}  "
                      f"{r['n_cuts']:>{col_w[6]}d}")
                if r.get("cut_diag"):
                    d = r["cut_diag"]
                    print(f"    cuts: selected={d.get('cut_selected',0)} "
                          f"committed={d.get('cut_committed',0)} "
                          f"pool_empty={d.get('pool_empty',0)} "
                          f"rej_integ={d.get('rejected_integrality',0)} "
                          f"rej_depth={d.get('rejected_depth',0)}")

        if results:
            nums = {k: [r[k] for r in results if isinstance(r.get(k), (int, float))]
                    for k in ("nodes", "wall_time", "primal", "dual_gap", "n_cuts")}
            avg = {k: float(np.mean(v)) if v else float("nan") for k, v in nums.items()}
            solved = sum(1 for r in results if r.get("status") == "optimal")
            print("-" * len(header))
            print(f"  {'AVG (solved '+str(solved)+'/'+str(len(results))+')':<{col_w[0]}}  "
                  f"{'':>{col_w[1]}}  {avg['nodes']:>{col_w[2]}.0f}  "
                  f"{avg['wall_time']:>{col_w[3]}.2f}  {avg['primal']:>{col_w[4]}.3f}  "
                  f"{avg['dual_gap']:>{col_w[5]}.4f}  {avg['n_cuts']:>{col_w[6]}.1f}")
            all_results[cfg_name] = {"avg": avg, "solved": solved,
                                     "n": len(results), "rows": results}
        print()

    # ---- Summary table -------------------------------------------------------
    print("\n" + "=" * 80)
    print("ABLATION SUMMARY")
    print("=" * 80)
    hdr2 = (f"{'Config':<20}  {'Solved':>6}  {'Nodes':>7}  {'Time':>7}  "
            f"{'Gap':>8}  {'Cuts':>6}  {'ΔNodes%':>8}  {'ΔGap%':>8}")
    print(hdr2)
    print("-" * len(hdr2))

    # Use SCIP as reference for deltas; fall back to Custom-Baseline
    ref_nodes = all_results.get("SCIP+HiGHS", {}).get("avg", {}).get("nodes", None)
    ref_gap   = all_results.get("SCIP+HiGHS", {}).get("avg", {}).get("dual_gap", None)
    if ref_nodes is None:
        ref_nodes = all_results.get("Custom-Baseline", {}).get("avg", {}).get("nodes", None)
        ref_gap   = all_results.get("Custom-Baseline", {}).get("avg", {}).get("dual_gap", None)

    for cfg_name, _, _ in CONFIGS:
        if cfg_name not in all_results:
            continue
        d = all_results[cfg_name]
        avg = d["avg"]
        dn = f"{(avg['nodes'] - ref_nodes) / (ref_nodes + 1e-8) * 100:+.1f}%" if ref_nodes else "—"
        dg = f"{(avg['dual_gap'] - ref_gap) / (ref_gap + 1e-8) * 100:+.1f}%" if ref_gap else "—"
        print(f"  {cfg_name:<20}  {str(d['solved'])+'/'+str(d['n']):>6}  "
              f"{avg['nodes']:>7.0f}  {avg['wall_time']:>7.2f}  "
              f"{avg['dual_gap']:>8.4f}  {avg['n_cuts']:>6.1f}  "
              f"{dn:>8}  {dg:>8}")

    print()
    print("Notes:")
    print("  ΔNodes%/ΔGap% relative to SCIP+HiGHS (or Custom-Baseline if SCIP absent).")
    print("  diag_mode=True: ORS + neural pruning disabled for clean node counts.")
    print("  Policy/Dynamics heads are OOD (trained on SCIP LP states) — expect")
    print("  calibration penalty on top of any structural improvement.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # make results JSON-serialisable
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj
    with open(out_path, "w") as f:
        json.dump(_clean({"args": vars(args), "results": all_results}), f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
