"""
benchmark.py — Full solver benchmark: Our BnBSolver vs SCIP+HiGHS.

Both solvers use HiGHS as their LP backend so wall-time is apples-to-apples.

Metrics collected per instance:
    nodes      — nodes traversed to optimality (or time limit)
    wall_time  — seconds to termination
    primal     — best primal bound at termination
    dual_gap   — |primal - dual| / (1 + |primal|)  [0 = optimal]
    n_cuts     — number of cuts applied (our solver) / SCIP separator rounds

Usage:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/benchmark.py \\
        --checkpoint checkpoints/phase3_best.pt \\
        --n_instances 20 \\
        --time_limit 120 \\
        --out results/benchmark.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",   default="checkpoints/phase3_best.pt")
parser.add_argument("--config",       default="configs/default.yaml")
parser.add_argument("--n_instances",  type=int,   default=20)
parser.add_argument("--time_limit",   type=float, default=120.0)
parser.add_argument("--node_limit",   type=int,   default=50_000)
parser.add_argument("--problem",      default="set_cover",
                    choices=["set_cover"])
parser.add_argument("--n_rows",       type=int,   default=500)
parser.add_argument("--n_cols",       type=int,   default=1000)
parser.add_argument("--density",      type=float, default=0.05)
parser.add_argument("--seed",         type=int,   default=42)
parser.add_argument("--cut_mode",     default="latent",
                    help="cut_mode for BnBSolver (latent|classic|none)")
parser.add_argument("--cut_beam",     type=int,   default=3)
parser.add_argument("--cut_rounds",   type=int,   default=2)
parser.add_argument("--rollout_depth",type=int,   default=3)
parser.add_argument("--out",          default="results/benchmark.json")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Imports (after sys.path set)
# ---------------------------------------------------------------------------
import yaml
from bnb_wm.model.world_model import BnBWorldModel
from bnb_wm.solver.bnb_solver import BnBSolver

try:
    import pyscipopt
    _SCIP_OK = True
except ImportError:
    _SCIP_OK = False
    print("[WARN] pyscipopt not found — SCIP baseline will be skipped.")

try:
    import ecole
    _ECOLE_OK = True
except ImportError:
    _ECOLE_OK = False
    print("[WARN] ecole not found — using scipy set-cover generator.")

# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------

def _generate_set_cover_scipy(rng, n_rows, n_cols, density):
    """Generate a set-cover LP in standard form: min c'x s.t. Ax >= 1, x in [0,1].
    Returns (A, b, c) where A is [n_rows, n_cols], b = ones(n_rows), c = rand.
    """
    A = (rng.random((n_rows, n_cols)) < density).astype(np.float64)
    # Ensure every row has at least one nonzero.
    for i in range(n_rows):
        if A[i].sum() == 0:
            A[i, rng.integers(n_cols)] = 1.0
    b = np.ones(n_rows, dtype=np.float64)
    c = rng.uniform(1.0, 10.0, size=n_cols)
    return A, b, c


def _ecole_to_lp(instance):
    """Convert an Ecole instance to (A, b, c) standard form for BnBSolver."""
    m = instance.as_pyscipopt()
    vars_ = m.getVars()
    cons_ = m.getConss()
    n_vars = len(vars_)
    n_cons = len(cons_)
    var_idx = {v.name: i for i, v in enumerate(vars_)}
    A = np.zeros((n_cons, n_vars), dtype=np.float64)
    b = np.zeros(n_cons, dtype=np.float64)
    for j, con in enumerate(cons_):
        row = m.getValsLinear(con)
        lhs = m.getLhs(con)
        rhs = m.getRhs(con)
        for vname, coeff in row.items():
            A[j, var_idx[vname]] = coeff
        b[j] = rhs if rhs < 1e29 else lhs
    c = np.array([m.getObjective().getCoefficients().get(v.name, 0.0)
                  for v in vars_])
    return A, b, c


def generate_instances(n, rng, args):
    instances = []
    if _ECOLE_OK and args.problem == "set_cover":
        gen = ecole.instance.SetCoverGenerator(
            n_rows=args.n_rows, n_cols=args.n_cols, density=args.density,
        )
        for _ in range(n):
            inst = next(gen)
            try:
                A, b, c = _ecole_to_lp(inst)
                instances.append((A, b, c, inst))
            except Exception as e:
                print(f"  [WARN] ecole→LP conversion failed: {e}")
    else:
        for _ in range(n):
            A, b, c = _generate_set_cover_scipy(rng, args.n_rows, args.n_cols, args.density)
            instances.append((A, b, c, None))
    return instances

# ---------------------------------------------------------------------------
# SCIP+HiGHS baseline
# ---------------------------------------------------------------------------

def run_scip_highs(A, b, c, ecole_inst, time_limit):
    """Run SCIP with HiGHS LP backend. Returns metric dict."""
    if not _SCIP_OK:
        return None

    if ecole_inst is not None and _ECOLE_OK:
        m = ecole_inst.as_pyscipopt()
    else:
        m = pyscipopt.Model()
        m.setMinimize()
        vars_ = [m.addVar(vtype="B", name=f"x{j}", obj=float(c[j]))
                 for j in range(len(c))]
        for i in range(len(b)):
            m.addCons(
                pyscipopt.quicksum(A[i, j] * vars_[j] for j in range(len(c)))
                >= b[i]
            )

    m.hideOutput()
    m.setParam("limits/time", time_limit)
    # Use HiGHS as LP solver if available in this SCIP build.
    try:
        m.setParam("lp/solver", "highs")
    except Exception:
        pass  # SCIP build without HiGHS plugin — falls back to default LP solver

    t0 = time.perf_counter()
    m.optimize()
    elapsed = time.perf_counter() - t0

    status = m.getStatus()          # "optimal" | "timelimit" | ...
    n_nodes = int(m.getNNodes())
    n_cuts  = int(m.getNCutsApplied()) if hasattr(m, "getNCutsApplied") else 0

    primal = m.getObjVal() if status in ("optimal", "timelimit") else np.inf
    try:
        dual = m.getDualbound()
    except Exception:
        dual = primal
    gap = abs(primal - dual) / (1.0 + abs(primal))

    return {
        "status":    status,
        "nodes":     n_nodes,
        "wall_time": elapsed,
        "primal":    float(primal),
        "dual_gap":  float(gap),
        "n_cuts":    n_cuts,
    }

# ---------------------------------------------------------------------------
# Our solver
# ---------------------------------------------------------------------------

def run_our_solver(A, b, c, solver):
    t0 = time.perf_counter()
    result = solver.solve(A, b, c)
    elapsed = time.perf_counter() - t0

    primal = float(result.objective)
    # BnBSolver stores best dual bound in result; fall back to primal if absent.
    dual   = float(getattr(result, "dual_bound", result.objective))
    gap    = abs(primal - dual) / (1.0 + abs(primal))

    # Count cuts: each committed cut is one entry in cut_latent_errors list
    # (even classic mode appends a 0.0 placeholder).
    n_cuts = len(result.cut_latent_errors) if result.cut_latent_errors else 0

    return {
        "status":    result.status,
        "nodes":     result.n_nodes,
        "wall_time": elapsed,
        "primal":    primal,
        "dual_gap":  float(result.optimality_gap),
        "n_cuts":    n_cuts,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    cfg = yaml.safe_load(open(args.config))
    model = BnBWorldModel(**cfg["model"]).to(device).eval()
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt = sd["model"] if "model" in sd else sd
    model.load_state_dict(ckpt, strict=False)
    print(f"Checkpoint: {args.checkpoint}")

    solver = BnBSolver(
        model=model,
        device=device,
        time_limit=args.time_limit,
        node_limit=args.node_limit,
        cut_mode=args.cut_mode,
        cut_beam=args.cut_beam,
        cut_rounds=args.cut_rounds,
        rollout_depth=args.rollout_depth,
    )

    print(f"\nGenerating {args.n_instances} {args.problem} instances "
          f"({args.n_rows}×{args.n_cols}, density={args.density})...\n")
    instances = generate_instances(args.n_instances, rng, args)

    results = {"ours": [], "scip_highs": []}
    header = f"{'#':>3}  {'Method':<12}  {'Status':<9}  {'Nodes':>7}  "
    header += f"{'Time':>7}  {'Primal':>10}  {'Gap':>8}  {'Cuts':>5}"
    print(header)
    print("-" * len(header))

    for i, (A, b, c, ecole_inst) in enumerate(instances):
        # --- Our solver ---
        r_ours = run_our_solver(A, b, c, solver)
        results["ours"].append(r_ours)
        print(f"{i+1:>3}  {'Ours':<12}  {r_ours['status']:<9}  "
              f"{r_ours['nodes']:>7d}  {r_ours['wall_time']:>7.2f}  "
              f"{r_ours['primal']:>10.4f}  {r_ours['dual_gap']:>8.4f}  "
              f"{r_ours['n_cuts']:>5d}")

        # --- SCIP+HiGHS ---
        r_scip = run_scip_highs(A, b, c, ecole_inst, args.time_limit)
        if r_scip is not None:
            results["scip_highs"].append(r_scip)
            print(f"{i+1:>3}  {'SCIP+HiGHS':<12}  {r_scip['status']:<9}  "
                  f"{r_scip['nodes']:>7d}  {r_scip['wall_time']:>7.2f}  "
                  f"{r_scip['primal']:>10.4f}  {r_scip['dual_gap']:>8.4f}  "
                  f"{r_scip['n_cuts']:>5d}")
        print()

    # Summary table
    print("=" * len(header))
    print("AVERAGES")
    print("=" * len(header))
    for method, res_list in results.items():
        if not res_list:
            continue
        avg = {k: np.mean([r[k] for r in res_list
                           if isinstance(r[k], (int, float))])
               for k in ("nodes", "wall_time", "primal", "dual_gap", "n_cuts")}
        print(f"  {method:<14}  nodes={avg['nodes']:>7.1f}  "
              f"time={avg['wall_time']:>6.2f}s  "
              f"primal={avg['primal']:>9.3f}  "
              f"gap={avg['dual_gap']:>7.4f}  "
              f"cuts={avg['n_cuts']:>5.1f}")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
