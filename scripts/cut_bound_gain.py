"""
cut_bound_gain.py — Is the cut problem GENERATION or SELECTION?

The cut half is net-negative (validate_cuts: learned 143.7 vs none 102.4 nodes,
and learned == heuristic). Two possible causes:
  * GENERATION: the Gomory cut POOL barely tightens the root LP -> selecting
    among it cannot help (Gomory is weak on set-covering).
  * SELECTION : the pool tightens well, but the head picks the wrong subset.

This measures ROOT LP BOUND-GAIN per selection mode. For each instance:
  1. Solve the root LP (no cuts) -> obj_none, x_lp.
  2. Generate the valid Gomory pool (solver.gomory.generate_root_gomory_cuts).
  3. For each mode, add the selected cuts, re-solve the LP, record the objective
     increase (a min LP relaxation's objective RISES toward the integer optimum
     as valid cuts are added, so gain = obj_mode - obj_none >= 0).

Modes:
  none      : 0 cuts (baseline)
  all       : every valid Gomory cut  (upper bound on what generation can give)
  random    : a random size-K subset  (selection-agnostic control)
  learned   : top-K by the trained cut head's score
  heuristic : top-K by violation/efficacy

Reading the result:
  * all-cuts gain ~= 0            -> GENERATION is the wall (Gomory too weak);
                                      no selection model can help -> NO-CUT gate
                                      + a better generator (cover/clique) is the fix.
  * all-cuts gain > 0 but
    learned << all                -> SELECTION matters -> HEM-style head worth it.
  * learned ~= random ~= all      -> selection is irrelevant; whether-to-cut is.

Node counts are NOT measured here (that is validate_cuts.py); this isolates the
bound-tightening signal only, which is cheap (a handful of LP solves/instance).

Usage:
    PYTHONPATH=. python scripts/cut_bound_gain.py \
        --checkpoint checkpoints/model_rl_best.pt \
        --n_instances 30 --n_rows 200 --n_cols 400 --top_k 10
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import ecole
import yaml
from scipy.optimize import linprog

from bnb_wm.model.world_model import BnBWorldModel
from bnb_wm.training.checkpoint import load_weights_only
from bnb_wm.solver.gomory import generate_root_gomory_cuts
from bnb_wm.evaluate.benchmark import _format_obs
from scripts.validate_cuts import _extract_Abc
from scripts.collect_with_cuts_v2 import _cut_features

try:
    import highspy
except Exception:
    highspy = None


def _solve_lp(c, A, b, cuts, n):
    """min c^T x  s.t.  A x >= b, (lhs x >= rhs) for cuts, 0 <= x <= 1.
    Returns (objective, x) or (None, None) on failure."""
    rows = [-A]
    rhs = [-b]
    for lhs, r in cuts:
        rows.append(-np.asarray(lhs, dtype=np.float64).reshape(1, -1))
        rhs.append(np.array([-float(r)]))
    A_ub = np.vstack(rows)
    b_ub = np.concatenate(rhs)
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0.0, 1.0)] * n,
                      method="highs")
    except Exception:
        return None, None
    if not res.success:
        return None, None
    return float(res.fun), res.x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n_instances", type=int, default=30)
    ap.add_argument("--n_rows", type=int, default=200)
    ap.add_argument("--n_cols", type=int, default=400)
    ap.add_argument("--density", type=float, default=0.05)
    ap.add_argument("--top_k", type=int, default=10, help="cuts selected by learned/heuristic/random")
    ap.add_argument("--max_pool", type=int, default=50, help="max Gomory cuts generated")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/cut_bound_gain.json")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    if highspy is None:
        raise SystemExit("highspy required (Gomory basis). Install highspy.")

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
    print(f"Loaded {args.checkpoint} | cut_head_trained={bool(model.cut_head_trained)}",
          flush=True)

    gen = ecole.instance.SetCoverGenerator(
        n_rows=args.n_rows, n_cols=args.n_cols, density=args.density)
    gen.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = ecole.environment.Branching(
        observation_function=ecole.observation.NodeBipartite(),
        scip_params={"separating/maxrounds": 0, "presolving/maxrounds": 0},
    )

    modes = ["all", "random", "learned", "heuristic"]
    gains = {m: [] for m in modes}
    ncuts = {m: [] for m in modes}
    pool_sizes = []

    for i in range(args.n_instances):
        inst = next(gen)
        try:
            A, b, c = _extract_Abc(inst.copy_orig().as_pyscipopt())
        except Exception as e:
            print(f"[{i+1}] extract failed: {e}", flush=True)
            continue
        n = len(c)

        obj_none, x_lp = _solve_lp(c, A, b, [], n)
        if obj_none is None:
            print(f"[{i+1}] root LP failed", flush=True)
            continue

        pool = generate_root_gomory_cuts(A, b, c, highspy, max_cuts=args.max_pool)
        pool_sizes.append(len(pool))
        if not pool:
            for m in modes:
                gains[m].append(0.0); ncuts[m].append(0)
            print(f"[{i+1}] pool=0 (no valid Gomory cuts)", flush=True)
            continue

        # cut features (same 6-dim as collector) for learned/heuristic scoring
        feats = np.array([_cut_features(np.asarray(lhs), float(r), x_lp, c, n)
                          for (lhs, r) in pool], dtype=np.float32)   # [P,6]

        # encode root node -> z for the learned head
        try:
            obs, _, _, _, _ = env.reset(inst.copy_orig())
            batch = _format_obs(obs, device)
            with torch.no_grad():
                _, z = model.encode(batch)
            z0 = z[0]
        except Exception:
            z0 = None

        K = min(args.top_k, len(pool))
        sel = {}
        sel["all"] = list(range(len(pool)))
        sel["random"] = list(rng.choice(len(pool), size=K, replace=False))
        # heuristic: violation / efficacy = feats[:,0] (violation), then [:,1]
        heur_order = np.argsort(-(feats[:, 0]))
        sel["heuristic"] = list(heur_order[:K])
        # learned: cut head score
        if z0 is not None:
            with torch.no_grad():
                scr = model.cut_scores(torch.tensor(feats, device=device), z0)
            sel["learned"] = list(torch.argsort(scr, descending=True)[:K].cpu().numpy())
        else:
            sel["learned"] = sel["heuristic"]

        for m in modes:
            chosen = [pool[j] for j in sel[m]]
            obj_m, _ = _solve_lp(c, A, b, chosen, n)
            gain = (obj_m - obj_none) if obj_m is not None else 0.0
            gains[m].append(max(0.0, gain))
            ncuts[m].append(len(chosen))

        print(f"[{i+1}/{args.n_instances}] pool={len(pool)} obj_none={obj_none:.3f} "
              + " | ".join(f"{m}:+{gains[m][-1]:.3f}({ncuts[m][-1]})" for m in modes),
              flush=True)

    print("\n" + "=" * 64, flush=True)
    print(f"instances={len(pool_sizes)}  mean pool size={np.mean(pool_sizes):.1f}",
          flush=True)
    print(f"{'mode':<12}{'mean bound-gain':>18}{'mean #cuts':>12}", flush=True)
    for m in modes:
        g = np.mean(gains[m]) if gains[m] else float("nan")
        k = np.mean(ncuts[m]) if ncuts[m] else float("nan")
        print(f"{m:<12}{g:>18.4f}{k:>12.1f}", flush=True)

    # verdict
    ga = np.mean(gains["all"]) if gains["all"] else 0.0
    gl = np.mean(gains["learned"]) if gains["learned"] else 0.0
    gr = np.mean(gains["random"]) if gains["random"] else 0.0
    print("\nVERDICT:", flush=True)
    if ga < 1e-3:
        print("  GENERATION is the wall: even ALL Gomory cuts barely tighten the "
              "root LP. Selection cannot help. Fix = NO-CUT gate + a set-cover "
              "generator (cover/clique) or SCIP's cut pool.", flush=True)
    elif gl >= 0.9 * ga and gl >= gr:
        print("  Pool tightens AND the learned head captures most of it. If cuts "
              "still hurt nodes, the issue is INTEGRATION with branching.", flush=True)
    elif gl < 0.7 * ga:
        print("  SELECTION matters: ALL-cuts tightens well but the learned head "
              "leaves gain on the table. HEM-style hierarchical selection is "
              "justified.", flush=True)
    else:
        print("  Mixed signal: inspect per-instance rows.", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    import json
    json.dump({"gains": gains, "ncuts": ncuts, "pool_sizes": pool_sizes},
              open(args.out, "w"), indent=2, default=str)
    print(f"\nSaved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
