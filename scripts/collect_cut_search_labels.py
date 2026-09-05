"""
collect_cut_search_labels.py — SEARCH-aware cut labels (ΔNodes), not LP-bound.

Our diagnostics showed LP-bound gain does NOT predict node reduction: even
optimally-selected Gomory cuts tighten the root LP yet increase the B&B tree.
To train a cut evaluator that targets what we actually care about (search cost),
we need labels measured in *nodes*, under *our* branching policy.

For each instance and each candidate cut c_j (root Gomory pool):
  N0     = nodes to solve under learned branching, NO cuts        (baseline)
  N_j    = nodes to solve with c_j added as a constraint (cut_mode='none',
           so ONLY c_j is present — no other cuts interfere)
  ΔNodes_j = N0 - N_j       (>0 => the cut helps the search)
  ΔLP_j    = root LP objective gain from c_j                       (for the scatter)

A correctness gate requires N0's and N_j's optima to match (a valid cut must not
change the optimum); mismatches are flagged and excluded.

Outputs an .npz with per-cut rows (features[6], z[H], dlp, dnodes, valid) for
training a search-aware NO-CUT-gated evaluator, plus a per-instance gate label
(did the heuristic top-k subset reduce nodes?). Prints corr(ΔLP, ΔNodes) — the
decisive check of whether LP-bound and search benefit diverge on our instances.

NOTE: node counts come from the standalone Python-LP solver (bnb_solver.py) under
the LEARNED branching policy. They are valid RELATIVE node measurements (with vs
without a cut, same solver) — this is a labelling tool, not a SCIP-time benchmark.

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/collect_cut_search_labels.py \
        --checkpoint checkpoints/model_rl_best.pt \
        --n_instances 30 --n_rows 200 --n_cols 400 \
        --node_limit 5000 --time_limit 45 --out data/cut_labels/labels.npz
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
from bnb_wm.solver.bnb_solver import BnBSolver
from bnb_wm.solver.gomory import generate_root_gomory_cuts
from bnb_wm.evaluate.benchmark import _format_obs
from scripts.validate_cuts import _extract_Abc
from scripts.collect_with_cuts_v2 import _cut_features

try:
    import highspy
except Exception:
    highspy = None


def _root_lp(c, A, b, cuts, n):
    rows = [-A]; rhs = [-b]
    for lhs, r in cuts:
        rows.append(-np.asarray(lhs, np.float64).reshape(1, -1)); rhs.append(np.array([-float(r)]))
    try:
        res = linprog(c, A_ub=np.vstack(rows), b_ub=np.concatenate(rhs),
                      bounds=[(0.0, 1.0)] * n, method="highs")
    except Exception:
        return None
    return float(res.fun) if res.success else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n_instances", type=int, default=30)
    ap.add_argument("--n_rows", type=int, default=200)
    ap.add_argument("--n_cols", type=int, default=400)
    ap.add_argument("--density", type=float, default=0.05)
    ap.add_argument("--max_pool", type=int, default=30, help="max Gomory cuts / instance to label")
    ap.add_argument("--node_limit", type=int, default=5000)
    ap.add_argument("--time_limit", type=int, default=45)
    ap.add_argument("--top_k", type=int, default=6, help="subset size for the per-instance gate label")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/cut_labels/labels.npz")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    if highspy is None:
        raise SystemExit("highspy required for Gomory generation.")

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
    H = cfg["hidden_dim"]
    print(f"Loaded {args.checkpoint} on {device}", flush=True)

    gen = ecole.instance.SetCoverGenerator(
        n_rows=args.n_rows, n_cols=args.n_cols, density=args.density)
    gen.seed(args.seed)
    env = ecole.environment.Branching(
        observation_function=ecole.observation.NodeBipartite(),
        scip_params={"separating/maxrounds": 0, "presolving/maxrounds": 0},
    )

    def solve_nodes(A, b, c):
        """Nodes + objective + solved flag under LEARNED branching, no cuts."""
        s = BnBSolver(model, device, time_limit=args.time_limit,
                      node_limit=args.node_limit, cut_mode="none")
        r = s.solve(A, b, c)
        return r.n_nodes, float(r.objective), (r.status == "optimal")

    feats_all, z_all, dlp_all, dnodes_all, valid_all = [], [], [], [], []
    gate_x, gate_y = [], []          # per-instance: (z, did-subset-help)
    dlp_scatter, dn_scatter = [], []

    for i in range(args.n_instances):
        inst = next(gen)
        try:
            A, b, c = _extract_Abc(inst.copy_orig().as_pyscipopt())
        except Exception as e:
            print(f"[{i+1}] extract failed: {e}", flush=True); continue
        n = len(c)

        N0, obj0, ok0 = solve_nodes(A, b, c)
        if not ok0:
            print(f"[{i+1}] baseline not solved (N0={N0}) -> skip", flush=True); continue

        obj_lp0 = _root_lp(c, A, b, [], n)
        pool = generate_root_gomory_cuts(A, b, c, highspy, max_cuts=args.max_pool)
        if not pool:
            print(f"[{i+1}] pool=0 -> gate label only (N0={N0})", flush=True)
            # still record a NO-CUT gate example (nothing to add -> subset can't help)
            continue

        # root LP solution for cut features
        try:
            res = linprog(c, A_ub=-A, b_ub=-b, bounds=[(0, 1)] * n, method="highs")
            x_lp = res.x if res.success else np.zeros(n)
        except Exception:
            x_lp = np.zeros(n)

        feats = np.array([_cut_features(np.asarray(l), float(r), x_lp, c, n)
                          for (l, r) in pool], dtype=np.float32)   # [P,6]
        # encode root -> z
        try:
            obs, _, _, _, _ = env.reset(inst.copy_orig())
            with torch.no_grad():
                _, z = model.encode(_format_obs(obs, device))
            z0 = z[0].detach().cpu().numpy().astype(np.float32)
        except Exception:
            z0 = np.zeros(H, np.float32)

        # per-cut ΔNodes (add ONLY that cut as a constraint)
        for j, (lhs, r) in enumerate(pool):
            Aj = np.vstack([A, np.asarray(lhs, np.float64).reshape(1, -1)])
            bj = np.concatenate([b, [float(r)]])
            Nj, objj, okj = solve_nodes(Aj, bj, c)
            valid = okj and abs(objj - obj0) < 1e-4      # cut must not change optimum
            dN = float(N0 - Nj)
            dLP = 0.0
            if obj_lp0 is not None:
                lpj = _root_lp(c, A, b, [(lhs, r)], n)
                if lpj is not None:
                    dLP = max(0.0, lpj - obj_lp0)
            feats_all.append(feats[j]); z_all.append(z0)
            dlp_all.append(dLP); dnodes_all.append(dN); valid_all.append(bool(valid))
            if valid:
                dlp_scatter.append(dLP); dn_scatter.append(dN)

        # per-instance gate label: does the heuristic top-k subset reduce nodes?
        K = min(args.top_k, len(pool))
        order = np.argsort(-(feats[:, 0]))[:K]            # by violation
        cuts_sub = [pool[k] for k in order]
        As = np.vstack([A] + [np.asarray(l, np.float64).reshape(1, -1) for l, _ in cuts_sub])
        bs = np.concatenate([b] + [[float(r)] for _, r in cuts_sub])
        Ns, objs, oks = solve_nodes(As, bs, c)
        helped = 1.0 if (oks and abs(objs - obj0) < 1e-4 and Ns < N0) else 0.0
        gate_x.append(z0); gate_y.append(helped)

        pos = sum(1 for k in range(len(pool)) if valid_all[-len(pool)+k] and dnodes_all[-len(pool)+k] > 0)
        print(f"[{i+1}/{args.n_instances}] N0={N0} pool={len(pool)} "
              f"cuts-helping={pos}/{len(pool)} | subset {int(Ns)} vs {N0} "
              f"({'HELP' if helped else 'hurt/none'})", flush=True)

    # ---- summary + decisive scatter correlation ----
    dlp_s = np.array(dlp_scatter); dn_s = np.array(dn_scatter)
    print("\n" + "=" * 64, flush=True)
    print(f"labeled cuts: {len(feats_all)} ({int(np.sum(valid_all))} valid) | "
          f"gate examples: {len(gate_y)} (pos={int(np.sum(gate_y))})", flush=True)
    if dn_s.size:
        frac_help = float((dn_s > 0).mean())
        corr = float(np.corrcoef(dlp_s, dn_s)[0, 1]) if dn_s.size > 2 and dlp_s.std() > 0 else float("nan")
        print(f"valid cuts with ΔNodes>0 (help search): {100*frac_help:.1f}%", flush=True)
        print(f"corr(ΔLP, ΔNodes) = {corr:.3f}  "
              f"(near 0 => LP gain does NOT predict search benefit; "
              f"our earlier hypothesis)", flush=True)
        print(f"ΔNodes: mean {dn_s.mean():+.1f}, median {np.median(dn_s):+.1f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        feats=np.array(feats_all, np.float32),
        z=np.array(z_all, np.float32),
        dlp=np.array(dlp_all, np.float32),
        dnodes=np.array(dnodes_all, np.float32),
        valid=np.array(valid_all, bool),
        gate_z=np.array(gate_x, np.float32),
        gate_y=np.array(gate_y, np.float32),
    )
    print(f"\nSaved labels -> {args.out}", flush=True)
    print("Next: train the search-aware NO-CUT-gated evaluator on these labels.", flush=True)


if __name__ == "__main__":
    main()
