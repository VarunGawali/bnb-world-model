"""
cut_oracle_subset.py — Does the current cut POOL contain a useful SUBSET?

Our per-cut labels showed individual root-Gomory cuts rarely help (11.9%), and a
violation-ranked top-k subset helped 0/6. But subset value != sum of individual
values: cut INTERACTIONS (individually-neutral cuts that jointly tighten) are
exactly what HEM/NeuralCut exploit. This script measures the BEST ACHIEVABLE
subset, to separate three outcomes:

  A  best-subset << baseline           -> selection is the problem (build selector)
  B  individuals <=0 but subset > 0     -> cut INTERACTIONS exist (HEM justified)
  C  even best subset ~= baseline       -> generation is the wall (go to SCIP pool)

Method (bounded, since each subset eval = one standalone B&B solve):
  * N0 = nodes under learned branching, no cuts.
  * Pre-filter the pool to the top `--pool_cap` cuts by violation (bounds cost).
  * GREEDY FORWARD selection with beam width `--beam`: at each step try adding
    each remaining cut, keep the `--beam` partial subsets with fewest nodes,
    up to cardinality `--k_max`. Track the best subset found at each k.
  * Every evaluated subset must preserve the optimum (obj == obj0), else discard.

Also a cheap POOL AUDIT (no B&B): pool size, per-instance N0, individual-cut
validity rate, redundancy (mean |cosine| between cut normals), #positive cuts.
This also investigates the pool=0 phenomenon (is it only trivial instances?).

NOTE: relative node counts from the standalone Python-LP solver under LEARNED
branching. A diagnosis tool, not a SCIP-time benchmark.

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/cut_oracle_subset.py \
        --checkpoint checkpoints/model_rl_best.pt \
        --n_instances 15 --n_rows 200 --n_cols 400 \
        --pool_cap 12 --k_max 6 --beam 2 --node_limit 5000 --time_limit 45
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
from bnb_wm.solver.gomory import generate_root_gomory_cuts
from scripts.validate_cuts import _extract_Abc
from scripts.collect_with_cuts_v2 import _cut_features

try:
    import highspy
except Exception:
    highspy = None


def _redundancy(pool):
    """Mean pairwise |cosine| between cut normals (1 = all parallel/redundant)."""
    if len(pool) < 2:
        return 0.0
    N = np.array([np.asarray(l, np.float64) for l, _ in pool])
    N = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)
    G = np.abs(N @ N.T)
    iu = np.triu_indices(len(pool), k=1)
    return float(G[iu].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n_instances", type=int, default=15)
    ap.add_argument("--n_rows", type=int, default=200)
    ap.add_argument("--n_cols", type=int, default=400)
    ap.add_argument("--density", type=float, default=0.05)
    ap.add_argument("--max_pool", type=int, default=40, help="max Gomory cuts generated")
    ap.add_argument("--pool_cap", type=int, default=12,
                    help="cap candidates entering the subset search (top by violation)")
    ap.add_argument("--k_max", type=int, default=6, help="max subset cardinality")
    ap.add_argument("--beam", type=int, default=2, help="beam width for greedy forward search")
    ap.add_argument("--node_limit", type=int, default=5000)
    ap.add_argument("--time_limit", type=int, default=45)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/cut_oracle_subset.json")
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
    print(f"Loaded {args.checkpoint} on {device}", flush=True)

    gen = ecole.instance.SetCoverGenerator(
        n_rows=args.n_rows, n_cols=args.n_cols, density=args.density)
    gen.seed(args.seed)

    def solve(A, b, c):
        s = BnBSolver(model, device, time_limit=args.time_limit,
                      node_limit=args.node_limit, cut_mode="none")
        r = s.solve(A, b, c)
        return r.n_nodes, float(r.objective), (r.status == "optimal")

    def eval_subset(A, b, c, cuts, obj0):
        if cuts:
            As = np.vstack([A] + [np.asarray(l, np.float64).reshape(1, -1) for l, _ in cuts])
            bs = np.concatenate([b] + [[float(r)] for _, r in cuts])
        else:
            As, bs = A, b
        N, obj, ok = solve(As, bs, c)
        valid = ok and abs(obj - obj0) < 1e-4
        return (N if valid else None)

    audit = []          # per instance: dict of pool stats
    oracle = []         # per instance with pool>0: N0, best1, best_by_k
    for i in range(args.n_instances):
        inst = next(gen)
        try:
            A, b, c = _extract_Abc(inst.copy_orig().as_pyscipopt())
        except Exception as e:
            print(f"[{i+1}] extract failed: {e}", flush=True); continue
        n = len(c)
        N0, obj0, ok0 = solve(A, b, c)
        pool = generate_root_gomory_cuts(A, b, c, highspy, max_cuts=args.max_pool) if ok0 else []

        rec = {"N0": N0, "solved": ok0, "pool": len(pool)}
        if not ok0:
            print(f"[{i+1}] baseline unsolved (N0={N0}) -> skip", flush=True)
            audit.append(rec); continue
        if not pool:
            print(f"[{i+1}] pool=0 | N0={N0} (baseline solved, generator produced no cuts)",
                  flush=True)
            audit.append(rec); continue

        rec["redundancy"] = _redundancy(pool)
        audit.append(rec)

        # pre-filter to top pool_cap by violation to bound the search cost
        try:
            import scipy.optimize as so
            res = so.linprog(c, A_ub=-A, b_ub=-b, bounds=[(0, 1)] * n, method="highs")
            x_lp = res.x if res.success else np.zeros(n)
        except Exception:
            x_lp = np.zeros(n)
        viol = np.array([_cut_features(np.asarray(l), float(r), x_lp, c, n)[0]
                         for (l, r) in pool])
        keep = list(np.argsort(-viol)[:args.pool_cap])
        cand = [pool[j] for j in keep]

        # individual best (best-1)
        indiv = {}
        for j, cut in enumerate(cand):
            Nj = eval_subset(A, b, c, [cut], obj0)
            if Nj is not None:
                indiv[j] = Nj
        best1 = min(indiv.values()) if indiv else N0

        # greedy forward with beam: beams = list of (subset_idx_tuple, nodes)
        beams = [((), N0)]
        best_by_k = {0: N0}
        for k in range(1, args.k_max + 1):
            candidates = []
            for (subset, _) in beams:
                for j in range(len(cand)):
                    if j in subset:
                        continue
                    new = tuple(sorted(subset + (j,)))
                    Nn = eval_subset(A, b, c, [cand[t] for t in new], obj0)
                    if Nn is not None:
                        candidates.append((new, Nn))
            if not candidates:
                break
            # dedup + keep best `beam`
            seen = {}
            for s, nn in candidates:
                if s not in seen or nn < seen[s]:
                    seen[s] = nn
            beams = sorted(seen.items(), key=lambda kv: kv[1])[:args.beam]
            best_by_k[k] = beams[0][1]

        best_subset = min(best_by_k.values())
        oracle.append({"N0": N0, "best1": best1, "best_subset": best_subset,
                       "best_by_k": best_by_k, "pool": len(pool),
                       "redundancy": rec["redundancy"]})
        print(f"[{i+1}] N0={N0} pool={len(pool)} cap={len(cand)} "
              f"best1={best1} best_subset={best_subset} "
              f"(Δ={N0-best_subset:+d}) redund={rec['redundancy']:.2f}", flush=True)

    # ---- summary ----
    print("\n" + "=" * 66, flush=True)
    n_pool0 = sum(1 for r in audit if r.get("solved") and r["pool"] == 0)
    n_solved = sum(1 for r in audit if r.get("solved"))
    print(f"AUDIT: {n_solved} baseline-solved | pool=0 on {n_pool0}/{n_solved} "
          f"(generator produced no cuts even though baseline solved)", flush=True)
    if oracle:
        N0m = np.mean([o["N0"] for o in oracle])
        b1m = np.mean([o["best1"] for o in oracle])
        bsm = np.mean([o["best_subset"] for o in oracle])
        headroom = np.mean([1.0 if o["best_subset"] < o["N0"] else 0.0 for o in oracle])
        interact = np.mean([1.0 if o["best_subset"] < o["best1"] - 1e-9 else 0.0 for o in oracle])
        print(f"\nORACLE over {len(oracle)} instances with pool>0:", flush=True)
        print(f"  mean N0            = {N0m:.1f}", flush=True)
        print(f"  mean best-1        = {b1m:.1f}  (best single cut)", flush=True)
        print(f"  mean best-subset   = {bsm:.1f}  (greedy/beam oracle)", flush=True)
        print(f"  instances where best-subset < N0        : {100*headroom:.0f}%", flush=True)
        print(f"  instances where best-subset < best-1    : {100*interact:.0f}%  "
              f"(cut-interaction evidence)", flush=True)
        red = np.mean([o["redundancy"] for o in oracle])
        print(f"  mean pool redundancy (|cos|)            : {red:.2f}", flush=True)
        print("\nVERDICT:", flush=True)
        if bsm < 0.9 * N0m:
            v = ("A/B: the pool CONTAINS a useful subset -> selection/interaction "
                 "is the problem. Build the hierarchical selector.")
        elif b1m < 0.98 * N0m or bsm < 0.98 * N0m:
            v = ("Weak headroom: subsets help only marginally; likely need a "
                 "richer pool (SCIP) to matter.")
        else:
            v = ("C: even the ORACLE subset ~= baseline -> GENERATION is the wall. "
                 "Move to SCIP's cut pool (or reconsider the pool/generator).")
        print("  " + v, flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    import json
    json.dump({"audit": audit, "oracle": oracle}, open(args.out, "w"),
              indent=2, default=str)
    print(f"\nSaved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
