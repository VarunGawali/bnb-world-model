"""
diagnose_policy.py — Is the policy bad, or is the solver feeding it bad features?

The benchmark shows the learned policy needing 3-15x more nodes than
most-fractional. There are only two explanations, and this separates them.

PART A — the model on its OWN training states
    Replays recorded trajectory nodes through build_pyg_data (the exact path
    training used) and measures top-1 / top-3 agreement with the stored strong
    branching labels, plus Kendall tau against the stored sb_scores.

    Reference points printed alongside:
        random          1 / |action_set|
        most-fractional the simple rule the policy is losing to
        Phase-1 ValAcc  ~0.39 reported during training

    If top-1 is near the Phase-1 number -> THE MODEL IS FINE. The fault is in
    how the solver builds features, so go to Part B.
    If top-1 is near random -> the checkpoint or encoder is broken, and no
    amount of solver work will help.

PART B — the solver's own feature construction
    Runs NeuralBnBSolver on fresh instances, captures the [n_vars, 19] matrix
    it actually hands the encoder, and compares it column-by-column against the
    training distribution. This is the check validate_feature_mapping.py does
    for the Ecole path and nobody has ever done for the solver path.

Usage
-----
    python tools/diagnose_policy.py \\
        --checkpoint checkpoints/highs_retrain/phase4_best.pt \\
        --data_dir data/highs_trajectories/medium \\
        --n_traj 10 --n_nodes 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from bnb_wm.model.world_model import BnBWorldModel
from bnb_wm.training.checkpoint import load_weights_only
from bnb_wm.data.datasets import build_pyg_data

N_COLS = 19
COL_NAMES = [
    "obj_coef_norm", "has_lb", "has_ub", "sol_is_at_lb", "sol_is_at_ub",
    "basis_status", "reduced_cost_norm", "zero_7", "n_rows_norm",
    "zero_9", "zero_10", "zero_11", "zero_12", "sol_val", "sol_frac",
    "lp_obj_norm", "n_rows_tight_norm", "lb_value", "ub_value",
]


def load_model(args, device):
    mc = yaml.safe_load(open(args.config))["model"]
    model = BnBWorldModel(
        hidden_dim=mc["hidden_dim"], n_gnn_layers=mc["n_gnn_layers"],
        n_gnn_heads=mc["n_gnn_heads"], n_dyn_layers=mc["n_dyn_layers"],
        n_dyn_heads=mc.get("n_dyn_heads", 4), max_seq=mc.get("max_seq", 512),
    ).to(device)
    load_weights_only(model, args.checkpoint, device=device)
    model.eval()

    enc = model.encoder
    vm, vs = enc.var_mean.detach().cpu().numpy(), enc.var_std.detach().cpu().numpy()
    fitted = not (np.allclose(vm, 0) and np.allclose(vs, 1))
    print(f"\nencoder input standardisation: "
          f"{'FITTED' if fitted else 'NOT fitted (identity)'}")
    if fitted:
        print(f"  var_mean[:6] = {np.round(vm[:6], 4)}")
        print(f"  var_std [:6] = {np.round(vs[:6], 4)}")
    else:
        print("  (0/1 buffers -> no standardisation; consistent between train "
              "and deploy, so not itself a bug)")
    return model


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------

def part_a(model, device, args):
    print("\n" + "=" * 74)
    print("PART A -- policy accuracy on recorded TRAINING states")
    print("=" * 74)

    files = sorted(Path(args.data_dir).glob("**/*.npz"))
    files = [f for f in files if not f.name.endswith("_cut.npz")][:args.n_traj]
    if not files:
        raise SystemExit(f"no trajectories under {args.data_dir}")

    top1 = top3 = mf_top1 = rnd = n = 0
    taus, aset_sizes = [], []
    try:
        from scipy.stats import kendalltau
    except ImportError:
        kendalltau = None

    _debug_done = False

    for f in files:
        d = np.load(f, allow_pickle=True)
        T = int(d["n_steps"])
        for t in range(T):
            if n >= args.n_nodes:
                break
            vf = np.asarray(d["var_features"][t], dtype=np.float32)
            cf = np.asarray(d["con_features"][t], dtype=np.float32)
            ei = np.asarray(d["edge_indices"][t], dtype=np.int64)
            ev = np.asarray(d["edge_values"][t], dtype=np.float32)
            aset = np.asarray(d["action_sets"][t], dtype=np.int64)
            label = int(d["local_branching_label"][t])      # index INTO aset

            # sb_scores may be absent in older npz files
            sb_raw = d["sb_scores"][t] if "sb_scores" in d else None
            sb = np.asarray(sb_raw, dtype=np.float64) if sb_raw is not None else None

            if len(aset) < 2:
                continue

            data = build_pyg_data(vf, cf, ei, ev)
            from torch_geometric.data import Batch
            batch = Batch.from_data_list([data]).to(device)

            with torch.no_grad():
                h_vars, z = model.encode(batch)
                var_mask = batch.node_type == 0
                scores = model.policy_scores(h_vars, z, batch.batch[var_mask])

            raw = scores.detach().cpu().numpy()

            # ---- one-time debug dump ----
            if not _debug_done:
                _debug_done = True
                nan_count = int(np.isnan(raw).sum())
                inf_count = int(np.isinf(raw).sum())
                print(f"\n[DEBUG node 0]")
                print(f"  scores shape={raw.shape}  NaN={nan_count}  Inf={inf_count}")
                print(f"  scores[:5]       = {np.round(raw[:5], 4)}")
                print(f"  scores[aset[:5]] = {np.round(raw[aset[:5]], 4)}")
                print(f"  aset[:5]={aset[:5]}  label(local)={label}")
                print(f"  sb_scores present: {'yes, len=' + str(len(sb)) if sb is not None else 'NO'}")
                print(f"  vf shape={vf.shape}, ei shape={ei.shape}")
                print(f"  vf[0,:6] = {np.round(vf[0,:6], 4)}")
                if sb is not None:
                    print(f"  sb[:5]={np.round(sb[:5], 4)}  len(sb)={len(sb)}  len(aset)={len(aset)}")
                    print(f"  NOTE: sb aligned to {'aset' if len(sb)==len(aset) else 'full vars (n_cols)'}")
                print()

            s = raw[aset]
            order = np.argsort(-s)                      # local indices, best first
            top1 += int(order[0] == label)
            top3 += int(label in order[:3])

            # most-fractional on the same candidate set, same features
            frac = vf[aset, 14]
            mf_top1 += int(int(np.argmax(frac)) == label)

            rnd += 1.0 / len(aset)
            aset_sizes.append(len(aset))

            if kendalltau is not None and sb is not None and len(aset) >= 4:
                # sb may be aligned to aset (len k) or to full vars (len n_cols)
                sb_cand = sb[aset] if len(sb) != len(aset) else sb
                tau = kendalltau(s, sb_cand).statistic
                if np.isfinite(tau):
                    taus.append(float(tau))
            n += 1
        if n >= args.n_nodes:
            break

    if n == 0:
        raise SystemExit("no usable nodes found")

    print(f"\nnodes evaluated: {n}   median |action_set|: "
          f"{int(np.median(aset_sizes))}")
    print(f"\n{'metric':<34}{'value':>10}")
    print("-" * 46)
    print(f"{'policy top-1 vs SB argmax':<34}{top1 / n:>10.4f}")
    print(f"{'policy top-3 vs SB argmax':<34}{top3 / n:>10.4f}")
    print(f"{'most-fractional top-1':<34}{mf_top1 / n:>10.4f}")
    print(f"{'random top-1 (1/|aset|)':<34}{rnd / n:>10.4f}")
    if taus:
        print(f"{'Kendall tau vs sb_scores':<34}{float(np.mean(taus)):>10.4f}")

    p = top1 / n
    print("\nVERDICT")
    if p <= 1.5 * (rnd / n):
        print("  Policy is at/near RANDOM on its own training states.")
        print("  The fault is upstream of the solver: checkpoint, encoder or")
        print("  policy head. Fixing the solver will not help.")
    elif p < 0.5 * (mf_top1 / n):
        print("  Policy is well BELOW most-fractional on its own training")
        print("  states. The learned rule itself is weak -- this is a training")
        print("  problem, not a deployment problem.")
    else:
        print("  Policy performs on its training states roughly as Phase 1")
        print("  reported. THE MODEL IS FINE -- the solver is feeding it")
        print("  something different. See Part B.")
    return p, mf_top1 / n


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------

def part_b(model, device, args, train_files):
    print("\n" + "=" * 74)
    print("PART B -- features the SOLVER builds vs the training distribution")
    print("=" * 74)

    try:
        from bnb_wm.solver.neural_bnb import NeuralBnBSolver
        from bnb_wm.solver.config import SolverConfig
    except Exception as exc:
        print(f"  SKIPPED: cannot import the solver ({exc})")
        return

    captured: list[np.ndarray] = []

    class _Probe(NeuralBnBSolver):
        def _encode(self, sol, vlb, vub):
            st = self._structure()
            from bnb_wm.features import var_features
            vf = var_features(st["A"], st["b"], self._c, sol,
                              st["n_rows_per_var"])
            captured.append(np.asarray(vf, dtype=np.float64))
            return super()._encode(sol, vlb, vub)

    rng = np.random.default_rng(args.seed)
    cfg = SolverConfig(branch_mode="policy", cut_mode="none",
                       time_limit=20.0, node_limit=200)
    solver = _Probe(model, device, cfg)

    for _ in range(args.n_solver_instances):
        A = (rng.random((args.n_rows, args.n_cols)) < args.density).astype(np.float64)
        for i in range(args.n_rows):
            if A[i].sum() == 0:
                A[i, rng.integers(args.n_cols)] = 1.0
        b = np.ones(args.n_rows)
        c = rng.uniform(1.0, 10.0, size=args.n_cols)
        try:
            solver.solve(A, b, c)
        except Exception as exc:
            print(f"  solve failed: {exc}")
            break

    if not captured:
        print("  no features captured")
        return

    S = np.concatenate(captured, axis=0)

    chunks = []
    for f in train_files:
        d = np.load(f, allow_pickle=True)
        for t in range(int(d["n_steps"])):
            arr = np.asarray(d["var_features"][t], dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] == N_COLS:
                chunks.append(arr)
    if not chunks:
        print("  no training var_features found to compare against")
        return
    Tm = np.concatenate(chunks, axis=0)

    print(f"\nsolver rows: {S.shape[0]}   training rows: {Tm.shape[0]}")
    print(f"\n{'col':>3} {'name':<20}{'train mean':>12}{'solver mean':>13}"
          f"{'train range':>22}{'solver range':>22}  verdict")
    print("-" * 110)

    bad = []
    for j in range(N_COLS):
        tv, sv = Tm[:, j], S[:, j]
        tconst = len(np.unique(np.round(tv, 6))) == 1
        sconst = len(np.unique(np.round(sv, 6))) == 1
        v = "PASS"
        if tconst != sconst:
            v = "FAIL"
        elif tconst and sconst and abs(tv[0] - sv[0]) > 1e-6:
            v = "FAIL"
        elif not tconst:
            if sv.min() > tv.max() + 1e-9 or sv.max() < tv.min() - 1e-9:
                v = "FAIL"
            else:
                den = max(abs(tv.mean()), abs(sv.mean()), 1e-8)
                if abs(tv.mean() - sv.mean()) / den > args.tol:
                    v = "WARN"
        if v == "FAIL":
            bad.append(COL_NAMES[j])
        print(f"{j:>3} {COL_NAMES[j]:<20}{tv.mean():>12.5f}{sv.mean():>13.5f}"
              f"{f'[{tv.min():.4g}, {tv.max():.4g}]':>22}"
              f"{f'[{sv.min():.4g}, {sv.max():.4g}]':>22}  {v}")

    print("\n" + ("SOLVER FEATURES MATCH TRAINING (no FAIL columns)."
                  if not bad else
                  f"MISMATCHED COLUMNS: {bad}\n"
                  "  The encoder is seeing values it never saw in training. "
                  "Fix these before\n  drawing any conclusion about the model."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=str(_REPO / "configs" / "default.yaml"))
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n_traj", type=int, default=10)
    ap.add_argument("--n_nodes", type=int, default=400)
    ap.add_argument("--n_solver_instances", type=int, default=3)
    ap.add_argument("--n_rows", type=int, default=100)
    ap.add_argument("--n_cols", type=int, default=200)
    ap.add_argument("--density", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tol", type=float, default=0.25)
    ap.add_argument("--skip_b", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)

    part_a(model, device, args)

    if not args.skip_b:
        files = sorted(Path(args.data_dir).glob("**/*.npz"))
        files = [f for f in files if not f.name.endswith("_cut.npz")][:args.n_traj]
        part_b(model, device, args, files)


if __name__ == "__main__":
    main()
