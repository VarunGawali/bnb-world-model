"""
validate_feature_mapping.py — Prove the Ecole->training feature remap is correct.

check_feature_layout.py identified the column layouts from a SINGLE root node.
This script validates the resulting mapping distributionally, across many nodes
on both sides, and flags every column where eval features still do not look like
training features.

What it checks
--------------
A. TRAINING side, across ALL recorded nodes of N trajectories (not just root):
     - which columns are genuinely constant everywhere (the remap hardcodes
       several constants that were only verified at the root)
     - whether sol_is_at_ub (col 4) is ever nonzero at depth
     - whether basis_status (col 5) ever takes the value 2
     - per-column min / max / mean / p05 / p95

B. EVAL side, stepping an Ecole env for M nodes and applying the remap:
     - the same per-column statistics
     - edge_index orientation
     - whether Ecole's raw edge values are 1.0 (as training assumes)

C. SIDE-BY-SIDE comparison, per column, with a PASS/WARN/FAIL verdict:
     - FAIL: one side is constant and the other is not, or the ranges are
       disjoint -- the model sees a feature it never saw in training
     - WARN: ranges overlap but means differ by more than --tol (relative)
     - PASS: distributions are comparable

Usage
-----
    python tools/validate_feature_mapping.py \\
        --data_dir data/highs_trajectories/medium \\
        --n_traj 20 --n_eval_nodes 200

    # Match the eval instance size to the training tier:
    python tools/validate_feature_mapping.py \\
        --data_dir data/highs_trajectories/medium \\
        --n_rows 500 --n_cols 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

N_COLS = 19

COL_NAMES = [
    "obj_coef_norm", "has_lb", "has_ub", "sol_is_at_lb", "sol_is_at_ub",
    "basis_status", "reduced_cost_norm", "zero_7", "n_rows_norm",
    "zero_9", "zero_10", "zero_11", "zero_12", "sol_val", "sol_frac",
    "lp_obj_norm", "n_rows_tight_norm", "lb_value", "ub_value",
]


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def column_stats(M: np.ndarray) -> list[dict]:
    """Per-column summary for a stacked [N, 19] feature matrix."""
    out = []
    for j in range(M.shape[1]):
        v = M[:, j]
        v = v[np.isfinite(v)]
        if v.size == 0:
            out.append(dict(min=np.nan, max=np.nan, mean=np.nan,
                            p05=np.nan, p95=np.nan, n_uniq=0, const=False))
            continue
        uniq = np.unique(np.round(v, 6))
        out.append(dict(
            min=float(v.min()), max=float(v.max()), mean=float(v.mean()),
            p05=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)),
            n_uniq=int(len(uniq)), const=bool(len(uniq) == 1),
        ))
    return out


def verdict(t: dict, e: dict, tol: float) -> tuple[str, str]:
    """Compare one column's training vs eval stats. Returns (tag, reason)."""
    if t["n_uniq"] == 0 or e["n_uniq"] == 0:
        return "FAIL", "no data on one side"

    if t["const"] != e["const"]:
        side = "train" if t["const"] else "eval"
        return "FAIL", f"{side} is constant, the other is not"

    if t["const"] and e["const"]:
        if abs(t["min"] - e["min"]) < 1e-6:
            return "PASS", ""
        return "FAIL", f"constants differ ({t['min']:.4g} vs {e['min']:.4g})"

    # Disjoint ranges -> the encoder sees values it never saw in training.
    if e["min"] > t["max"] + 1e-9 or e["max"] < t["min"] - 1e-9:
        return "FAIL", "ranges are disjoint"

    denom = max(abs(t["mean"]), abs(e["mean"]), 1e-8)
    rel = abs(t["mean"] - e["mean"]) / denom
    if rel > tol:
        return "WARN", f"means differ by {rel * 100:.0f}%"

    # Range-width mismatch catches normalisation errors the mean can hide.
    tw = t["p95"] - t["p05"]
    ew = e["p95"] - e["p05"]
    if max(tw, ew) > 1e-8:
        wr = abs(tw - ew) / max(tw, ew, 1e-8)
        if wr > tol:
            return "WARN", f"p05-p95 spread differs by {wr * 100:.0f}%"

    return "PASS", ""


# ---------------------------------------------------------------------------
# Part A -- training side
# ---------------------------------------------------------------------------

def load_training_features(data_dir: Path, n_traj: int):
    files = sorted(data_dir.glob("**/*.npz"))
    files = [f for f in files if not f.name.endswith("_cut.npz")][:n_traj]
    if not files:
        raise SystemExit(f"No trajectory .npz files under {data_dir}")

    print(f"\n[Part A] Reading {len(files)} trajectory file(s) from {data_dir}")

    chunks, n_nodes = [], 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        vfs = d["var_features"]
        for t in range(len(vfs)):
            arr = np.asarray(vfs[t], dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] == N_COLS:
                chunks.append(arr)
                n_nodes += 1
    if not chunks:
        raise SystemExit("No usable var_features found.")

    M = np.concatenate(chunks, axis=0)
    print(f"[Part A] {n_nodes} recorded nodes, {M.shape[0]} variable rows total")

    # The remap hardcodes constants that were only checked at the root. Verify
    # them against EVERY recorded node.
    print("\n[Part A] Hardcoded-constant audit (across ALL nodes, not just root)")
    hardcoded = {1: 1.0, 2: 1.0, 4: 0.0, 7: 0.0, 9: 0.0, 10: 0.0,
                 11: 0.0, 12: 0.0, 15: 1.0, 17: 0.0, 18: 1.0}
    any_bad = False
    for j, expected in sorted(hardcoded.items()):
        v = M[:, j]
        ok = bool(np.allclose(v, expected, atol=1e-6))
        if ok:
            print(f"   col {j:>2} {COL_NAMES[j]:<20} == {expected:<5} OK")
        else:
            any_bad = True
            frac_off = float((np.abs(v - expected) > 1e-6).mean())
            print(f"   col {j:>2} {COL_NAMES[j]:<20} == {expected:<5} "
                  f"** NOT CONSTANT ** ({frac_off*100:.2f}% of rows differ, "
                  f"range [{v.min():.4g}, {v.max():.4g}])")
    if any_bad:
        print("   -> the remap must COMPUTE these columns, not hardcode them.")

    b = M[:, 5]
    print(f"\n[Part A] basis_status values seen: "
          f"{np.unique(np.round(b, 6))[:8]}")
    if not np.any(np.abs(b - 2.0) < 1e-6):
        print("   value 2 (at upper bound) never occurs -> "
              "mapping from ecole col 17 is unexercised but harmless")
    else:
        print("   value 2 DOES occur -> ecole col 17 (is_basis_upper) matters")

    return M


# ---------------------------------------------------------------------------
# Part B -- eval side
# ---------------------------------------------------------------------------

def collect_eval_features(args):
    try:
        import ecole
    except ImportError:
        print("\n[Part B] SKIPPED: ecole is not importable here.")
        return None

    from bnb_wm.features import ecole_to_train_layout

    print(f"\n[Part B] Stepping an Ecole env for up to {args.n_eval_nodes} nodes "
          f"({args.n_rows}x{args.n_cols})")

    gen = ecole.instance.SetCoverGenerator(
        n_rows=args.n_rows, n_cols=args.n_cols, density=args.density)
    gen.seed(args.seed)

    env = ecole.environment.Branching(
        observation_function=ecole.observation.NodeBipartite(),
        scip_params={"separating/maxrounds": 0, "presolving/maxrounds": 0,
                     "limits/time": args.time_limit},
    )

    chunks = []
    n_nodes = 0
    checked_edges = False
    rng = np.random.default_rng(args.seed)

    while n_nodes < args.n_eval_nodes:
        obs, action_set, _, done, _ = env.reset(next(gen))
        while not done and action_set is not None and len(action_set) > 0:
            if obs is None:
                break
            vf_raw = np.array(obs.variable_features, dtype=np.float32)
            cf_ec = np.array(obs.constraint_features, dtype=np.float32)
            ei = np.array(obs.edge_features.indices, dtype=np.int64)
            ev = np.array(obs.edge_features.values, dtype=np.float32).flatten()

            if not checked_edges:
                checked_edges = True
                nv, nc = vf_raw.shape[0], cf_ec.shape[0]
                print(f"\n[Part B] edge_index orientation check "
                      f"(n_vars={nv}, n_cons={nc})")
                print(f"   ei[0].max()={ei[0].max()}  ei[1].max()={ei[1].max()}")
                if ei[0].max() < nc and ei[1].max() < nv:
                    print("   -> ei[0]=constraint, ei[1]=variable  CORRECT")
                elif ei[1].max() < nc and ei[0].max() < nv:
                    print("   -> ** FLIPPED **: ei[0]=variable, ei[1]=constraint.")
                    print("      n_rows_norm and n_rows_tight_norm are WRONG.")
                else:
                    print("   -> ambiguous (n_vars and n_cons too close to tell)")

                uniq_ev = np.unique(np.round(ev, 6))
                print(f"\n[Part B] Ecole edge values: {len(uniq_ev)} unique, "
                      f"first few {uniq_ev[:5]}")
                if len(uniq_ev) == 1 and abs(uniq_ev[0] - 1.0) < 1e-6:
                    print("   -> all 1.0, matches training edge_attr. OK")
                else:
                    print("   -> ** NOT all 1.0 **. Training edge_attr is "
                          "[1,1,1] for set-cover; eval will differ.")
                    print("      Consider ev = np.sign(ev) in _format_obs.")

            vf, _cf = ecole_to_train_layout(
                np.nan_to_num(vf_raw), np.nan_to_num(cf_ec), ei)
            chunks.append(np.asarray(vf, dtype=np.float64))
            n_nodes += 1
            if n_nodes >= args.n_eval_nodes:
                break

            action = int(action_set[rng.integers(len(action_set))])
            obs, action_set, _, done, _ = env.step(action)

    if not chunks:
        print("[Part B] No eval nodes collected.")
        return None

    M = np.concatenate(chunks, axis=0)
    print(f"\n[Part B] {n_nodes} eval nodes, {M.shape[0]} variable rows total")
    return M


# ---------------------------------------------------------------------------
# Part C -- comparison
# ---------------------------------------------------------------------------

def compare(train_M, eval_M, tol):
    print("\n" + "=" * 100)
    print("PART C -- training vs remapped-eval, per column")
    print("=" * 100)

    ts = column_stats(train_M)
    es = column_stats(eval_M)

    print(f"{'col':>3} {'name':<20} "
          f"{'train mean':>11} {'eval mean':>11} "
          f"{'train range':>20} {'eval range':>20}  {'verdict':<6} reason")
    print("-" * 125)

    fails, warns = [], []
    for j in range(N_COLS):
        t, e = ts[j], es[j]
        tag, why = verdict(t, e, tol)
        if tag == "FAIL":
            fails.append((j, why))
        elif tag == "WARN":
            warns.append((j, why))
        tr = f"[{t['min']:.4g}, {t['max']:.4g}]"
        er = f"[{e['min']:.4g}, {e['max']:.4g}]"
        print(f"{j:>3} {COL_NAMES[j]:<20} "
              f"{t['mean']:>11.5f} {e['mean']:>11.5f} "
              f"{tr:>20} {er:>20}  {tag:<6} {why}")

    print("\n" + "=" * 100)
    if not fails and not warns:
        print("ALL COLUMNS PASS -- eval features now look like training features.")
    else:
        if fails:
            print(f"{len(fails)} FAIL column(s): "
                  f"{[COL_NAMES[j] for j, _ in fails]}")
            print("  These are features the encoder never saw in this form "
                  "during training. Fix before trusting any eval number.")
        if warns:
            print(f"{len(warns)} WARN column(s): "
                  f"{[COL_NAMES[j] for j, _ in warns]}")
            print("  Semantics look right but scale/distribution differs. Often "
                  "a normalisation mismatch, or a genuine difference between "
                  "the training instance generator and Ecole's SetCoverGenerator.")
    print("=" * 100)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True,
                   help="Directory of collect_highs trajectory .npz files")
    p.add_argument("--n_traj", type=int, default=20)
    p.add_argument("--n_eval_nodes", type=int, default=200)
    p.add_argument("--n_rows", type=int, default=500)
    p.add_argument("--n_cols", type=int, default=1000)
    p.add_argument("--density", type=float, default=0.05)
    p.add_argument("--time_limit", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=0.25,
                   help="Relative tolerance on mean/spread before WARN (0.25 = 25%%)")
    args = p.parse_args()

    train_M = load_training_features(Path(args.data_dir), args.n_traj)
    eval_M = collect_eval_features(args)
    if eval_M is None:
        print("\nPart B unavailable; skipping comparison.")
        return
    compare(train_M, eval_M, args.tol)


if __name__ == "__main__":
    main()
