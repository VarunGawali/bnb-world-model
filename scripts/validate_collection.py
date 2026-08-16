#!/usr/bin/env python
"""
validate_collection.py — Robustness check for collected trajectory .npz files.

Runs a battery of assertions over every field the training pipeline consumes, so
data problems are caught HERE (loudly, per-file) instead of silently corrupting
training. Use it after any collection run, especially before a full recollection.

Usage:
    python scripts/validate_collection.py data/trajectories            # all files
    python scripts/validate_collection.py data/trajectories --limit 50 # first 50
    python scripts/validate_collection.py data/trajectories --strict   # warnings fail

Exit code is non-zero if any hard check fails, so it can gate a pipeline.
"""

from __future__ import annotations

import argparse
import glob
import sys
from collections import Counter

import numpy as np

REQUIRED_KEYS = [
    "n_steps", "var_features", "con_features", "edge_indices", "edge_values",
    "action_sets", "branching_vars", "local_branching_label", "dual_bounds",
    "depths", "node_ids", "parent_ids", "branch_dirs", "next_is_leaf",
    "root_bound", "primal_bound", "optimal_valid", "n_cuts", "cut_features",
    "cut_labels",
]


def _root_to_leaf_ok(node_ids, parent_ids):
    """Every recorded non-root node's parent must resolve, or be an unrecorded
    fragment root; and the graph must be acyclic (no node is its own ancestor)."""
    id2idx = {int(n): i for i, n in enumerate(node_ids)}
    for i in range(len(node_ids)):
        seen, cur = set(), i
        while cur is not None and cur not in seen:
            seen.add(cur)
            pid = int(parent_ids[cur])
            cur = id2idx.get(pid)
        if cur is not None:      # loop terminated by revisiting -> cycle
            return False
    return True


def validate_file(path, strict=False):
    """Return (hard_errors, warnings) lists of strings for one file."""
    errs, warns = [], []
    d = np.load(path, allow_pickle=True)

    missing = [k for k in REQUIRED_KEYS if k not in d]
    if missing:
        return [f"missing keys: {missing}"], warns   # can't check further

    T = int(d["n_steps"])
    if T < 1:
        return [f"n_steps={T}"], warns

    def _len(k):
        return len(np.asarray(d[k]))

    for k in ("var_features", "branching_vars", "local_branching_label",
              "dual_bounds", "depths", "node_ids", "parent_ids", "branch_dirs",
              "next_is_leaf", "n_cuts"):
        if _len(k) != T:
            errs.append(f"{k} length {_len(k)} != n_steps {T}")

    # depths: non-negative small integers (true tree depth, NOT node ids).
    depths = np.asarray(d["depths"], dtype=np.int64)
    if depths.min() < 0:
        errs.append(f"negative depth {depths.min()}")
    if depths.max() > 1e5:
        errs.append(f"depth {depths.max()} looks like a node id, not tree depth")
    node_ids = np.asarray(d["node_ids"], dtype=np.int64)
    if depths.max() == node_ids.max() and node_ids.max() > 100:
        warns.append("max depth == max node_id — depth may still be the node id")

    # branch_dirs strictly in {-1,0,1}.
    bd = set(np.unique(np.asarray(d["branch_dirs"])).tolist())
    if not bd <= {-1, 0, 1}:
        errs.append(f"branch_dirs has values outside {{-1,0,1}}: {bd}")

    # next_is_leaf in {0,1}.
    nil = set(np.unique(np.asarray(d["next_is_leaf"]).astype(int)).tolist())
    if not nil <= {0, 1}:
        errs.append(f"next_is_leaf not binary: {nil}")

    # bounds finite.
    db = np.asarray(d["dual_bounds"], dtype=np.float64)
    if not np.all(np.isfinite(db)):
        errs.append("non-finite dual_bounds")

    # primal anchor consistency (only when the collector proved optimality).
    optimal_valid = bool(np.asarray(d["optimal_valid"]))
    root_bound = float(np.asarray(d["root_bound"]))
    primal = float(np.asarray(d["primal_bound"]))
    if optimal_valid:
        # A single node's local LP bound may LEGITIMATELY exceed the optimum
        # (prune-able nodes B&B branched before proving optimality). The real
        # invariant for a minimisation problem is that the BEST (lowest) observed
        # bound — a valid global lower bound, ~the root relaxation — is <= optimum.
        global_lb = float(db.min())
        if not np.isfinite(primal):
            errs.append("optimal_valid=True but primal_bound not finite")
        elif global_lb > primal + 1e-6 * (abs(primal) + 1):
            errs.append(f"global LB {global_lb:.4g} exceeds optimum {primal:.4g} "
                        "(space mismatch — anchor should be invalid)")
    else:
        warns.append("optimal_valid=False (optimum not proven; anchor fallback)")

    # branching label indexes its action set (P0.1).
    for t in range(T):
        aset = np.asarray(d["action_sets"][t])
        lbl = int(d["local_branching_label"][t])
        if not (0 <= lbl < len(aset)):
            errs.append(f"step {t}: local label {lbl} out of range [0,{len(aset)})")
            break
        if int(d["branching_vars"][t]) != int(aset[lbl]):
            errs.append(f"step {t}: branching_var != action_set[label]")
            break

    # tree is reconstructable and acyclic (P0.3).
    if not _root_to_leaf_ok(node_ids, np.asarray(d["parent_ids"], dtype=np.int64)):
        errs.append("parent_ids contain a cycle — path reconstruction unsafe")

    # full-tree capture (collector v3 Stage 1), only when present.
    if "full_node_ids" in d:
        full = set(np.asarray(d["full_node_ids"]).tolist())
        rec = set(node_ids.tolist())
        missing = rec - full
        if missing:
            errs.append(f"{len(missing)} recorded nodes absent from full tree "
                        "(event handler missed nodes — widen event mask)")
        if "true_next_is_leaf" in d:
            tnl = set(np.unique(np.asarray(d["true_next_is_leaf"]).astype(int)).tolist())
            if not tnl <= {0, 1}:
                errs.append(f"true_next_is_leaf not binary: {tnl}")
        else:
            errs.append("full_node_ids present but true_next_is_leaf missing")

    # cut sanity (P0.4/P0.7).
    ncuts = np.asarray(d["n_cuts"], dtype=np.int64)
    total_cuts = int(ncuts.sum())
    if total_cuts == 0:
        warns.append("no cuts recorded (highspy missing? branching-only?)")
    else:
        for t in range(T):
            if int(ncuts[t]) > 0:
                cf = np.asarray(d["cut_features"][t])
                if cf.ndim != 2 or cf.shape[1] != 6:
                    errs.append(f"step {t}: cut_features shape {cf.shape} != [_,6]")
                    break
    return errs, warns


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", help="directory of traj_*.npz (searched recursively)")
    ap.add_argument("--limit", type=int, default=0, help="check only first N files")
    ap.add_argument("--strict", action="store_true", help="warnings count as failures")
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.root}/**/traj_*.npz", recursive=True))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"No traj_*.npz under {args.root}")
        sys.exit(2)

    n_bad = 0
    err_counter = Counter()
    warn_counter = Counter()
    for f in files:
        try:
            errs, warns = validate_file(f, strict=args.strict)
        except Exception as e:
            errs, warns = [f"exception: {e}"], []
        for e in errs:
            err_counter[e.split(":")[0].split("(")[0].strip()] += 1
        for w in warns:
            warn_counter[w.split("(")[0].strip()] += 1
        if errs or (args.strict and warns):
            n_bad += 1
            print(f"[BAD] {f}")
            for e in errs:
                print(f"       ERROR: {e}")
            if args.strict:
                for w in warns:
                    print(f"       WARN:  {w}")

    print(f"\nChecked {len(files)} files | {len(files) - n_bad} clean | {n_bad} flagged")
    if err_counter:
        print("Error summary:")
        for k, v in err_counter.most_common():
            print(f"  {v:5d}  {k}")
    if warn_counter:
        print("Warning summary:")
        for k, v in warn_counter.most_common():
            print(f"  {v:5d}  {k}")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
