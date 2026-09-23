#!/usr/bin/env python
"""
check_feature_layout.py — Identify the column layout of Ecole's NodeBipartite
observation and compare it against the layout collect_highs.py writes (which is
what the model was actually trained on).

Why this exists
---------------
Three different 19-dim variable-feature layouts exist in this project:

  1. collect_highs.py  -> what the model was TRAINED on (via build_pyg_data)
  2. Ecole NodeBipartite -> what ablation.py serves at eval (via _format_obs)
  3. bnb_solver._encode_node -> what the Python solver serves

If (2) and (3) do not match (1), the encoder receives permuted inputs and every
downstream head is reading the wrong feature in every slot.

This script identifies (1) and (2) EMPIRICALLY -- no reliance on documentation
or on remembered column orderings -- and prints the permutation needed to map
Ecole columns onto the training layout.

How identification works
------------------------
Fractionality is self-identifying: there is exactly one pair of columns (a, b)
in any LP-based layout where  b == |a - round(a)|  elementwise and b is bounded
in [0, 0.5].  That pins sol_val and sol_frac without any external reference.
Basis-status columns are the 0/1 columns that are mutually exclusive and sum to
1 across the basis group.  Reduced cost is the continuous column that takes
negative values and is not sol_val.

Usage
-----
    # Part A only (Ecole layout) -- needs ecole + pyscipopt:
    python tools/check_feature_layout.py

    # Part A + Part B (also verify the collector layout from a real trajectory):
    python tools/check_feature_layout.py --npz data/highs_trajectories/medium/traj_00000.npz

    # Smaller/faster instance:
    python tools/check_feature_layout.py --n_rows 100 --n_cols 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# The layout collect_highs.py writes, i.e. what the model was trained on.
# (name, is_constant) -- constant columns carry a fixed value, not a feature.
TRAIN_LAYOUT = [
    (0,  "obj_coef_norm",      None),
    (1,  "has_lb",             1.0),
    (2,  "has_ub",             1.0),
    (3,  "sol_is_at_lb",       None),
    (4,  "sol_is_at_ub",       None),
    (5,  "basis_status_012",   None),
    (6,  "reduced_cost_norm",  None),
    (7,  "zero",               0.0),
    (8,  "n_rows_norm",        None),
    (9,  "zero_obj_sense",     0.0),
    (10, "zero_col_age",       0.0),
    (11, "zero_incumbent",     0.0),
    (12, "zero_avg_incumbent", 0.0),
    (13, "sol_val",            None),
    (14, "sol_frac",           None),
    (15, "lp_obj_norm",        None),
    (16, "n_rows_tight_norm",  None),
    (17, "lb_value",           0.0),
    (18, "ub_value",           1.0),
]


# ---------------------------------------------------------------------------
# Column fingerprinting
# ---------------------------------------------------------------------------

def describe_columns(M: np.ndarray, label: str) -> None:
    """Print per-column statistics for a [n, d] feature matrix."""
    print(f"\n--- {label}: {M.shape[0]} rows x {M.shape[1]} cols ---")
    print(f"{'col':>4} {'min':>10} {'max':>10} {'mean':>10} "
          f"{'#uniq':>7}  {'guess':<28}")
    print("-" * 78)
    for j in range(M.shape[1]):
        v = M[:, j]
        finite = v[np.isfinite(v)]
        if finite.size == 0:
            print(f"{j:>4} {'all-nonfinite':>32}")
            continue
        uniq = np.unique(np.round(finite, 6))
        guess = _guess_column(finite, uniq)
        print(f"{j:>4} {finite.min():>10.4f} {finite.max():>10.4f} "
              f"{finite.mean():>10.4f} {len(uniq):>7}  {guess:<28}")


def _guess_column(v: np.ndarray, uniq: np.ndarray) -> str:
    """Cheap heuristic label for one column, from its value distribution."""
    if len(uniq) == 1:
        return f"CONSTANT {uniq[0]:.4g}"
    if len(uniq) == 2 and set(np.round(uniq, 6)).issubset({0.0, 1.0}):
        return "binary 0/1 flag"
    if len(uniq) <= 4 and v.min() >= 0 and v.max() <= 3 and \
            np.allclose(v, np.round(v)):
        return "small integer code"
    if v.min() >= -1e-9 and v.max() <= 0.5 + 1e-9:
        return "in [0, 0.5] -> frac candidate"
    if v.min() >= -1e-9 and v.max() <= 1.0 + 1e-9:
        return "in [0, 1] -> sol_val candidate"
    if v.min() < 0:
        return "signed continuous"
    return "continuous"


def find_solval_frac_pair(M: np.ndarray, tol: float = 1e-4):
    """
    Find (i, j) such that M[:, j] == |M[:, i] - round(M[:, i])|.

    This uniquely identifies (sol_val, sol_frac) in any LP feature layout,
    with no reliance on documented column orderings.
    Returns a list of matching (i, j) pairs.
    """
    d = M.shape[1]
    hits = []
    for i in range(d):
        a = M[:, i]
        if not np.all(np.isfinite(a)):
            continue
        implied = np.abs(a - np.round(a))
        if implied.max() < tol:          # column is already integral: no info
            continue
        for j in range(d):
            if i == j:
                continue
            b = M[:, j]
            if not np.all(np.isfinite(b)):
                continue
            if np.allclose(implied, b, atol=tol):
                hits.append((i, j))
    return hits


def find_basis_group(M: np.ndarray, tol: float = 1e-6):
    """
    Find groups of 0/1 columns that are mutually exclusive and sum to 1.

    Ecole encodes basis status as separate one-hot flags; collect_highs encodes
    it as a single 0/1/2 integer code. Returns candidate column index tuples.
    """
    d = M.shape[1]
    binary_cols = []
    for j in range(d):
        u = np.unique(np.round(M[:, j], 6))
        if set(u).issubset({0.0, 1.0}) and len(u) == 2:
            binary_cols.append(j)

    groups = []
    # Try contiguous runs of 3 and 4 binary columns (basis one-hots are adjacent
    # in every layout we know of).
    for size in (4, 3):
        for start in range(len(binary_cols) - size + 1):
            cols = binary_cols[start:start + size]
            if cols != list(range(cols[0], cols[0] + size)):
                continue                      # not contiguous
            s = M[:, cols].sum(axis=1)
            if np.allclose(s, 1.0, atol=tol):
                groups.append(tuple(cols))
    return groups


# ---------------------------------------------------------------------------
# Part A -- Ecole
# ---------------------------------------------------------------------------

def part_a(args) -> dict | None:
    try:
        import ecole
    except ImportError:
        print("\n[Part A] SKIPPED: ecole is not importable in this environment.")
        return None

    print("\n" + "=" * 78)
    print("PART A -- Ecole NodeBipartite observation layout")
    print("=" * 78)

    gen = ecole.instance.SetCoverGenerator(
        n_rows=args.n_rows, n_cols=args.n_cols, density=args.density
    )
    gen.seed(args.seed)
    instance = next(gen)

    env = ecole.environment.Branching(
        observation_function=ecole.observation.NodeBipartite(),
        scip_params={
            "separating/maxrounds": 0,
            "presolving/maxrounds": 0,
        },
    )
    obs, action_set, _, done, _ = env.reset(instance)
    if obs is None:
        print("Root observation was None (instance solved in presolve?). "
              "Try a larger instance.")
        return None

    vf = np.asarray(
        obs.variable_features if hasattr(obs, "variable_features")
        else obs.column_features, dtype=np.float64)
    cf = np.asarray(
        obs.constraint_features if hasattr(obs, "constraint_features")
        else obs.row_features, dtype=np.float64)

    describe_columns(vf, "Ecole variable_features")
    describe_columns(cf, "Ecole constraint_features")

    print("\n--- Positive identification ---")
    pairs = find_solval_frac_pair(vf)
    if pairs:
        for (i, j) in pairs:
            print(f"  sol_val = col {i}   sol_frac = col {j}   "
                  f"(verified: col{j} == |col{i} - round(col{i})|)")
    else:
        print("  !! Could not identify a (sol_val, sol_frac) pair.")
        print("     The root LP may be integral. Try --n_rows/--n_cols larger,")
        print("     or step the env a few times before reading the observation.")

    groups = find_basis_group(vf)
    if groups:
        for g in groups:
            print(f"  basis one-hot group: cols {g} (mutually exclusive, sum=1)")
    else:
        print("  (no contiguous one-hot basis group found)")

    print(f"\n  action_set size at root: "
          f"{0 if action_set is None else len(action_set)}")

    ecole_sol_val  = pairs[0][0] if pairs else None
    ecole_sol_frac = pairs[0][1] if pairs else None

    print("\n--- What ablation.py currently assumes ---")
    print(f"  _pick_action reads column 14 as sol_frac "
          f"(integrality gate + most_fractional baseline)")
    if ecole_sol_frac is None:
        print("  -> could not verify")
    elif ecole_sol_frac == 14:
        print("  -> CORRECT for Ecole. No permutation needed on this column.")
    else:
        print(f"  -> WRONG. Ecole's sol_frac is column {ecole_sol_frac}, "
              f"not 14.")
        print(f"     Column 14 is: {_guess_column(vf[:,14], np.unique(np.round(vf[:,14],6)))}")
        print(f"     Every learned method AND the most_fractional baseline "
              f"are affected.")

    return {"vf": vf, "cf": cf,
            "sol_val": ecole_sol_val, "sol_frac": ecole_sol_frac,
            "basis_groups": groups}


# ---------------------------------------------------------------------------
# Part B -- the collector's own trajectories (= the training layout)
# ---------------------------------------------------------------------------

def part_b(args) -> dict | None:
    if not args.npz:
        print("\n[Part B] SKIPPED: pass --npz <trajectory.npz> to verify "
              "the training layout.")
        return None

    path = Path(args.npz)
    if not path.exists():
        print(f"\n[Part B] SKIPPED: {path} does not exist.")
        return None

    print("\n" + "=" * 78)
    print(f"PART B -- collect_highs trajectory layout ({path.name})")
    print("=" * 78)

    d = np.load(path, allow_pickle=True)
    print(f"\nFields present: {sorted(d.files)}")

    for key in ("node_ids", "parent_ids", "subtree_size", "sb_scores",
                "branch_dirs", "next_is_leaf"):
        print(f"  {key:<16} {'PRESENT' if key in d.files else 'MISSING'}")

    vf_all = d["var_features"]
    vf = np.asarray(vf_all[0], dtype=np.float64)   # root node
    describe_columns(vf, "collector var_features (root node)")

    print("\n--- Positive identification ---")
    pairs = find_solval_frac_pair(vf)
    if pairs:
        for (i, j) in pairs:
            print(f"  sol_val = col {i}   sol_frac = col {j}")
        if (13, 14) in pairs:
            print("  -> matches the documented collector layout (13, 14). Good.")
        else:
            print("  -> DOES NOT match the documented (13, 14). Investigate.")
    else:
        print("  !! Could not identify a (sol_val, sol_frac) pair.")

    print("\n--- Constant columns (must be reproduced exactly at inference) ---")
    for idx, name, const in TRAIN_LAYOUT:
        if const is None:
            continue
        col = vf[:, idx]
        ok = np.allclose(col, const, atol=1e-6)
        print(f"  col {idx:>2} {name:<20} expected {const:<5} "
              f"{'OK' if ok else f'MISMATCH (saw {np.unique(np.round(col,4))[:4]})'}")

    return {"vf": vf, "fields": set(d.files)}


# ---------------------------------------------------------------------------
# Part C -- proposed mapping
# ---------------------------------------------------------------------------

def part_c(a: dict | None) -> None:
    print("\n" + "=" * 78)
    print("PART C -- Ecole -> training-layout mapping")
    print("=" * 78)

    if a is None or a["sol_frac"] is None:
        print("\nPart A did not complete; cannot propose a mapping.")
        return

    sv, sf = a["sol_val"], a["sol_frac"]
    print(f"""
Confirmed from Part A:
    Ecole sol_val  = column {sv}
    Ecole sol_frac = column {sf}

Training layout needs these 19 columns, in this order:
""")
    for idx, name, const in TRAIN_LAYOUT:
        if const is not None:
            src = f"CONSTANT {const}"
        elif idx == 13:
            src = f"ecole col {sv}"
        elif idx == 14:
            src = f"ecole col {sf}"
        elif idx == 8:
            src = "DERIVED from edge_index (variable degree / n_cons)"
        elif idx == 16:
            src = "DERIVED from edge_index + constraint is_tight column"
        elif idx == 15:
            src = "CONSTANT 1.0  (lp_obj/(|lp_obj|+eps) for positive objective)"
        elif idx == 5:
            grp = a["basis_groups"][0] if a["basis_groups"] else None
            src = (f"ENCODE from ecole basis one-hots {grp} -> 0/1/2"
                   if grp else "ecole basis one-hots -> 0/1/2 (group not found)")
        else:
            src = "<-- fill in from the Part A table above"
        print(f"    train[{idx:>2}] {name:<20} <- {src}")

    print("""
Columns marked "fill in" are ones this script cannot pin down automatically
(objective coefficient, at-lb / at-ub flags, reduced cost). Read them off the
Part A table: match by value distribution against the collector's Part B table.

IMPORTANT -- normalisation also differs, not just ordering. collect_highs
normalises the objective by max|c| and reduced cost by max|rc| within each node.
Ecole applies its own normalisation. Permuting the columns aligns the SEMANTICS;
matching the SCALE needs the same per-node normalisation applied after the
permutation. Check whether the encoder's set_feature_stats() standardisation was
fitted on collector data -- if so it absorbs some, but not all, of the gap.
""")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_rows", type=int, default=500)
    p.add_argument("--n_cols", type=int, default=1000)
    p.add_argument("--density", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--npz", default=None,
                   help="Path to one collect_highs trajectory .npz "
                        "(enables Part B)")
    args = p.parse_args()

    a = part_a(args)
    part_b(args)
    part_c(a)

    print("\nDone. Paste the full output back and the permutation maps can be "
          "written against your verified indices.\n")


if __name__ == "__main__":
    main()
