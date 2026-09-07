"""
gen_cut_transitions.py — Generate cut transition pairs for dynamics training.

For each trajectory file (at the ROOT node, t=0):
  1. Load LP matrices (A, b, c) and GNN features.
  2. Generate one Gomory cut via generate_cg_cuts().
  3. Inject the cut row, re-solve LP (HiGHS).
  4. Build the post-cut bipartite graph by appending a new constraint node
     (the cut row) with proper edges.
  5. Store raw graph features — NOT latent vectors — so the training loop
     can encode with the *current* Phase-3 encoder rather than a stale one.

Stored per file (data/cut_transitions/<stem>_cut.npz):
    vf_before   [n_vars, 19]  variable features (root, before cut)
    cf_before   [n_cons, 5]   constraint features (before cut)
    ei_before   [2, E_b]      edge index (before cut)
    ev_before   [E_b]         edge values (before cut)

    vf_after    [n_vars, 19]  variable features (updated x_lp / fractionality)
    cf_after    [n_cons+1, 5] constraint features (+cut row)
    ei_after    [2, E_a]      edge index (+cut edges)
    ev_after    [E_a]         edge values (+cut edges)

    cut_feats   [6]           [violation, norm_viol, density, obj_norm,
                               gain_proxy, frac_support]
    cut_coeffs  [n_vars]      Gomory cut LHS coefficients
    cut_rhs     scalar        Gomory cut RHS
    lp_obj_before scalar      LP objective before cut
    lp_obj_after  scalar      LP objective after cut
    delta_lb      scalar      lp_obj_after - lp_obj_before
    latent_dist   scalar      placeholder (filled by training loop)

Run ONCE before Phase-3 retraining:
    uv run python scripts/gen_cut_transitions.py \\
        --data_dir data/trajectories \\
        --out_dir  data/cut_transitions

No --checkpoint needed: latents are NOT cached here.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Cut feature vector
# ---------------------------------------------------------------------------

def _build_cut_features(cut, x_lp, c):
    """6-dim Gomory cut feature vector."""
    coeffs = np.asarray(cut["coeffs"], dtype=np.float64)
    rhs    = float(cut["rhs"])

    violation    = float(np.dot(coeffs, x_lp) - rhs)
    norm_viol    = violation / (np.linalg.norm(coeffs) + 1e-8)
    density      = float((coeffs != 0).mean())
    obj_dot      = float(np.dot(coeffs, c))
    obj_norm     = obj_dot / (np.linalg.norm(coeffs) * np.linalg.norm(c) + 1e-8)
    active       = coeffs != 0
    frac         = np.abs(x_lp[active] - np.round(x_lp[active]))
    frac_support = float(frac.mean()) if active.any() else 0.0
    gain_proxy   = float(violation * (np.linalg.norm(coeffs) + 1e-8))

    return np.array([violation, norm_viol, density, obj_norm,
                     gain_proxy, frac_support], dtype=np.float32)


# ---------------------------------------------------------------------------
# LP solve with HiGHS (no warm-start; add warm-start later if speed matters)
# ---------------------------------------------------------------------------

def _solve_lp(A, b, c):
    """Solve min c·x s.t. A·x ≤ b, x ≥ 0.  Returns (obj, x) or (None, None)."""
    try:
        import highspy
    except ImportError:
        raise ImportError("highspy required: uv pip install highspy")

    n = A.shape[1]
    m = A.shape[0]

    h = highspy.Highs()
    h.silent()

    inf = highspy.kHighsInf
    h.addVars(n, np.zeros(n), np.full(n, inf))
    h.changeColsCostByRange(0, n - 1, c.astype(np.float64))

    for i in range(m):
        row = A[i]
        nz  = row.nonzero()[0]
        h.addRow(-inf, float(b[i]), len(nz),
                 nz.astype(np.int32), row[nz].astype(np.float64))

    h.run()
    info = h.getInfoValue("primal_solution_status")[1]
    if info != 2:
        return None, None

    sol = h.getSolution()
    x   = np.asarray(sol.col_value, dtype=np.float64)
    obj = float(h.getInfoValue("objective_function_value")[1])
    return obj, x


def _inject_cut_and_resolve(A, b, c, cut):
    """Append Gomory cut row and solve. Returns (obj_new, x_new, A_new, b_new)."""
    coeffs = np.asarray(cut["coeffs"], dtype=np.float64)
    rhs    = float(cut["rhs"])
    A_new  = np.vstack([A, coeffs[np.newaxis, :]])
    b_new  = np.append(b, rhs)
    obj, x = _solve_lp(A_new, b_new, c)
    return obj, x, A_new, b_new


# ---------------------------------------------------------------------------
# Post-cut graph construction
# ---------------------------------------------------------------------------

def _con_features_for_cut(coeffs, rhs, x_lp_after, n_vars):
    """
    Build a 5-dim constraint feature row for the new Gomory cut constraint,
    matching the Ecole row-feature layout (5 columns):
        0: bias / RHS
        1: obj_cos_sim  (reused as: cosine alignment with LP solution)
        2: is_tight     (1 if the constraint is active at the LP solution)
        3: dual_val     (approximated as 0; not available without solving dual)
        4: scaled_age   (0; cut is brand-new)
    """
    nz_mask = coeffs != 0
    rhs_safe = float(rhs) if abs(rhs) > 1e-10 else 1.0

    activity = float(np.dot(coeffs, x_lp_after))
    is_tight  = float(abs(activity - rhs) < 1e-6)
    density   = float(nz_mask.mean())

    return np.array([rhs_safe, density, is_tight, 0.0, 0.0], dtype=np.float32)


def _build_after_graph(var_feats, con_feats, edge_idx, edge_vals,
                       cut, x_lp_after):
    """
    Return (vf_after, cf_after, ei_after, ev_after):
      - var_feats with updated LP values (col 0) and fractionality (col 14)
      - con_feats with a new row for the cut constraint
      - edge_idx/edge_vals with new edges for the cut
    """
    coeffs = np.asarray(cut["coeffs"], dtype=np.float64)
    rhs    = float(cut["rhs"])
    n_vars = var_feats.shape[0]
    n_cons = con_feats.shape[0]

    # Updated variable features: LP solution and fractionality columns.
    vf_after = var_feats.copy().astype(np.float32)
    n_use    = min(n_vars, len(x_lp_after))
    vf_after[:n_use, 0]  = x_lp_after[:n_use].astype(np.float32)
    frac = np.abs(x_lp_after[:n_use] - np.round(x_lp_after[:n_use])).astype(np.float32)
    vf_after[:n_use, 14] = frac

    # New constraint node: append to con_feats.
    cut_con_feats = _con_features_for_cut(coeffs, rhs, x_lp_after, n_vars)
    cf_after = np.vstack([con_feats, cut_con_feats[np.newaxis, :]])

    # New edges: connect the new constraint (index n_cons) to variables with
    # non-zero cut coefficients.
    nz_vars  = np.where(coeffs[:n_vars] != 0)[0].astype(np.int64)
    new_ei   = np.stack([
        np.full(len(nz_vars), n_cons, dtype=np.int64),  # constraint row
        nz_vars,                                          # variable col
    ], axis=0)
    new_ev   = coeffs[:n_vars][nz_vars].astype(np.float32)

    ei_after = np.concatenate([edge_idx, new_ei], axis=1)
    ev_after = np.concatenate([edge_vals, new_ev])

    return vf_after, cf_after, ei_after, ev_after


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(fpath, out_dir):
    d = np.load(fpath, allow_pickle=True)
    T = int(d["n_steps"])
    if T == 0:
        return False

    # Use root node (t=0).
    var_feats = d["var_features"][0]
    con_feats = d["con_features"][0]
    edge_idx  = d["edge_indices"][0]
    edge_vals = d["edge_values"][0]

    # LP matrices — present in files collected with collect_with_cuts_v2.
    if "A" not in d or "b" not in d or "c" not in d:
        return False
    A = np.asarray(d["A"], dtype=np.float64)
    b = np.asarray(d["b"], dtype=np.float64)
    c = np.asarray(d["c"], dtype=np.float64)

    # x_lp is required for valid Gomory cut generation; don't substitute zeros.
    if "x_lp" not in d:
        return False
    x_lp = np.asarray(d["x_lp"], dtype=np.float64)

    # LP objective before cut.
    lp_obj_before = float(np.dot(c, x_lp))

    # Generate one Gomory cut from the root LP solution.
    try:
        from bnb_wm.cuts.cg_cuts import generate_cg_cuts
        cuts = generate_cg_cuts(A, b, c, x_lp=x_lp, max_cuts=1)
    except Exception:
        return False
    if not cuts:
        return False
    cut = cuts[0]

    cut_feats  = _build_cut_features(cut, x_lp, c)
    cut_coeffs = np.asarray(cut["coeffs"], dtype=np.float32)
    cut_rhs    = float(cut["rhs"])

    # Inject cut and re-solve.
    lp_obj_after, x_new, _, _ = _inject_cut_and_resolve(A, b, c, cut)
    if x_new is None:
        return False

    delta_lb = (lp_obj_after - lp_obj_before) if lp_obj_after is not None else 0.0

    # Build post-cut graph (proper: add constraint node + edges for cut row).
    vf_after, cf_after, ei_after, ev_after = _build_after_graph(
        var_feats, con_feats, edge_idx, edge_vals, cut, x_new
    )

    stem = Path(fpath).stem
    out_path = out_dir / f"{stem}_cut.npz"
    np.savez_compressed(
        out_path,
        # Before-cut raw graph (for on-the-fly encoding).
        vf_before=var_feats.astype(np.float32),
        cf_before=con_feats.astype(np.float32),
        ei_before=edge_idx.astype(np.int64),
        ev_before=edge_vals.astype(np.float32),
        # After-cut raw graph (proper constraint added).
        vf_after=vf_after,
        cf_after=cf_after.astype(np.float32),
        ei_after=ei_after.astype(np.int64),
        ev_after=ev_after,
        # Cut description.
        cut_feats=cut_feats,
        cut_coeffs=cut_coeffs,
        cut_rhs=np.float32(cut_rhs),
        # LP objective information for auditing.
        lp_obj_before=np.float32(lp_obj_before),
        lp_obj_after=np.float32(lp_obj_after if lp_obj_after is not None else lp_obj_before),
        delta_lb=np.float32(delta_lb),
    )
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",  required=True)
    ap.add_argument("--out_dir",   required=True)
    ap.add_argument("--max_files", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.data_dir).rglob("*.npz"))
    if args.max_files:
        files = files[:args.max_files]

    ok = fail = 0
    for f in files:
        try:
            if process_file(f, out_dir):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            warnings.warn(f"skip {f.name}: {e}")
            fail += 1

    print(f"\nDone: {ok} cut transitions written, {fail} skipped → {out_dir}")
    print("NOTE: latent vectors are NOT cached here. The training loop")
    print("      encodes vf/cf/ei/ev on-the-fly using the current Phase-3 encoder.")
    print("NOTE: cut dynamics was trained on ROOT-STATE transitions only.")


if __name__ == "__main__":
    main()
