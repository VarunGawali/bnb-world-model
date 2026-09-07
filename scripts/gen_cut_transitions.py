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
import multiprocessing as mp
import os
import warnings
from functools import partial
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
    c64 = c.astype(np.float64)
    lb  = np.zeros(n, dtype=np.float64)
    ub  = np.full(n, inf, dtype=np.float64)

    # API differs across highspy versions: try new passLp first, fall back to
    # addVars/addRow for older builds.
    try:
        from scipy.sparse import csr_matrix
        A_sp = csr_matrix(A)
        h.passLp(
            m, n, A_sp.nnz,
            1,          # a_format: row-wise
            1,          # sense: minimise
            0.0,        # offset
            c64,
            np.full(m, -inf), b.astype(np.float64),
            lb, ub,
            A_sp.indptr.astype(np.int32),
            A_sp.indices.astype(np.int32),
            A_sp.data.astype(np.float64),
        )
    except Exception:
        # Fallback: column-by-column / row-by-row via addVar/addRow.
        for j in range(n):
            h.addVar(0.0, inf)
        for j in range(n):
            h.changeColCost(j, c64[j])
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

def _con_features_for_cut(coeffs, rhs, x_lp_after, n_vars, c=None):
    """
    Build a 5-dim constraint feature row for the new Gomory cut constraint.

    Column layout MUST match Ecole's NodeBipartite row features exactly, because
    build_pyg_data reads col 1 as the RHS for edge normalisation:
        0: obj_cosine_sim  cosine(cut_coeffs, c); 0.0 when c unavailable
        1: bias / RHS      — build_pyg_data reads edge RHS from THIS column
        2: is_tight        1 if the cut is active at the new LP solution
        3: dual_val        0.0 — not available without a dual solve
        4: scaled_age      0.0 — cut is brand-new
    """
    rhs_val  = float(rhs)

    activity = float(np.dot(coeffs, x_lp_after))
    is_tight = float(abs(activity - rhs_val) < 1e-6)

    if c is not None:
        obj_cos = float(np.dot(coeffs, c) /
                        (np.linalg.norm(coeffs) * np.linalg.norm(c) + 1e-8))
    else:
        obj_cos = 0.0

    return np.array([obj_cos, rhs_val, is_tight, 0.0, 0.0], dtype=np.float32)


def _build_after_graph(var_feats, con_feats, edge_idx, edge_vals,
                       cut, x_lp_after, c=None):
    """
    Return (vf_after, cf_after, ei_after, ev_after):
      - var_feats with updated LP values (col 13) and fractionality (col 14),
        matching the Ecole NodeBipartite variable feature layout
      - con_feats with a new row for the cut constraint (5-col Ecole layout)
      - edge_idx/edge_vals with new edges for the cut
    """
    coeffs = np.asarray(cut["coeffs"], dtype=np.float64)
    rhs    = float(cut["rhs"])
    n_vars = var_feats.shape[0]
    n_cons = con_feats.shape[0]

    # Updated variable features.
    # Ecole NodeBipartite variable feature layout (19 cols):
    #   col 13 = sol_val  (LP solution value)   ← updated here
    #   col 14 = sol_frac (fractionality)        ← updated here
    # Do NOT write to col 0 — that is obj_cosine_similarity, fixed at collection.
    vf_after = var_feats.copy().astype(np.float32)
    n_use    = min(n_vars, len(x_lp_after))
    if vf_after.shape[1] > 14:
        vf_after[:n_use, 13] = x_lp_after[:n_use].astype(np.float32)
        frac = np.abs(
            x_lp_after[:n_use] - np.round(x_lp_after[:n_use])
        ).astype(np.float32)
        vf_after[:n_use, 14] = frac
    # If the feature matrix is narrower than 15 cols (shouldn't happen with
    # Ecole 19-dim features), skip the update rather than corrupting data.

    # New constraint node (5-dim, matching Ecole row-feature layout).
    cut_con_feats = _con_features_for_cut(coeffs, rhs, x_lp_after, n_vars, c=c)
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
# LP reconstruction from bipartite graph features
# ---------------------------------------------------------------------------

def _reconstruct_lp(var_feats, con_feats, edge_idx, edge_vals):
    """
    Reconstruct (A, b, c, x_lp) from Ecole NodeBipartite graph features.

    Ecole normalises each constraint row by its L1 norm (sum of |A[i,:]|) and
    flips the sign for >= constraints → -Ax <= -b.  The stored values are
    therefore already a valid LP in HiGHS form (Ax <= b, x >= 0) because
    A_stored = A_orig/scale with sign-flip and b_stored = b_orig/scale with
    the same sign-flip.  Solving the scaled LP gives the same optimal x as the
    original.

    Columns used (Ecole NodeBipartite 19-dim variable features):
        col 13 : sol_val  — LP solution at this node
        col  5 : obj (normed objective coefficient) used as proxy for c

    The RHS is read from con_features[:, 1] — the same column build_pyg_data
    uses for edge normalisation.
    """
    n_vars = var_feats.shape[0]
    n_cons = con_feats.shape[0]

    # Build sparse A from edges; both A and b already carry correct signs.
    A = np.zeros((n_cons, n_vars), dtype=np.float64)
    rows = edge_idx[0].astype(int)
    cols = edge_idx[1].astype(int)
    mask = (rows < n_cons) & (cols < n_vars)
    A[rows[mask], cols[mask]] = edge_vals[mask].astype(np.float64)

    b = con_feats[:, 1].astype(np.float64)

    # Objective: use normalised obj coefficient (col 5) as proxy.
    # For uniform-cost problems this equals the true c up to a positive scale,
    # which is sufficient because LP argmin is scale-invariant.  Fall back to
    # all-ones (valid for set-cover) when col 5 is identically zero.
    if var_feats.shape[1] > 5:
        c_proxy = var_feats[:, 5].astype(np.float64)
        if np.all(c_proxy == 0):
            c_proxy = np.ones(n_vars, dtype=np.float64)
    else:
        c_proxy = np.ones(n_vars, dtype=np.float64)

    # LP solution at this node (Ecole col 13 = sol_val).
    if var_feats.shape[1] > 13:
        x_lp = var_feats[:, 13].astype(np.float64)
    else:
        x_lp = np.zeros(n_vars, dtype=np.float64)

    return A, b, c_proxy, x_lp


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(fpath, out_dir, resume=False):
    stem = Path(fpath).stem
    out_path = Path(out_dir) / f"{stem}_cut.npz"
    if resume and out_path.exists():
        return None   # already done

    d = np.load(fpath, allow_pickle=True)
    T = int(d["n_steps"])
    if T == 0:
        return False

    # Use root node (t=0).
    var_feats = d["var_features"][0]
    con_feats = d["con_features"][0]
    edge_idx  = d["edge_indices"][0]
    edge_vals = d["edge_values"][0]

    # ---- Obtain (A, b, c, x_lp) ----------------------------------------
    # Prefer explicitly stored LP matrices (collect_with_cuts_v2 files).
    # Fall back to graph-based reconstruction for traj_sc_ files which carry
    # pre-computed cuts but not the raw LP matrices.
    if "A" in d and "b" in d and "c" in d and "x_lp" in d:
        A    = np.asarray(d["A"],    dtype=np.float64)
        b    = np.asarray(d["b"],    dtype=np.float64)
        c    = np.asarray(d["c"],    dtype=np.float64)
        x_lp = np.asarray(d["x_lp"], dtype=np.float64)
        use_precomputed_cuts = False
    elif "cut_lhs" in d and "cut_rhs" in d:
        # traj_sc_ style: reconstruct LP from graph, use stored cuts directly.
        A, b, c, x_lp = _reconstruct_lp(var_feats, con_feats, edge_idx, edge_vals)
        use_precomputed_cuts = True
    else:
        return False   # no LP data and no pre-computed cuts → skip

    # Sanity: x_lp must be finite; all-zeros means no LP info was stored.
    if not np.all(np.isfinite(x_lp)):
        return False

    # For the reconstruction path, Ecole's sol_val (col 13) is not reliably
    # populated — always re-solve the LP so x_lp matches the reconstructed
    # A/b matrices and the cut violation check is meaningful.
    if use_precomputed_cuts or np.all(x_lp == 0):
        _, x_lp_solved = _solve_lp(A, b, c)
        if x_lp_solved is None:
            return False
        x_lp = x_lp_solved

    # Final check: must still be finite and non-trivial after any solve.
    if not np.all(np.isfinite(x_lp)) or np.all(x_lp == 0):
        return False

    lp_obj_before = float(np.dot(c, x_lp))

    # ---- Obtain cut coefficients -----------------------------------------
    if use_precomputed_cuts:
        # Use first pre-computed cut stored at root (t=0).
        cl = d["cut_lhs"][0]   # may be object array wrapping a 2-D array
        cr = d["cut_rhs"][0]
        cf_store = d["cut_features"][0] if "cut_features" in d else None

        cl = np.asarray(cl, dtype=np.float64)
        if cl.ndim == 2:
            cl = cl[0]    # first cut row
        cr = np.asarray(cr, dtype=np.float64)
        cut_rhs_val = float(cr[0]) if cr.ndim > 0 else float(cr)

        # Align coefficient length to n_vars (may differ if problem was padded).
        n_vars = var_feats.shape[0]
        if cl.shape[0] > n_vars:
            cl = cl[:n_vars]
        elif cl.shape[0] < n_vars:
            cl = np.pad(cl, (0, n_vars - cl.shape[0]))

        cut_coeffs_arr = cl

        # Use stored 6-dim cut features when available; otherwise compute.
        if cf_store is not None:
            cf_store = np.asarray(cf_store, dtype=np.float32)
            cut_feats = cf_store[0] if cf_store.ndim == 2 else cf_store
        else:
            cut = {"coeffs": cut_coeffs_arr, "rhs": cut_rhs_val}
            cut_feats = _build_cut_features(cut, x_lp, c)
    else:
        # Generate one Gomory cut from the root LP solution.
        try:
            from bnb_wm.cuts.cg_cuts import generate_cg_cuts
            cuts = generate_cg_cuts(A, b, c, x_lp=x_lp, max_cuts=1)
        except Exception:
            return False
        if not cuts:
            return False
        cut = cuts[0]
        cut_coeffs_arr = np.asarray(cut["coeffs"], dtype=np.float64)
        cut_rhs_val    = float(cut["rhs"])
        cut_feats      = _build_cut_features(cut, x_lp, c)

    # Sanity: the cut must violate the current LP solution.
    violation = float(np.dot(cut_coeffs_arr, x_lp) - cut_rhs_val)
    if violation <= 1e-6:
        return False

    cut_coeffs = cut_coeffs_arr.astype(np.float32)
    cut_rhs    = cut_rhs_val

    # ---- Inject cut and re-solve -----------------------------------------
    cut_dict = {"coeffs": cut_coeffs_arr, "rhs": cut_rhs_val}
    lp_obj_after, x_new, _, _ = _inject_cut_and_resolve(A, b, c, cut_dict)
    if x_new is None:
        return False

    # For minimisation, LP obj cannot decrease after adding a feasibility cut.
    delta_lb = lp_obj_after - lp_obj_before
    if delta_lb < -1e-6:
        return False

    # ---- Build post-cut graph --------------------------------------------
    vf_after, cf_after, ei_after, ev_after = _build_after_graph(
        var_feats, con_feats, edge_idx, edge_vals, cut_dict, x_new, c=c
    )

    np.savez_compressed(
        out_path,
        vf_before=var_feats.astype(np.float32),
        cf_before=con_feats.astype(np.float32),
        ei_before=edge_idx.astype(np.int64),
        ev_before=edge_vals.astype(np.float32),
        vf_after=vf_after,
        cf_after=cf_after.astype(np.float32),
        ei_after=ei_after.astype(np.int64),
        ev_after=ev_after,
        cut_feats=cut_feats,
        cut_coeffs=cut_coeffs,
        cut_rhs=np.float32(cut_rhs),
        lp_obj_before=np.float32(lp_obj_before),
        lp_obj_after=np.float32(lp_obj_after if lp_obj_after is not None else lp_obj_before),
        delta_lb=np.float32(delta_lb),
    )
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _worker(args):
    fpath, out_dir, resume = args
    try:
        result = process_file(fpath, out_dir, resume=resume)
        return result   # True = written, False = skipped, None = already done
    except Exception as e:
        warnings.warn(f"skip {Path(fpath).name}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",  required=True)
    ap.add_argument("--out_dir",   required=True)
    ap.add_argument("--max_files", type=int, default=None)
    ap.add_argument("--workers",   type=int, default=os.cpu_count(),
                    help="Parallel worker processes (default: all CPUs)")
    ap.add_argument("--resume",    action="store_true",
                    help="Skip files whose output .npz already exists")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.data_dir).rglob("*.npz"))
    # Exclude output files if they end up in the same tree.
    files = [f for f in files if "_cut.npz" not in f.name]
    if args.max_files:
        files = files[:args.max_files]

    print(f"Processing {len(files)} trajectory files with {args.workers} workers"
          + (" (resume mode)" if args.resume else "") + " ...")

    work = [(str(f), str(out_dir), args.resume) for f in files]

    ok = fail = skipped = 0
    if args.workers == 1:
        for item in work:
            r = _worker(item)
            if r is True:    ok += 1
            elif r is None:  skipped += 1
            else:            fail += 1
    else:
        with mp.Pool(args.workers) as pool:
            for r in pool.imap_unordered(_worker, work, chunksize=4):
                if r is True:    ok += 1
                elif r is None:  skipped += 1
                else:            fail += 1

    print(f"\nDone: {ok} written, {skipped} already done, {fail} skipped → {out_dir}")
    print("NOTE: latent vectors are NOT cached here. The training loop")
    print("      encodes vf/cf/ei/ev on-the-fly using the current Phase-3 encoder.")
    print("NOTE: cut dynamics trained on ROOT-STATE transitions only.")


if __name__ == "__main__":
    main()
