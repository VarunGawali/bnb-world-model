"""
features.py — Shared feature-layout utilities.

Single source of truth for the 19-dim variable / 5-dim constraint layout
that collect_highs.py writes and the model was trained on.  Both _format_obs
(Ecole eval path) and bnb_solver._encode_node (HiGHS solver path) should
call ecole_to_train_layout() instead of hand-rolling their own remappings.

Training layout (collect_highs.py._var_features):
  0  obj_coef_norm     c / max|c|
  1  has_lb            CONSTANT 1
  2  has_ub            CONSTANT 1
  3  sol_is_at_lb      LP value <= lb + eps
  4  sol_is_at_ub      LP value >= ub - eps  (≈ 0 at root for binary)
  5  basis_status      0=lower  1=basic  2=upper
  6  reduced_cost_norm rc / max|rc|
  7  (zero)            CONSTANT 0
  8  n_rows_norm       degree_j / n_cons  (derived from edge_index)
  9–12 (zeros)         CONSTANT 0
 13  sol_val           LP solution value
 14  sol_frac          |sol_val − round(sol_val)|   <- integrality gate
 15  lp_obj_norm       CONSTANT 1
 16  n_rows_tight_norm #tight_cons_for_j / n_cons  (derived)
 17  lb_value          CONSTANT 0
 18  ub_value          CONSTANT 1

Ecole NodeBipartite confirmed layout (check_feature_layout.py, 500×1000
set-cover, action_set=112 fractional vars):
  0  obj_coef          ‖c‖-normalised by Ecole
  1  type_binary       CONSTANT 1
  2–4 other type flags CONSTANT 0
  5  has_lb            CONSTANT 1
  6  has_ub            CONSTANT 1
  7  normed_reduced_cost signed, Ecole-normalised
  8  solution_value    LP solution value [0, 1]
  9  solution_frac     x − floor(x)  (NOT |x − round(x)|)
 10  is_at_lower_bound binary  mean≈0.888
 11  (other binary)    mean≈0.686
 12  scaled_age        5 unique values
 13  incumbent_value   binary flag
 14  avg_incumbent     continuous  (NOT sol_frac — max > 0.5)
 15  is_basis_lower    one-hot lower  mean≈0.888
 16  is_basis_basic    one-hot basic  mean≈0.112
 17  is_basis_upper    CONSTANT 0 for binary vars
 18  is_basis_zero     CONSTANT 0
"""

from __future__ import annotations

import numpy as np


def ecole_to_train_layout(
    vf_raw: np.ndarray,
    cf_ecole: np.ndarray,
    ei: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remap Ecole NodeBipartite features to the training layout.

    Args:
        vf_raw:   [n_vars, ≥19]  float32 — Ecole variable_features
        cf_ecole: [n_cons, ≥5]   float32 — Ecole constraint_features
        ei:       [2, E]         int64   — bipartite edge_index
                                           ei[0]=constraint idx, ei[1]=var idx

    Returns:
        vf: [n_vars, 19]  float32  training-layout variable features
        cf: [n_cons,  5]  float32  training-layout constraint features
    """
    n_vars = vf_raw.shape[0]
    n_cons = cf_ecole.shape[0]

    # --- Variable features ---

    # col 0: obj_coef_norm  (re-normalise to c/max|c|)
    obj_raw = vf_raw[:, 0]
    obj_max = float(np.abs(obj_raw).max()) + 1e-8
    obj_coef_norm = (obj_raw / obj_max).astype(np.float32)

    # col 13: sol_val  (Ecole col 8, continuous [0,1])
    sol_val = np.clip(vf_raw[:, 8], 0.0, 1.0).astype(np.float32)

    # col 14: sol_frac  |sol_val − round(sol_val)|
    # Ecole col 9 is x−floor(x), not the distance to nearest integer.
    # For x=0.94: Ecole gives 0.94; training needs 0.06.  Must recompute.
    sol_frac = np.abs(sol_val - np.round(sol_val)).astype(np.float32)

    # col 3: sol_is_at_lb  (Ecole col 10 = is_solution_at_lower_bound)
    sol_is_at_lb = vf_raw[:, 10].astype(np.float32)

    # col 5: basis_status  0=lower 1=basic 2=upper
    # Ecole: col15=is_basis_lower, col16=is_basis_basic, col17=is_basis_upper
    basis_status = (
        vf_raw[:, 16] * 1.0 + vf_raw[:, 17] * 2.0
    ).astype(np.float32)

    # col 6: reduced_cost_norm  (re-normalise to rc/max|rc|)
    rc_raw = vf_raw[:, 7]
    rc_max = float(np.abs(rc_raw).max()) + 1e-8
    rc_norm = (rc_raw / rc_max).astype(np.float32)

    # col 8: n_rows_norm = degree_j / n_cons
    n_rows_per_var = np.bincount(ei[1], minlength=n_vars).astype(np.float32)
    n_rows_norm = n_rows_per_var / max(n_cons, 1)

    # col 16: n_rows_tight_norm
    is_tight = cf_ecole[:, 2] if cf_ecole.shape[1] > 2 else np.zeros(n_cons, np.float32)
    tight_per_var = np.zeros(n_vars, dtype=np.float32)
    np.add.at(tight_per_var, ei[1], is_tight[ei[0]])
    n_rows_tight_norm = tight_per_var / max(n_cons, 1)

    # assemble
    vf = np.zeros((n_vars, 19), dtype=np.float32)
    vf[:, 0]  = obj_coef_norm
    vf[:, 1]  = 1.0               # has_lb
    vf[:, 2]  = 1.0               # has_ub
    vf[:, 3]  = sol_is_at_lb
    # col 4 sol_is_at_ub: constant 0 in all training data (binary vars at root)
    vf[:, 5]  = basis_status
    vf[:, 6]  = rc_norm
    # col 7 zero
    vf[:, 8]  = n_rows_norm
    # cols 9–12 zeros
    vf[:, 13] = sol_val
    vf[:, 14] = sol_frac
    vf[:, 15] = 1.0               # lp_obj_norm
    vf[:, 16] = n_rows_tight_norm
    # col 17 lb = 0
    vf[:, 18] = 1.0               # ub

    # --- Constraint features ---
    # Training layout: [obj_cos, rhs=1, is_tight, dual_norm, n_vars_norm]
    # Ecole layout:    [obj_cos, normed_RHS (negative, ≠ raw b), is_tight,
    #                   dual_value, n_vars_per_row/n_vars]
    # For set-cover (Ax >= 1) the raw RHS is always 1.0, so we hard-code it.
    # This keeps edge normalisation consistent: norm_ev = ev/(|RHS|+eps) = ev.
    cf = np.zeros((n_cons, 5), dtype=np.float32)
    if cf_ecole.shape[1] >= 1:
        cf[:, 0] = cf_ecole[:, 0]      # obj_cos
    cf[:, 1] = 1.0                     # raw RHS (set-cover constant)
    if cf_ecole.shape[1] >= 3:
        cf[:, 2] = cf_ecole[:, 2]      # is_tight
    if cf_ecole.shape[1] >= 4:
        y_raw = cf_ecole[:, 3]
        y_max = float(np.abs(y_raw).max()) + 1e-8
        cf[:, 3] = (y_raw / y_max).astype(np.float32)   # dual_norm
    if cf_ecole.shape[1] >= 5:
        cf[:, 4] = cf_ecole[:, 4]      # n_vars_norm

    return vf, cf
