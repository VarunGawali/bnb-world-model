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
  4  sol_is_at_ub      LP value >= ub - eps  (rare but nonzero at depth)
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

    # col 4: sol_is_at_ub — NOT constant across the tree (0.33% of training rows
    # have value 1, at depth when a var is fixed to its upper bound).
    # Compute from sol_val: for binary vars ub=1.0.
    sol_is_at_ub = (sol_val >= 1.0 - 1e-6).astype(np.float32)

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
    vf[:, 4]  = sol_is_at_ub
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


# ---------------------------------------------------------------------------
# Native feature construction (raw LP data -> training layout)
# ---------------------------------------------------------------------------
#
# These are lifted VERBATIM from collect_highs.py._var_features / _con_features.
# They are the definition of the training distribution: whatever the collector
# wrote is what the encoder learned to read. The solver must call these rather
# than hand-rolling a layout, so train and serve cannot drift apart again.
#
# collect_highs.py should import these instead of keeping its own copies.

_FRAC_TOL = 1e-6
_TIGHT_TOL = 1e-4


def var_features(A, b, c, sol, n_rows_per_var):
    """Build [n_vars, 19] variable features in the training layout.

    Args:
        A: [m, n] constraint matrix (dense).
        b: [m] right-hand sides.
        c: [n] objective coefficients.
        sol: dict with keys x, rc, y, slack, obj, at_lb, at_ub, basis_status
             (see sol_from_highs).
        n_rows_per_var: [n] number of constraints each variable appears in.

    Returns:
        [n_vars, 19] float32
    """
    n_vars = len(c)
    x = np.asarray(sol["x"], dtype=np.float64)
    rc = np.asarray(sol["rc"], dtype=np.float64)
    lp_obj = float(sol["obj"])

    obj_max = max(float(np.abs(c).max()), 1e-8)
    rc_max = max(float(np.abs(rc).max()), 1e-8)

    sol_frac = np.abs(x - np.round(np.clip(x, 0.0, 1.0)))

    slack = np.asarray(sol["slack"], dtype=np.float64) - np.asarray(b, dtype=np.float64)
    tight_rows = np.abs(slack) < _TIGHT_TOL                      # [m]
    # Vectorised equivalent of the collector's per-variable Python loop.
    n_tight_per_var = (tight_rows.astype(np.float64) @ (A != 0)).astype(np.float32)
    n_rows_f = max(A.shape[0], 1)

    vf = np.stack([
        c / obj_max,                                      # 0  obj_coef_norm
        np.ones(n_vars, dtype=np.float32),                # 1  has_lb
        np.ones(n_vars, dtype=np.float32),                # 2  has_ub
        np.asarray(sol["at_lb"], dtype=np.float32),       # 3  sol_is_at_lb
        np.asarray(sol["at_ub"], dtype=np.float32),       # 4  sol_is_at_ub
        np.asarray(sol["basis_status"], dtype=np.float32),# 5  basis_status
        rc / rc_max,                                      # 6  reduced_cost_norm
        np.zeros(n_vars, dtype=np.float32),               # 7  (zero)
        np.asarray(n_rows_per_var, np.float32) / n_rows_f,# 8  n_rows_norm
        np.zeros(n_vars, dtype=np.float32),               # 9  obj_sense
        np.zeros(n_vars, dtype=np.float32),               # 10 col_age
        np.zeros(n_vars, dtype=np.float32),               # 11 incumbent_value
        np.zeros(n_vars, dtype=np.float32),               # 12 avg_incumbent
        x.astype(np.float32),                             # 13 sol_val
        sol_frac.astype(np.float32),                      # 14 sol_frac
        np.full(n_vars, lp_obj / (abs(lp_obj) + 1e-8), dtype=np.float32),  # 15
        n_tight_per_var / n_rows_f,                       # 16 n_rows_tight_norm
        np.zeros(n_vars, dtype=np.float32),               # 17 lb
        np.ones(n_vars, dtype=np.float32),                # 18 ub
    ], axis=1).astype(np.float32)

    return vf


def con_features(A, b, c, sol):
    """Build [n_cons, 5] constraint features in the training layout."""
    n_cons, n_vars = A.shape
    y = np.asarray(sol["y"], dtype=np.float64)
    slack = np.asarray(sol["slack"], dtype=np.float64) - np.asarray(b, dtype=np.float64)

    y_max = max(float(np.abs(y).max()), 1e-8)
    obj_norm = np.asarray(c, dtype=np.float64) / (np.linalg.norm(c) + 1e-8)

    row_norms = np.linalg.norm(A, axis=1) + 1e-8
    obj_cos = (A @ obj_norm) / row_norms

    tight = (np.abs(slack) < _TIGHT_TOL).astype(np.float32)
    n_vars_f = max(n_vars, 1)

    cf = np.stack([
        obj_cos.astype(np.float32),                            # 0 obj_cos
        np.asarray(b, dtype=np.float32),                       # 1 rhs
        tight,                                                  # 2 is_tight
        (y / y_max).astype(np.float32),                        # 3 dual_value_norm
        (A != 0).sum(axis=1).astype(np.float32) / n_vars_f,    # 4 n_vars_norm
    ], axis=1).astype(np.float32)

    return cf


def edge_arrays(A):
    """Bipartite edges of A: (edge_indices [2,E] as (con, var), values [E])."""
    rows, cols = np.where(np.abs(A) > 1e-12)
    ei = np.stack([rows.astype(np.int64), cols.astype(np.int64)], axis=0)
    ev = A[rows, cols].astype(np.float32)
    return ei, ev


def sol_from_highs(highspy, h, n_vars, n_cons):
    """Build the `sol` dict var_features/con_features expect from a HiGHS solve.

    Mirrors collect_highs._HiGHSLP.solve() exactly, including the 0/1/2
    encoding of basis_status, so solver features match training features.
    """
    sol_obj = h.getSolution()
    x = np.array(sol_obj.col_value[:n_vars], dtype=np.float64)
    rc = np.array(sol_obj.col_dual[:n_vars], dtype=np.float64)
    y = np.array(sol_obj.row_dual[:n_cons], dtype=np.float64)
    slack = np.array(sol_obj.row_value[:n_cons], dtype=np.float64)

    basis = h.getBasis()
    kBasic = highspy.HighsBasisStatus.kBasic
    kLower = highspy.HighsBasisStatus.kLower
    kUpper = highspy.HighsBasisStatus.kUpper
    col_status = list(basis.col_status)

    at_lb = np.array([1 if col_status[j] == kLower else 0
                      for j in range(n_vars)], dtype=np.int8)
    at_ub = np.array([1 if col_status[j] == kUpper else 0
                      for j in range(n_vars)], dtype=np.int8)
    basis_status = np.array(
        [1 if col_status[j] == kBasic else (2 if col_status[j] == kUpper else 0)
         for j in range(n_vars)], dtype=np.int8)

    return {
        "x": x, "rc": rc, "y": y, "slack": slack,
        "obj": float(h.getInfoValue("objective_function_value")[1]),
        "at_lb": at_lb, "at_ub": at_ub, "basis_status": basis_status,
        "col_status": col_status,
        "row_status": list(basis.row_status),
    }
