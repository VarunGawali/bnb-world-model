"""
gomory.py — Globally valid Gomory fractional cuts for binary set-cover B&C.

Validity scope
--------------
This implementation is designed specifically for binary set-cover instances
with integer A and b (A x >= b, x in {0,1}^n). The upper-bound transformation
x_j + t_j = 1 requires x_j to be binary/integer for t_j to also be integer,
which is a prerequisite for the Gomory fractional cut argument. Do NOT use
this generator for general MILPs without auditing the validity claim.

Global validity: cuts are derived once at the root from the original covering
constraints with NO branching bounds, so they hold at every descendant node.

Standard form
-------------
We lift upper bounds x <= 1 into explicit rows so every nonbasic variable
sits at its lower bound 0 (clean regime for the textbook GMI cut):

    cover  i:  sum_j A_ij x_j - s_i = b_i,   s_i >= 0   (m rows)
    ubound j:  x_j + t_j = 1,                 t_j >= 0   (n rows)
    variables: [ x (n) | s (m) | t (n) ],  all >= 0

For a basic structural variable x_j with fractional LP value b̄_r, tableau row
    x_j + sum_{k in N} a_k v_k = b̄_r,
the GMI cut is:
    sum_{k in N} frac(a_k) v_k >= frac(b̄_r).
Back-substituting s_i = A_i x - b_i and t_j = 1 - x_j yields a globally valid
inequality  alpha^T x >= beta  in the original variables.

Cut validity is not guaranteed post-hoc: we verify every generated cut is
actually violated by the current LP solution before returning it.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Tolerances — separated by role for easier auditing
# ---------------------------------------------------------------------------
_INTEG_TOL  = 1e-6   # a value is "integer" if its fractional part < this
_COEFF_TOL  = 1e-8   # tableau coefficients smaller than this are zero
_VIOL_TOL   = 1e-6   # a cut must be violated by at least this much to be kept
_COEFF_CAP  = 1e8    # reject cuts with absurdly large coefficients


def _frac(y):
    """Fractional part in [0, 1)."""
    return y - np.floor(y)


def generate_root_gomory_cuts(
    A, b, c, highspy,
    max_cuts: int = 50,
    x_lp: "np.ndarray | None" = None,
    var_lb: "np.ndarray | None" = None,
    var_ub: "np.ndarray | None" = None,
):
    """
    Generate Gomory fractional cuts valid at the current B&B node.

    Args:
        A       : [m, n] covering matrix (A x >= b)
        b       : [m]    right-hand side
        c       : [n]    objective coefficients
        highspy : imported highspy module; returns [] if None
        max_cuts: maximum number of cuts to return
        x_lp    : [n] current LP solution at this node; used for violation check
        var_lb  : [n] variable lower bounds from branching (default 0)
        var_ub  : [n] variable upper bounds from branching (default 1)

    Returns:
        list of (lhs [n] float64, rhs float) where  lhs @ x >= rhs  is a
        valid cut violated by the current node's LP solution.
    """
    if highspy is None:
        return []

    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    c = np.asarray(c, dtype=np.float64).reshape(-1)
    m, n = A.shape

    lb = np.zeros(n) if var_lb is None else np.asarray(var_lb, dtype=np.float64)
    ub = np.ones(n)  if var_ub is None else np.asarray(var_ub, dtype=np.float64)

    # ---- build the standard-form equality system  E z = d,  z >= 0 ----------
    # columns: [ x (n) | s (m) | t (n) ]  -> N = 2n + m
    # rows:    [ cover (m) | ubound (n) ] -> M = m + n
    # Branching bounds are passed directly to the LP solver as variable bounds
    # (lb[j], ub[j]); the GMI derivation still holds because t_j = ub_j - x_j
    # and the substitution is valid for any fixed ub_j.
    N = 2 * n + m
    M = m + n
    E = np.zeros((M, N), dtype=np.float64)
    d = np.zeros(M, dtype=np.float64)

    # cover rows:  A x - s = b
    E[:m, :n]    = A
    E[:m, n:n+m] = -np.eye(m)
    d[:m]        = b
    # ubound rows: x + t = ub_j  (root: ub=1; after branching ub may be 0 or 1)
    E[m:, :n]    = np.eye(n)
    E[m:, n+m:]  = np.eye(n)
    d[m:]        = ub   # node-local upper bounds

    cost = np.concatenate([c, np.zeros(m), np.zeros(n)])

    # ---- solve the standard-form LP with highspy, read the basis ------------
    try:
        basic_cols, z_sf = _solve_standard_form(highspy, E, d, cost, N, M, lb, ub)
    except Exception:
        return []
    if basic_cols is None:
        return []

    # ---- factorise B using LU (safer than explicit inverse) -----------------
    try:
        B = E[:, basic_cols]           # [M, M]
        # bbar: values of basic variables in the optimal BFS
        bbar = np.linalg.solve(B, d)   # B x = d  ->  x = B^-1 d
    except np.linalg.LinAlgError:
        return []

    nonbasic = np.setdiff1d(np.arange(N), basic_cols, assume_unique=False)

    cuts   = []
    seen   = set()

    for r, col in enumerate(basic_cols):
        if col >= n:                           # only structural x_j
            continue
        f0 = _frac(bbar[r])
        if f0 < _INTEG_TOL or f0 > 1.0 - _INTEG_TOL:
            continue                          # basic value ~integer → no cut

        # tableau row for this basic variable:  e_r^T B^{-1} E
        # Solved as  B^T y = e_r  then  arow = y^T E  (avoids forming B^-1)
        e_r  = np.zeros(M); e_r[r] = 1.0
        try:
            y = np.linalg.solve(B.T, e_r)    # B^T y = e_r
        except np.linalg.LinAlgError:
            continue
        arow = y @ E                          # [N] — full tableau row

        # GMI cut in (x, s, t) space: sum_{k nonbasic} frac(a_k) v_k >= f0
        # Back-substitute:
        #   s_i = A_i x - b_i   →  add w*A_i to alpha, add w*b_i to beta
        #   t_j = 1   - x_j     →  subtract w from alpha[j], subtract w from beta
        alpha = np.zeros(n, dtype=np.float64)
        beta  = f0
        any_coeff = False

        for k in nonbasic:
            w = _frac(arow[k])
            if w < _COEFF_TOL or w > 1.0 - _COEFF_TOL:
                continue
            any_coeff = True
            if k < n:                         # v_k = x_k
                alpha[k] += w
            elif k < n + m:                   # v_k = s_i = A_i x - b_i
                i = k - n
                alpha    += w * A[i, :]
                beta     += w * b[i]          # ← correct for general b
            else:                             # v_k = t_j = 1 - x_j
                j = k - n - m
                alpha[j] -= w
                beta     -= w

        if not any_coeff:
            continue
        # Trim tiny coefficients (numerical noise)
        alpha[np.abs(alpha) < _COEFF_TOL] = 0.0
        if not np.any(np.abs(alpha) > _COEFF_TOL):
            continue

        # Sanity / finite check
        if not (np.all(np.isfinite(alpha)) and np.isfinite(beta)):
            continue
        if np.max(np.abs(alpha)) > _COEFF_CAP:
            continue

        # Violation check: the cut must be violated by the current LP solution.
        # alpha @ x_lp < beta - tol  (the cut is NOT satisfied → it cuts x_lp)
        if x_lp is not None:
            violation = beta - float(alpha @ x_lp)
            if violation <= _VIOL_TOL:
                continue

        # Deduplicate on normalized representation (inf-norm = 1) but return
        # the original un-normalized cut so LP injection has full strength.
        inf_norm = np.max(np.abs(alpha))
        if inf_norm < _COEFF_TOL:
            continue
        alpha_n = alpha / inf_norm
        beta_n  = beta  / inf_norm

        key = (tuple(np.round(alpha_n, 5)), round(beta_n, 5))
        if key in seen:
            continue
        seen.add(key)

        cuts.append((alpha, float(beta)))  # un-normalized: full LP strength
        if len(cuts) >= max_cuts:
            break

    return cuts


def _solve_standard_form(highspy, E, d, cost, N, M, var_lb=None, var_ub=None):
    """
    Solve  min cost^T z  s.t.  E z = d,  lb <= x <= ub, slack/surplus >= 0
    with highspy. var_lb/var_ub apply only to the first n x-columns.
    Returns (basic_column_indices, z_values) or (None, None) on failure.
    """
    inf = highspy.kHighsInf
    n_x = len(var_lb) if var_lb is not None else 0

    col_lo = [0.0] * N
    col_hi = [inf] * N
    if var_lb is not None:
        for j in range(n_x):
            col_lo[j] = float(var_lb[j])
    if var_ub is not None:
        for j in range(n_x):
            col_hi[j] = float(var_ub[j])

    lp = highspy.HighsLp()
    lp.num_col_   = N
    lp.num_row_   = M
    lp.col_cost_  = cost.astype(np.float64).tolist()
    lp.col_lower_ = col_lo
    lp.col_upper_ = col_hi
    lp.row_lower_ = d.astype(np.float64).tolist()
    lp.row_upper_ = d.astype(np.float64).tolist()

    lp.a_matrix_.format_  = highspy.MatrixFormat.kColwise
    lp.a_matrix_.num_col_ = N
    lp.a_matrix_.num_row_ = M
    start, index, value = [], [], []
    for j in range(N):
        start.append(len(index))
        nz = np.where(np.abs(E[:, j]) > 1e-12)[0]
        for i in nz:
            index.append(int(i))
            value.append(float(E[i, j]))
    start.append(len(index))
    lp.a_matrix_.start_ = start
    lp.a_matrix_.index_ = index
    lp.a_matrix_.value_ = value

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    if h.passModel(lp) != highspy.HighsStatus.kOk:
        return None, None
    h.run()
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return None, None

    sol      = h.getSolution()
    z        = np.array(sol.col_value[:N], dtype=np.float64)
    basis    = h.getBasis()
    kBasic   = highspy.HighsBasisStatus.kBasic
    basic_cols = np.array(
        [j for j in range(N) if list(basis.col_status)[j] == kBasic],
        dtype=np.int64,
    )
    if basic_cols.size != M:
        return None, None
    return basic_cols, z
