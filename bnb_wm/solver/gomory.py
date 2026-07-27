"""
gomory.py — Provably valid Gomory fractional cuts for the neural B&C solver.

Replaces the previous pairwise "intersection >= 1" cut, which was mathematically
INVALID for a covering system (it rounded a >= constraint's coefficients down,
which can cut off feasible integer points). Gomory fractional cuts, by contrast,
are valid for *every* integer-feasible point of the problem.

Global validity
---------------
A Gomory cut is globally valid only if it is derived from globally valid rows,
not from a node's local branching bounds. We therefore derive cuts ONCE at the
root, from the original covering constraints A x >= 1 together with the box
0 <= x <= 1, with NO branching bounds applied. The resulting cuts hold at every
node and are safely propagated to descendants.

Standard form (all variables >= 0, equalities)
----------------------------------------------
We lift the upper bounds x <= 1 into explicit rows so that every nonbasic
variable sits at its lower bound 0 — the clean regime for the textbook Gomory
fractional cut (no bounded-variable book-keeping):

    cover  i :  sum_j A_ij x_j - s_i = 1,     s_i >= 0     (m rows)
    ubound j :  x_j + t_j        = 1,          t_j >= 0     (n rows)
    variables ordered [ x (n) | s (m) | t (n) ],  all >= 0

For a basic *structural* variable x_j whose LP value is fractional, with tableau
row  x_j + sum_{k in N} a_k v_k = b̄  (v_k the nonbasic variables, all >= 0), the
Gomory fractional cut is

    sum_{k in N} frac(a_k) v_k  >=  frac(b̄).

We then substitute the slack definitions  s_i = (A x)_i - 1  and  t_j = 1 - x_j
to express the cut purely in the structural variables x, yielding a globally
valid inequality  alpha^T x >= beta.

Robustness
----------
This routine requires highspy (to read the optimal basis). If highspy is
unavailable or anything numerically doubtful occurs, it returns [] — i.e. NO
cuts — so the solver stays correct (no cuts is provably correct) and never emits
an invalid cut. Validity is the invariant; coverage is best-effort.
"""

import numpy as np


def _frac(y):
    """Fractional part in [0, 1)."""
    return y - np.floor(y)


def generate_root_gomory_cuts(A, b, c, highspy, max_cuts=50, tol=1e-6):
    """
    Generate globally valid Gomory fractional cuts at the root.

    Args:
        A       : [m, n] covering matrix (A x >= b, b = 1 for Set Cover)
        b       : [m]    right-hand side
        c       : [n]    objective
        highspy : the imported highspy module (basis access); if None -> []
        max_cuts: cap on the number of cuts returned
        tol     : numerical tolerance for "fractional" and "nonzero"

    Returns:
        list of (lhs [n] float64, rhs float) with the meaning  lhs @ x >= rhs.
        Empty if highspy is unavailable or nothing valid could be derived.
    """
    if highspy is None:
        return []

    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    c = np.asarray(c, dtype=np.float64).reshape(-1)
    m, n = A.shape

    # ---- build the standard-form equality system  E z = d,  z >= 0 ----------
    # columns: [ x (n) | s (m) | t (n) ]  -> N = 2n + m
    # rows:    [ cover (m) | ubound (n) ] -> M = m + n
    N = 2 * n + m
    M = m + n
    E = np.zeros((M, N), dtype=np.float64)
    d = np.zeros(M, dtype=np.float64)

    # cover rows:  A x - s = b
    E[:m, :n] = A
    E[:m, n:n + m] = -np.eye(m)
    d[:m] = b
    # ubound rows: x + t = 1
    E[m:, :n] = np.eye(n)
    E[m:, n + m:] = np.eye(n)
    d[m:] = 1.0

    # objective over [x | s | t]: only x has cost
    cost = np.concatenate([c, np.zeros(m), np.zeros(n)])

    # ---- solve the standard-form LP with highspy, read the basis -----------
    try:
        basic_cols, zval = _solve_standard_form(highspy, E, d, cost, N, M, tol)
    except Exception:
        return []
    if basic_cols is None:
        return []

    # ---- assemble B, B^{-1}, and generate cuts from fractional basic x_j ----
    try:
        B = E[:, basic_cols]                       # [M, M]
        Binv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        return []

    bbar = Binv @ d                                # basic-variable values
    nonbasic = np.setdiff1d(np.arange(N), basic_cols, assume_unique=False)

    cuts = []
    seen = set()
    for r, col in enumerate(basic_cols):
        if col >= n:                               # only structural x_j
            continue
        f0 = _frac(bbar[r])
        if f0 < tol or f0 > 1.0 - tol:             # basic value ~integer -> skip
            continue

        # tableau row over ALL columns: e_r^T B^{-1} E
        arow = Binv[r, :] @ E                      # [N]

        # Gomory cut in (x, s, t) space: sum_{k in N} frac(a_k) v_k >= f0
        alpha = np.zeros(n, dtype=np.float64)      # coefficients on x
        beta = f0                                  # rhs accumulator
        any_coeff = False
        for k in nonbasic:
            w = _frac(arow[k])
            if w < tol or w > 1.0 - tol:
                continue
            any_coeff = True
            if k < n:                              # v_k = x_k
                alpha[k] += w
            elif k < n + m:                        # v_k = s_i = (A x)_i - 1
                i = k - n
                alpha += w * A[i, :]
                beta += w                          # move -w constant to rhs
            else:                                  # v_k = t_j = 1 - x_j
                j = k - n - m
                alpha[j] -= w
                beta -= w                          # move +w constant to rhs

        if not any_coeff or beta < tol:
            continue
        # clean tiny coefficients
        alpha[np.abs(alpha) < tol] = 0.0
        if not np.any(np.abs(alpha) > tol):
            continue

        key = (tuple(np.round(alpha, 6)), round(beta, 6))
        if key in seen:
            continue
        seen.add(key)
        cuts.append((alpha, float(beta)))
        if len(cuts) >= max_cuts:
            break

    return cuts


def _solve_standard_form(highspy, E, d, cost, N, M, tol):
    """
    Solve  min cost^T z  s.t.  E z = d,  z >= 0  with highspy, and return
    (basic_column_indices, z_values). Returns (None, None) if not solved.

    Uses the version-stable passModel(HighsLp) API (the incremental
    addVars/changeCols* API differs across highspy releases).
    """
    inf = highspy.kHighsInf

    lp = highspy.HighsLp()
    lp.num_col_ = N
    lp.num_row_ = M
    lp.col_cost_  = cost.astype(np.float64).tolist()
    lp.col_lower_ = [0.0] * N
    lp.col_upper_ = [inf] * N
    # equality rows: lower == upper == d
    lp.row_lower_ = d.astype(np.float64).tolist()
    lp.row_upper_ = d.astype(np.float64).tolist()

    # Constraint matrix in column-wise (CSC) form.
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.num_col_ = N
    lp.a_matrix_.num_row_ = M
    start, index, value = [], [], []
    for j in range(N):
        start.append(len(index))
        col = E[:, j]
        nz = np.where(np.abs(col) > 1e-12)[0]
        for i in nz:
            index.append(int(i))
            value.append(float(col[i]))
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

    sol = h.getSolution()
    z = np.array(sol.col_value[:N], dtype=np.float64)

    basis = h.getBasis()
    kBasic = highspy.HighsBasisStatus.kBasic
    col_status = list(basis.col_status)
    basic_cols = np.array([j for j in range(N) if col_status[j] == kBasic],
                          dtype=np.int64)
    # A valid basis for M equality rows has exactly M basic columns.
    if basic_cols.size != M:
        return None, None
    return basic_cols, z
