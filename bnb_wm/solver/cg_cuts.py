"""
cg_cuts.py — GNN-guided Chvátal-Gomory cuts for binary set cover.

Validity scope
--------------
Binary set cover: A x >= b, x ∈ {0,1}^n, A ∈ {0,1}^{m×n}, b ∈ Z^m.
For any nonneg multipliers λ ∈ R^m, the aggregated row

    (λᵀA) x >= λᵀb

is valid. Since x ∈ {0,1}, rounding down the LHS coefficients gives the
Chvátal-Gomory cut:

    ⌊λᵀA⌋ · x >= ⌈λᵀb⌉

which is valid for all integer x and potentially cuts off the LP optimum x_lp.

GNN guidance
------------
Row multipliers λ are chosen using a per-variable importance score:

  Phase A (early rounds):  importance_j = attn_j × frac(x_lp_j)
      attn_j from CrossAttentionPool.forward_attn(z_current, h_vars)
      → targets rows around structurally important + LP-loose variables

  Phase B (later rounds):  importance_j = π_j × frac(x_lp_j)
      π_j from softmax(policy_scores) under current z
      → targets rows around variables the policy is still uncertain about

For each selected row subset, λ is concentrated on those rows (uniform over
the subset), giving a cut direction aligned with the GNN's learned geometry.

Cut embedding
-------------
The cut embedding is a weighted sum of GNN variable embeddings:

    cut_embed = Σ_j coeff_j * h_vars[j]   (H-dim, same space as branch actions)

This makes cuts first-class actions in the dynamics model alongside branches.
No new nn.Module is needed: the embedding is computed in numpy/torch arithmetic.
"""

import itertools
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
_VIOL_TOL  = 1e-5   # cut must be violated by at least this much
_COEFF_TOL = 1e-8   # cut coefficients below this are treated as zero
_COEFF_CAP = 1e6    # reject cuts with absurdly large coefficients


def generate_cg_cuts(
    A: np.ndarray,
    b: np.ndarray,
    x_lp: np.ndarray,
    importance: np.ndarray,
    h_vars: torch.Tensor,
    n_cuts: int = 6,
    top_rows: int = 15,
    max_subset_size: int = 2,
) -> list:
    """Generate GNN-guided Chvátal-Gomory cuts for binary set cover.

    Args:
        A           : [m, n] binary covering matrix (A x >= b)
        b           : [m]    integer right-hand side
        x_lp        : [n]    current LP optimal solution
        importance  : [n]    per-variable importance score (attn×frac or π×frac)
        h_vars      : [n, H] GNN variable embeddings (torch, CPU or GPU)
        n_cuts      : maximum cuts to return
        top_rows    : number of rows to consider for aggregation
        max_subset_size : maximum row subset size for aggregation

    Returns:
        list of dicts, each with:
            'coeff'     : [n] np.float64 — cut LHS coefficients (integer-valued)
            'rhs'       : float           — cut RHS (ceiling of aggregated b)
            'violation' : float           — how much x_lp violates the cut
            'embed'     : torch.Tensor [H] — GNN-structured cut embedding
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x_lp = np.asarray(x_lp, dtype=np.float64)
    importance = np.asarray(importance, dtype=np.float64)
    m, n = A.shape

    # Score each row by the max importance among its nonzero variables.
    # Rows covering high-importance variables get aggregated first.
    row_score = np.array([
        importance[A[i] > 0.5].max() if (A[i] > 0.5).any() else 0.0
        for i in range(m)
    ])
    selected_rows = np.argsort(-row_score)[:top_rows]

    cuts = []
    seen = set()

    # Try subsets of selected rows in increasing size
    for size in range(1, max_subset_size + 1):
        if len(cuts) >= n_cuts:
            break
        # Limit combinations: single rows + pairs only (covers most useful CG cuts).
        # C(15,1)=15, C(15,2)=105 — fast numpy arithmetic throughout.
        row_pool = selected_rows[:top_rows]
        for subset in itertools.islice(itertools.combinations(row_pool, size), 200):
            if len(cuts) >= n_cuts:
                break

            subset = list(subset)
            # λ = row importance scores (unnormalized; only ratios matter)
            lam = row_score[subset]
            lam = lam / (lam.sum() + 1e-12)

            agg_lhs = lam @ A[subset]        # [n] float
            agg_rhs = float(lam @ b[subset]) # float

            # CG rounding: valid for binary x
            coeff = np.floor(agg_lhs)        # [n] integer-valued float
            rhs_int = float(np.ceil(agg_rhs))

            # Skip trivial or numerically bad cuts
            if not np.any(coeff > _COEFF_TOL):
                continue
            if np.max(np.abs(coeff)) > _COEFF_CAP:
                continue

            # Violation check: cut must be violated by x_lp
            lp_val = float(coeff @ x_lp)
            violation = rhs_int - lp_val
            if violation <= _VIOL_TOL:
                continue

            # Deduplication on normalized coefficient vector
            inf_norm = np.max(np.abs(coeff))
            key = tuple(np.round(coeff / inf_norm, 4))
            if key in seen:
                continue
            seen.add(key)

            # Cut embedding: importance-weighted sum of GNN variable embeddings
            # for variables with nonzero cut coefficients — same space as h_vars
            pos_mask = coeff > _COEFF_TOL
            weights = torch.tensor(
                coeff[pos_mask] / (coeff[pos_mask].sum() + 1e-12),
                dtype=h_vars.dtype, device=h_vars.device,
            )
            cut_embed = (h_vars[pos_mask] * weights.unsqueeze(-1)).sum(0)  # [H]

            cuts.append({
                "coeff":     coeff,
                "rhs":       rhs_int,
                "violation": violation,
                "embed":     cut_embed,
            })

    # Sort by violation descending (most violated first)
    cuts.sort(key=lambda c: -c["violation"])
    return cuts[:n_cuts]


def importance_from_attn(
    attn: torch.Tensor,
    x_lp: np.ndarray,
    device: torch.device | None = None,
) -> np.ndarray:
    """Phase A importance: attention × fractionality.

    Args:
        attn  : [V] CrossAttentionPool attention weights (torch)
        x_lp  : [V] LP solution values
        device: unused, kept for API symmetry
    Returns:
        importance : [V] numpy array
    """
    frac = np.minimum(x_lp, 1.0 - x_lp)          # ∈ [0, 0.5], max at 0.5
    return attn.cpu().numpy() * frac


def importance_from_policy(
    policy_scores: torch.Tensor,
    x_lp: np.ndarray,
) -> np.ndarray:
    """Phase B importance: policy probability × fractionality.

    Args:
        policy_scores : [V] raw policy logits (torch)
        x_lp          : [V] LP solution values
    Returns:
        importance : [V] numpy array
    """
    pi = torch.softmax(policy_scores, dim=0).cpu().numpy()
    frac = np.minimum(x_lp, 1.0 - x_lp)
    return pi * frac
