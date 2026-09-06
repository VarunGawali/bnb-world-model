"""
heads.py — Prediction heads for the BnB World Model.

Architecture:
    PolicyHead        : Pointer Network — scores candidates jointly via global z
    ValueHead         : MLP(z || frac_mean) — dual bound with fractional context
    SubtreeSizeHead   : same input as ValueHead, predicts log1p(subtree nodes)
    CostToGoHead      : same input as ValueHead, predicts log1p(remaining nodes)
    IntegralityHead   : MLP(z || depth || n_frac) — leaf logit with aux scalars
    ZeroShotCutScorer : parameter-free cut scorer via GNN-native embeddings

Key design decisions
--------------------
PolicyHead z-projection factorisation:
    z is broadcast to z_per_var [V, H] at the call site, so W_q and W_z were
    each being applied V times to identical vectors. PolicyHead now accepts
    z [B, H] and var_batch [V], applies both projections once per graph (O(B·H²))
    and indexes the result — not O(V·H²).

_frac_mean caching:
    ValueHead, SubtreeSizeHead and CostToGoHead all need the same fractional-
    variable mean. When all three are called for the same node (e.g. in the
    rollout scoring loop) the scatter is redundant. Each head accepts an optional
    precomputed_frac_mean; callers that invoke multiple heads in sequence should
    compute it once via compute_frac_mean() and pass it through.

ZeroShotCutScorer:
    Replaces the trained CuttingPlaneHead. CG cuts already carry a GNN-native
    embedding (Σ coeff_j · h_vars[j] ∈ ℝ^H), so scoring by cosine similarity
    with z requires no parameters and no Phase 5 training. The violation bonus
    ensures violated cuts score above non-violated ones of similar alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter


# ---------------------------------------------------------------------------
# Shared fractional-mean helper
# ---------------------------------------------------------------------------

def compute_frac_mean(
    z: torch.Tensor,
    h_vars: torch.Tensor,
    batch_vec: torch.Tensor,
    frac_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Mean embedding of fractional variables per graph, vectorised.

    Segment-means h_vars over the fractional variables of each graph and falls
    back to z for graphs with no fractional variable.

    Args:
        z         : [B, H]  graph-level embeddings (fallback)
        h_vars    : [V, H]  per-variable embeddings
        batch_vec : [V]     batch assignment for each variable
        frac_mask : [V] bool or None

    Returns:
        frac_mean : [B, H]

    Call this ONCE and pass the result as `precomputed_frac_mean` to each head
    that needs it to avoid redundant scatter operations.
    """
    batch_size = z.size(0)
    if frac_mask is None or not frac_mask.any():
        return z
    idx  = batch_vec[frac_mask]
    vals = h_vars[frac_mask]
    summ = scatter(vals, idx, dim=0, dim_size=batch_size, reduce="sum")
    cnt  = scatter(torch.ones(idx.size(0), dtype=z.dtype, device=z.device),
                   idx, dim=0, dim_size=batch_size, reduce="sum")
    mean = summ / cnt.clamp_min(1.0).unsqueeze(-1)
    return torch.where(cnt.unsqueeze(-1) > 0, mean, z)


# Keep _frac_mean as an alias for internal backward compatibility.
_frac_mean = compute_frac_mean


# ---------------------------------------------------------------------------
# PolicyHead
# ---------------------------------------------------------------------------

class PolicyHead(nn.Module):
    """
    Pointer Network that scores branching candidates jointly.

    score_i = v · (W_q(z)[i] * tanh(W_k(h_var_i) + W_z(z)[i])) / sqrt(H)

    Both W_q and W_z project the graph-level z, not the broadcast z_per_var.
    z [B, H] is projected once per graph; the results are indexed by var_batch
    to build the [V, H] tensors — O(B·H²) instead of O(V·H²).

    Interface change vs old API:
        Old: forward(h_vars [V,H], z_per_var [V,H])
        New: forward(h_vars [V,H], z [B,H], var_batch [V])
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_z = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v   = nn.Linear(hidden_dim, 1, bias=False)
        self.scale = hidden_dim ** -0.5

    def forward(
        self,
        h_vars: torch.Tensor,    # [V, H]  per-variable embeddings
        z: torch.Tensor,          # [B, H]  graph-level embedding (one per graph)
        var_batch: torch.Tensor,  # [V]     batch assignment for each variable
    ) -> torch.Tensor:            # [V]     branching scores
        # Project z once per graph, then broadcast via indexing — not expand().
        z_q = self.W_q(z)[var_batch]                           # [V, H]
        z_k = self.W_z(z)[var_batch]                           # [V, H]
        key  = torch.tanh(self.W_k(h_vars) + z_k)             # [V, H]
        return self.v(z_q * key * self.scale).squeeze(-1)      # [V]


# ---------------------------------------------------------------------------
# Value / SubtreeSize / CostToGo heads (shared structure)
# ---------------------------------------------------------------------------

class _EnrichedHead(nn.Module):
    """Base for heads that use concat(z, frac_mean) as input."""

    def __init__(self, hidden_dim: int, output_activation=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.output_activation = output_activation

    def _run(
        self,
        z: torch.Tensor,
        h_vars: torch.Tensor,
        batch_vec: torch.Tensor,
        frac_mask: torch.Tensor | None,
        precomputed_frac_mean: torch.Tensor | None,
    ) -> torch.Tensor:
        if precomputed_frac_mean is not None:
            fm = precomputed_frac_mean
        else:
            fm = compute_frac_mean(z, h_vars, batch_vec, frac_mask)
        out = self.net(torch.cat([z, fm], dim=-1)).squeeze(-1)
        if self.output_activation is not None:
            out = self.output_activation(out)
        return out


class ValueHead(_EnrichedHead):
    """Dual bound predictor: MLP(z || frac_mean) → scalar per graph."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__(hidden_dim, output_activation=None)

    def forward(
        self,
        z: torch.Tensor,
        h_vars: torch.Tensor,
        batch_vec: torch.Tensor,
        frac_mask: torch.Tensor | None = None,
        precomputed_frac_mean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._run(z, h_vars, batch_vec, frac_mask, precomputed_frac_mean)


class SubtreeSizeHead(_EnrichedHead):
    """Predicts log1p(subtree node count) rooted at the current node.

    The target is log1p(subtree_size) — well-conditioned because true subtree
    sizes span orders of magnitude. Softplus keeps predicted log-size ≥ 0.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__(hidden_dim, output_activation=F.softplus)

    def forward(
        self,
        z: torch.Tensor,
        h_vars: torch.Tensor,
        batch_vec: torch.Tensor,
        frac_mask: torch.Tensor | None = None,
        precomputed_frac_mean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._run(z, h_vars, batch_vec, frac_mask, precomputed_frac_mean)


class CostToGoHead(_EnrichedHead):
    """Predicts log1p(remaining B&B nodes) — the cost-to-go value.

    Training target: steps_to_go(t) = n_steps - t, read directly from the
    collected trajectory. Trainable on non-DFS traces (no DFS ordering needed).
    Softplus keeps predicted log-cost ≥ 0.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__(hidden_dim, output_activation=F.softplus)

    def forward(
        self,
        z: torch.Tensor,
        h_vars: torch.Tensor,
        batch_vec: torch.Tensor,
        frac_mask: torch.Tensor | None = None,
        precomputed_frac_mean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._run(z, h_vars, batch_vec, frac_mask, precomputed_frac_mean)


# ---------------------------------------------------------------------------
# IntegralityHead
# ---------------------------------------------------------------------------

class IntegralityHead(nn.Module):
    """
    Leaf-probability predictor with auxiliary scalar inputs.

    depth and n_frac are the strongest predictors of leaf proximity and
    cannot be reliably inferred from the GNN embedding alone.

    Input  : z [batch, H], depth [batch] (optional), n_frac [batch] (optional)
    Output : logit [batch]
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        depth: torch.Tensor | None = None,
        n_frac: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = z.size(0)
        if depth is None:
            depth = torch.zeros(batch_size, device=z.device, dtype=z.dtype)
        if n_frac is None:
            n_frac = torch.zeros(batch_size, device=z.device, dtype=z.dtype)
        inp = torch.cat([z, depth.float().unsqueeze(-1),
                         n_frac.float().unsqueeze(-1)], dim=-1)
        return self.net(inp).squeeze(-1)


# ---------------------------------------------------------------------------
# ZeroShotCutScorer  (replaces CuttingPlaneHead)
# ---------------------------------------------------------------------------

class ZeroShotCutScorer(nn.Module):
    """Parameter-free cut scorer using GNN-native cut embeddings.

    CG cuts from cg_cuts.generate_cg_cuts() already carry an H-dim embedding:
        cut_embed = Σ_j coeff_j · h_vars[j]   ∈ ℝ^H

    This embedding lives in the same latent space as z (both are functions of
    the GNN variable embeddings). Scoring by cosine similarity with z therefore
    asks: "does this cut target the variables the model currently finds most
    structurally important?" — a zero-shot alignment score.

    A violation bonus ensures violated cuts score above non-violated ones of
    similar alignment, matching the standard CG validity priority.

    No parameters → no Phase 5 training required.

    score_c = cos(cut_embed_c, z) + violation_weight · violation_c

    Args:
        violation_weight: coefficient for the LP violation bonus (default 1.0).
                          Higher values prioritise maximally violated cuts over
                          geometrically aligned ones.
    """

    def __init__(self, violation_weight: float = 1.0):
        super().__init__()
        self.violation_weight = violation_weight

    def forward(
        self,
        cut_embeds: torch.Tensor,   # [C, H]  GNN-native cut embeddings
        z: torch.Tensor,             # [H] or [1, H]
        violations: torch.Tensor,    # [C]  LP violation amounts (≥ 0)
    ) -> torch.Tensor:               # [C]  cut scores (higher = prefer)
        if z.dim() == 2:
            z = z.squeeze(0)
        cos = F.cosine_similarity(cut_embeds, z.unsqueeze(0), dim=-1)  # [C]
        return cos + self.violation_weight * violations
