"""
heads.py — Prediction heads for the BnB World Model.

Architecture (current):
    PolicyHead      : Pointer Network — scores candidates jointly via global z
    ValueHead       : MLP(z || frac_mean) — dual bound with fractional context
    IntegralityHead : MLP(z || depth || n_frac) — leaf logit with aux scalars
    CuttingPlaneHead: Pointer Network — scores candidate cuts jointly via z
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter


def _frac_mean(
    z: torch.Tensor,
    h_vars: torch.Tensor,
    batch_vec: torch.Tensor,
    frac_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Mean embedding of fractional variables, per graph, vectorised.

    Segment-means h_vars over the fractional variables of each graph and falls
    back to z for graphs with no fractional variable. Mathematically identical
    to the old `for b in range(batch_size)` loop shared by the value/subtree/
    cost-to-go heads, but runs as a couple of scatter kernels instead of
    batch_size Python iterations (those loops were a CPU-bound bottleneck).
    """
    batch_size = z.size(0)
    if frac_mask is None or not frac_mask.any():
        return z
    idx  = batch_vec[frac_mask]                      # [n_frac]
    vals = h_vars[frac_mask]                         # [n_frac, H]
    summ = scatter(vals, idx, dim=0, dim_size=batch_size, reduce="sum")
    cnt  = scatter(torch.ones_like(idx, dtype=z.dtype), idx, dim=0,
                   dim_size=batch_size, reduce="sum")            # [batch_size]
    mean = summ / cnt.clamp_min(1.0).unsqueeze(-1)
    has  = (cnt > 0).unsqueeze(-1)
    return torch.where(has, mean, z)


class PolicyHead(nn.Module):
    """
    Pointer Network that scores branching candidates jointly.

    score_i = v · tanh(W_k·h_var_i + W_z·z_per_var_i) / sqrt(H)

    Input  : h_vars [total_vars, H], z_per_var [total_vars, H]
    Output : scores [total_vars]
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_z = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v   = nn.Linear(hidden_dim, 1, bias=False)
        self.scale = hidden_dim ** -0.5

    def forward(self, h_vars: torch.Tensor, z_per_var: torch.Tensor) -> torch.Tensor:
        query  = self.W_q(z_per_var)
        key    = torch.tanh(self.W_k(h_vars) + self.W_z(z_per_var))
        return self.v(query * key * self.scale).squeeze(-1)


class ValueHead(nn.Module):
    """
    Dual bound predictor with enriched input.

    Receives concat(z, frac_mean) where frac_mean is the mean embedding
    of currently fractional variables. Falls back to z when no frac_mask.

    Input  : z [batch, H], h_vars [total_vars, H],
             batch_vec [total_vars], frac_mask [total_vars] bool (optional)
    Output : v [batch]
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        h_vars: torch.Tensor,
        batch_vec: torch.Tensor,
        frac_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        frac_mean = _frac_mean(z, h_vars, batch_vec, frac_mask)
        return self.net(torch.cat([z, frac_mean], dim=-1)).squeeze(-1)


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

        inp = torch.cat([
            z,
            depth.float().unsqueeze(-1),
            n_frac.float().unsqueeze(-1),
        ], dim=-1)
        return self.net(inp).squeeze(-1)


class SubtreeSizeHead(nn.Module):
    """
    Predicts the size (node count) of the B&B subtree rooted at the current
    node, in log space.

    This is the decision-relevant quantity for branching: the solver's cost
    IS the number of nodes explored, so a model that predicts how many nodes
    a subtree will take lets us branch to *minimise predicted tree growth* —
    a direct latent-space approximation of strong branching's subtree
    evaluation.

    The target is log1p(subtree_size), which is well-conditioned because true
    subtree sizes span several orders of magnitude. The head shares the same
    enriched input as the value head (z + fractional mean) since the same
    signals — fractional state, dual gap — drive subtree growth.

    Input  : z [batch, H], h_vars [total_vars, H],
             batch_vec [total_vars], frac_mask [total_vars] bool (optional)
    Output : log_size [batch]   predicted log1p(subtree node count)
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        h_vars: torch.Tensor,
        batch_vec: torch.Tensor,
        frac_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        frac_mean = _frac_mean(z, h_vars, batch_vec, frac_mask)
        # Softplus keeps the predicted log-size non-negative (size >= 1).
        out = self.net(torch.cat([z, frac_mean], dim=-1)).squeeze(-1)
        return F.softplus(out)


class CostToGoHead(nn.Module):
    """
    Predicts the *cost-to-go* at the current node: the expected number of B&B
    nodes still to explore before the search terminates, in log space.

    This is the decision-relevant value in the B&B MDP — the objective is to
    close the gap in as few nodes as possible, so a value that estimates
    remaining work (rather than the dual bound, a proxy) directly targets what
    we care about. The training target is a Monte-Carlo return read straight
    from the trajectory: steps_to_go(t) = n_steps - t. Crucially this needs no
    DFS ordering (unlike subtree size), so it is trainable on the collected
    non-DFS traces.

    Same enriched input as the value head (z + fractional mean); softplus keeps
    the predicted log-cost non-negative.

    Input  : z [batch, H], h_vars [total_vars, H],
             batch_vec [total_vars], frac_mask [total_vars] bool (optional)
    Output : log_ctg [batch]   predicted log1p(remaining node count)
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        h_vars: torch.Tensor,
        batch_vec: torch.Tensor,
        frac_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        frac_mean = _frac_mean(z, h_vars, batch_vec, frac_mask)
        out = self.net(torch.cat([z, frac_mean], dim=-1)).squeeze(-1)
        return F.softplus(out)


class CuttingPlaneHead(nn.Module):
    """
    Pointer Network that scores candidate cuts jointly for branch-and-cut.

    Each cut k is represented by a d_cut-dim feature vector capturing:
        [violation, efficacy, density, parallelism, obj_cutoff, support_frac]

    The global node embedding z provides tree-search context so the head
    can learn to prefer cuts with lasting tightening value across the
    subtree, not just cuts that are locally tight.

    This is architecturally identical to PolicyHead but operates on cuts
    rather than variables: the global context z attends over the candidate
    pool and scores each cut relative to the current B&B node state.

    score_k = v · tanh(W_k · cut_emb_k + W_z · z) / sqrt(H)

    where cut_emb_k = ReLU(W_in · cut_feat_k) projects raw features to H-dim.

    Input  : cut_feats [n_cuts, d_cut], z [H]  (single graph, not batched)
    Output : scores    [n_cuts]
    """

    def __init__(self, hidden_dim: int = 128, cut_feat_dim: int = 6):
        super().__init__()
        self.cut_proj = nn.Linear(cut_feat_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_z = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v   = nn.Linear(hidden_dim, 1, bias=False)
        self.scale = hidden_dim ** -0.5

    def forward(self, cut_feats: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cut_feats : [n_cuts, d_cut]   per-cut features
            z         : [H]               graph-level embedding for current node
        Returns:
            scores    : [n_cuts]          unbounded cut selection logits
        """
        cut_emb = F.relu(self.cut_proj(cut_feats))              # [n_cuts, H]
        z_exp   = z.unsqueeze(0).expand(cut_emb.size(0), -1)    # [n_cuts, H]
        key     = torch.tanh(self.W_k(cut_emb) + self.W_z(z_exp))
        return self.v(key * self.scale).squeeze(-1)              # [n_cuts]
