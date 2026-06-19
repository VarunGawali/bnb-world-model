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
        batch_size = z.size(0)

        if frac_mask is not None and frac_mask.any():
            frac_mean = torch.zeros_like(z)
            for b in range(batch_size):
                sel = (batch_vec == b) & frac_mask
                frac_mean[b] = h_vars[sel].mean(0) if sel.any() else z[b]
        else:
            frac_mean = z

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
