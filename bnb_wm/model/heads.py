"""
heads.py — Prediction heads for the BnB World Model.

Architecture changes vs. original:
    PolicyHead      : MLP -> Pointer Network
                      The global context z modulates attention over each
                      candidate variable embedding, scoring them jointly
                      rather than independently.
    ValueHead       : MLP -> MLP with enriched input
                      Receives z (global) + mean of fractional variable
                      embeddings (local) so the dual-bound prediction is
                      anchored to the actual fractional structure.
    IntegralityHead : MLP -> MLP with auxiliary scalar inputs
                      Depth and number-of-fractional-variables are strong
                      predictors of leaf proximity that the GNN cannot
                      easily encode; they are concatenated to z before scoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyHead(nn.Module):
    """
    Pointer Network that scores branching candidates jointly.

    Instead of scoring each variable independently with a plain MLP,
    a learned query derived from the global graph embedding z attends
    over all per-variable embeddings h_vars. This means the policy sees
    all candidates simultaneously and scores them relative to each other,
    closely mirroring how strong branching evaluates candidates.

    score_i = (W_q · z) · tanh(W_k · h_var_i + W_z · z) / sqrt(H)

    Input  : h_vars [total_vars, H], z_per_var [total_vars, H]
             (z_per_var is the graph embedding z broadcast to each variable)
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
        """
        Args:
            h_vars    : [total_vars, H]  per-variable embeddings from encoder
            z_per_var : [total_vars, H]  graph embedding z repeated for each var
        Returns:
            scores    : [total_vars]     unbounded branching logits
        """
        query  = self.W_q(z_per_var)                     # [total_vars, H]
        key    = torch.tanh(self.W_k(h_vars) + self.W_z(z_per_var))  # [total_vars, H]
        scores = self.v(query * key * self.scale).squeeze(-1)         # [total_vars]
        return scores


class ValueHead(nn.Module):
    """
    Dual bound predictor with enriched input.

    Receives the concatenation of:
        z              : global graph embedding [batch, H]
        frac_mean      : mean embedding of fractional variables [batch, H]

    Fractional variable structure is the primary determinant of the LP
    relaxation bound; giving the value head direct access to it removes
    a representational bottleneck present in the original single-z MLP.

    If no fractional mask is provided, frac_mean falls back to z (safe
    for batches where all variables are integer).

    Input  : z [batch, H], h_vars [total_vars, H],
             batch_vec [total_vars], frac_mask [total_vars] (optional bool)
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

        inp = torch.cat([z, frac_mean], dim=-1)   # [batch, 2H]
        return self.net(inp).squeeze(-1)


class IntegralityHead(nn.Module):
    """
    Leaf-probability predictor with auxiliary scalar inputs.

    Depth in the B&B tree and number of fractional variables are among
    the strongest predictors of whether a node is a leaf. These cannot
    be reliably inferred from the GNN embedding alone (depth is not
    encoded in any node feature; fractional count requires counting).
    Concatenating them as scalars gives the head a direct signal.

    Input  : z [batch, H], depth [batch], n_frac [batch]
    Output : logit [batch]   (raw; apply sigmoid for probability)
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        # +2 for the two scalar auxiliary inputs
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

        depth = depth.float().unsqueeze(-1)   # [batch, 1]
        n_frac = n_frac.float().unsqueeze(-1) # [batch, 1]

        inp = torch.cat([z, depth, n_frac], dim=-1)   # [batch, H+2]
        return self.net(inp).squeeze(-1)
