"""
dynamics.py — GRU-based latent dynamics model.

Predicts the next latent graph embedding z_{t+1} given the current
embedding z_t and an action embedding a_t. Enables 1-step lookahead
in latent space without running another LP relaxation.

Trained with MSE + cosine loss against the true next embedding
produced by the encoder on the actual next B&B node.
"""

import torch
import torch.nn as nn


class DynamicsGRU(nn.Module):
    """
    z_{t+1}, h_{t+1} = GRU([z_t || a_t], h_t)

    The action embedding a_t should be the h_vars slice for the
    chosen branching variable (same hidden_dim as z_t).

    Args:
        hidden_dim : must match the encoder's hidden_dim

    Inputs:
        z_t    : [batch, hidden_dim]  current graph embedding
        a_emb_t: [batch, hidden_dim]  action (branching var) embedding
        h_prev : [batch, hidden_dim]  GRU hidden state (None = zeros)

    Returns:
        z_next : [batch, hidden_dim]  predicted next embedding
        h_new  : [batch, hidden_dim]  updated GRU hidden state
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(2 * hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, z_t, a_emb_t, h_prev=None):
        x = torch.cat([z_t, a_emb_t], dim=-1)

        if h_prev is None:
            if x.dim() == 1:
                h_prev = torch.zeros(self.hidden_dim, device=x.device)
            else:
                h_prev = torch.zeros(x.size(0), self.hidden_dim, device=x.device)

        h_new = self.gru(x, h_prev)
        z_next = self.proj(h_new)

        return z_next, h_new
