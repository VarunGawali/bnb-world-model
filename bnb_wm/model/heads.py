"""
heads.py — Prediction heads for the BnB World Model.

Three independent heads, each operating on different embeddings:
    PolicyHead      : scores each variable node -> branching logits
    ValueHead       : predicts normalised dual bound from graph embedding z
    IntegralityHead : predicts P(next node is leaf) from graph embedding z
"""

import torch.nn as nn


class PolicyHead(nn.Module):
    """
    Maps per-variable embeddings h_vars -> unbounded scores.
    During training, scores are masked to the candidate action set
    and trained with cross-entropy against the strong-branching label.

    Input  : h_vars [num_vars, hidden_dim]
    Output : scores [num_vars]
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_vars):
        return self.net(h_vars).squeeze(-1)


class ValueHead(nn.Module):
    """
    Maps graph-level embedding z -> scalar normalised dual bound prediction.
    Trained with Huber loss against the normalised dual bound recorded
    at each B&B node during dataset generation.

    Input  : z [batch_size, hidden_dim]
    Output : v [batch_size]
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


class IntegralityHead(nn.Module):
    """
    Maps graph-level embedding z -> logit for P(next node is leaf).
    Trained with BCE + pos_weight to handle class imbalance
    (leaf nodes are rare early in the tree).

    Input  : z [batch_size, hidden_dim]
    Output : logit [batch_size]  (raw, apply sigmoid for probability)
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)
