"""
world_model.py — Full BnB World Model.

Composes the encoder, policy head, value head, integrality head,
and dynamics model into a single nn.Module with named helper methods
for each use case (training vs. inference).
"""

import torch.nn as nn
from .encoder import BipartiteGNN
from .heads import PolicyHead, ValueHead, IntegralityHead
from .dynamics import DynamicsGRU


class BnBWorldModel(nn.Module):
    """
    Branch-and-Bound World Model.

    Components
    ----------
    encoder     : BipartiteGNN  — encodes a B&B node into h_vars, z
    policy      : PolicyHead    — branching variable scores
    value       : ValueHead     — normalised dual bound prediction
    integrality : IntegralityHead — leaf probability logit
    dynamics    : DynamicsGRU   — latent transition model

    Training phases
    ---------------
    Phase 1 : policy head only (imitation learning from strong branching)
    Phase 2 : value head only  (encoder + policy frozen)
    Phase 3 : dynamics model   (encoder frozen, trained on sequences)
    Phase 4 : joint fine-tuning of all components end-to-end
    """

    def __init__(self, hidden_dim: int = 128, n_gnn_layers: int = 3):
        super().__init__()
        self.encoder = BipartiteGNN(hidden_dim=hidden_dim, n_layers=n_gnn_layers)
        self.policy = PolicyHead(hidden_dim)
        self.value = ValueHead(hidden_dim)
        self.integrality = IntegralityHead(hidden_dim)
        self.dynamics = DynamicsGRU(hidden_dim)

    # ------------------------------------------------------------------
    # Primary forward (used during Phase 1 training)
    # Returns policy scores and graph embedding for the batch.
    # ------------------------------------------------------------------
    def forward(self, batch):
        """
        Args:
            batch : PyG Batch with fields x, edge_index, node_type, batch

        Returns:
            scores : [total_vars]      policy logits
            z      : [batch_size, H]   graph-level embeddings
        """
        h_vars, z = self.encoder(
            batch.x, batch.edge_index, batch.node_type, batch.batch
        )
        scores = self.policy(h_vars)
        return scores, z

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def encode(self, batch):
        """Run encoder only. Returns (h_vars, z)."""
        return self.encoder(
            batch.x, batch.edge_index, batch.node_type, batch.batch
        )

    def policy_scores(self, h_vars):
        """Score variable nodes for branching."""
        return self.policy(h_vars)

    def value_pred(self, z):
        """Predict normalised dual bound."""
        return self.value(z)

    def integrality_logit(self, z):
        """Predict raw logit for P(next node is leaf)."""
        return self.integrality(z)

    def dynamics_step(self, z_t, a_emb_t, h_prev=None):
        """Predict next latent state given current state + action embedding."""
        return self.dynamics(z_t, a_emb_t, h_prev)
