"""
world_model.py — Full BnB World Model.

Composes the encoder, policy head, value head, integrality head,
and dynamics model into a single nn.Module with named helper methods
for each use case (training vs. inference).

Architecture changes vs. original:
    - encoder  : BipartiteGNN now uses GATv2Conv + edge features +
                 CrossAttentionPool readout
    - policy   : PolicyHead is now a Pointer Network (requires z_per_var)
    - value    : ValueHead now accepts h_vars + frac_mask for richer input
    - integrality : IntegralityHead now accepts depth + n_frac scalars
    - dynamics : DynamicsGRU replaced by DynamicsTransformer (causal)
"""

import torch
import torch.nn as nn
from .encoder import BipartiteGNN
from .heads import PolicyHead, ValueHead, IntegralityHead
from .dynamics import DynamicsTransformer


class BnBWorldModel(nn.Module):
    """
    Branch-and-Bound World Model.

    Components
    ----------
    encoder     : BipartiteGNN        — GATv2 + edge features + attention pool
    policy      : PolicyHead          — Pointer Network branching scores
    value       : ValueHead           — dual bound prediction (z + frac mean)
    integrality : IntegralityHead     — leaf logit (z + depth + n_frac)
    dynamics    : DynamicsTransformer — causal Transformer latent transition

    Training phases
    ---------------
    Phase 1 : policy head  (imitation learning from strong branching)
    Phase 2 : value head   (encoder + policy frozen)
    Phase 3 : dynamics     (encoder frozen, trained on trajectory sequences)
    Phase 4 : joint fine-tuning of all components end-to-end
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_gnn_layers: int = 3,
        n_gnn_heads: int = 4,
        n_dyn_layers: int = 4,
        n_dyn_heads: int = 4,
        max_seq: int = 512,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder     = BipartiteGNN(
            hidden_dim=hidden_dim,
            n_layers=n_gnn_layers,
            n_heads=n_gnn_heads,
        )
        self.policy      = PolicyHead(hidden_dim)
        self.value       = ValueHead(hidden_dim)
        self.integrality = IntegralityHead(hidden_dim)
        self.dynamics    = DynamicsTransformer(
            hidden_dim=hidden_dim,
            n_layers=n_dyn_layers,
            n_heads=n_dyn_heads,
            max_seq=max_seq,
        )

    # ------------------------------------------------------------------
    # Primary forward (used during Phase 1 training)
    # ------------------------------------------------------------------
    def forward(self, batch):
        """
        Args:
            batch : PyG Batch with fields:
                x, edge_index, node_type, batch
                edge_attr  (optional [E, 3] edge features)

        Returns:
            scores : [total_vars]      policy pointer-network logits
            z      : [batch_size, H]   graph-level embeddings
        """
        edge_attr = getattr(batch, "edge_attr", None)
        h_vars, z = self.encoder(
            batch.x, batch.edge_index, batch.node_type, batch.batch,
            edge_attr=edge_attr,
        )

        # Broadcast z to each variable for the Pointer Network
        var_mask  = batch.node_type == 0
        z_per_var = z[batch.batch[var_mask]]

        scores = self.policy(h_vars, z_per_var)
        return scores, z

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def encode(self, batch):
        """Run encoder only. Returns (h_vars, z)."""
        edge_attr = getattr(batch, "edge_attr", None)
        return self.encoder(
            batch.x, batch.edge_index, batch.node_type, batch.batch,
            edge_attr=edge_attr,
        )

    def policy_scores(self, h_vars: torch.Tensor, z: torch.Tensor,
                      var_batch: torch.Tensor) -> torch.Tensor:
        """
        Score variable nodes for branching.

        Args:
            h_vars    : [total_vars, H]
            z         : [batch_size, H]
            var_batch : [total_vars]  batch index for each variable node
        """
        z_per_var = z[var_batch]
        return self.policy(h_vars, z_per_var)

    def value_pred(
        self,
        z: torch.Tensor,
        h_vars: torch.Tensor,
        batch_vec: torch.Tensor,
        frac_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict normalised dual bound."""
        return self.value(z, h_vars, batch_vec, frac_mask)

    def integrality_logit(
        self,
        z: torch.Tensor,
        depth: torch.Tensor | None = None,
        n_frac: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict raw logit for P(next node is leaf)."""
        return self.integrality(z, depth, n_frac)

    def dynamics_forward(
        self,
        z_seq: torch.Tensor,
        a_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parallel training forward over full trajectories.

        Args:
            z_seq : [B, T, H]  encoder embeddings along trajectory
            a_seq : [B, T, H]  action embeddings along trajectory

        Returns:
            z_pred : [B, T, H]  predicted next embeddings
        """
        return self.dynamics(z_seq, a_seq)

    def dynamics_step(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        past_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step inference (replaces the old GRU step interface).

        Args:
            z_t         : [B, H]
            a_t         : [B, H]
            past_tokens : [B, t, H] or None

        Returns:
            z_next      : [B, H]
            past_tokens : [B, t+1, H]  updated buffer
        """
        return self.dynamics.step(z_t, a_t, past_tokens)
