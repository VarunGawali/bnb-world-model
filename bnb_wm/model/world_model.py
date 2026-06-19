"""
world_model.py — Full BnB World Model.

Composes encoder, policy head, value head, integrality head,
cutting-plane head, and dynamics model into a single nn.Module.

Components
----------
encoder         : BipartiteGNN        — GATv2 + edge features + attention pool
policy          : PolicyHead          — Pointer Network branching scores
value           : ValueHead           — dual bound (z + fractional mean)
integrality     : IntegralityHead     — leaf logit (z + depth + n_frac)
cutting_planes  : CuttingPlaneHead    — cut selection scores (z + cut features)
dynamics        : DynamicsTransformer — causal Transformer latent transition

Training phases
---------------
Phase 1 : policy head     (imitation from strong branching)
Phase 2 : value head      (encoder + policy frozen)
Phase 3 : dynamics model  (encoder frozen, trajectory sequences)
Phase 4 : joint fine-tune (all components end-to-end)
Phase 5 : cut selection   (encoder frozen, cut imitation from SCIP)
"""

import torch
import torch.nn as nn
from .encoder import BipartiteGNN
from .heads import PolicyHead, ValueHead, IntegralityHead, CuttingPlaneHead
from .dynamics import DynamicsTransformer


class BnBWorldModel(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 128,
        n_gnn_layers: int = 3,
        n_gnn_heads: int = 4,
        n_dyn_layers: int = 4,
        n_dyn_heads: int = 4,
        max_seq: int = 512,
        cut_feat_dim: int = 6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder        = BipartiteGNN(
            hidden_dim=hidden_dim, n_layers=n_gnn_layers, n_heads=n_gnn_heads,
        )
        self.policy         = PolicyHead(hidden_dim)
        self.value          = ValueHead(hidden_dim)
        self.integrality    = IntegralityHead(hidden_dim)
        self.cutting_planes = CuttingPlaneHead(hidden_dim, cut_feat_dim)
        self.dynamics       = DynamicsTransformer(
            hidden_dim=hidden_dim, n_layers=n_dyn_layers,
            n_heads=n_dyn_heads, max_seq=max_seq,
        )

    # ------------------------------------------------------------------
    # Primary forward (Phase 1 training)
    # ------------------------------------------------------------------
    def forward(self, batch):
        """
        Args:
            batch : PyG Batch — x, edge_index, node_type, batch, edge_attr (opt)
        Returns:
            scores : [total_vars]   policy logits
            z      : [batch_size, H]
        """
        edge_attr = getattr(batch, "edge_attr", None)
        h_vars, z = self.encoder(
            batch.x, batch.edge_index, batch.node_type, batch.batch,
            edge_attr=edge_attr,
        )
        var_mask  = batch.node_type == 0
        z_per_var = z[batch.batch[var_mask]]
        scores    = self.policy(h_vars, z_per_var)
        return scores, z

    # ------------------------------------------------------------------
    # Encode only
    # ------------------------------------------------------------------
    def encode(self, batch):
        """Returns (h_vars [total_vars, H], z [batch_size, H])."""
        edge_attr = getattr(batch, "edge_attr", None)
        return self.encoder(
            batch.x, batch.edge_index, batch.node_type, batch.batch,
            edge_attr=edge_attr,
        )

    # ------------------------------------------------------------------
    # Individual head helpers
    # ------------------------------------------------------------------
    def policy_scores(
        self,
        h_vars: torch.Tensor,
        z: torch.Tensor,
        var_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Score variable nodes for branching.

        Args:
            h_vars    : [total_vars, H]
            z         : [batch_size, H]
            var_batch : [total_vars]  batch index per variable node
        """
        return self.policy(h_vars, z[var_batch])

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

    def cut_scores(
        self,
        cut_feats: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidate cuts for branch-and-cut selection.

        Args:
            cut_feats : [n_cuts, cut_feat_dim]  per-cut features
            z         : [H]  graph embedding for current node (single graph)
        Returns:
            scores    : [n_cuts]
        """
        return self.cutting_planes(cut_feats, z)

    # ------------------------------------------------------------------
    # Dynamics helpers
    # ------------------------------------------------------------------
    def dynamics_forward(
        self,
        z_seq: torch.Tensor,
        a_seq: torch.Tensor,
    ) -> torch.Tensor:
        """Parallel training forward over full trajectories.

        Args:
            z_seq : [B, T, H]  encoder embeddings along trajectory
            a_seq : [B, T, H]  action embeddings along trajectory
        Returns:
            z_pred : [B, T, H]  predicted next embeddings (z_{t+1})
        """
        return self.dynamics(z_seq, a_seq)

    def dynamics_step(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        past_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step inference with token buffer (replaces GRU step).

        Args:
            z_t         : [B, H]
            a_t         : [B, H]
            past_tokens : [B, t, H] or None
        Returns:
            z_next      : [B, H]
            past_tokens : [B, t+1, H]
        """
        return self.dynamics.step(z_t, a_t, past_tokens)
