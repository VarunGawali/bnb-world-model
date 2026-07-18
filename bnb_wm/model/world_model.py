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
from .heads import (
    PolicyHead, ValueHead, IntegralityHead, CuttingPlaneHead, SubtreeSizeHead,
    CostToGoHead,
)
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
        self.subtree_size   = SubtreeSizeHead(hidden_dim)
        self.cost_to_go     = CostToGoHead(hidden_dim)
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

    def subtree_size_pred(
        self,
        z: torch.Tensor,
        h_vars: torch.Tensor,
        batch_vec: torch.Tensor,
        frac_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict log1p(subtree node count) rooted at the current node."""
        return self.subtree_size(z, h_vars, batch_vec, frac_mask)

    def cost_to_go_pred(
        self,
        z: torch.Tensor,
        h_vars: torch.Tensor,
        batch_vec: torch.Tensor,
        frac_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict log1p(remaining B&B nodes) — the cost-to-go value."""
        return self.cost_to_go(z, h_vars, batch_vec, frac_mask)

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

    def dynamics_step_full(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        h_vars_t: torch.Tensor,
        past_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step latent transition that also predicts next h_vars.

        Returns (z_next [B,H], h_vars_next [V,H], past_tokens [B,t+1,H]).
        """
        return self.dynamics.step_full(z_t, a_t, h_vars_t, past_tokens)

    # ------------------------------------------------------------------
    # Real latent rollout for candidate selection
    # ------------------------------------------------------------------
    def rollout_candidate(
        self,
        z: torch.Tensor,
        h_vars: torch.Tensor,
        cand_idx: int,
        depth: int,
        gamma: float,
        valid_mask: torch.Tensor | None = None,
        past_tokens: torch.Tensor | None = None,
        size_weight: float = 1.0,
        ctg_weight: float = 0.0,
    ) -> float:
        """
        Estimate the quality of branching on `cand_idx` by rolling the learned
        dynamics forward `depth` steps in latent space.

        Unlike the earlier heuristic (which reused the same action embedding
        at every step), this performs a genuine rollout:

            1. branch on cand_idx  -> predict z_1, h_vars_1
            2. run the policy on h_vars_1 to pick the *next* branching var
            3. roll forward with that chosen action -> z_2, h_vars_2
            4. repeat; accumulate discounted value estimates

        The score combines two learned signals about the simulated subtree:
            + discounted value        (higher dual bound is better)
            - predicted subtree size  (fewer nodes to close is better)

        Predicting subtree size is the decision-relevant quantity — the
        solver's cost is node count — so branching to minimise predicted tree
        growth directly targets the metric we care about. The subtree-size
        estimate is read at the candidate's immediate predicted child (the
        root of the subtree that branching on this candidate creates).

        Args:
            z           : [1, H]   current graph latent
            h_vars      : [V, H]   current per-variable embeddings
            cand_idx    : int      first action (candidate under evaluation)
            depth       : int      rollout horizon
            gamma       : float    per-step discount
            valid_mask  : [V] bool valid branching candidates (fractional vars)
            past_tokens : token buffer for the dynamics Transformer
            size_weight : float    weight on the predicted-subtree-size penalty
                                   (0 recovers the pure value-based rollout)
            ctg_weight  : float    weight on the predicted cost-to-go (remaining
                                   nodes). Lower cost-to-go is better, so it is
                                   subtracted. This is the decision-relevant
                                   signal; set value contribution and ctg_weight
                                   to taste for the ablation.

        Returns:
            score : float   higher is better (branch on the max-score candidate)
        """
        z_cur      = z
        h_cur      = h_vars
        tokens_cur = past_tokens
        a_idx      = cand_idx
        # Per-variable batch vector (single graph): every variable maps to graph 0.
        bvec = torch.zeros(h_vars.size(0), dtype=torch.long, device=z.device)

        discounted_return = 0.0
        discounted_ctg = 0.0
        size_estimate = 0.0
        g = 1.0
        for step in range(depth):
            a_emb = h_cur[a_idx].unsqueeze(0)                    # [1, H]
            z_cur, h_cur, tokens_cur = self.dynamics.step_full(
                z_cur, a_emb, h_cur, tokens_cur
            )
            # Value of the predicted future state, with reconstructed
            # per-variable fractional context (fixes the frac_mask=None gap).
            v = self.value(
                z_cur, h_cur, bvec,
                frac_mask=valid_mask,
            ).item()
            discounted_return += g * v

            # Predicted cost-to-go (remaining nodes) at the predicted state.
            if ctg_weight != 0.0:
                ctg = self.cost_to_go(
                    z_cur, h_cur, bvec, frac_mask=valid_mask
                ).item()
                discounted_ctg += g * ctg

            g *= gamma

            # Predicted subtree size at the immediate child of this candidate.
            if step == 0 and size_weight != 0.0:
                size_estimate = self.subtree_size(
                    z_cur, h_cur, bvec, frac_mask=valid_mask
                ).item()

            # Pick the next branching action ON THE PREDICTED STATE.
            scores = self.policy(h_cur, z_cur.expand(h_cur.size(0), -1))
            if valid_mask is not None:
                masked = torch.full_like(scores, -1e4)
                masked[valid_mask] = scores[valid_mask]
                a_idx = int(masked.argmax())
            else:
                a_idx = int(scores.argmax())

        return (
            discounted_return
            - size_weight * size_estimate
            - ctg_weight * discounted_ctg
        )
