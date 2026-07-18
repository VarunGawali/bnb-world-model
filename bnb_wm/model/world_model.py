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
        # Grounding head (Gap 2): predicts the next node's normalised dual bound
        # from the predicted latent, so the dynamics is anchored to a real
        # solver quantity instead of drifting as a free self-supervised latent.
        self.dyn_bound      = nn.Linear(hidden_dim, 1)

        # Global search-state context (Gap 1): projects scalar frontier/bound
        # features and adds them to the node embedding z, so heads can see the
        # global search state (open-node count, bounds, gap) not just the local
        # node. Zero-initialised, so until fine-tuned it is an exact no-op and
        # cannot degrade a model trained without it.
        self.n_global       = 6
        self.global_proj    = nn.Linear(self.n_global, hidden_dim)
        nn.init.zeros_(self.global_proj.weight)
        nn.init.zeros_(self.global_proj.bias)

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

    def add_global_context(
        self,
        z: torch.Tensor,
        global_ctx: torch.Tensor | None,
    ) -> torch.Tensor:
        """Add the projected global search-state context to z (Gap 1).

        Args:
            z          : [batch, H]
            global_ctx : [batch, n_global] scalar features, or None (no-op)
        """
        if global_ctx is None:
            return z
        return z + self.global_proj(global_ctx)

    def dynamics_bound_pred(self, z: torch.Tensor) -> torch.Tensor:
        """Predict the normalised dual bound from a (predicted) latent (Gap 2).

        Accepts z of shape [..., H]; returns [...] (last dim squeezed).
        """
        return self.dyn_bound(z).squeeze(-1)

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
        branch_factor: int = 1,
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
            branch_factor : int    number of next actions expanded at each
                                   rollout step (Gap 4). 1 = single greedy path
                                   (the original behaviour); >1 expands a
                                   predicted branching tree and averages child
                                   continuations, a richer subtree estimate.

        Returns:
            score : float   higher is better (branch on the max-score candidate)
        """
        # Per-variable batch vector (single graph): every variable maps to graph 0.
        bvec = torch.zeros(h_vars.size(0), dtype=torch.long, device=z.device)
        b = max(1, branch_factor)
        size_estimate = [0.0]   # captured from the candidate's immediate child

        def expand(z_cur, h_cur, tokens_cur, a_idx, depth_left, g, is_root):
            # Apply the action -> predicted next state (z and per-variable h).
            a_emb = h_cur[a_idx].unsqueeze(0)                    # [1, H]
            z_n, h_n, tok = self.dynamics.step_full(
                z_cur, a_emb, h_cur, tokens_cur
            )
            v = self.value(z_n, h_n, bvec, frac_mask=valid_mask).item()
            node_score = g * v
            if ctg_weight != 0.0:
                ctg = self.cost_to_go(z_n, h_n, bvec, frac_mask=valid_mask).item()
                node_score -= ctg_weight * g * ctg
            if is_root and size_weight != 0.0:
                size_estimate[0] = self.subtree_size(
                    z_n, h_n, bvec, frac_mask=valid_mask
                ).item()

            if depth_left <= 1:
                return node_score

            # Expand the top-b next actions on the PREDICTED state and average
            # their continuations (b=1 recovers the single greedy path).
            scores = self.policy(h_n, z_n.expand(h_n.size(0), -1))
            if valid_mask is not None:
                masked = torch.full_like(scores, -1e4)
                masked[valid_mask] = scores[valid_mask]
            else:
                masked = scores
            k = min(b, masked.size(0))
            next_actions = masked.topk(k).indices
            child = [
                expand(z_n, h_n, tok, int(na), depth_left - 1, g * gamma, False)
                for na in next_actions
            ]
            return node_score + sum(child) / len(child)

        total = expand(z, h_vars, past_tokens, cand_idx, depth, 1.0, True)
        return total - size_weight * size_estimate[0]
