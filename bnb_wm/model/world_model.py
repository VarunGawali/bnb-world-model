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
        dyn_residual: bool = True,
        dyn_heteroscedastic: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder = BipartiteGNN(
            hidden_dim=hidden_dim, n_layers=n_gnn_layers, n_heads=n_gnn_heads,
        )
        self.policy = PolicyHead(hidden_dim)
        self.value = ValueHead(hidden_dim)
        self.subtree_size = SubtreeSizeHead(hidden_dim)
        self.cost_to_go = CostToGoHead(hidden_dim)
        self.integrality = IntegralityHead(hidden_dim)
        self.cutting_planes = CuttingPlaneHead(hidden_dim, cut_feat_dim)

        # P0.8: persisted flag — set True only when Phase 5 (cut selection) has
        # trained this head. Saved/restored with the state_dict so the solver can
        # refuse `cut_mode=learned` on a model whose cut head is untrained.
        self.register_buffer("cut_head_trained", torch.tensor(False))

        self.dynamics = DynamicsTransformer(
            hidden_dim=hidden_dim, n_layers=n_dyn_layers,
            n_heads=n_dyn_heads, max_seq=max_seq,
            residual=dyn_residual, heteroscedastic=dyn_heteroscedastic,
        )

        # Grounding head (Gap 2): predicts the next node's normalised dual bound
        # from the predicted latent.
        self.dyn_bound = nn.Linear(hidden_dim, 1)

        # Reward head (Fix 3): predicts per-step reward from the predicted latent.
        self.dyn_reward = nn.Linear(hidden_dim, 1)


    # ------------------------------------------------------------------
    # Primary forward (Phase 1 training)
    # ------------------------------------------------------------------
    def forward(self, batch):
        edge_attr = getattr(batch, "edge_attr", None)
        h_vars, z = self.encoder(
            batch.x, batch.edge_index, batch.node_type, batch.batch,
            edge_attr=edge_attr,
        )
        var_mask = batch.node_type == 0
        z_per_var = z[batch.batch[var_mask]]
        scores = self.policy(h_vars, z_per_var)
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
        """Score variable nodes for branching."""
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
        """Score candidate cuts for branch-and-cut selection."""
        return self.cutting_planes(cut_feats, z)

    # ------------------------------------------------------------------
    # Dynamics helpers
    # ------------------------------------------------------------------
    def dynamics_forward(
        self,
        z_seq: torch.Tensor,
        a_seq: torch.Tensor,
        d_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Parallel training forward over full trajectories."""
        return self.dynamics(z_seq, a_seq, d_seq)

    def dynamics_step(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        past_tokens: torch.Tensor | None = None,
        d_t: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step inference with token buffer."""
        return self.dynamics.step(z_t, a_t, past_tokens, d_t)

    def dynamics_bound_pred(self, z: torch.Tensor) -> torch.Tensor:
        """Predict the normalised dual bound from a predicted latent."""
        return self.dyn_bound(z).squeeze(-1)

    def dynamics_reward_pred(self, z: torch.Tensor) -> torch.Tensor:
        """Predict the per-step reward from a predicted latent."""
        return self.dyn_reward(z).squeeze(-1)

    def dynamics_step_full(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        h_vars_t: torch.Tensor,
        past_tokens: torch.Tensor | None = None,
        d_t: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step latent transition that also predicts next h_vars."""
        return self.dynamics.step_full(
            z_t, a_t, h_vars_t, past_tokens, d_t
        )

    def dynamics_step_full_batched(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        h_vars_t: torch.Tensor,
        past_tokens: torch.Tensor | None = None,
        d_t: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched latent transition for rollout frontiers.

        Args:
            z_t: [B, H]
            a_t: [B, H]
            h_vars_t: [B, V, H] (or [V, H] when B == 1)
            past_tokens: [B, T, H] or None
            d_t: [B], scalar, or None

        Returns:
            z_next: [B, H]
            h_vars_next: [B, V, H]
            new_tokens: [B, T+1, H]
        """
        return self.dynamics.step_full_batched(
            z_t, a_t, h_vars_t, past_tokens, d_t
        )

    # ------------------------------------------------------------------
    # Batched rollout utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _expand_frontier_tensor(x: torch.Tensor, n: int) -> torch.Tensor:
        """Repeat a frontier tensor along its batch dimension without copying."""
        if n == 1:
            return x
        return x.unsqueeze(1).expand(-1, n, *x.shape[1:]).reshape(
            x.shape[0] * n, *x.shape[1:]
        )

    def _policy_topk_batched(
        self,
        h_vars: torch.Tensor,
        z: torch.Tensor,
        masks: torch.Tensor | None,
        k: int,
    ) -> torch.Tensor:
        """Select top-k variable indices independently for each frontier node.

        Args:
            h_vars: [B, V, H]
            z: [B, H]
            masks: [B, V] bool or None
            k: requested number of actions

        Returns:
            indices: [B, k_eff]

        Notes:
            Every frontier element must have at least one valid candidate.
            Callers should filter terminal/no-candidate nodes before invoking.
        """
        B, V, H = h_vars.shape
        z_per_var = z.unsqueeze(1).expand(-1, V, -1)
        scores = self.policy(
            h_vars.reshape(B * V, H),
            z_per_var.reshape(B * V, H),
        ).reshape(B, V)

        if masks is not None:
            scores = scores.masked_fill(~masks, float("-inf"))
            valid_counts = masks.sum(dim=1)
            k_eff = min(k, int(valid_counts.min().item()))
        else:
            k_eff = min(k, V)

        if k_eff <= 0:
            raise RuntimeError("No valid branching candidates in rollout frontier.")

        return scores.topk(k_eff, dim=1).indices

    def rollout_candidate_batched(
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
        use_reward_return: bool = False,
        expand_both_children: bool = True,
    ) -> torch.Tensor:
        """Batched level-wise version of rollout_candidate.

        The candidate being evaluated remains fixed across the batch dimension
        at the root. After the root, all child continuations are represented as
        a frontier tensor and expanded together at each depth.

        This removes recursive Python calls from the expensive dynamics/policy
        portion of the rollout while preserving:
          * both-child (+1/-1) expansion;
          * shrinking candidate masks;
          * branch-factor top-k policy selection;
          * value / reward-return / CTG scoring;
          * immediate-child subtree-size penalty;
          * per-frontier Transformer token histories.

        The return is a scalar tensor. Existing callers expecting a Python float
        can call `.item()` once at the outermost boundary.
        """
        if z.dim() != 2 or z.size(0) != 1:
            raise ValueError("z must have shape [1, H].")
        if h_vars.dim() != 2:
            raise ValueError("h_vars must have shape [V, H].")
        if not (0 <= cand_idx < h_vars.size(0)):
            raise IndexError("cand_idx is outside the variable range.")
        if depth < 1:
            raise ValueError("depth must be >= 1.")

        device = z.device
        V = h_vars.size(0)
        b = max(1, int(branch_factor))
        directions = (1.0, -1.0) if expand_both_children else (0.0,)
        n_dirs = len(directions)

        # Candidate mask at the root.
        if valid_mask is None:
            root_mask = None
        else:
            if valid_mask.dim() != 1 or valid_mask.numel() != V:
                raise ValueError("valid_mask must have shape [V].")
            root_mask = valid_mask.to(device=device, dtype=torch.bool).clone()
            if not bool(root_mask[cand_idx]):
                raise ValueError("cand_idx must be valid under valid_mask.")

        bvec = torch.zeros(V, dtype=torch.long, device=device)

        # Root transition: one tensorized dynamics call for both directions.
        z_root = z.expand(n_dirs, -1)
        a_root = h_vars[cand_idx].unsqueeze(0).expand(n_dirs, -1)
        h_root = h_vars.unsqueeze(0).expand(n_dirs, -1, -1)

        if root_mask is None:
            child_masks = None
            fm_root = None
        else:
            child_mask = root_mask.clone()
            child_mask[cand_idx] = False
            child_masks = child_mask.unsqueeze(0).expand(n_dirs, -1).clone()
            fm_root = child_mask if bool(child_mask.any()) else None

        d_root = torch.tensor(directions, dtype=z.dtype, device=device)
        z_front, h_front, tok_front = self.dynamics_step_full_batched(
            z_root, a_root, h_root, past_tokens, d_root
        )

        # Flatten [F,V,H] -> [F*V,H] and build proper per-graph batch index.
        F_root = z_front.size(0)
        h_front_flat = h_front.reshape(F_root * V, -1)
        bvec_root = torch.arange(F_root, device=device).repeat_interleave(V)
        fm_root_flat = (
            fm_root.unsqueeze(0).expand(F_root, -1).reshape(-1)
            if fm_root is not None else None
        )

        # Accumulate root-child scores. Keep everything tensor-valued until
        # the final scalar conversion so no .item() calls force GPU sync.
        g = 1.0
        score_front = []

        if use_reward_return:
            score_front.append(
                g * self.dynamics_reward_pred(z_front)
            )
        else:
            score_front.append(
                g * self.value_pred(
                    z_front, h_front_flat, bvec_root, frac_mask=fm_root_flat
                )
            )

        if ctg_weight != 0.0:
            score_front[-1] = score_front[-1] - (
                ctg_weight * g *
                self.cost_to_go_pred(
                    z_front, h_front_flat, bvec_root, frac_mask=fm_root_flat
                )
            )

        total_score = torch.stack(score_front).sum()

        if size_weight != 0.0:
            size_root = self.subtree_size_pred(
                z_front, h_front_flat, bvec_root, frac_mask=fm_root_flat
            )
            total_score = total_score - size_weight * size_root.sum()

        # At depth 1, there are no continuations.
        if depth == 1:
            if use_reward_return:
                leaf_value = self.value_pred(
                    z_front, h_front_flat, bvec_root, frac_mask=fm_root_flat
                )
                # frontier_weights are all 1.0 here (root directions, no K averaging)
                total_score = total_score + g * leaf_value.sum()
            return total_score

        # Frontier state:
        # z_front       [F,H]
        # h_front       [F,V,H]
        # tok_front     [F,T,H]
        # child_masks   [F,V] or None
        #
        # Each root direction is an independent frontier element.
        frontier_z = z_front
        frontier_h = h_front
        frontier_tok = tok_front
        frontier_masks = child_masks

        # frontier_weights [F]: cumulative averaging weight for each element.
        # Root directions are summed (weight=1); each K-expansion divides by K
        # to replicate the recursive `sum(cont) / len(cont)` averaging. This
        # ensures the batched version is numerically identical to the recursive
        # one for all branch_factor values, not just branch_factor=1.
        frontier_weights = torch.ones(F_root, dtype=z.dtype, device=device)

        # Continuation discount starts at 1.0; multiplied by gamma at the top
        # of each loop iteration so level l contributes gamma^l.
        continuation_discount = 1.0

        for level in range(1, depth):
            continuation_discount *= gamma

            F = frontier_z.size(0)

            # Determine which frontier elements can actually expand.
            if frontier_masks is None:
                expandable = torch.ones(F, dtype=torch.bool, device=device)
            else:
                expandable = frontier_masks.any(dim=1)

            # Terminal frontiers contribute a reward-return bootstrap only.
            terminal = ~expandable

            if bool(terminal.any()):
                if use_reward_return:
                    z_term = frontier_z[terminal]
                    h_term = frontier_h[terminal]
                    masks_term = (
                        frontier_masks[terminal]
                        if frontier_masks is not None else None
                    )
                    N_term = z_term.size(0)
                    h_term_flat = h_term.reshape(N_term * V, -1)
                    bvec_term = torch.arange(N_term, device=device).repeat_interleave(V)
                    fm_term_flat = masks_term.reshape(-1) if masks_term is not None else None
                    terminal_score = self.value_pred(
                        z_term, h_term_flat, bvec_term, frac_mask=fm_term_flat,
                    )
                    w_term = frontier_weights[terminal]
                    total_score = total_score + (
                        continuation_discount * (w_term * terminal_score).sum()
                    )

            if not bool(expandable.any()):
                break

            # Filter to expandable frontier before top-k selection.
            exp_idx = expandable.nonzero(as_tuple=False).squeeze(1)
            z_exp = frontier_z[exp_idx]
            h_exp = frontier_h[exp_idx]
            tok_exp = frontier_tok[exp_idx]
            masks_exp = (
                frontier_masks[exp_idx]
                if frontier_masks is not None else None
            )
            w_exp = frontier_weights[exp_idx]  # [E]

            k = b
            next_idx = self._policy_topk_batched(
                h_exp, z_exp, masks_exp, k
            )  # [E,k_eff]

            E, K = next_idx.shape

            # Gather the selected action embeddings.
            H = h_exp.size(-1)
            h_expanded = h_exp.unsqueeze(1).expand(-1, K, -1, -1)
            gather_idx = next_idx.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, 1, H
            )
            a_exp = torch.gather(
                h_expanded, 2, gather_idx
            ).squeeze(2)  # [E,K,H]

            # Each selected action generates both child directions.
            z_parent = z_exp.unsqueeze(1).expand(-1, K, -1).reshape(
                E * K, H
            )
            a_flat = a_exp.reshape(E * K, H)
            h_parent = h_exp.unsqueeze(1).expand(
                -1, K, -1, -1
            ).reshape(E * K, V, H)

            tok_parent = tok_exp.unsqueeze(1).expand(
                -1, K, -1, -1
            ).reshape(E * K, tok_exp.size(1), H)

            if masks_exp is not None:
                masks_parent = masks_exp.unsqueeze(1).expand(
                    -1, K, -1
                ).reshape(E * K, V).clone()
                row = torch.arange(E * K, device=device)
                chosen_flat = next_idx.reshape(-1)
                masks_parent[row, chosen_flat] = False
            else:
                masks_parent = None

            if masks_parent is not None:
                fm_parent = masks_parent
            else:
                fm_parent = None

            # Expand directions without a Python loop over child nodes.
            z_child_in = z_parent.unsqueeze(1).expand(
                -1, n_dirs, -1
            ).reshape(E * K * n_dirs, H)
            a_child_in = a_flat.unsqueeze(1).expand(
                -1, n_dirs, -1
            ).reshape(E * K * n_dirs, H)
            h_child_in = h_parent.unsqueeze(1).expand(
                -1, n_dirs, -1, -1
            ).reshape(E * K * n_dirs, V, H)
            tok_child_in = tok_parent.unsqueeze(1).expand(
                -1, n_dirs, -1, -1
            ).reshape(E * K * n_dirs, tok_parent.size(1), H)

            d = torch.tensor(
                directions, dtype=z.dtype, device=device
            ).view(1, n_dirs).expand(E * K, -1).reshape(-1)

            if fm_parent is not None:
                fm_child = fm_parent.unsqueeze(1).expand(
                    -1, n_dirs, -1
                ).reshape(E * K * n_dirs, V)
            else:
                fm_child = None

            z_next, h_next, tok_next = self.dynamics_step_full_batched(
                z_child_in, a_child_in, h_child_in, tok_child_in, d
            )

            N_step = z_next.size(0)
            h_next_flat = h_next.reshape(N_step * V, -1)
            bvec_step = torch.arange(N_step, device=device).repeat_interleave(V)
            fm_step_flat = fm_child.reshape(-1) if fm_child is not None else None

            if use_reward_return:
                step_score = self.dynamics_reward_pred(z_next)
            else:
                step_score = self.value_pred(
                    z_next, h_next_flat, bvec_step, frac_mask=fm_step_flat,
                )

            if ctg_weight != 0.0:
                ctg = self.cost_to_go_pred(
                    z_next, h_next_flat, bvec_step, frac_mask=fm_step_flat,
                )
                step_score = step_score - ctg_weight * ctg

            # Weights for step children: divide by K (averaging over K actions)
            # and replicate for n_dirs (directions are summed, not averaged).
            # Shape: [E*K*n_dirs], ordering (i, k, d).
            w_step = (w_exp / K).unsqueeze(1).expand(
                -1, K * n_dirs
            ).reshape(E * K * n_dirs)

            total_score = total_score + continuation_discount * (
                w_step * step_score
            ).sum()

            # The frontier after this level consists of all direction-expanded
            # children, carrying their cumulative averaging weights.
            frontier_z = z_next
            frontier_h = h_next
            frontier_tok = tok_next
            frontier_masks = fm_child
            frontier_weights = w_step

        # For reward-return mode, the final frontier gets a single value
        # bootstrap, matching sum(rewards) + gamma^k V(leaf).
        if use_reward_return and frontier_z.size(0) > 0:
            N_leaf = frontier_z.size(0)
            h_leaf_flat = frontier_h.reshape(N_leaf * V, -1)
            bvec_leaf = torch.arange(N_leaf, device=device).repeat_interleave(V)
            fm_leaf_flat = frontier_masks.reshape(-1) if frontier_masks is not None else None
            leaf_value = self.value_pred(
                frontier_z, h_leaf_flat, bvec_leaf, frac_mask=fm_leaf_flat,
            )
            total_score = total_score + continuation_discount * (
                frontier_weights * leaf_value
            ).sum()

        return total_score

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
        use_reward_return: bool = False,
        expand_both_children: bool = True,
        batched: bool = True,
    ) -> float:
        """Estimate candidate quality using latent rollout.

        By default this dispatches to the level-wise batched implementation.
        Set batched=False to retain the original recursive implementation for
        exact A/B equivalence testing.
        """
        if batched:
            return float(
                self.rollout_candidate_batched(
                    z=z,
                    h_vars=h_vars,
                    cand_idx=cand_idx,
                    depth=depth,
                    gamma=gamma,
                    valid_mask=valid_mask,
                    past_tokens=past_tokens,
                    size_weight=size_weight,
                    ctg_weight=ctg_weight,
                    branch_factor=branch_factor,
                    use_reward_return=use_reward_return,
                    expand_both_children=expand_both_children,
                ).detach().cpu().item()
            )

        # ------------------------------------------------------------------
        # Reference recursive implementation retained for equivalence testing.
        # ------------------------------------------------------------------
        bvec = torch.zeros(
            h_vars.size(0), dtype=torch.long, device=z.device
        )
        b = max(1, branch_factor)
        directions = (1.0, -1.0) if expand_both_children else (0.0,)
        size_estimate = [0.0]
        neg_inf = float("-inf")

        def branch(z_cur, h_cur, tokens_cur, a_idx, depth_left, g, is_root, mask):
            a_emb = h_cur[a_idx].unsqueeze(0)

            if mask is not None:
                child_mask = mask.clone()
                child_mask[a_idx] = False
                fm = child_mask if bool(child_mask.any()) else None
            else:
                child_mask, fm = None, None

            subtree = 0.0
            for direction in directions:
                z_n, h_n, tok = self.dynamics.step_full(
                    z_cur, a_emb, h_cur, tokens_cur, direction
                )

                if use_reward_return:
                    child_score = g * self.dynamics_reward_pred(z_n).item()
                else:
                    child_score = g * self.value(
                        z_n, h_n, bvec, frac_mask=fm
                    ).item()

                if ctg_weight != 0.0:
                    ctg = self.cost_to_go(
                        z_n, h_n, bvec, frac_mask=fm
                    ).item()
                    child_score -= ctg_weight * g * ctg

                if is_root and size_weight != 0.0:
                    size_estimate[0] += self.subtree_size(
                        z_n, h_n, bvec, frac_mask=fm
                    ).item()

                can_expand = (
                    depth_left > 1
                    and (child_mask is None or bool(child_mask.any()))
                )

                if not can_expand:
                    if use_reward_return:
                        child_score += g * self.value(
                            z_n, h_n, bvec, frac_mask=fm
                        ).item()
                else:
                    scores = self.policy(
                        h_n, z_n.expand(h_n.size(0), -1)
                    )
                    if child_mask is not None:
                        scores = scores.masked_fill(~child_mask, neg_inf)
                        k = min(
                            b, int(child_mask.sum().item())
                        )
                    else:
                        k = min(b, scores.size(0))

                    next_actions = scores.topk(k).indices
                    cont = [
                        branch(
                            z_n, h_n, tok, int(na),
                            depth_left - 1, g * gamma,
                            False, child_mask
                        )
                        for na in next_actions
                    ]
                    child_score += sum(cont) / len(cont)

                subtree += child_score

            return subtree

        init_mask = (
            valid_mask.clone() if valid_mask is not None else None
        )
        total = branch(
            z, h_vars, past_tokens, cand_idx, depth,
            1.0, True, init_mask
        )
        return total - size_weight * size_estimate[0]

