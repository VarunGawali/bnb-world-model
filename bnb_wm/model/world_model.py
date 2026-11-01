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
        gnn_use_checkpoint: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder = BipartiteGNN(
            hidden_dim=hidden_dim, n_layers=n_gnn_layers, n_heads=n_gnn_heads,
            use_checkpoint=gnn_use_checkpoint,
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

        # Global search-state context (Gap 1).
        self.n_global = 6
        self.global_proj = nn.Linear(self.n_global, hidden_dim)
        nn.init.zeros_(self.global_proj.weight)
        nn.init.zeros_(self.global_proj.bias)

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

    def add_global_context(
        self,
        z: torch.Tensor,
        global_ctx: torch.Tensor | None,
    ) -> torch.Tensor:
        """Add projected global search-state context to z."""
        if global_ctx is None:
            return z
        return z + self.global_proj(global_ctx)

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
        """Batched level-wise version of rollout_candidate (Eq. 17).

        Scores branching on `cand_idx` by expanding an imagined tree K = `depth`
        transitions deep. Transition j (1-indexed) produces both children
        (+1/-1) of every frontier node; from depth 2 on, each node's next
        action is each of the policy's top-b variables, and the b continuations
        are AVERAGED (an expectation over the policy), while the two children
        of one branching are SUMMED (both subtrees must be searched):

            Q(s, a, j) = sum_dir [ gamma^(j-1) * (r_hat - w_C * C_hat)(s'_dir)
                                   + cont(s'_dir, j) ]
            cont(s', j) = mean_{a' in topb(s')} Q(s', a', j + 1)   if j < K
                        = gamma^j * V_hat(s')                       otherwise

        so the leaf is bootstrapped with gamma^K (gamma^j for a node that runs
        out of candidates at depth j). With use_reward_return=False the value
        replaces the reward at every step and there is no leaf bootstrap.

        Each frontier node carries an expectation weight (the product of 1/b
        factors along its path); every term is multiplied by it, which makes
        this numerically identical to the recursive reference implementation
        while expanding a whole level in one batched dynamics call.

        Returns a scalar tensor.
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
        V, H = h_vars.shape
        b = max(1, int(branch_factor))
        directions = (1.0, -1.0) if expand_both_children else (0.0,)
        n_dirs = len(directions)
        dir_vec = torch.tensor(directions, dtype=z.dtype, device=device)

        if valid_mask is None:
            mask_f = None
        else:
            if valid_mask.dim() != 1 or valid_mask.numel() != V:
                raise ValueError("valid_mask must have shape [V].")
            root_mask = valid_mask.to(device=device, dtype=torch.bool)
            if not bool(root_mask[cand_idx]):
                raise ValueError("cand_idx must be valid under valid_mask.")
            child_mask = root_mask.clone()
            child_mask[cand_idx] = False
            mask_f = child_mask.unsqueeze(0).expand(n_dirs, -1).clone()

        # Root transition: both children of the candidate in one call.
        # past_tokens from the inference loop is [1,T,H]; expand to the batch.
        z_f, h_f, tok_f = self.dynamics_step_full_batched(
            z.expand(n_dirs, -1),
            h_vars[cand_idx].unsqueeze(0).expand(n_dirs, -1),
            h_vars.unsqueeze(0).expand(n_dirs, -1, -1),
            past_tokens.expand(n_dirs, -1, -1) if past_tokens is not None else None,
            dir_vec,
        )
        weight_f = torch.ones(n_dirs, dtype=z.dtype, device=device)
        total = z.new_zeros(())

        if size_weight != 0.0:
            h_flat, bvec, fm = self._flatten_frontier(h_f, mask_f)
            total = total - size_weight * self.subtree_size_pred(
                z_f, h_flat, bvec, frac_mask=fm).sum()

        for level in range(1, depth + 1):
            h_flat, bvec, fm = self._flatten_frontier(h_f, mask_f)

            # Per-step score of the nodes produced by transition `level`.
            if use_reward_return:
                step = self.dynamics_reward_pred(z_f)
            else:
                step = self.value_pred(z_f, h_flat, bvec, frac_mask=fm)
            if ctg_weight != 0.0:
                step = step - ctg_weight * self.cost_to_go_pred(
                    z_f, h_flat, bvec, frac_mask=fm)
            total = total + gamma ** (level - 1) * (weight_f * step).sum()

            if level == depth:
                expandable = torch.zeros(z_f.size(0), dtype=torch.bool,
                                         device=device)
            elif mask_f is None:
                expandable = torch.ones(z_f.size(0), dtype=torch.bool,
                                        device=device)
            else:
                expandable = mask_f.any(dim=1)

            # Leaf bootstrap (reward-return mode): gamma^level * V(s').
            terminal = ~expandable
            if use_reward_return and bool(terminal.any()):
                v = self.value_pred(z_f, h_flat, bvec, frac_mask=fm)
                total = total + gamma ** level * (
                    weight_f[terminal] * v[terminal]).sum()

            if not bool(expandable.any()):
                break

            idx = expandable.nonzero(as_tuple=False).squeeze(1)
            z_e, h_e, tok_e, w_e = z_f[idx], h_f[idx], tok_f[idx], weight_f[idx]
            m_e = mask_f[idx] if mask_f is not None else None

            nxt = self._policy_topk_batched(h_e, z_e, m_e, b)        # [E,K]
            E, K = nxt.shape
            N = E * K * n_dirs
            rows = torch.arange(E, device=device).unsqueeze(1)
            a = h_e[rows, nxt]                                       # [E,K,H]

            def rep(x):
                # [E, ...] -> [E*K*n_dirs, ...], parent repeated per child.
                return x.view(E, 1, 1, *x.shape[1:]).expand(
                    E, K, n_dirs, *x.shape[1:]).reshape(N, *x.shape[1:])

            if m_e is not None:
                m_c = m_e.unsqueeze(1).expand(-1, K, -1).clone()     # [E,K,V]
                m_c.scatter_(2, nxt.unsqueeze(-1), False)
                mask_f = m_c.unsqueeze(2).expand(-1, -1, n_dirs, -1).reshape(N, V)

            a_in = a.unsqueeze(2).expand(E, K, n_dirs, H).reshape(N, H)
            d_in = dir_vec.view(1, 1, n_dirs).expand(E, K, n_dirs).reshape(N)
            weight_f = (w_e / K).view(E, 1, 1).expand(E, K, n_dirs).reshape(N)
            z_f, h_f, tok_f = self.dynamics_step_full_batched(
                rep(z_e), a_in, rep(h_e), rep(tok_e), d_in)

        return total

    @staticmethod
    def _flatten_frontier(h_f: torch.Tensor, mask_f: torch.Tensor | None):
        """[F,V,H] frontier -> (h_flat [F*V,H], batch index, frac mask or None)."""
        F_, V, _ = h_f.shape
        bvec = torch.arange(F_, device=h_f.device).repeat_interleave(V)
        fm = mask_f.reshape(-1) if mask_f is not None else None
        return h_f.reshape(F_ * V, -1), bvec, fm

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
        exact A/B equivalence testing. Both compute Eq. 17 as documented on
        rollout_candidate_batched (tests/test_rollout.py checks they agree).
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
                    # Leaf bootstrap one transition past this step's reward:
                    # gamma^j * V at depth j (gamma^K at the horizon).
                    if use_reward_return:
                        child_score += g * gamma * self.value(
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
