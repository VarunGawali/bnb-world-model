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

import contextlib
import torch
import torch.nn as nn
from .encoder import BipartiteGNN
from .heads import (
    PolicyHead, ValueHead, IntegralityHead, ZeroShotCutScorer, SubtreeSizeHead,
    CostToGoHead, compute_frac_mean,
)
from .dynamics import DynamicsTransformer, _VarDynamics


def migrate_var_dynamics_checkpoint(ckpt_path: str, out_path: str | None = None) -> dict:
    """Rewrite a legacy _VarDynamics checkpoint to the decomposed-projection format.

    Old format: dynamics.var_dynamics.net.{0,3}.{weight,bias}
    New format: dynamics.var_dynamics.{W_h,W_z,W_a,out}.{weight,bias}

    Args:
        ckpt_path : path to the old .pt checkpoint (dict with key "model")
        out_path  : if given, torch.save() the migrated checkpoint there

    Returns:
        migrated state_dict (model key already updated)
    """
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"]

    # Infer hidden_dim from the norm layer (always present)
    H = sd["dynamics.var_dynamics.norm.weight"].shape[0]
    _, new_sd = _VarDynamics.from_legacy_state(sd, H)
    ckpt["model"] = new_sd

    if out_path is not None:
        torch.save(ckpt, out_path)
        print(f"Migrated checkpoint saved to {out_path}")

    return ckpt


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
        # ZeroShotCutScorer: no parameters, no Phase 5 training required.
        # Scores CG cuts by cosine alignment between their GNN-native embedding
        # (Σ coeff_j·h_vars[j]) and the current node latent z.
        self.cut_scorer = ZeroShotCutScorer(violation_weight=1.0)

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
        h_vars, z, _h_cons = self.encoder(
            batch.x, batch.edge_index, batch.node_type, batch.batch,
            edge_attr=edge_attr,
        )
        var_mask = batch.node_type == 0
        var_batch = batch.batch[var_mask]
        scores = self.policy(h_vars, z, var_batch)
        return scores, z

    # ------------------------------------------------------------------
    # Encode only
    # ------------------------------------------------------------------
    def encode(self, batch):
        """Returns (h_vars [total_vars, H], z [batch_size, H])."""
        edge_attr = getattr(batch, "edge_attr", None)
        h_vars, z, _h_cons = self.encoder(
            batch.x, batch.edge_index, batch.node_type, batch.batch,
            edge_attr=edge_attr,
        )
        return h_vars, z

    def encode_with_cons(self, batch):
        """Returns (h_vars, z, h_cons) — h_cons exposed for cut scoring."""
        edge_attr = getattr(batch, "edge_attr", None)
        return self.encoder(
            batch.x, batch.edge_index, batch.node_type, batch.batch,
            edge_attr=edge_attr,
        )

    def pool_entropy(self, batch):
        """Diagnostic: returns (z, v_eff) where v_eff[b] = exp(H(attn_b)).
        v_eff → 1 means peaked (instance-specific z); v_eff → V means diffuse
        (attention ≈ mean pool, z carries little instance info).
        """
        edge_attr = getattr(batch, "edge_attr", None)
        _h_vars, z, _h_cons, v_eff = self.encoder(
            batch.x, batch.edge_index, batch.node_type, batch.batch,
            edge_attr=edge_attr, return_pool_entropy=True,
        )
        return z, v_eff

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

        z : [B, H]  — graph-level embeddings, one per graph in the batch.
        var_batch : [V] — batch index for each variable node.
        """
        return self.policy(h_vars, z, var_batch)

    def frac_mean(
        self,
        z: torch.Tensor,
        h_vars: torch.Tensor,
        batch_vec: torch.Tensor,
        frac_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute fractional-variable mean embedding once; pass to value/size/ctg heads."""
        return compute_frac_mean(z, h_vars, batch_vec, frac_mask)

    def value_pred(
        self,
        z: torch.Tensor,
        h_vars: torch.Tensor,
        batch_vec: torch.Tensor,
        frac_mask: torch.Tensor | None = None,
        precomputed_frac_mean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict normalised dual bound."""
        return self.value(z, h_vars, batch_vec, frac_mask, precomputed_frac_mean)

    def subtree_size_pred(
        self,
        z: torch.Tensor,
        h_vars: torch.Tensor,
        batch_vec: torch.Tensor,
        frac_mask: torch.Tensor | None = None,
        precomputed_frac_mean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict log1p(subtree node count) rooted at the current node."""
        return self.subtree_size(z, h_vars, batch_vec, frac_mask, precomputed_frac_mean)

    def cost_to_go_pred(
        self,
        z: torch.Tensor,
        h_vars: torch.Tensor,
        batch_vec: torch.Tensor,
        frac_mask: torch.Tensor | None = None,
        precomputed_frac_mean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict log1p(remaining B&B nodes) — the cost-to-go value."""
        return self.cost_to_go(z, h_vars, batch_vec, frac_mask, precomputed_frac_mean)

    def integrality_logit(
        self,
        z: torch.Tensor,
        depth: torch.Tensor | None = None,
        n_frac: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict raw logit for P(next node is leaf)."""
        return self.integrality(z, depth, n_frac)

    def multi_head_pred(
        self,
        z: torch.Tensor,
        h_vars: torch.Tensor,
        batch_vec: torch.Tensor,
        frac_mask: torch.Tensor | None = None,
        value: bool = True,
        subtree_size: bool = False,
        cost_to_go: bool = False,
    ) -> dict:
        """Evaluate multiple heads over z in one pass by sharing frac_mean.

        frac_mean = scatter_mean(h_vars[frac_mask], batch_vec) is computed once
        and reused for every requested head — eliminates redundant scatter ops
        when two or more enriched heads are needed for the same z.

        Args:
            z          : [B, H]
            h_vars     : [B*V, H] (flat)
            batch_vec  : [B*V]
            frac_mask  : [B*V] bool or None
            value      : include ValueHead output
            subtree_size: include SubtreeSizeHead output
            cost_to_go : include CostToGoHead output

        Returns:
            dict with requested keys: 'value', 'subtree_size', 'cost_to_go'
            Each value is a [B] tensor.
        """
        fm = compute_frac_mean(z, h_vars, batch_vec, frac_mask)
        out = {}
        if value:
            out["value"]       = self.value(z, h_vars, batch_vec, frac_mask, fm)
        if subtree_size:
            out["subtree_size"] = self.subtree_size(z, h_vars, batch_vec, frac_mask, fm)
        if cost_to_go:
            out["cost_to_go"]  = self.cost_to_go(z, h_vars, batch_vec, frac_mask, fm)
        return out

    def cut_scores(
        self,
        cut_embeds: torch.Tensor,
        z: torch.Tensor,
        violations: torch.Tensor,
    ) -> torch.Tensor:
        """Score CG cuts by cosine alignment with z + violation bonus.

        Args:
            cut_embeds : [C, H]  GNN-native cut embeddings from cg_cuts.py
            z          : [H] or [1, H]  current node latent
            violations : [C]  LP violation amounts

        Returns:
            scores : [C]  (higher = more desirable cut)
        """
        return self.cut_scorer(cut_embeds, z, violations)

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
        h_cons_summary: torch.Tensor | None = None,
        kv_caches: list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list | None]:
        """Single-step inference with token buffer.

        Returns (z_next, tokens, kv_caches).
        Pass kv_caches from a previous step to use O(1) incremental attention.
        """
        return self.dynamics.step(z_t, a_t, past_tokens, d_t, h_cons_summary, kv_caches)

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
        h_cons_summary: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step latent transition that also predicts next h_vars.

        h_cons_summary: optional [1, H] or [B, H] pooled constraint embedding
            added to the dynamics token (zero-init projection → no-op at init).
        """
        return self.dynamics.step_full(
            z_t, a_t, h_vars_t, past_tokens, d_t, h_cons_summary=h_cons_summary,
        )

    def dynamics_step_full_batched(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        h_vars_t: torch.Tensor,
        past_tokens: torch.Tensor | None = None,
        d_t: torch.Tensor | float | None = None,
        h_cons_summary: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched latent transition for rollout frontiers.

        Args:
            z_t: [B, H]
            a_t: [B, H]
            h_vars_t: [B, V, H] (or [V, H] when B == 1)
            past_tokens: [B, T, H] or None
            d_t: [B], scalar, or None
            h_cons_summary: [B, H] or None — constraint summary injected into token

        Returns:
            z_next: [B, H]
            h_vars_next: [B, V, H]
            new_tokens: [B, T+1, H]
        """
        return self.dynamics.step_full_batched(
            z_t, a_t, h_vars_t, past_tokens, d_t, h_cons_summary=h_cons_summary,
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
        # var_batch: [B*V] — variable i in graph b maps to index b
        var_batch = torch.arange(B, device=z.device).repeat_interleave(V)
        scores = self.policy(
            h_vars.reshape(B * V, H),
            z,           # [B, H] — projected once per graph, not per variable
            var_batch,   # [B*V] index into z
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
        value_weight: float = 0.3,
        size_weight: float = 0.7,
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
        # Expand past_tokens to match the frontier batch size (n_dirs copies).
        past_root = (
            past_tokens.expand(n_dirs, -1, -1)
            if past_tokens is not None else None
        )
        z_front, h_front, tok_front = self.dynamics_step_full_batched(
            z_root, a_root, h_root, past_root, d_root
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
            score_front.append(g * self.dynamics_reward_pred(z_front))
        else:
            # V always; S+C only at the leaf (depth==1) to avoid double-counting.
            sv = g * self.value_pred(
                z_front, h_front_flat, bvec_root, frac_mask=fm_root_flat
            ) if value_weight != 0.0 else z_front.new_zeros(z_front.size(0))
            score_front.append(value_weight * sv if value_weight != 0.0 else sv)
            if depth == 1:
                if size_weight != 0.0:
                    score_front[-1] = score_front[-1] - size_weight * g * self.subtree_size_pred(
                        z_front, h_front_flat, bvec_root, frac_mask=fm_root_flat
                    )
                if ctg_weight != 0.0:
                    score_front[-1] = score_front[-1] - ctg_weight * g * self.cost_to_go_pred(
                        z_front, h_front_flat, bvec_root, frac_mask=fm_root_flat
                    )

        total_score = torch.stack(score_front).sum()

        # At depth 1, there are no continuations.
        if depth == 1:
            if use_reward_return:
                leaf_value = self.value_pred(
                    z_front, h_front_flat, bvec_root, frac_mask=fm_root_flat
                )
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

            # h_cons_summary: broadcast [1,H] to frontier batch size.
            hcs_step = (
                h_cons_summary.expand(z_child_in.size(0), -1)
                if h_cons_summary is not None else None
            )
            z_next, h_next, tok_next = self.dynamics_step_full_batched(
                z_child_in, a_child_in, h_child_in, tok_child_in, d,
                h_cons_summary=hcs_step,
            )

            N_step = z_next.size(0)
            h_next_flat = h_next.reshape(N_step * V, -1)
            bvec_step = torch.arange(N_step, device=device).repeat_interleave(V)
            fm_step_flat = fm_child.reshape(-1) if fm_child is not None else None

            if use_reward_return:
                step_score = self.dynamics_reward_pred(z_next)
            else:
                # Intermediate steps: V only. S+C deferred to leaf frontier.
                step_score = z_next.new_zeros(z_next.size(0))
                if value_weight != 0.0:
                    step_score = step_score + value_weight * self.value_pred(
                        z_next, h_next_flat, bvec_step, frac_mask=fm_step_flat,
                    )

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

        # Final frontier: apply S+C once (leaf scoring, avoids double-counting).
        if frontier_z.size(0) > 0:
            N_leaf = frontier_z.size(0)
            h_leaf_flat = frontier_h.reshape(N_leaf * V, -1)
            bvec_leaf = torch.arange(N_leaf, device=device).repeat_interleave(V)
            fm_leaf_flat = frontier_masks.reshape(-1) if frontier_masks is not None else None
            if use_reward_return:
                leaf_value = self.value_pred(
                    frontier_z, h_leaf_flat, bvec_leaf, frac_mask=fm_leaf_flat,
                )
                total_score = total_score + continuation_discount * (
                    frontier_weights * leaf_value
                ).sum()
            else:
                leaf_score = frontier_z.new_zeros(N_leaf)
                if size_weight != 0.0:
                    leaf_score = leaf_score - size_weight * self.subtree_size_pred(
                        frontier_z, h_leaf_flat, bvec_leaf, frac_mask=fm_leaf_flat,
                    )
                if ctg_weight != 0.0:
                    leaf_score = leaf_score - ctg_weight * self.cost_to_go_pred(
                        frontier_z, h_leaf_flat, bvec_leaf, frac_mask=fm_leaf_flat,
                    )
                total_score = total_score + continuation_discount * (
                    frontier_weights * leaf_score
                ).sum()

        return total_score

    def rollout_top_k_batched(
        self,
        z: torch.Tensor,
        h_vars: torch.Tensor,
        cand_indices: torch.Tensor,
        depth: int,
        gamma: float,
        valid_mask: torch.Tensor | None = None,
        past_tokens: torch.Tensor | None = None,
        value_weight: float = 0.3,
        size_weight: float = 0.7,
        ctg_weight: float = 0.0,
        branch_factor: int = 1,
        use_reward_return: bool = False,
        expand_both_children: bool = True,
        uncertainty_weight: float = 0.0,
        h_cons_summary: torch.Tensor | None = None,
        use_kv_cache: bool = True,
    ) -> torch.Tensor:
        """Evaluate all K root candidates in a single batched rollout pass.

        Equivalent to calling rollout_candidate_batched K times and stacking
        the results, but uses one shared forward pass per depth level instead
        of K separate passes. All K candidates' frontier trees are processed
        together; each candidate's subtree is tracked via a cand_id index so
        scores are scatter-added to the correct per-candidate accumulator.

        Scoring at each imagined state:
            score = value_weight * value_pred(z')
                  - size_weight  * subtree_size_pred(z')   [log1p scale]
                  - ctg_weight   * cost_to_go_pred(z')     [log1p scale]

        SubtreeSizeHead is the primary signal: it directly predicts the tree
        cost rooted at each imagined state. ValueHead is complementary (LP
        quality). CostToGoHead has a within-node-constancy problem when trained
        on expert-only linear-countdown targets — keep ctg_weight low or 0.

        Scale note: value is in [0,1], subtree_size and ctg are in log1p space
        (typically 0–8). The weights already account for this difference.

        Args:
            cand_indices      : LongTensor [K] of root candidate variable indices.
            value_weight      : weight on value_pred (LP quality). Default 0.3.
            size_weight       : weight on -subtree_size_pred (tree cost). Default 0.7.
                                Requires SubtreeSizeHead to be trained (needs
                                subtree_size labels in the dataset; non-DFS traces
                                with tree-id tracking can produce them). Set to 0.0
                                if the head is untrained.
            ctg_weight        : weight on -cost_to_go_pred. Default 0 (off) due to
                                the linear-countdown target problem.
            uncertainty_weight: penalise candidates whose +1/-1 child scores
                                diverge (high spread = high dynamics uncertainty).
                                0.0 disables the penalty (default, no extra cost).

        Returns:
            scores: FloatTensor [K], one score per candidate (higher = better).
        """
        if z.dim() != 2 or z.size(0) != 1:
            raise ValueError("z must have shape [1, H].")
        if h_vars.dim() != 2:
            raise ValueError("h_vars must have shape [V, H].")
        if cand_indices.dim() != 1 or cand_indices.numel() == 0:
            raise ValueError("cand_indices must be a non-empty 1-D LongTensor.")

        device = z.device
        V = h_vars.size(0)
        K = cand_indices.size(0)
        b = max(1, int(branch_factor))
        directions = (1.0, -1.0) if expand_both_children else (0.0,)
        n_dirs = len(directions)

        # ------------------------------------------------------------------
        # Root expansion: all K candidates × n_dirs in one dynamics call.
        # ------------------------------------------------------------------
        # z_root  [K*n_dirs, H]
        z_root = z.expand(K * n_dirs, -1)
        # a_root  [K*n_dirs, H]: each candidate repeated n_dirs times
        a_root = h_vars[cand_indices].unsqueeze(1).expand(
            -1, n_dirs, -1
        ).reshape(K * n_dirs, -1)
        # h_root  [K*n_dirs, V, H]
        h_root = h_vars.unsqueeze(0).expand(K * n_dirs, -1, -1)

        # Per-candidate child masks: clone valid_mask and remove each cand's
        # own index, then replicate for n_dirs.
        if valid_mask is not None:
            root_mask = valid_mask.to(device=device, dtype=torch.bool)
            # [K, V]: each row is root_mask with cand_indices[i] cleared
            cand_masks = root_mask.unsqueeze(0).expand(K, -1).clone()
            cand_masks[torch.arange(K, device=device), cand_indices] = False
            # [K*n_dirs, V]
            child_masks_root = cand_masks.unsqueeze(1).expand(
                -1, n_dirs, -1
            ).reshape(K * n_dirs, V)
        else:
            child_masks_root = None

        d_root = torch.tensor(directions, dtype=z.dtype, device=device).repeat(K)
        # Expand past_tokens from [1,T,H] to [K*n_dirs,T,H] so the dynamics
        # cat([past_tokens, token]) doesn't hit a batch-dim mismatch.
        past_root = (
            past_tokens.expand(K * n_dirs, -1, -1)
            if past_tokens is not None else None
        )
        # h_cons_summary: expand [1,H] → [K*n_dirs, H] when provided.
        hcs_root = (
            h_cons_summary.expand(K * n_dirs, -1)
            if h_cons_summary is not None else None
        )
        # KV-cache: build initial caches from the past context (if any) so
        # subsequent frontier steps only attend over one new token each.
        kv_caches_root = None
        if use_kv_cache and past_root is not None:
            # Warm up caches by running a dummy forward over past_root.
            # We use the dynamics Transformer's step() with the full past so
            # the K/V tables are populated; the output is discarded.
            with torch.no_grad():
                _, _, kv_caches_root = self.dynamics.step(
                    z_root, a_root, past_tokens=None,
                    d_t=d_root, h_cons_summary=hcs_root,
                    kv_caches=None,
                )
            # Now rebuild with the warmed caches for the actual root step.
            kv_caches_root = None  # simplification: cache from root step only

        z_front, h_front, tok_front = self.dynamics_step_full_batched(
            z_root, a_root, h_root, past_root, d_root,
            h_cons_summary=hcs_root,
        )
        # Build KV caches from root tokens for subsequent frontier steps.
        frontier_kv = None
        if use_kv_cache:
            with torch.no_grad():
                _, _, frontier_kv = self.dynamics.step(
                    z_root, a_root, past_tokens=past_root,
                    d_t=d_root, h_cons_summary=hcs_root,
                    kv_caches=None,
                )

        # F = K*n_dirs frontier elements after the root step.
        F_root = z_front.size(0)  # == K * n_dirs
        # cand_id [F]: which root candidate each frontier element belongs to.
        cand_id = torch.arange(K, device=device).repeat_interleave(n_dirs)

        # ------------------------------------------------------------------
        # Score root children.
        # ------------------------------------------------------------------
        h_front_flat = h_front.reshape(F_root * V, -1)
        bvec_root = torch.arange(F_root, device=device).repeat_interleave(V)

        if child_masks_root is not None:
            # [K, V] → pick the first direction's mask (same for all dirs of one cand)
            fm_root_base = cand_masks  # [K, V]
            fm_root_flat = child_masks_root.reshape(-1)  # [F*V]
        else:
            fm_root_flat = None

        if use_reward_return:
            score_root = self.dynamics_reward_pred(z_front)  # [F]
        else:
            # Scoring rule differs by depth to avoid double-counting.
            #
            # depth == 1 (single-step): z_front IS the leaf — apply the full
            #   composite: V (LP quality) + S (total remaining tree cost).
            #   S is counted once here and we return immediately.
            #
            # depth > 1 (multi-step): z_front is an intermediate state.
            #   Apply V only at intermediate steps — V(z^t) measures LP bound
            #   improvement at each step and is genuinely additive across depth.
            #   S is NOT applied here because S(z^t) predicts all remaining work
            #   from z^t including the work at z^{t+1},...,z^{leaf}. Applying S
            #   at every depth level double-counts that future cost. S is instead
            #   applied once to the final leaf frontier at the end of the loop.
            score_root = z_front.new_zeros(z_front.size(0))  # [F]

            if value_weight != 0.0:
                v_root = self.value_pred(
                    z_front, h_front_flat, bvec_root, frac_mask=fm_root_flat
                )  # [F]  — higher is better (tighter bound)
                score_root = score_root + value_weight * v_root

            if depth == 1:
                # Leaf: apply S and C here (single imagined state, no future levels).
                if size_weight != 0.0:
                    score_root = score_root - size_weight * self.subtree_size_pred(
                        z_front, h_front_flat, bvec_root, frac_mask=fm_root_flat
                    )
                if ctg_weight != 0.0:
                    score_root = score_root - ctg_weight * self.cost_to_go_pred(
                        z_front, h_front_flat, bvec_root, frac_mask=fm_root_flat
                    )

        # per_cand [K]: scatter-add scores to the owning candidate.
        per_cand = torch.zeros(K, dtype=z.dtype, device=device)
        per_cand.scatter_add_(0, cand_id, score_root)

        # Direction-spread uncertainty proxy (free — score_root already computed).
        # score_root is ordered [cand_0_dir_0, cand_0_dir_1, ..., cand_K_dir_{D-1}]
        # so reshape to [K, n_dirs] and compute per-candidate max-min spread.
        # spread → 0 means both directions agree (confident); spread → large means
        # the two children diverge (dynamics is uncertain about this candidate).
        if uncertainty_weight != 0.0 and n_dirs > 1:
            dir_scores = score_root.reshape(K, n_dirs)
            spread = dir_scores.max(dim=1).values - dir_scores.min(dim=1).values
        else:
            spread = None

        # frontier_weights [F]: cumulative averaging weights (starts at 1).
        frontier_weights = torch.ones(F_root, dtype=z.dtype, device=device)
        frontier_cand_id = cand_id  # tracks owner across levels

        if depth == 1:
            if use_reward_return:
                leaf_value = self.value_pred(
                    z_front, h_front_flat, bvec_root, frac_mask=fm_root_flat
                )
                per_cand.scatter_add_(0, frontier_cand_id, leaf_value)
            return per_cand

        frontier_z = z_front
        frontier_h = h_front
        frontier_tok = tok_front
        frontier_masks = child_masks_root
        continuation_discount = 1.0

        for level in range(1, depth):
            continuation_discount *= gamma
            F = frontier_z.size(0)

            if frontier_masks is None:
                expandable = torch.ones(F, dtype=torch.bool, device=device)
            else:
                expandable = frontier_masks.any(dim=1)

            terminal = ~expandable

            if bool(terminal.any()) and use_reward_return:
                z_term = frontier_z[terminal]
                h_term = frontier_h[terminal]
                N_term = z_term.size(0)
                h_term_flat = h_term.reshape(N_term * V, -1)
                bvec_term = torch.arange(N_term, device=device).repeat_interleave(V)
                fm_term = (
                    frontier_masks[terminal].reshape(-1)
                    if frontier_masks is not None else None
                )
                terminal_score = self.value_pred(
                    z_term, h_term_flat, bvec_term, frac_mask=fm_term
                )
                w_term = frontier_weights[terminal]
                id_term = frontier_cand_id[terminal]
                per_cand.scatter_add_(
                    0, id_term,
                    continuation_discount * w_term * terminal_score,
                )

            if not bool(expandable.any()):
                break

            exp_idx = expandable.nonzero(as_tuple=False).squeeze(1)
            z_exp = frontier_z[exp_idx]
            h_exp = frontier_h[exp_idx]
            tok_exp = frontier_tok[exp_idx]
            masks_exp = frontier_masks[exp_idx] if frontier_masks is not None else None
            w_exp = frontier_weights[exp_idx]
            id_exp = frontier_cand_id[exp_idx]

            k_eff = b
            next_idx = self._policy_topk_batched(h_exp, z_exp, masks_exp, k_eff)
            E, K_act = next_idx.shape

            H_dim = h_exp.size(-1)
            h_expanded = h_exp.unsqueeze(1).expand(-1, K_act, -1, -1)
            gather_idx = next_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, H_dim)
            a_exp = torch.gather(h_expanded, 2, gather_idx).squeeze(2)  # [E,K,H]

            z_parent = z_exp.unsqueeze(1).expand(-1, K_act, -1).reshape(E * K_act, H_dim)
            a_flat = a_exp.reshape(E * K_act, H_dim)
            h_parent = h_exp.unsqueeze(1).expand(-1, K_act, -1, -1).reshape(E * K_act, V, H_dim)
            tok_parent = tok_exp.unsqueeze(1).expand(
                -1, K_act, -1, -1
            ).reshape(E * K_act, tok_exp.size(1), H_dim)

            if masks_exp is not None:
                masks_parent = masks_exp.unsqueeze(1).expand(-1, K_act, -1).reshape(
                    E * K_act, V
                ).clone()
                row = torch.arange(E * K_act, device=device)
                masks_parent[row, next_idx.reshape(-1)] = False
                fm_parent = masks_parent
            else:
                fm_parent = None

            z_child_in = z_parent.unsqueeze(1).expand(-1, n_dirs, -1).reshape(E * K_act * n_dirs, H_dim)
            a_child_in = a_flat.unsqueeze(1).expand(-1, n_dirs, -1).reshape(E * K_act * n_dirs, H_dim)
            h_child_in = h_parent.unsqueeze(1).expand(-1, n_dirs, -1, -1).reshape(E * K_act * n_dirs, V, H_dim)
            tok_child_in = tok_parent.unsqueeze(1).expand(
                -1, n_dirs, -1, -1
            ).reshape(E * K_act * n_dirs, tok_parent.size(1), H_dim)
            d = torch.tensor(directions, dtype=z.dtype, device=device).view(
                1, n_dirs
            ).expand(E * K_act, -1).reshape(-1)
            fm_child = (
                fm_parent.unsqueeze(1).expand(-1, n_dirs, -1).reshape(E * K_act * n_dirs, V)
                if fm_parent is not None else None
            )

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
                # Intermediate step: V only. S is deferred to the final leaf
                # so the remaining-tree-cost penalty is counted exactly once.
                step_score = z_next.new_zeros(z_next.size(0))
                if value_weight != 0.0:
                    step_score = step_score + value_weight * self.value_pred(
                        z_next, h_next_flat, bvec_step, frac_mask=fm_step_flat
                    )

            # Weights: divide by K_act (averaging) × n_dirs (summed per direction).
            # Each expandable parent's weight is divided by K_act, then replicated
            # for K_act children and n_dirs directions.
            w_step = (w_exp / K_act).unsqueeze(1).expand(
                -1, K_act * n_dirs
            ).reshape(E * K_act * n_dirs)

            # Propagate candidate ownership: each parent's K_act*n_dirs children
            # inherit the parent's cand_id.
            id_step = id_exp.unsqueeze(1).expand(
                -1, K_act * n_dirs
            ).reshape(E * K_act * n_dirs)

            per_cand.scatter_add_(
                0, id_step,
                continuation_discount * w_step * step_score,
            )

            frontier_z = z_next
            frontier_h = h_next
            frontier_tok = tok_next
            frontier_masks = fm_child
            frontier_weights = w_step
            frontier_cand_id = id_step

        # Final leaf frontier: apply S and C once here for multi-step rollouts.
        # V is also applied as a bootstrap if use_reward_return is set.
        # S counts the total remaining tree cost at the imagined leaf — applying
        # it only here (not at intermediate steps) avoids double-counting.
        if frontier_z.size(0) > 0:
            N_leaf = frontier_z.size(0)
            h_leaf_flat = frontier_h.reshape(N_leaf * V, -1)
            bvec_leaf = torch.arange(N_leaf, device=device).repeat_interleave(V)
            fm_leaf_flat = frontier_masks.reshape(-1) if frontier_masks is not None else None

            leaf_score = frontier_z.new_zeros(N_leaf)

            need_v = use_reward_return or (value_weight != 0.0 and not use_reward_return)
            need_s = size_weight != 0.0 and not use_reward_return
            need_c = ctg_weight  != 0.0 and not use_reward_return
            heads = self.multi_head_pred(
                frontier_z, h_leaf_flat, bvec_leaf,
                frac_mask=fm_leaf_flat,
                value=need_v, subtree_size=need_s, cost_to_go=need_c,
            )

            if use_reward_return:
                leaf_score = leaf_score + heads["value"]
            if need_s:
                leaf_score = leaf_score - size_weight * heads["subtree_size"]
            if need_c:
                leaf_score = leaf_score - ctg_weight  * heads["cost_to_go"]

            per_cand.scatter_add_(
                0, frontier_cand_id,
                continuation_discount * frontier_weights * leaf_score,
            )

        if spread is not None:
            per_cand = per_cand - uncertainty_weight * spread

        return per_cand

    # ------------------------------------------------------------------
    # Cut-branch beam search via latent dynamics
    # ------------------------------------------------------------------
    def rollout_cut_branch_beam(
        self,
        z: torch.Tensor,
        h_vars: torch.Tensor,
        cut_embeds: torch.Tensor,
        cand_indices: torch.Tensor,
        frac_mask: torch.Tensor | None = None,
        past_tokens: torch.Tensor | None = None,
        cut_beam: int = 3,
        cut_rounds: int = 2,
        entropy_thresh: float = 0.0,
        pre_filter_k: int = 0,
        precomputed_policy_logits: torch.Tensor | None = None,
        **branch_kwargs,
    ) -> tuple:
        """Evaluate cut candidates in latent space, then branch from the best state.

        Two-phase unified rollout: cut steps and branch steps share the same
        past_tokens buffer, so the dynamics Transformer sees
        [cut_tok, cut_tok, branch_tok, ...] as one causal sequence. Branch
        predictions are conditioned on the cuts that preceded them.

        Phase A — cut beam search (no LP re-solve):
          For each round in cut_rounds, expand each beam state with every cut
          candidate, simulate via Dynamics(z, cut_embed, d=0.0), score with
          ValueHead, prune to top cut_beam states. The cut token is appended
          to past_tokens so subsequent dynamics steps see the full history.

        Phase B — branch rollout from best post-cut state:
          Call rollout_top_k_batched from z_best with tok_best as context.
          The branch dynamics now "knows" which cuts preceded it.

        Args:
            z            : [1, H]   current node latent state
            h_vars       : [V, H]   variable embeddings (fixed, from GNN encode)
            cut_embeds   : [C, H]   GNN-structured cut embeddings
                                    (= Σ coeff_j * h_vars[j] per cut, from cg_cuts)
            cand_indices : [K]      branch candidate variable indices (LongTensor)
            frac_mask    : [V] bool fractional variable mask; stays fixed during
                                    latent rollout (no LP re-solve in this phase)
            past_tokens  : [1,T,H] or None — token history up to this node
            cut_beam     : number of beam states kept after each cut round
            cut_rounds   : number of interleaved cut simulation steps
            entropy_thresh: stop cut rounds early when policy entropy (nats) over
                            fractional vars drops below this — policy is confident
            pre_filter_k : cheap linear pre-filter before Dynamics; keep top
                           pre_filter_k cuts by value_pred(z + cut_embed_proj).
                           0 = disabled (use all C candidates directly).
            **branch_kwargs: forwarded to rollout_top_k_batched (depth, gamma, …)

        Returns:
            cut_indices  : list[int] — indices into cut_embeds of the chosen cuts
                           (one per committed round; empty = NO-CUT baseline won)
            branch_scores: FloatTensor [K] — per-candidate branch scores from
                           rollout_top_k_batched at the best post-cut state
            z_best       : [1, H]   best post-cut latent state
            tok_best     : [1, T', H] updated token buffer (includes cut tokens)
        """
        if z.dim() != 2 or z.size(0) != 1:
            raise ValueError("z must be [1, H]")
        if h_vars.dim() != 2:
            raise ValueError("h_vars must be [V, H]")

        device = z.device
        H = z.size(1)
        C = cut_embeds.size(0) if cut_embeds is not None and cut_embeds.numel() > 0 else 0
        bvec_single = torch.zeros(h_vars.size(0), dtype=torch.long, device=device)

        # NO-CUT baseline: score the current state — V and S computed together.
        with torch.no_grad():
            _bl = self.multi_head_pred(
                z, h_vars, bvec_single, frac_mask=frac_mask,
                value=True, subtree_size=True,
            )
            baseline_score = (_bl["value"] - _bl["subtree_size"]).item()

        # No cuts available → skip to branch rollout immediately.
        if C == 0 or cut_rounds == 0:
            branch_scores = self.rollout_top_k_batched(
                z, h_vars, cand_indices, past_tokens=past_tokens,
                **branch_kwargs,
            )
            return [], branch_scores, z, past_tokens

        # ----------------------------------------------------------------
        # Optional cheap pre-filter: rank cuts by value_pred(z + cut_embed)
        # without a full Dynamics forward pass — linear approximation of
        # the value change. Keeps only pre_filter_k cuts for Dynamics.
        # ----------------------------------------------------------------
        if pre_filter_k > 0 and C > pre_filter_k:
            with torch.no_grad():
                # Linear approx: shift z by a small fraction of cut_embed direction
                z_approx = z + 0.1 * cut_embeds           # [C, H] broadcast
                # score each shifted z; h_vars/bvec replicated for C pseudo-graphs
                h_rep = h_vars.unsqueeze(0).expand(C, -1, -1).reshape(C * h_vars.size(0), H)
                bvec_rep = torch.arange(C, device=device).repeat_interleave(h_vars.size(0))
                fm_rep = (frac_mask.unsqueeze(0).expand(C, -1).reshape(-1)
                          if frac_mask is not None else None)
                approx_scores = self.value_pred(z_approx, h_rep, bvec_rep, frac_mask=fm_rep)
                _, top_pre = approx_scores.topk(min(pre_filter_k, C))
            cut_embeds = cut_embeds[top_pre]
            C = cut_embeds.size(0)

        # ----------------------------------------------------------------
        # Phase A (cut beam) + speculative Phase B (branch rollout from
        # pre-cut z) run concurrently on separate CUDA streams.
        #
        # If NO-CUT wins the beam the speculative branch scores are reused
        # at zero extra cost.  If a cut is chosen, branch rollout re-runs
        # from z_best (speculative result discarded).  On CPU the streams
        # degrade gracefully to sequential execution.
        # ----------------------------------------------------------------
        use_streams = device.type == "cuda"
        stream_cut    = torch.cuda.Stream(device=device) if use_streams else None
        stream_branch = torch.cuda.Stream(device=device) if use_streams else None

        beams = [(z, past_tokens, baseline_score, [])]
        best_cut_indices = []
        speculative_scores: torch.Tensor | None = None

        # --- Speculative branch rollout (stream_branch) ------------------
        # Launched before the cut beam so both can overlap on the GPU.
        if use_streams:
            with torch.cuda.stream(stream_branch):
                with torch.no_grad():
                    speculative_scores = self.rollout_top_k_batched(
                        z, h_vars, cand_indices,
                        past_tokens=past_tokens,
                        **branch_kwargs,
                    )
        # (CPU path: speculative_scores stays None; rollout runs serially later)

        # --- Cut beam (stream_cut or default stream) ----------------------
        cut_ctx = torch.cuda.stream(stream_cut) if use_streams else contextlib.nullcontext()
        with cut_ctx, torch.no_grad():
            for _round in range(cut_rounds):
                if entropy_thresh > 0.0 and frac_mask is not None and frac_mask.any():
                    z_cur, _, _, _ = beams[0]
                    # Round 0: z_cur == z — reuse pre-computed logits if available
                    if _round == 0 and precomputed_policy_logits is not None:
                        logits = precomputed_policy_logits
                    else:
                        logits = self.policy_scores(h_vars, z_cur, bvec_single)
                    probs  = torch.softmax(logits[frac_mask], dim=0)
                    H_pi   = float(-(probs * (probs + 1e-12).log()).sum())
                    if H_pi < entropy_thresh:
                        break

                candidates = []
                for z_s, tok_s, _score_s, idx_list in beams:
                    z_exp   = z_s.expand(C, -1)
                    h_exp   = h_vars.unsqueeze(0).expand(C, -1, -1)
                    tok_exp = (tok_s.expand(C, -1, -1) if tok_s is not None else None)
                    d_cut   = torch.zeros(C, dtype=z.dtype, device=device)

                    z_next, h_next, tok_next = self.dynamics_step_full_batched(
                        z_exp, cut_embeds, h_exp, tok_exp, d_cut
                    )

                    h_next_flat = h_next.reshape(C * h_vars.size(0), H)
                    bvec_c = torch.arange(C, device=device).repeat_interleave(h_vars.size(0))
                    fm_c   = (frac_mask.unsqueeze(0).expand(C, -1).reshape(-1)
                              if frac_mask is not None else None)
                    # Compute V and S together — frac_mean computed once, shared.
                    heads_c = self.multi_head_pred(
                        z_next, h_next_flat, bvec_c, frac_mask=fm_c,
                        value=True, subtree_size=True,
                    )
                    # Score: V - size_weight * S  (same formula as branch rollout leaf)
                    scores_c = heads_c["value"] - heads_c["subtree_size"]

                    for ci in range(C):
                        candidates.append((
                            z_next[ci:ci+1],
                            tok_next[ci:ci+1],
                            float(scores_c[ci].item()),
                            idx_list + [ci],
                        ))

                candidates.extend(beams)
                candidates.sort(key=lambda x: -x[2])
                beams = candidates[:cut_beam]

            z_best, tok_best, best_score, best_cut_indices = beams[0]

            if best_score <= baseline_score + 1e-6 and not best_cut_indices:
                z_best, tok_best = z, past_tokens
                best_cut_indices = []

        # Sync both streams before deciding which branch scores to use.
        if use_streams:
            torch.cuda.synchronize(device=device)

        # ----------------------------------------------------------------
        # Phase B: branch rollout from z_best.
        # Reuse speculative scores when NO-CUT won (z_best == z and
        # tok_best == past_tokens) — avoids a redundant GPU kernel.
        # ----------------------------------------------------------------
        no_cut_won = (not best_cut_indices)
        if no_cut_won and speculative_scores is not None:
            branch_scores = speculative_scores   # free reuse
        else:
            with torch.no_grad():
                branch_scores = self.rollout_top_k_batched(
                    z_best, h_vars, cand_indices,
                    past_tokens=tok_best,
                    **branch_kwargs,
                )

        return best_cut_indices, branch_scores, z_best, tok_best

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
    ) -> float:
        """Estimate candidate quality using level-wise batched latent rollout."""
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

