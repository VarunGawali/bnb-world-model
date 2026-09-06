"""
dynamics.py — Causal Transformer latent dynamics model.

Replaces the original GRUCell with a causal Transformer decoder that
treats the B&B trajectory as a token sequence.

Each token = concat(z_t, a_t) where z_t is the graph embedding at step t
and a_t is the embedding of the branching action taken at step t.

A causal (masked) self-attention layer ensures the model only attends to
past context, so it can be used auto-regressively at inference while still
being trained in parallel on full trajectories.

Inference supports both single-state and batched latent rollouts. The
batched path expands a whole rollout frontier at once, avoiding the
Python-recursive GPU launch pattern in the world-model rollout.

NOTE:
    The Transformer still recomputes the token buffer at every step. This
    is intentionally NOT a KV-cache rewrite; true KV caching is a separate
    optimization and should be validated independently.

Architecture change vs. original:
    GRUCell (single hidden vector, exponential forgetting)
    -> Causal Transformer decoder (full receptive field, multi-step lookahead)

This makes the 'world model' claim concrete: the model can plan multiple
steps ahead in latent space by unrolling forward without touching the LP.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _CausalSelfAttention(nn.Module):
    """Single causal multi-head self-attention block (pre-norm).

    KV-cache mode: when kv_cache is provided it holds (K_past, V_past) tensors
    for all T_0 history tokens. Only the new token's Q/K/V are projected; K/V
    are concatenated and the result K_full/V_full are returned for the next step.
    Attention is then O(1) in new-token compute instead of O(T_0).
    """

    def __init__(self, d_model: int, n_heads: int, max_seq: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # Fixed causal mask (upper-triangular = -inf)
        mask = torch.triu(
            torch.full((max_seq, max_seq), float("-inf")),
            diagonal=1,
        )
        self.register_buffer("causal_mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Args:
            x        : [B, T, D] — full sequence (training) or
                       [B, 1, D] — single new token (kv_cache inference)
            kv_cache : (K_past [B,H,T_past,head_dim], V_past [B,H,T_past,head_dim])
                       or None.  When not None, x is expected to be [B,1,D].

        Returns:
            out      : [B, T, D]
            new_kv   : updated (K_full, V_full) when kv_cache is not None, else None
        """
        B, T, D = x.shape
        residual = x
        x_ln = self.norm(x)

        Q, K, V = self.qkv(x_ln).chunk(3, dim=-1)

        def split(t, seq):
            return t.view(B, seq, self.n_heads, self.head_dim).transpose(1, 2)

        Q = split(Q, T)
        K = split(K, T)
        V = split(V, T)

        new_kv = None
        if kv_cache is not None:
            K_past, V_past = kv_cache
            K_full = torch.cat([K_past, K], dim=2)   # [B, H, T_past+1, head_dim]
            V_full = torch.cat([V_past, V], dim=2)
            new_kv = (K_full, V_full)
            # Q is [B, H, 1, head_dim]; attend over full history
            attn = (Q @ K_full.transpose(-2, -1)) * self.scale
            # no causal mask needed: new token can attend to all past (causal by construction)
            attn = self.drop(F.softmax(attn, dim=-1))
            out = (attn @ V_full).transpose(1, 2).contiguous().view(B, T, D)
        else:
            attn = (Q @ K.transpose(-2, -1)) * self.scale
            attn = attn + self.causal_mask[:T, :T]
            attn = self.drop(F.softmax(attn, dim=-1))
            out = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)

        return residual + self.proj(out), new_kv


class _FFN(nn.Module):
    """Position-wise feed-forward block (pre-norm)."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))




class _VarDynamics(nn.Module):
    """
    Per-variable latent transition head — decomposed first layer.

    Supports arbitrary leading batch dimensions.

        h_vars: [..., V, H]
        z_next: [..., H]
        a:      [..., H]

    The original formulation expanded z_next and a to [..., V, H] and
    concatenated them with h_vars before the MLP, running the first
    Linear(3H → 2H) V times even though 2/3 of its input was identical
    across variables. For V = 100k, H = 128 this wastes ≈200k redundant
    matmuls per dynamics step.

    Fix: split the first Linear(3H → 2H) into three H → 2H projections,
    exploiting linearity:
        W @ [h, z, a] = W_h(h) + W_z(z) + W_a(a)

    W_z(z) and W_a(a) are computed once (O(H²)), then broadcast-added to
    the per-variable W_h(h) result (O(V·H²)).  Total first-layer cost drops
    from O(V·3H²) to O(V·H² + 2H²) ≈ O(V·H²) with a 3× coefficient saving.

    The second Linear(2H → H) is unavoidable (output is per-variable) but is
    applied to h_vars.size(0)*H² flops regardless of formulation.

    Checkpoint migration:
        Old:  dynamics.var_dynamics.net.{0,3}.{weight,bias}
        New:  dynamics.var_dynamics.{W_h,W_z,W_a,out}.{weight,bias}
        Use _VarDynamics.from_legacy_state(old_sd, hidden_dim) to convert.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        H2 = 2 * hidden_dim
        # Per-variable projection (bias lives here)
        self.W_h = nn.Linear(hidden_dim, H2, bias=True)
        # Shared projections (no bias — bias is absorbed into W_h)
        self.W_z = nn.Linear(hidden_dim, H2, bias=False)
        self.W_a = nn.Linear(hidden_dim, H2, bias=False)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.out  = nn.Linear(H2, hidden_dim, bias=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h_vars: torch.Tensor,   # [..., V, H]
        z_next: torch.Tensor,   # [..., H]
        a: torch.Tensor,        # [..., H]
    ) -> torch.Tensor:
        # Shared contributions: computed once, O(H²) per batch element.
        shared = self.W_z(z_next) + self.W_a(a)              # [..., H2]
        # Per-variable first hidden: broadcast-add shared to W_h output.
        hidden = self.act(
            self.W_h(h_vars) + shared.unsqueeze(-2)           # [..., V, H2]
        )
        delta = self.out(self.drop(hidden))                    # [..., V, H]
        return self.norm(h_vars + delta)

    @classmethod
    def from_legacy_state(
        cls, legacy_sd: dict, hidden_dim: int, dropout: float = 0.1
    ) -> "tuple[_VarDynamics, dict]":
        """Construct a new _VarDynamics and migrate old net.{0,3} weights.

        Args:
            legacy_sd  : full model state_dict from an old checkpoint
            hidden_dim : hidden dimension used by the model
            dropout    : dropout rate

        Returns:
            (module, new_sd) where new_sd is legacy_sd with the
            var_dynamics keys rewritten to the new format.
        """
        import copy
        sd = copy.copy(legacy_sd)
        pfx = "dynamics.var_dynamics."

        old_w0 = sd.pop(pfx + "net.0.weight")   # [2H, 3H]
        old_b0 = sd.pop(pfx + "net.0.bias")     # [2H]
        old_w3 = sd.pop(pfx + "net.3.weight")   # [H, 2H]
        old_b3 = sd.pop(pfx + "net.3.bias")     # [H]

        H = hidden_dim
        sd[pfx + "W_h.weight"] = old_w0[:, :H].contiguous()
        sd[pfx + "W_h.bias"]   = old_b0
        sd[pfx + "W_z.weight"] = old_w0[:, H:2*H].contiguous()
        sd[pfx + "W_a.weight"] = old_w0[:, 2*H:].contiguous()
        sd[pfx + "out.weight"] = old_w3
        sd[pfx + "out.bias"]   = old_b3
        # norm keys (norm.weight / norm.bias) are unchanged

        mod = cls(hidden_dim, dropout)
        return mod, sd


class DynamicsTransformer(nn.Module):
    """
    Causal Transformer dynamics model.

    Each B&B trajectory is a sequence of (state, action) pairs:
        token_t = Linear([z_t || a_t])   ->  d_model

    The model predicts z_{t+1} from the full causal context
    [token_0, ..., token_t]. In addition, a per-variable head (_VarDynamics)
    predicts the next per-variable embeddings h_vars_{t+1}, so the policy can
    be re-run on the predicted state and a genuine multi-step branching
    rollout can be performed in latent space (no LP solves).

    Training (parallel, teacher-forced):
        inputs  : token sequence [B, T, d_model]
        targets : z_{1}, ..., z_{T}  (one-step shifted)
                  and optionally h_vars_{1}, ..., h_vars_{T}

    Inference:
        - step(): single/batched state transition
        - step_full(): single/batched state + variable transition
        - rollout(): autoregressive latent overshooting

    The explicit token buffer is retained. It is not a true KV cache.

    Args:
        hidden_dim : must match encoder's hidden_dim
        n_layers   : transformer depth (default 4)
        n_heads    : attention heads (default 4)
        max_seq    : maximum trajectory length supported (default 512)
        dropout    : attention + FFN dropout rate (default 0.1)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        max_seq: int = 512,
        dropout: float = 0.1,
        residual: bool = True,
        heteroscedastic: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq = max_seq

        self.residual = residual
        self.heteroscedastic = heteroscedastic

        # Project [z_t || a_t || dir_t] -> d_model.
        self.input_proj = nn.Linear(2 * hidden_dim + 1, hidden_dim)

        # Constraint-summary injection: pooled h_cons → token addend.
        # Zero-init → identity at load; warm-start safe.
        self.h_cons_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.h_cons_proj.weight)

        # Learned positional embeddings
        self.pos_emb = nn.Embedding(max_seq, hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.ModuleList([
                _CausalSelfAttention(hidden_dim, n_heads, max_seq, dropout),
                _FFN(hidden_dim, dropout),
            ])
            for _ in range(n_layers)
        ])

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.res_norm = nn.LayerNorm(hidden_dim)

        self.logvar_proj = (
            nn.Linear(hidden_dim, hidden_dim)
            if heteroscedastic else None
        )

        # Per-variable transition head
        self.var_dynamics = _VarDynamics(hidden_dim, dropout)

    # ------------------------------------------------------------------
    # Per-variable prediction helper
    # ------------------------------------------------------------------
    def predict_vars(
        self,
        h_vars: torch.Tensor,
        z_next: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        """Predict h_vars_{t+1} given current h_vars, z_next and action."""
        return self.var_dynamics(h_vars, z_next, a)

    # ------------------------------------------------------------------
    # Token / decoder helpers
    # ------------------------------------------------------------------
    def _tokens(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        d: torch.Tensor | float | None,
        h_cons_summary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Build input tokens from latents, actions and direction.

        Supports:
            z/a: [B, H]
            z/a: [B, T, H]
            z/a: [..., H]

        d can be scalar, [...], [..., 1], or None.
        h_cons_summary: optional [..., H] pooled constraint embedding;
            added to the token after projection (zero-init projection → no-op at init).
        """
        if d is None:
            d = z.new_zeros(*z.shape[:-1], 1)
        elif not torch.is_tensor(d):
            d = z.new_full((*z.shape[:-1], 1), float(d))
        else:
            d = d.to(device=z.device, dtype=z.dtype)
            if d.dim() == z.dim() - 1:
                d = d.unsqueeze(-1)
            elif d.dim() == 0:
                d = d.expand(*z.shape[:-1], 1)

        tok = self.input_proj(torch.cat([z, a, d], dim=-1))
        if h_cons_summary is not None:
            tok = tok + self.h_cons_proj(h_cons_summary)
        return tok

    def _decode(
        self,
        feat: torch.Tensor,
        z_in: torch.Tensor,
    ) -> torch.Tensor:
        """Map transformer features to the next latent."""
        delta = self.out_proj(feat)
        if self.residual:
            return self.res_norm(z_in + delta)
        return delta

    def _decode_sequence(
        self,
        x: torch.Tensor,
        z_in: torch.Tensor,
        kv_caches: list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list | None]:
        """Run Transformer blocks and decode the final sequence.

        When kv_caches is provided (list of (K,V) per layer or list of None),
        each attention layer uses cached K/V and returns an updated cache.

        Returns:
            z_pred     : decoded next latent
            feat       : out_norm output (for logvar head, etc.)
            new_caches : updated per-layer kv caches, or None in training mode
        """
        new_caches = [] if kv_caches is not None else None
        for i, (attn, ffn) in enumerate(self.layers):
            kvc = kv_caches[i] if kv_caches is not None else None
            x, new_kv = attn(x, kv_cache=kvc)
            x = ffn(x)
            if new_caches is not None:
                new_caches.append(new_kv)

        feat = self.out_norm(x)
        # Training forward: z_in is [B,T,H] matching feat — decode all timesteps.
        # Inference (step): z_in is [B,H] while feat is [B,T,H] — use last token only.
        feat_for_decode = feat if feat.shape == z_in.shape else feat[..., -1, :]
        return self._decode(feat_for_decode, z_in), feat, new_caches

    # ------------------------------------------------------------------
    # Parallel training forward
    # ------------------------------------------------------------------
    def forward(
        self,
        z_seq: torch.Tensor,
        a_seq: torch.Tensor,
        d_seq: torch.Tensor | None = None,
        return_logvar: bool = False,
    ):
        """
        Parallel (training) forward over a full trajectory.

        Args:
            z_seq : [B, T, H]
            a_seq : [B, T, H]
            d_seq : [B, T]

        Returns:
            z_pred : [B, T, H]
            optionally (z_pred, logvar)
        """
        B, T, _ = z_seq.shape
        if T > self.max_seq:
            raise ValueError(
                f"Sequence length T={T} exceeds max_seq={self.max_seq}."
            )

        tokens = self._tokens(z_seq, a_seq, d_seq)
        pos = self.pos_emb(torch.arange(T, device=z_seq.device))
        x = tokens + pos

        z_pred, feat, _ = self._decode_sequence(x, z_seq, kv_caches=None)

        if return_logvar and self.logvar_proj is not None:
            return z_pred, self.logvar_proj(feat)
        return z_pred

    # ------------------------------------------------------------------
    # Single / batched single-step inference
    # ------------------------------------------------------------------
    def step(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        past_tokens: torch.Tensor | None = None,
        d_t: torch.Tensor | float | None = None,
        h_cons_summary: torch.Tensor | None = None,
        kv_caches: list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list | None]:
        """
        Single-step inference.

        z_t/a_t may have arbitrary leading batch dimensions ending in H,
        while past_tokens must be [B, T, H] for the common batched path.

        Args:
            h_cons_summary: optional [..., H] pooled constraint embedding injected
                            into the token before the Transformer (zero-init proj).
            kv_caches     : list of (K_past, V_past) per layer from the previous step,
                            or None for the first step / full-sequence (training) mode.
                            When provided, only the new token is processed (O(1) attn).

        Returns:
            z_next       : predicted next latent
            tokens       : updated full token buffer [B, T+1, H] (appended)
            new_kv_caches: updated per-layer KV caches, or None when kv_caches=None
        """
        token = self._tokens(z_t, a_t, d_t, h_cons_summary).unsqueeze(-2)  # [..., 1, H]

        if past_tokens is None:
            tokens = token
        else:
            tokens = torch.cat([past_tokens, token], dim=-2)

        if tokens.size(-2) > self.max_seq:
            tokens = tokens[..., -self.max_seq:, :]

        T_full = tokens.size(-2)

        if kv_caches is not None:
            # KV-cache path: new token only, attend over cached past K/V.
            # pos for the new token = T_full - 1 (its position in the full seq).
            pos_new = self.pos_emb(
                torch.tensor([T_full - 1], device=z_t.device)
            ).view(*([1] * (token.dim() - 2)), 1, self.hidden_dim)
            x = token + pos_new
            _, feat_kv, new_kv_caches = self._decode_sequence(x, z_t, kv_caches=kv_caches)
            feat = feat_kv[..., -1, :]   # extract last (only) token
        else:
            pos = self.pos_emb(
                torch.arange(T_full, device=z_t.device)
            ).view(*([1] * (tokens.dim() - 2)), T_full, self.hidden_dim)
            x = tokens + pos
            _, feat_seq, new_kv_caches = self._decode_sequence(x, z_t, kv_caches=None)
            feat = feat_seq[..., -1, :]   # last token's feature

        z_next = self._decode(feat, z_t)
        return z_next, tokens, new_kv_caches

    # ------------------------------------------------------------------
    # Autoregressive latent rollout
    # ------------------------------------------------------------------
    def rollout(
        self,
        z0: torch.Tensor,
        a_seq: torch.Tensor,
        past_tokens: torch.Tensor | None = None,
        d_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive latent rollout for training-time latent overshooting.

        Shapes:
            z0    : [B, H]
            a_seq : [B, K, H]
            d_seq : [B, K], optional

        Returns:
            preds : [B, K, H]
        """
        preds = []
        z_cur = z0
        tokens = past_tokens

        for j in range(a_seq.size(1)):
            d_j = d_seq[:, j] if d_seq is not None else None
            z_cur, tokens, _ = self.step(z_cur, a_seq[:, j], tokens, d_j)
            preds.append(z_cur)

        return torch.stack(preds, dim=1)

    # ------------------------------------------------------------------
    # Full single / batched state transition
    # ------------------------------------------------------------------
    def step_full(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        h_vars_t: torch.Tensor,
        past_tokens: torch.Tensor | None = None,
        d_t: torch.Tensor | float | None = None,
        h_cons_summary: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-step inference that also predicts next per-variable embeddings.

        Batch-safe shapes:
            z_t       : [B, H]
            a_t       : [B, H]
            h_vars_t  : [B, V, H]
            past      : [B, T, H]

        Single-graph compatibility:
            z_t       : [1, H]
            a_t       : [1, H]
            h_vars_t  : [V, H]
            past      : [1, T, H] or None

        Returns:
            z_next       : [B, H]
            h_vars_next  : [B, V, H] for batched input
                           [V, H] for single-graph compatibility
            new_tokens   : [B, T+1, H]
        """
        single_vars = h_vars_t.dim() == 2

        if single_vars:
            if z_t.dim() != 2 or z_t.size(0) != 1:
                raise ValueError(
                    "For h_vars_t shaped [V,H], z_t must be [1,H]."
                )
            h_vars_b = h_vars_t.unsqueeze(0)
        else:
            if h_vars_t.dim() != 3:
                raise ValueError(
                    "h_vars_t must have shape [V,H] or [B,V,H]."
                )
            h_vars_b = h_vars_t

        z_next, tokens, _ = self.step(z_t, a_t, past_tokens, d_t, h_cons_summary)

        # IMPORTANT: preserve the batch dimension. The old implementation
        # used z_next[0] and a_t[0], which made step_full effectively
        # single-graph only and prevented clean frontier batching.
        h_vars_next = self.var_dynamics(
            h_vars_b,
            z_next,
            a_t,
        )

        if single_vars:
            h_vars_next = h_vars_next.squeeze(0)

        return z_next, h_vars_next, tokens

    # ------------------------------------------------------------------
    # Fully batched full transition helper
    # ------------------------------------------------------------------
    def step_full_batched(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        h_vars_t: torch.Tensor,
        past_tokens: torch.Tensor | None = None,
        d_t: torch.Tensor | float | None = None,
        h_cons_summary: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Explicit batch-oriented alias for step_full().

        Shapes:
            z_t       : [B,H]
            a_t       : [B,H]
            h_vars_t  : [B,V,H]
            past      : [B,T,H]

        This method exists to make the intended rollout-frontier API explicit.
        It does not introduce a separate implementation, so step_full() and
        step_full_batched() remain numerically identical.
        """
        if z_t.dim() != 2 or a_t.dim() != 2 or h_vars_t.dim() != 3:
            raise ValueError(
                "Batched step_full requires z_t [B,H], a_t [B,H], "
                "and h_vars_t [B,V,H]."
            )

        if z_t.size(0) != a_t.size(0) or z_t.size(0) != h_vars_t.size(0):
            raise ValueError(
                "Batch dimensions of z_t, a_t and h_vars_t must match."
            )

        if past_tokens is not None and (
            past_tokens.dim() != 3 or past_tokens.size(0) != z_t.size(0)
        ):
            raise ValueError(
                "past_tokens must be [B,T,H] with the same B as z_t."
            )

        return self.step_full(
            z_t, a_t, h_vars_t, past_tokens, d_t, h_cons_summary
        )

    # ------------------------------------------------------------------
    # Parallel training with per-variable predictions
    # ------------------------------------------------------------------
    def forward_with_vars(
        self,
        z_seq: torch.Tensor,
        a_seq: torch.Tensor,
        h_vars_seq: torch.Tensor,
        d_seq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parallel (training) forward returning both next graph latents and
        next per-variable embeddings.

        Args:
            z_seq      : [B, T, H]
            a_seq      : [B, T, H]
            h_vars_seq : [B, T, V, H]
            d_seq      : [B, T]

        Returns:
            z_pred      : [B, T, H]
            h_vars_pred : [B, T, V, H]
        """
        z_pred = self.forward(z_seq, a_seq, d_seq)
        h_vars_pred = self.var_dynamics(
            h_vars_seq,
            z_pred,
            a_seq,
        )
        return z_pred, h_vars_pred
