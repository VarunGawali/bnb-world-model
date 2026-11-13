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
    """Single causal multi-head self-attention block (pre-norm)."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [B, T, D]
        B, T, D = x.shape
        residual = x
        x = self.norm(x)

        Q, K, V = self.qkv(x).chunk(3, dim=-1)

        def split(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        Q, K, V = split(Q), split(K), split(V)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn + self.causal_mask[:T, :T]
        attn = self.drop(F.softmax(attn, dim=-1))

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
        return residual + self.proj(out)


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
    Per-variable latent transition head.

    Supports arbitrary leading batch dimensions.

        h_vars: [..., V, H]
        z_next: [..., H]
        a:      [..., H]

    The shared MLP therefore works unchanged for both the original single
    graph rollout and a batched rollout frontier.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h_vars: torch.Tensor,   # [..., V, H]
        z_next: torch.Tensor,   # [..., H]
        a: torch.Tensor,        # [..., H]
    ) -> torch.Tensor:
        z_b = z_next.unsqueeze(-2).expand_as(h_vars)
        a_b = a.unsqueeze(-2).expand_as(h_vars)
        delta = self.net(torch.cat([h_vars, z_b, a_b], dim=-1))
        return self.norm(h_vars + delta)


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
    ) -> torch.Tensor:
        """
        Build input tokens from latents, actions and direction.

        Supports:
            z/a: [B, H]
            z/a: [B, T, H]
            z/a: [..., H]

        d can be scalar, [...], [..., 1], or None.
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

        return self.input_proj(torch.cat([z, a, d], dim=-1))

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
    ) -> torch.Tensor:
        """Run Transformer blocks and decode the final sequence."""
        for attn, ffn in self.layers:
            x = attn(x)
            x = ffn(x)

        feat = self.out_norm(x)
        return self._decode(feat, z_in)

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

        for attn, ffn in self.layers:
            x = attn(x)
            x = ffn(x)

        feat = self.out_norm(x)
        z_pred = self._decode(feat, z_seq)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step inference.

        z_t/a_t may have arbitrary leading batch dimensions ending in H,
        while past_tokens must be [B, T, H] for the common batched path.

        Returns:
            z_next
            new token buffer
        """
        token = self._tokens(z_t, a_t, d_t).unsqueeze(-2)

        if past_tokens is None:
            tokens = token
        else:
            tokens = torch.cat([past_tokens, token], dim=-2)

        if tokens.size(-2) > self.max_seq:
            tokens = tokens[..., -self.max_seq:, :]

        T = tokens.size(-2)
        pos = self.pos_emb(
            torch.arange(T, device=z_t.device)
        ).view(*([1] * (tokens.dim() - 2)), T, self.hidden_dim)
        x = tokens + pos

        for attn, ffn in self.layers:
            x = attn(x)
            x = ffn(x)

        feat = self.out_norm(x[..., -1, :])
        z_next = self._decode(feat, z_t)
        return z_next, tokens

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
            z_cur, tokens = self.step(
                z_cur, a_seq[:, j], tokens, d_j
            )
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

        z_next, tokens = self.step(z_t, a_t, past_tokens, d_t)

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
            z_t, a_t, h_vars_t, past_tokens, d_t
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
