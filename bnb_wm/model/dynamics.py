"""
dynamics.py — Causal Transformer latent dynamics model.

Replaces the original GRUCell with a causal Transformer decoder that
treats the B&B trajectory as a token sequence.

Each token = concat(z_t, a_t) where z_t is the graph embedding at step t
and a_t is the embedding of the branching action taken at step t.

A causal (masked) self-attention layer ensures the model only attends to
past context, so it can be used auto-regressively at inference while still
being trained in parallel on full trajectories.

At inference, past key-value pairs are cached so each new step costs O(1)
transformer work rather than O(T).

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
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = self.head_dim ** -0.5

        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # Fixed causal mask (upper-triangular = -inf)
        mask = torch.triu(torch.full((max_seq, max_seq), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [B, T, D]
        B, T, D = x.shape
        residual = x
        x = self.norm(x)

        Q, K, V = self.qkv(x).chunk(3, dim=-1)
        # Reshape to [B, n_heads, T, head_dim]
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
        self.net  = nn.Sequential(
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

    Predicts how each variable-node embedding evolves after a branching
    decision. Between two consecutive B&B nodes the variable *set* is
    unchanged (only bounds tighten), so the update is naturally expressed
    as a per-variable residual conditioned on:

        - the variable's current embedding   h_var_i^t
        - the predicted next graph latent     z_{t+1}
        - the branching action embedding      a_t

        h_var_i^{t+1} = h_var_i^t + MLP([h_var_i^t || z_{t+1} || a_t])

    The MLP is shared across variables and count-agnostic, so it applies
    to any number of variables and any problem size. This is the head that
    lets the policy be re-run on a *predicted* future state — the missing
    ingredient for a real latent rollout.
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
        # Broadcast graph-level z_next and action a across the V variables.
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

    Inference (auto-regressive, O(1) per step):
        Maintain a growing buffer of past tokens; feed the full buffer
        and read the last output position.

    Args:
        hidden_dim : must match encoder's hidden_dim
        n_layers   : transformer depth (default 4)
        n_heads    : attention heads   (default 4)
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
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project [z_t || a_t] (2*hidden_dim) -> d_model
        self.input_proj = nn.Linear(2 * hidden_dim, hidden_dim)

        # Learned positional embeddings
        self.pos_emb = nn.Embedding(max_seq, hidden_dim)

        # Transformer layers (each = causal attention + FFN)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                _CausalSelfAttention(hidden_dim, n_heads, max_seq, dropout),
                _FFN(hidden_dim, dropout),
            ])
            for _ in range(n_layers)
        ])

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Per-variable transition head (enables real latent rollout)
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
        """Predict h_vars_{t+1} given current h_vars, predicted z_{t+1}, action a."""
        return self.var_dynamics(h_vars, z_next, a)

    def forward(
        self,
        z_seq: torch.Tensor,
        a_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parallel (training) forward over a full trajectory.

        Args:
            z_seq : [B, T, H]  graph embeddings at steps 0..T-1
            a_seq : [B, T, H]  action embeddings at steps 0..T-1

        Returns:
            z_pred : [B, T, H]  predicted z at steps 1..T
                     z_pred[:, t, :] is the prediction for z_{t+1}
        """
        B, T, _ = z_seq.shape
        tokens = self.input_proj(torch.cat([z_seq, a_seq], dim=-1))  # [B, T, H]
        pos    = self.pos_emb(torch.arange(T, device=z_seq.device))  # [T, H]
        x = tokens + pos

        for attn, ffn in self.layers:
            x = attn(x)
            x = ffn(x)

        return self.out_proj(self.out_norm(x))   # [B, T, H]

    def step(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        past_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step inference. Maintains an explicit token buffer so
        this is functionally equivalent to a cached KV implementation.

        Args:
            z_t         : [B, H]        current graph embedding
            a_t         : [B, H]        current action embedding
            past_tokens : [B, t, H]     buffer of previous tokens (or None)

        Returns:
            z_next      : [B, H]        predicted next embedding
            new_tokens  : [B, t+1, H]   updated token buffer
        """
        token = self.input_proj(
            torch.cat([z_t, a_t], dim=-1)
        ).unsqueeze(1)                             # [B, 1, H]

        if past_tokens is None:
            tokens = token
        else:
            tokens = torch.cat([past_tokens, token], dim=1)  # [B, t+1, H]

        T = tokens.size(1)
        pos = self.pos_emb(torch.arange(T, device=z_t.device))
        x = tokens + pos

        for attn, ffn in self.layers:
            x = attn(x)
            x = ffn(x)

        z_next = self.out_proj(self.out_norm(x[:, -1, :]))   # [B, H]
        return z_next, tokens

    def step_full(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        h_vars_t: torch.Tensor,
        past_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-step inference that also predicts the next per-variable
        embeddings — the primitive for a *real* latent rollout.

        Args:
            z_t         : [B, H]        current graph embedding
            a_t         : [B, H]        current action embedding
            h_vars_t    : [V, H]        current per-variable embeddings
            past_tokens : [B, t, H]     token buffer (or None)

        Returns:
            z_next      : [B, H]        predicted next graph embedding
            h_vars_next : [V, H]        predicted next per-variable embeddings
            new_tokens  : [B, t+1, H]   updated token buffer
        """
        z_next, tokens = self.step(z_t, a_t, past_tokens)
        # Predict per-variable evolution conditioned on the new graph latent
        # and the action taken. z_next[0]/a_t[0]: single-graph rollout.
        h_vars_next = self.var_dynamics(h_vars_t, z_next[0], a_t[0])
        return z_next, h_vars_next, tokens

    def forward_with_vars(
        self,
        z_seq: torch.Tensor,
        a_seq: torch.Tensor,
        h_vars_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parallel (training) forward returning BOTH the next graph latents and
        the next per-variable embeddings.

        Args:
            z_seq      : [B, T, H]      graph embeddings at steps 0..T-1
            a_seq      : [B, T, H]      action embeddings at steps 0..T-1
            h_vars_seq : [B, T, V, H]   per-variable embeddings at steps 0..T-1
                         (padded to a common V; caller supplies a var mask
                          when computing the reconstruction loss)

        Returns:
            z_pred      : [B, T, H]      predicted z at steps 1..T
            h_vars_pred : [B, T, V, H]   predicted h_vars at steps 1..T
        """
        z_pred = self.forward(z_seq, a_seq)                          # [B, T, H]
        h_vars_pred = self.var_dynamics(h_vars_seq, z_pred, a_seq)   # broadcast over V
        return z_pred, h_vars_pred
