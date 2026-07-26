# CutWorld — Model Architecture

A faithful walkthrough of the model as implemented in `bnb_wm/model/`
(`encoder.py`, `heads.py`, `dynamics.py`, `world_model.py`). All components
share one latent width `hidden_dim = 128`.

## 1. The big picture

The model is a **learned latent dynamics model of Branch-and-Cut search**. At
any B&B node it:

1. **Encodes** the node (its constraint–variable graph) into a latent state `z`
   and per-variable embeddings `h_vars` — `encoder.py`
2. **Predicts** decision-relevant quantities via lightweight heads (which
   variable to branch on, dual bound, cost-to-go, leaf probability, which cuts
   to add) — `heads.py`
3. **Imagines the future** by rolling the latent state forward under
   hypothetical branching decisions — *without touching the LP solver* —
   `dynamics.py`
4. Uses those imagined rollouts to **score branching candidates** and pick the
   best one — `world_model.rollout_candidate`

## 2. Text architecture diagram

```
                        ┌──────────────────────────────────────────────┐
                        │            ONE B&B NODE (from SCIP)           │
                        │  bipartite graph: variables ── constraints    │
                        │  var feats [Nv,19]   con feats [Nc,5]         │
                        │  edges [2,E]   edge feats [E,3]               │
                        └───────────────────────┬──────────────────────┘
                                                │
              ╔═════════════════════════════════▼═══════════════════════════════╗
              ║                 ENCODER  — BipartiteGNN (encoder.py)             ║
              ║                                                                  ║
              ║   var_proj(19→128)      con_proj(5→128)                          ║
              ║        │                     │                                   ║
              ║        ▼   ×3 layers, each:  ▼                                   ║
              ║   ┌───────────────────────────────────────┐                     ║
              ║   │ GATv2Conv  constraints → variables     │  (uses edge feats)  ║
              ║   │ GATv2Conv  variables → constraints     │  4 attention heads  ║
              ║   │ ReLU + LayerNorm + residual            │                     ║
              ║   └───────────────────────────────────────┘                     ║
              ║        │                                                         ║
              ║        ├──────────────► h_vars [Nv,128]  (per-variable emb)      ║
              ║        │                                                         ║
              ║        ▼                                                         ║
              ║   CrossAttentionPool  (learned [CLS] query attends over vars)    ║
              ║        │                                                         ║
              ║        └──────────────► z [128]          (graph-level latent)    ║
              ╚═══════════════════╤═══════════════════════════╤═════════════════╝
                                  │                           │
             ┌────────────────────┴──────────┐                │
             │        PREDICTION HEADS        │                │
             │           (heads.py)           │                │
             │  ┌──────────────────────────┐  │                │
   h_vars,z ─┼─►│ PolicyHead (Pointer Net) │──┼─► branch var scores  [Nv]
             │  ├──────────────────────────┤  │
        z ───┼─►│ ValueHead    (z‖fracmean)│──┼─► norm. dual bound   [1]
             │  ├──────────────────────────┤  │
        z ───┼─►│ CostToGoHead (softplus)  │──┼─► log(remaining nodes)[1]
             │  ├──────────────────────────┤  │
        z ───┼─►│ SubtreeSizeHead          │──┼─► log(subtree size)  [1]  (gated off)
             │  ├──────────────────────────┤  │
        z ───┼─►│ IntegralityHead(z,depth, │──┼─► P(next = leaf)     [1]
             │  │              n_frac)      │  │
             │  ├──────────────────────────┤  │
 cutfeats,z ─┼─►│ CuttingPlaneHead(Pointer)│──┼─► cut scores        [Ncuts]
             │  └──────────────────────────┘  │
             └────────────────────────────────┘
                                  │
              ╔═══════════════════▼══════════════════════════════════════════╗
              ║        DYNAMICS  — DynamicsTransformer (dynamics.py)          ║
              ║                                                               ║
              ║  token_t = Linear([ z_t ‖ a_t ]) + positional_emb            ║
              ║  a_t = embedding of the branched variable (a column of h_vars)║
              ║                                                               ║
              ║   ┌─────────────────────────────────────────┐  ×4 layers     ║
              ║   │ Causal multi-head self-attention (masked)│                ║
              ║   │ Feed-forward (GELU, 4× width)            │                ║
              ║   └─────────────────────────────────────────┘                ║
              ║        │                                                      ║
              ║        ├──► z_{t+1}     predicted next graph latent           ║
              ║        │                                                      ║
              ║        ▼                                                      ║
              ║   _VarDynamics:  h_i^{t+1} = LN(h_i + MLP([h_i‖z_{t+1}‖a_t])) ║
              ║        └──► h_vars_{t+1}  predicted next per-variable emb      ║
              ║                                                               ║
              ║  grounding heads on the PREDICTED latent:                     ║
              ║    dyn_bound(z_{t+1})  → next dual bound   (anchors dynamics)  ║
              ║    dyn_reward(z_{t+1}) → step reward Δbound (MuZero return)    ║
              ╚═══════════════════════════════════════════════════════════════╝
                                  │
              ╔═══════════════════▼══════════════════════════════════════════╗
              ║   PLANNING — rollout_candidate (world_model.py)               ║
              ║                                                               ║
              ║   for each top-k policy candidate c:                          ║
              ║     branch on c → (z1,h1) → run policy on h1 → next action    ║
              ║       → (z2,h2) → ... depth steps, branch_factor wide         ║
              ║     return = Σ γ^t r_t  +  γ^k V(leaf)  −  w·cost_to_go        ║
              ║   pick argmax-return candidate  ← THE BRANCHING DECISION       ║
              ╚═══════════════════════════════════════════════════════════════╝
```

## 3. Component-by-component

### Encoder — `BipartiteGNN` (encoder.py)

- **Input:** a B&B node as a bipartite graph. Variables carry Ecole's 19-dim
  column features; constraints carry 5-dim row features (padded to 19 to share
  one node tensor). Each edge carries 3 features: raw coefficient, coefficient
  normalized by RHS, and its sign.
- **Message passing:** 3 rounds. Each round does two directed **GATv2Conv**
  passes — constraints→variables, then variables→constraints — with 4 attention
  heads and edge features injected into every message. ReLU + LayerNorm +
  residual per round. (Standard Gasse-style bipartite GNN, upgraded from
  SAGEConv to attention.)
- **Two outputs:**
  - `h_vars [Nv,128]` — one embedding per variable (used for branching and as
    action embeddings).
  - `z [128]` — a graph-level latent from **CrossAttentionPool**: a single
    learned query vector attends over all variable embeddings (a [CLS]-style
    readout) rather than mean-pooling, so structurally important variables
    dominate `z`.

### Prediction heads (heads.py)

- **PolicyHead** — a Pointer Network: `score_i = v·tanh(W_k·h_var_i + W_z·z)`.
  Scores each variable for branching. Trained by imitating strong branching
  (Phase 1).
- **ValueHead** — MLP on `[z ‖ frac_mean]`, where `frac_mean` is the mean
  embedding of currently-fractional variables. Predicts the normalized dual
  bound.
- **CostToGoHead** — MLP + softplus, predicts `log1p(remaining B&B nodes)`. The
  *decision-relevant* value — remaining work — with target
  `steps_to_go = n_steps − t`, which reads off any trajectory (no DFS ordering
  needed).
- **SubtreeSizeHead** — same shape, predicts `log1p(subtree size)`. **Gated
  off** because the collected traces are non-DFS, so exact subtree labels can't
  be derived.
- **IntegralityHead** — MLP on `[z ‖ depth ‖ n_frac]`, predicts P(next node is a
  leaf). Depth and fraction count are strong leaf predictors the GNN can't infer
  alone. Used at inference to *skip rollouts* near leaves (speed).
- **CuttingPlaneHead** — a Pointer Network over candidate cuts (6-dim cut
  features projected to 128, scored against `z`). Makes it branch-and-*cut*.

### Dynamics — `DynamicsTransformer` (dynamics.py)

The "world model" core. The B&B trajectory is treated as a **token sequence**,
one token per node: `token_t = Linear([z_t ‖ a_t])`, where the action embedding
`a_t` is the column of `h_vars` for the branched variable.

- **4-layer causal Transformer decoder** (masked self-attention + GELU FFN,
  learned positional embeddings). Causal masking means it trains in parallel
  over whole trajectories (teacher-forced) but runs autoregressively at
  inference.
- Predicts **`z_{t+1}`** (next graph latent).
- **`_VarDynamics`** predicts **`h_vars_{t+1}`** as a per-variable residual:
  `h_i^{t+1} = LayerNorm(h_i + MLP([h_i ‖ z_{t+1} ‖ a_t]))`. The MLP is shared
  across variables and size-agnostic. *This is the piece that makes a real
  rollout possible* — without predicted per-variable embeddings, the policy
  couldn't be re-applied to an imagined state.
- **Two grounding heads on the predicted latent:** `dyn_bound` (predict next
  dual bound — anchors the latent to a real solver quantity so it doesn't
  drift) and `dyn_reward` (predict the per-step reward = dual-bound improvement,
  enabling the MuZero-style return).
- Inference keeps a **sliding token buffer** capped at `max_seq=512` so long
  solves don't index the positional embedding out of range.

### Planning — `rollout_candidate` (world_model.py)

The inference-time decision procedure, run for each of the top-k policy
candidates:

1. Branch on candidate `c` → dynamics predicts `(z_1, h_vars_1)`.
2. Run the **policy on the predicted `h_vars_1`** to pick the next imagined
   branching variable.
3. Roll forward `depth` steps; at each step optionally expand `branch_factor`
   children and average their continuations (a small imagined tree).
4. Score the rollout with the **return**:

   ```
   R(c) = Σ_t γ^t · r_t   +   γ^k · V(leaf)   −   ctg_weight · cost_to_go
   ```

   (reward from `dyn_reward`, leaf value from ValueHead, cost-to-go from
   CostToGoHead).
5. **Branch on the candidate with the highest return.**

The evaluation ablation rows are exactly this function with pieces switched off
(`branch_factor=1`, `ctg_weight=0`, or `use_reward_return=False`).

## 4. How it's trained (5-phase curriculum)

| Phase | Trains | Frozen | Objective |
|---|---|---|---|
| 1 Policy | encoder + PolicyHead | — | imitate strong branching |
| 2 Value | ValueHead | encoder, policy | regress normalized dual bound |
| 3 Dynamics | DynamicsTransformer (+ `_VarDynamics`, `dyn_bound`, `dyn_reward`) | encoder | predict `z_{t+1}`, `h_vars_{t+1}`; grounding + reward + latent-overshoot losses |
| 4 Joint | all end-to-end | — | + cost-to-go, value-consistency |
| 5 Cuts | CuttingPlaneHead | encoder | imitate SCIP's cut choices |

Phase 3 is the dynamics model — the heart of the "imagining" claim.

## 5. What's on vs. gated off

- **Active:** encoder, policy, value, cost-to-go, integrality, cutting planes,
  full dynamics with per-variable prediction, grounding + reward heads, MuZero
  return, tree rollout.
- **Gated off (no-ops until future work):** `SubtreeSizeHead` (needs DFS traces)
  and `add_global_context` / `global_proj` (zero-initialized; needs per-node
  frontier features to train). Both are wired in but contribute nothing to
  current results — noted here to avoid overclaiming.
