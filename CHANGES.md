# Model Improvements

This document summarises the key changes made to the model relative to the
original notebook implementation (`MTP_model_code_1.ipynb`).

---

## GNN Encoder

- **SAGEConv → GATv2Conv (4 attention heads)**
  SAGEConv averages all neighbour messages equally. GATv2 learns per-edge
  attention weights, so the encoder can focus on the constraints that are
  actually binding at the current LP solution rather than treating all of
  them equally.

- **Added 3-dimensional edge features**
  The original model only saw graph topology. The encoder now receives
  the constraint coefficient, its normalised value (coefficient / RHS), and
  its sign on every edge. This gives the GNN the actual numerical values of
  the LP, which directly determine branching importance.

- **global_mean_pool → CrossAttentionPool**
  Mean pooling treats every variable node as equally important when forming
  the graph-level embedding z. CrossAttentionPool uses a single learnable
  query vector to attend selectively over variable nodes, so the embedding
  can focus on the fractional, high-objective variables that matter most.

---

## Prediction Heads

- **PolicyHead: MLP → Pointer Network**
  The original MLP scored each variable independently with no awareness of
  other candidates. The Pointer Network scores every variable relative to
  the global graph embedding z, which means candidates are compared in
  context rather than in isolation — much closer to how strong branching
  actually works.

- **ValueHead: MLP(z) → MLP(z ‖ frac_mean)**
  The LP dual bound is driven by fractional variables. Appending the mean
  embedding of fractional variable nodes to z gives the value head a direct
  signal about the current fractional state, the strongest predictor of node
  quality.

- **IntegralityHead: MLP(z) → MLP(z ‖ depth ‖ n_frac)**
  Depth and number of fractional variables are the two strongest predictors
  of whether a node is near a leaf. The GNN embedding alone cannot reliably
  infer either (depth has no node-level encoding; n_frac is destroyed by
  mean pooling). Adding them as explicit scalars is a cheap, targeted fix.

- **CuttingPlaneHead (new)**
  A Pointer Network that scores candidate cutting planes against z. Enables
  the model to decide which cuts are worth adding at each B&B node, replacing
  fixed heuristics like maximum violation or minimum density.

---

## Dynamics Model

- **GRUCell → 4-layer Causal Transformer**
  The GRU compresses the entire branching history into a single hidden vector
  which forgets early decisions exponentially. In B&B, early branching
  decisions constrain the entire subtree, so long-range dependencies matter.
  The Transformer attends over the full trajectory so any past token can
  influence the current prediction. It also trains in parallel on full
  trajectories (one forward pass) instead of T sequential GRU steps.

- **True multi-step latent rollout at inference (real world-model lookahead)**
  Earlier the rollout reused the same action embedding at every step and the
  dynamics model only predicted the graph latent z — so it could not choose a
  next action and was really a value re-ranking heuristic. The dynamics model
  now has a per-variable head that also predicts h_vars_{t+1}. Each rollout
  step therefore: (1) predicts z_{t+1} and h_vars_{t+1}, (2) re-runs the
  policy on the predicted state to select the *next* branching variable,
  (3) rolls forward with that chosen action, accumulating discounted value.
  This simulates a genuine branching *sequence* in latent space with no LP
  solves — the actual world-model claim, analogous to MuZero planning on a
  learned model rather than re-encoding real states.

- **Value head trained on its own predicted latents**
  Phase 3 now supervises the per-variable head (reconstruction loss against
  the real encoder outputs) so predicted h_vars stay on the encoder manifold.
  Phase 4 adds a value-consistency term: the value head is required to read
  dynamics-predicted latents the same way it reads real ones. Together these
  remove the distribution shift the value estimates would otherwise face
  during rollout.

- **SubtreeSizeHead — branch to minimise predicted tree growth**
  A new head predicts log1p(subtree node count) rooted at the current node.
  Because the solver's cost *is* node count, this is the decision-relevant
  quantity: during the latent rollout the model reads the predicted subtree
  size at each candidate's immediate child and the branching score becomes
      score = discounted_value  -  size_weight * predicted_subtree_size
  so the solver branches toward the candidate expected to close its subtree
  in the fewest nodes — a latent-space approximation of strong branching's
  subtree evaluation. The head is trained *fully supervised* (Phase 4) on the
  true subtree sizes recorded in the collected B&B traces; no proxy label.

  DATA REQUIREMENT: each per-node meta must carry `subtree_size` (the true
  number of B&B nodes in that node's subtree). It can be computed from an
  existing trace: for a depth-first visitation order, the subtree size of the
  node at position t is the count of subsequent nodes whose recorded depth is
  greater than depth[t], up to the first node whose depth returns to <=
  depth[t] (a standard stack pass over the `depths` array). If the field is
  absent, Phase 4 silently skips the subtree-size term and the rollout uses
  size_weight only if the head has been trained — otherwise set size_weight=0.

---

## Solver

- **Python-owned Branch-and-Cut loop replacing SCIP**
  The neural model previously only replaced SCIP's branching score
  computation. Now it controls all search decisions — branching, cut
  selection, node priority, and near-leaf detection — through a Python
  branch-and-cut loop that uses HiGHS (via scipy) for LP relaxations only.

- **Globally valid Chvátal-Gomory cuts propagated to all descendants**
  Pairwise CG intersection cuts are generated at each node and inherited by
  all descendant nodes. This is true branch-and-cut (not cut-and-branch):
  cuts tighten the LP bound across the entire subtree, not just locally.

- **LP warmstarting via HiGHS direct API**
  Child node LP solves previously cold-started from scratch. With the highspy
  API, the parent's simplex basis is passed to the child; HiGHS repairs it
  via dual simplex, reducing re-solves to typically a few pivots instead of
  a full solve.

---

## Steps Toward the Ideal B&B World Model

These target the gaps between the current model and a MuZero-style B&B world
model (see `RESEARCH_ROADMAP.md`). Each is gated/configurable so the AAAI run
stays reproducible and the untrainable ones are safe no-ops.

- **Gap 3 — cost-to-go value (`CostToGoHead`)**
  Predicts log1p(remaining B&B nodes). Target is the Monte-Carlo return
  `n_steps - t`, which needs no DFS ordering and so trains on the collected
  non-DFS traces (unlike subtree size). The rollout subtracts a discounted
  cost-to-go term (`ctg_weight`); 0 recovers the pure dual-bound-value rollout.
  This replaces the dual-bound *proxy* with the decision-relevant value.

- **Gap 2 — grounded dynamics (`dyn_bound`)**
  A linear head predicts the next node's normalised dual bound from the
  predicted latent, anchoring the dynamics to a real solver quantity. Trained
  in Phase 3 when the loader supplies `bound_next_seq`.

- **Gap 4 — tree rollout (`branch_factor`)**
  The rollout expands the top-`branch_factor` next actions at each step and
  averages child continuations, forming a predicted branching tree instead of a
  single greedy path. `branch_factor=1` reproduces the prior behaviour.

- **Gap 5 — learned node selection (`node_selection: cost_to_go`)**
  Orders the search frontier by each child's predicted cost-to-go (one dynamics
  step ahead) instead of the LP bound — learned best-first search. Node order
  affects only efficiency, never correctness, so exactness holds.

- **Gap 1 — global search-state context (`use_global_context`)**
  Projects scalar frontier/bound features (open-node count, gap, depth,
  incumbent, bounds) and adds them to z. Zero-initialised and gated off: a safe
  no-op until a future run with frontier data fine-tunes the projection.

## Bug Fixes

- **Feature index misalignment between training data and solver**
  Ecole NodeBipartite stores sol_val at index 13 and sol_frac at index 14.
  The trainer was reading index 9 (basis_lower) to compute the fractional
  mask, and the solver was writing LP values to indices 9/10. Both are fixed
  to match the exact Ecole layout, eliminating train/inference distribution
  shift.

- **Edge normalisation using wrong constraint feature**
  The benchmark was normalising edge coefficients by constraint feature
  index 3 (dual_solution_value). The correct index is 1 (bias/RHS).
  Fixed.

---

## Baseline Results (original model, for reference)

Trained on Phase 1 (policy) and Phase 2 (value) only, 300 training files,
8 epochs each:

- Policy Top-1 accuracy: **16.4%**
- Policy Top-3 accuracy: **35.0%**
- Policy Top-5 accuracy: **47.8%**
- Value head Spearman ρ: **0.682**

The improved model should be retrained to measure the delta.
The key result for publication is **node count reduction** during actual solving
compared to SCIP pseudocost branching, with an ablation comparing
policy-only vs +lookahead to justify the world model framing.
