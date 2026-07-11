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

- **Multi-step latent rollout at inference (1-step → 3-step)**
  Previously the Transformer was unrolled only one step per branching
  candidate. It now rolls out 3 steps in latent space per candidate,
  accumulates discounted value estimates, and picks the candidate with the
  best predicted discounted return. This is the core "world model" behaviour:
  simulate consequences of a decision before committing to it.

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
