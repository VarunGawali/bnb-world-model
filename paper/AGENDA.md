# Paper Agenda — CutWorld (AAAI full paper)

## Thesis (framing chosen: vision-first, bold but anchored)
> **We propose *world models for optimization* — learning a solver's decision
> dynamics so its discrete choices can be *planned in imagination* — and realize
> the first concrete instance inside Branch-and-Cut.**

Open with the broad vision, immediately anchor in B&C as the first realized
instance, and let a rigorous ablation + honest four-metric evaluation carry the
evidence. Do **not** stake the paper on beating SCIP.

## The genuine technical novelty (what earns the "world model" title)
- **Per-variable latent transition** (`_VarDynamics`): predicts `h_i^{t+1}`, so
  the policy can be re-applied to an *imagined* state — the enabling trick for a
  real rollout. Prior learn-to-branch is model-free imitation with no forward
  model.
- **Grounded, reward-bearing dynamics**: `dyn_bound` (anchors latent to dual
  bound) + `dyn_reward` (per-step reward) enabling a MuZero-style planning return.
- **Cost-to-go value** trained from non-DFS traces (target `T - t`).

## Section plan
1. **Introduction** — vision (solvers as sequential decision makers ripe for
   world models) → anchor in B&C → contributions.
2. **Related work** — learn-to-branch (model-free imitation) vs. model-based RL
   (MuZero/Dreamer). Wedge: *we introduce the forward model.*
3. **Background** — B&C as an MDP (state=node, action=branch/cut, reward=bound
   progress, cost=nodes). Earns the world-model vocabulary.
4. **Method** — encoder → heads → dynamics transformer → per-variable transition
   → grounding/reward → rollout return (Eq. 6) + Algorithm 1 → cuts → 5-phase
   curriculum. (Lift from `docs/ARCHITECTURE.md`.)
5. **Experiments** — four metrics (optimality, nodes, time, cuts); ablation
   ladder; both-solved fair subset; Wilcoxon. Headline deferred until retrain.
6. **Analysis** — dynamics fidelity vs. planning benefit; per-node cost as the
   SCIP gap; gated-off components (subtree size, global context). Turns a mixed
   result into a design-space map.
7. **Conclusion** — reconnect to the vision; LNS as the next instance.

## Metrics (supervisor requirement — report in this order)
1. **Optimality** (%solved) — first, the validity gate.
2. **Nodes** — on the both-solved subset only.
3. **Time** — on the both-solved subset only.
4. **Cuts** — meaningful only with separation on (`--separate` run).

## Honest headline, decided after retrain (one of):
- Monotonic ladder + %solved ≈ SCIP → "latent imagination improves branching."
- Planning helps only after good dynamics → "viability bottlenecked by dynamics
  fidelity; we quantify it." (Still a real contribution.)
- Beats heuristics, not SCIP → "first forward-model approach to B&C, matches
  classical heuristics via pure latent planning."

## Status
- §1–§4: drafted (results-independent). Vision framing added to intro +
  conclusion; contributions name the agenda.
- §5: setup + metrics + baselines + table skeleton done; **numbers pending
  retrain**.
- §6 analysis: to be written around the actual ablation ladder once results land.

## Blocking dependency
Retrained `checkpoints_warm/model_final.pt` → ablation runs (branching + the
`--separate` branch-and-cut run) → fill Table 1 → pick headline → write §6.
