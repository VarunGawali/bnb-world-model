# Research Roadmap — World Models for Optimization

This document records the project's north star and the deliberate two-phase
plan, so the near-term B&B work is understood as a stepping stone, not the
destination.

---

## The original question

> Can a **world model** — a learned model of environment dynamics used for
> planning by imagination — be used to *solve* optimization problems?

A world model has exactly one superpower: **when a real step in the
environment is expensive, imagined rollouts let you plan cheaply instead of
acting.** The whole project is about finding where in optimization that
superpower pays off, and putting the world model there.

---

## Three ways to frame optimization as an environment

| Framing | State | Action | Expensive step | World model's role |
|---|---|---|---|---|
| **1. Search tree (B&B)** | a B&B node | which var to branch on | the child LP solve | predict LP outcome without solving |
| **2. Construction** | partial assignment | assign next var | (cheap) | weak fit — nothing costly to amortize |
| **3. Improvement (LNS)** | a complete feasible solution | destroy + repair a neighborhood | the repair (a sub-MIP/LP) | **imagine a move's outcome; plan which neighborhood to destroy** |

- **Framing 1** makes the world model a *subordinate advisor* to a classical
  solver that still does the real work. Ceiling: "learned strong branching
  without LP solves." Legitimate but incremental; crowded field.
- **Framing 2** doesn't need a world model — the real step is cheap.
- **Framing 3** makes the world model the *protagonist*: encode solution →
  imagine consequences of moves → pick the best imagined move → apply → repeat.
  This is Dreamer/MuZero applied to combinatorial optimization, and it
  directly answers the original question. **This is the thesis.**

---

## The decision (2026-07)

- **Now (AAAI, abstract Jul 21 / paper Jul 28):** ship **Framing 1**.
  The learned-latent-rollout branching work is done and defensible. Frame it
  in the paper/abstract as **"instantiation #1 of world-model planning in
  optimization,"** with Framing 3 (LNS) named as the natural extension. The
  abstract is written around the *general* thesis so the door stays open.
- **Later (NeurIPS / ICML / journal):** build **Framing 3** as the real
  contribution — a world model that drives Large Neighborhood Search.

Rationale: the deadline forces a submission and stops the drift; the general
framing gives a north star so the B&B work is explicitly a stepping stone.

---

## What Framing 1 currently is (AAAI submission)

A Python branch-and-cut loop where a neural world model makes the search
decisions. Core claim: **latent-space rollout replaces LP-based strong
branching.** For each candidate the model rolls the learned dynamics forward,
predicting both the graph latent and per-variable embeddings, re-runs the
policy on the predicted state to choose the next action, and scores the
candidate by discounted value minus predicted subtree size.

Headline ablation:

| Method | Branching signal |
|---|---|
| policy-only | current-node score |
| + value rollout | discounted latent value |
| + subtree-size rollout | predicted nodes-to-close |

Result to report: **node-count reduction vs. SCIP pseudocost**, with the
ablation isolating the contribution of the latent rollout.

Status / gaps: no trained weights yet; `bnb_wm/data/` module not built;
subtree-size labels must be added to data collection (recipe in `CHANGES.md`).

---

## What Framing 3 will be (future thesis)

**Loop:** encode a feasible solution + instance → the world model imagines the
objective/feasibility outcome of many candidate destroy-repair moves in latent
space → pick the move with the best imagined improvement → actually repair only
that one → repeat.

**Why it's the right project:**
- The world model *drives* the solve instead of assisting a solver.
- It genuinely uses the superpower: repair is expensive; imagination amortizes it.
- It generalizes across MIP families (any problem with a feasible solution and
  a neighborhood structure), including instances where B&B struggles.
- The story is uncrowded: learned LNS exists as move-*policies*
  (Sonnerat et al. 2021, Song et al.), but **not as world models that imagine
  move outcomes and plan** over them.

**Component reuse from Framing 1:**

| Component | Framing 1 role | Framing 3 role |
|---|---|---|
| Bipartite GNN encoder | encode a B&B node | encode a *solution + instance* |
| Dynamics Transformer | predict next LP latent | predict next *solution* latent after a move |
| Value head | predict dual bound | predict *post-repair objective* |
| Policy / pointer head | pick branching var | pick which variables to *destroy* |
| Subtree-size head | predict nodes-to-close | (repurpose to predict repair cost, or drop) |

**New pipeline needed:** solution trajectories + repair outcomes (destroy-set →
repaired objective), not branching traces. This is the main lift and the reason
it is not attempted before the AAAI deadline.

---

## One-line north star

> A world model that plans in optimization by imagining the consequences of its
> moves. B&B branching is instantiation #1; LNS-driven improvement is the goal.
