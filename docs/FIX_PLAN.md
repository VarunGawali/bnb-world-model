# CutWorld — Fix & Re-evaluation Plan (later-venue submission)

Goal: a **correct** unified Branch-and-Cut model, trained on all difficulty
tiers at full capacity, evaluated on the metrics below, targeting a genuine
comparison against SCIP (including cutting).

---

## 1. Training data — all tiers, full capacity

- Train on **easy, medium, and hard** instances together (not medium only).
  - easy   ~ 200 x 400
  - medium ~ 500 x 1000
  - hard   ~ 1000 x 2000
- **Stratified splits** so each tier is represented in train/val/test; report
  each tier separately at eval (P1.2).
- **Full training**, not truncated: run every curriculum phase to convergence
  with early stopping on a validation metric, no epoch/time cutoffs that stop
  a phase early. Phase-5 (cuts) is **mandatory**, not optional (P0.8).
- Recollect trajectories with the corrected collector (see §4) before training.

## 2. Metrics (the ones that count)

Report all of these per method, per tier, on a fixed held-out set with
identical time limits and seeds for every baseline (P1.9):

| Metric | Definition |
|---|---|
| **Optimality** | Solution quality vs. proven optimum. A solve counts as "optimal" if its objective is within **5%** of the true optimum (optimality gap <= 5%), not only exact ties. |
| **Nodes explored** | Number of B&B nodes processed. Median reported (heavy-tailed). |
| **Total time** | Wall-clock solve time per instance. |
| **# solved to optimality** | Count / fraction of instances closed to within the 5% band inside the time limit. |
| **Cut metric** | Cutting cost, reported as **total number of cuts applied** and/or **peak memory** used over the whole solve (stored+active cuts, nnz, bytes; peak process/GPU RAM). (P2.5) |

Optimality/solved is the gate: node and time reductions only count on instances
actually solved (to the 5% band) by both the method and the baseline.

## 3. Unified Branch-and-Cut (the headline change)

Today branching is evaluated inside SCIP (no learned cuts) and the learned cut
head only runs in a separate standalone solver on tiny instances — the two never
meet, so the "final model" does no learned cutting.

Fix: **one controller** that does learned branching AND learned cut selection in
the same solve.
- Implement a pyscipopt **separator callback** (`Sepa`) that generates valid cuts
  (Gomory/GMI/MIR/cover — never the old invalid pairwise "intersection >= 1")
  and applies the **learned selection head** in the SAME SCIP solve as learned
  branching. (P1.7, P0.4)
- Same generator / filters / features / normalization at collect, train, val,
  and deploy (P0.6).
- Cut labels from **measured bound gain** (select -> add -> reoptimize LP ->
  record effect), not top-k violation (P0.7, P1.4).

## 4. Confirmed correctness fixes (verified in code)

- **P0.2 Bidirectional edges** — `build_pyg_data` (and solver/benchmark graph
  construction) currently emit constraint->variable edges only, so the encoder's
  variable->constraint pass runs on an empty edge set. Add reverse edges +
  matching edge_attr; unit-test both directions; **retrain**.
- **P0.3 True branch transitions** — dynamics is trained on consecutive *visited*
  nodes (`z_next = z[1:]`), not parent->child branch transitions. Recollect with
  `node_id, parent_id, branch_var, branch_dir` and both child LP states; train
  dynamics only on real parent->child edges.
- **P0.9 History propagation** — children are created with `past_tokens=None`;
  write the updated token history onto each child (or disable history until the
  P0.3 data exists).
- **P1.6 Per-child scoring** — score `x_j=0` and `x_j=1` children separately
  instead of giving both the same priority.
- **P0.11 Leaf / bound targets** — derive `next_is_leaf` from the true post-step
  outcome; dual/value targets from the local LP with a fixed cross-instance norm.
- **P2.6 Masked cosine** — mask the cosine term in `var_reconstruction_loss`.
- **P0.10 CLIs/docs** — restore/repair the missing `bnb_wm.data.generate` entry
  point; one canonical CLI; delete dead scripts.
- **P1.10 Docs** — sync solver docstrings (still claim invalid pairwise CG) to
  Gomory + GATv2 + Transformer.

## 5. Reproducibility & hygiene (P2)

- Seed torch/CUDA/DataLoader/SCIP; log versions, commit hash, dataset/ckpt
  hashes (P2.3).
- Mandatory highspy for cut experiments; fail loud if missing (P2.1).
- Real tests beyond smoke: cut validity, parent-child transitions, bidirectional
  edges, schema, Phase-5 load, SCIP-objective match (P2.4).
- One per-instance benchmark record: obj, gap, proven-optimal, nodes, time,
  cuts, LP iters, peak RAM/GPU (P1.8).

## 6. Blocked-on

- The data collector (`bnb_wm/data/generate.py`) is **not in the repo** — it must
  be recovered from the IITD machine before P0.1/P0.3/P0.4/P0.5/P0.7/P0.11 can be
  fixed, since they live in collection code.
