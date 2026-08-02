# Code Fixes — adjudicated against the supervisor's review

Verdicts are backed by the actual source (file:line). Legend:
**REAL** = confirmed bug · **REAL(subtle)** = real but narrow · **DONE-in-solver**
= already fixed in deploy path, only collector/docs lag · **CONTEXT** = mischar-
acterised, not a bug as stated · **ENH** = not a bug, a paper-grade improvement.

---

## P0 — must fix before retraining

| # | Verdict | Where | Fix |
|---|---|---|---|
| P0.1 SB indexing | **REAL(subtle)** | `collect_*_benchmark_sets.py:970-976` — shape-based disambiguation `if scores.shape[0]==len(action_set)` is wrong when every var is a candidate (set-cover roots). | Always `best_local = int(scores[action_set].argmax())`; assert `0 <= best_local < len(action_set)`. |
| P0.2 one-way edges | **FIXED** | `datasets.py:147` emitted constraint->variable only; `encoder.py:161` `v2c_mask` selected 0 edges, so constraints never saw variables. | Reverse (variable->constraint) edges + duplicated `edge_attr` added in all three builders — `build_pyg_data`, `benchmark._format_obs`, `bnb_solver` graph construction — so train/deploy match. Unit test `test_build_pyg_data_edges_are_bidirectional` asserts both passes see every edge. Requires retrain (done automatically on the new run). |
| P0.3 wrong transitions | **FIXED (Option A)** | Collector records Ecole `Branching` visitation order; npz has `depths` but no `node_id/parent_id`. `datasets.py` used `z[1:]` as "next". | Collector records `node_ids`/`parent_ids` (v2). `SequenceDataset` now reconstructs every true **root->leaf path** (`_root_to_leaf_paths`) and emits one training sequence per path, so each causal-Transformer position is a real parent->child lineage. One dataset item = one path (was one file); paths capped to `max_path_len=64` nearest ancestors. Legacy files without ids fall back to visitation-order (one sequence). Tested: `tests/test_dataset.py` (path topology, capping, cycle-safety, index counts). |

**P0.12 branch-direction action encoding — FIXED.** The action `a_t` was the *chosen branching variable's* embedding only, with no branch **direction** (up/down), so a node with both children recorded yielded two transitions with identical `(z_t, a_t)` but different `z_{t+1}` — the deterministic head could only learn their average. Fix (per request: append the direction scalar to the action token):
- **Collector:** `NodeIdentity` now also reads `node.getParentBranchings()` and emits `direction` (+1 up / -1 down / 0 root); saved as `branch_dirs` per node. Free — added before the 6k run, no recollection.
- **Dataset:** each transition carries `dir_seq` = the child's incoming direction; collate pads it to `[B,T]`.
- **Model:** `input_proj` widened to `2H+1`; `forward/forward_with_vars/step/rollout/step_full` (and the `world_model` wrappers) take an optional `d_seq`/`d_t` concatenated into the token. Optional + zero-default, so legacy batches and existing inference callers are unaffected.
- **Trainer:** threads `dir_seq` into the dynamics forward and the overshoot rollout.

**Inference rollout — WIRED.** `rollout_candidate` now expands **both** children of every branching: the up branch (`+1`) and the down branch (`-1`), each fed the matching direction token, with their subtrees **summed** (both must be solved) while the average over candidate *variables* (`branch_factor`) is unchanged. The candidate's `subtree_size` estimate now sums over both root children. A flag `expand_both_children` (default `True`) retains the pre-P0.12 single direction-0 child as a "direction-ignored" ablation baseline — note that after direction-conditioned training, direction 0 is off-distribution, so only the default is faithful. Compute per candidate scales ~`(2·branch_factor)^depth`; fine at the small depths used.

Note: widening `input_proj` changes the dynamics checkpoint shape — Phase-3 weights must be retrained (encoder/policy checkpoints unaffected).
| P0.4 invalid cuts | **REAL (serious)** | `collect_*.py:386-455` `_generate_intersection_candidates` builds `sum x_j >= 1` — cuts off feasible points. | Delete it. Use valid Gomory cuts from `bnb_wm/solver/gomory.py` (same generator as deploy). |
| P0.5 cuts not applied | **REAL** | `collect_*.py:919` `separating/maxrounds=0`; cuts only scored in a `startDive`/`endDive` that is undone (`:789-834`). Recorded state never contains a cut. | For cut labels: score valid cuts by add-row -> re-solve LP bound gain (dive is fine for *labels*). For the unified model: apply selected cuts for real via a SCIP separator (see FIX_PLAN §3). |
| P0.6 train/serve cut mismatch | **REAL** | Collector = intersection cuts; deploy (`gomory.py`) = Gomory cuts. Different generators & feature distributions. | One generator (Gomory/GMI) with identical filters/features/normalisation at collect, train, val, deploy. |
| P0.7 cut labels | **CONTEXT** | `collect_*.py:685-700` labels top-k by measured **bound improvement**, not violation (violation only pre-filters, `:454`). | His "top-k violation" wording is off, but labels are moot until cuts are valid (P0.4). Keep bound-gain labels, recompute on valid cuts. |
| P0.8 Phase-5 optional | **FIXED** | Was: `cut_mode=learned` ran with an untrained cut head. | `BnBWorldModel` carries a persisted `cut_head_trained` buffer set True by Phase-5 training and saved in the checkpoint; the solver warns and falls back to `heuristic` when `cut_mode=learned` on a model whose cut head is untrained. |
| P0.9 history not propagated | **REAL** | `bnb_solver.py:358` builds child `Node(...)` with no `past_tokens` (defaults None, `:61`). | Write updated token history onto each child; or disable history until P0.3 data exists (then it's honest single-step). |
| P0.10 broken CLI / missing generate | **FIXED** | Was: `scripts/generate_data.py` imported non-existent `bnb_wm.data.generate`; `scripts/train.py` had a broken one-arg `SequenceDataset` call. | Deleted both dead scripts. Canonical CLIs: `generate_instances.py` (instances), `collect_with_cuts_v2.py` (trajectories), root `train.py` (training). README Quickstart rewritten to match. |
| P0.11 leaf/bound targets | **REAL** | `collect_*.py:1041-1042` `next_is_leaf[-1]=1` only; `:1040` bound min-max normalised **per trajectory**. | Derive real leaf labels from tree (`node with no recorded children` = leaf). Store raw `dual_bounds` + `root_bound` + `primal_bound`; normalise with a fixed cross-instance scheme (e.g. gap to optimum). |

## P1 — needed for paper-grade results

| # | Verdict | Fix |
|---|---|---|
| P1.1 stale frac_mask in lookahead | **REAL** | Imagined states must not use the live fractional mask; use predicted `h_vars` (or predict the mask). |
| P1.2 tier imbalance | **ENH** | Stratified train/val/test per tier; report tiers separately; inverse-frequency sampling. |
| P1.3 independent cut head | **ENH** | Budgeted slate policy (diversity/parallelism + marginal gain) instead of independent per-cut BCE. |
| P1.4 single-cut labels | **ENH** | Record bound gain, LP time, stability, persistence, descendant effect, set-level gain. |
| P1.5 checkpoint by Top-1 | **REAL(minor)** | Gate selection on exactness first, then gap/nodes/time/cuts/memory — not policy Top-1. |
| P1.6 both children same score | **FIXED** | Was: one `child_priority` shared by both children. Now scored per child inside the branch loop, rolling the dynamics forward in each child's branch direction (+1 up / -1 down), reusing the P0.12 direction-conditioned step. |
| P1.7 branch & cut separate | **REAL** | Unified controller: cut slate -> re-solve -> branch, shared search-cost objective (needs P0). |
| P1.8 fragmented benchmarks | **ENH** | One per-instance record: obj, gap, proven-optimal, nodes, time, cuts, LP iters, peak RAM/GPU. |
| P1.9 wrong eval split | **REAL** | Held-out Easy/Medium/Hard files, fixed seeds, identical limits for every baseline. |
| P1.10 stale docs | **FIXED** | `bnb_solver.py` docstrings synced: valid Gomory cuts (not "pairwise CG intersection"), bidirectional GATv2 encoder, direction-conditioned DynamicsTransformer. |

## P2 — reproducibility & quality

| # | Verdict | Fix |
|---|---|---|
| P2.1 silent no-cuts if highspy missing | **REAL** | Mandatory highspy for cut experiments; fail loud. |
| P2.2 warm-start disabled | **REAL** | Enable/validate HiGHS warm-start; report LP iters/time with vs without. |
| P2.3 seeding/provenance | **ENH** | Seed torch/CUDA/DataLoader/SCIP; log versions, commit, dataset/ckpt hashes. |
| P2.4 smoke tests only | **ENH** | Tests: cut validity, parent-child transitions, bidirectional edges, schema, Phase-5 load, SCIP-obj match. |
| P2.5 cut memory unmeasured | **ENH** | Track stored/active cuts, nnz, bytes, peak process/GPU memory. |
| P2.6 masked-cosine bug | **FIXED** | `var_reconstruction_loss` cosine term now averages over valid (`var_mask`) positions only, instead of including zeroed padding rows. |
| P2.7 config ignored | **ENH** | Wire YAML into evaluate/benchmark/solver. |
| P2.8 build backend | **ENH** | `setuptools.build_meta`; clean ecole/pyscipopt/highspy extras. |

---

## Recommended order of execution

1. **Collector rewrite** (this doc's P0.1/P0.3/P0.4/P0.5/P0.6/P0.7/P0.11) — foundation.
2. **Recollect all tiers** with the new collector.
3. **P0.2 reverse edges** + **P2.6 mask** + **P1.6 per-child** + **P1.10 docs** (code-only, quick).
4. **Retrain full capacity** (all tiers, Phase-5 mandatory).
5. **Unified branch-and-cut** separator (P1.7) + **new metrics** (5% optimality, cuts/memory).
6. **Held-out eval** (P1.9) + provenance/tests (P2).
