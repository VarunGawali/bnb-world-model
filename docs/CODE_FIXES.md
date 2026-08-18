# Code Fixes — adjudicated against the supervisor's review

Verdicts are backed by the actual source (file:line). Legend:
**REAL** = confirmed bug · **REAL(subtle)** = real but narrow · **DONE-in-solver**
= already fixed in deploy path, only collector/docs lag · **CONTEXT** = mischar-
acterised, not a bug as stated · **ENH** = not a bug, a paper-grade improvement.

---

## P0 — must fix before retraining

| # | Verdict | Where | Fix |
|---|---|---|---|
| P0.1 SB indexing | **FIXED** | Was: shape-based disambiguation picked the global argmax. | Collector uses `best_local = int(scores[aset].argmax())` with a range assert (`collect_with_cuts_v2.py`). Invariant covered by `test_sb_local_label_indexes_candidates_only` (label is argmax over candidates, never the global argmax). |
| P0.2 one-way edges | **FIXED** | `datasets.py:147` emitted constraint->variable only; `encoder.py:161` `v2c_mask` selected 0 edges, so constraints never saw variables. | Reverse (variable->constraint) edges + duplicated `edge_attr` added in all three builders — `build_pyg_data`, `benchmark._format_obs`, `bnb_solver` graph construction — so train/deploy match. Unit test `test_build_pyg_data_edges_are_bidirectional` asserts both passes see every edge. Requires retrain (done automatically on the new run). |
| P0.3 wrong transitions | **PARTIAL** | Collector records Ecole `Branching` visitation order; npz has `depths` but no `node_id/parent_id`. `datasets.py` used `z[1:]` as "next". | Path reconstruction real; SequenceDataset now REFUSES id-less files (no silent visitation-order fallback). STILL OPEN: terminal/fathomed children not captured (Ecole yields only branchable nodes), so both child transitions aren't guaranteed — needs SCIP-brancher rewrite. |

**P0.12 branch-direction action encoding — FIXED.** The action `a_t` was the *chosen branching variable's* embedding only, with no branch **direction** (up/down), so a node with both children recorded yielded two transitions with identical `(z_t, a_t)` but different `z_{t+1}` — the deterministic head could only learn their average. Fix (per request: append the direction scalar to the action token):
- **Collector:** `NodeIdentity` now also reads `node.getParentBranchings()` and emits `direction` (+1 up / -1 down / 0 root); saved as `branch_dirs` per node. Free — added before the 6k run, no recollection.
- **Dataset:** each transition carries `dir_seq` = the child's incoming direction; collate pads it to `[B,T]`.
- **Model:** `input_proj` widened to `2H+1`; `forward/forward_with_vars/step/rollout/step_full` (and the `world_model` wrappers) take an optional `d_seq`/`d_t` concatenated into the token. Optional + zero-default, so legacy batches and existing inference callers are unaffected.
- **Trainer:** threads `dir_seq` into the dynamics forward and the overshoot rollout.

**Inference rollout — WIRED.** `rollout_candidate` now expands **both** children of every branching: the up branch (`+1`) and the down branch (`-1`), each fed the matching direction token, with their subtrees **summed** (both must be solved) while the average over candidate *variables* (`branch_factor`) is unchanged. The candidate's `subtree_size` estimate now sums over both root children. A flag `expand_both_children` (default `True`) retains the pre-P0.12 single direction-0 child as a "direction-ignored" ablation baseline — note that after direction-conditioned training, direction 0 is off-distribution, so only the default is faithful. Compute per candidate scales ~`(2·branch_factor)^depth`; fine at the small depths used.

Note: widening `input_proj` changes the dynamics checkpoint shape — Phase-3 weights must be retrained (encoder/policy checkpoints unaffected).
| P0.4 invalid cuts | **REAL (serious)** | `collect_*.py:386-455` `_generate_intersection_candidates` builds `sum x_j >= 1` — cuts off feasible points. | Delete it. Use valid Gomory cuts from `bnb_wm/solver/gomory.py` (same generator as deploy). |
| P0.5 cuts not applied | **REAL** | `collect_*.py:919` `separating/maxrounds=0`; cuts only scored in a `startDive`/`endDive` that is undone (`:789-834`). Recorded state never contains a cut. | For cut labels: score valid cuts by add-row -> re-solve LP bound gain (dive is fine for *labels*). For the unified model: apply selected cuts for real via a SCIP separator (see FIX_PLAN §3). |
| P0.6 train/serve cut mismatch | **PARTIAL** | Support feature aligned: solver now uses `abs(lhs)>1e-9` to match the collector (was `lhs>0.5`). STILL OPEN: candidate distribution differs (collector labels root candidates; deploy generates violated candidates at descendants) — closes with the unified B&C collector (P0.5). | One generator (Gomory/GMI) with identical filters/features/normalisation at collect, train, val, deploy. |
| P0.7 cut labels | **CONTEXT** | `collect_*.py:685-700` labels top-k by measured **bound improvement**, not violation (violation only pre-filters, `:454`). | His "top-k violation" wording is off, but labels are moot until cuts are valid (P0.4). Keep bound-gain labels, recompute on valid cuts. |
| P0.8 Phase-5 optional | **FIXED** | Was: `cut_mode=learned` ran with an untrained cut head. | `BnBWorldModel` carries a persisted `cut_head_trained` buffer set True by Phase-5 training and saved in the checkpoint; the solver warns and falls back to `heuristic` when `cut_mode=learned` on a model whose cut head is untrained. |
| P0.9 history not propagated | **FIXED (option a)** | Was: children built with `past_tokens=None`. | Each child now carries the updated token buffer from a direction-conditioned `dynamics_step_full`, so its latent lookahead has the true history that led to it. Legitimate now that P0.3 gives real parent->child transitions. |
| P0.10 broken CLI / missing generate | **FIXED** | Was: `scripts/generate_data.py` imported non-existent `bnb_wm.data.generate`; `scripts/train.py` had a broken one-arg `SequenceDataset` call. | Deleted both dead scripts. Canonical CLIs: `generate_instances.py` (instances), `collect_with_cuts_v2.py` (trajectories), root `train.py` (training). README Quickstart rewritten to match. |
| P0.11 leaf/bound targets | **PARTIAL** | FIXED: depth=`getDepth()`; per-node target=node-local LP bound (`getLowerbound()`); primal anchor=true optimum from independent solve (`optimal_valid`-gated), else fallback. OPEN: true leaf labels from terminal outcome + both-child (incl. fathomed) capture need the SCIP-brancher rewrite (P0.3/P0.5). |

## P1 — needed for paper-grade results

| # | Verdict | Fix |
|---|---|---|
| P1.1 stale frac_mask in lookahead | **FIXED** | `rollout_candidate` no longer passes the real node's `valid_mask` to value/cost-to-go/subtree heads on imagined states (uses `frac_mask=None`, relying on predicted `h_vars`), and ranks imagined next-branching vars by the policy's scores over all vars instead of the stale mask. `valid_mask` kept for caller compatibility. |
| P1.2 tier imbalance | **PARTIAL** | `split_files` now **stratifies by tier** (parent dir) so SC-easy/medium/hard are proportional in train/val/test (was random-pooled, starving thin SC-hard). Tier-wise *reporting* is post-training (eval harness). |
| P1.3 independent cut head | **ENH** | Budgeted slate policy (diversity/parallelism + marginal gain) instead of independent per-cut BCE. |
| P1.4 single-cut labels | **ENH** | Record bound gain, LP time, stability, persistence, descendant effect, set-level gain. |
| P1.5 checkpoint by Top-1 | **PARTIAL** | `evaluate/selection.py` gate (exactness -> gap -> nodes -> time -> cuts -> memory) exists + tested, but **not yet wired into the trainer** — wiring needs solver-quality numbers, so it lands post-training when the eval harness runs. |
| P1.6 both children same score | **FIXED** | Both node-selection modes now score the **predicted child state** (`z_child`/`h_child`, distinct per branch direction) with `frac_mask=None`. Previously only cost-to-go mode differed; default best-bound mode scored both children with the shared parent latent. |
| P1.7 branch & cut separate | **REAL** | Unified controller: cut slate -> re-solve -> branch, shared search-cost objective (needs P0). |
| P1.8 fragmented benchmarks | **ENH** | One per-instance record: obj, gap, proven-optimal, nodes, time, cuts, LP iters, peak RAM/GPU. |
| P1.9 wrong eval split | **REAL** | Held-out Easy/Medium/Hard files, fixed seeds, identical limits for every baseline. |
| P1.10 stale docs | **FIXED** | Synced `bnb_solver.py` docstrings, **README** (GATv2 + cross-attention [CLS] readout + DynamicsTransformer, not SAGEConv/GRU), and **CHANGES** (valid Gomory/GMI, not "pairwise CG intersection"). |

## P2 — reproducibility & quality

| # | Verdict | Fix |
|---|---|---|
| P2.1 silent no-cuts if highspy missing | **FIXED** | Collector now fails loud at startup (`SystemExit`) if `highspy` is unimportable, instead of silently writing empty cut labels. Opt out with `--allow-missing-highspy` for a deliberate branching-only run. |
| P2.2 warm-start disabled | **REAL** | Enable/validate HiGHS warm-start; report LP iters/time with vs without. |
| P2.3 seeding/provenance | **DONE** | `training/repro.py`: `seed_everything` (Python/NumPy/torch/CUDA + DataLoader `worker_init_fn`) and `write_provenance` (git commit+dirty, versions, argv, seed, config, **dataset fingerprint** [files/bytes/md5] + **config md5**). Wired into `train.py`; provenance.json in the ckpt dir. SCIP-seed capture still open. |
| P2.4 smoke tests only | **PARTIAL** | Added: bidirectional edges, root->leaf path reconstruction, SB-index invariant, selection gate, subtree sizes, **cut validity** (cuts never remove a feasible integer point), **schema** (catches depth-as-node-id). Still open: Phase-5-load, SCIP-obj-match. |
| P2.5 cut memory unmeasured | **ENH** | Track stored/active cuts, nnz, bytes, peak process/GPU memory. |
| P2.6 masked-cosine bug | **FIXED** | `var_reconstruction_loss` cosine term now averages over valid (`var_mask`) positions only, instead of including zeroed padding rows. |
| P2.7 config ignored | **DONE** | `benchmark.apply_config(cfg)` overrides the hardcoded lookahead/rollout constants from the YAML `benchmark:`/`solver:` sections; `run_macro_benchmark(config=...)` applies it. |
| P2.8 build backend | **DONE** | `pyproject.toml` build-backend set to `setuptools.build_meta` (was `setuptools.backends.legacy:build`). Optional-deps extras hygiene still open. |

---

## Recommended order of execution

1. **Collector rewrite** (this doc's P0.1/P0.3/P0.4/P0.5/P0.6/P0.7/P0.11) — foundation.
2. **Recollect all tiers** with the new collector.
3. **P0.2 reverse edges** + **P2.6 mask** + **P1.6 per-child** + **P1.10 docs** (code-only, quick).
4. **Retrain full capacity** (all tiers, Phase-5 mandatory).
5. **Unified branch-and-cut** separator (P1.7) + **new metrics** (5% optimality, cuts/memory).
6. **Held-out eval** (P1.9) + provenance/tests (P2).
