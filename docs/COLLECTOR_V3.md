# Collector v3 — full-tree + applied-cut collection (design)

Closes the remaining data gaps that require going beyond Ecole's branchable-node
stream: **true leaf outcomes**, **both children including fathomed**, and
**applied-cut next states** (P0.3 / P0.5 / P0.11-leaf). Folded into the planned
recollection so there is only one more collection run.

## Root cause
Ecole's `Branching` environment only surfaces nodes that need a branching
decision. Fathomed children (integer-feasible, infeasible, or bound-pruned) are
never handed back, so:
- leaf labels are a proxy ("no recorded child"), not the real terminal outcome;
- a node's second child is often missing, so both-child transitions aren't guaranteed;
- cuts are only scored in a discarded dive, never applied to a recorded state.

## Chosen architecture — Option B: Ecole + pyscipopt event handler
Augment the validated collector instead of rewriting it.

1. **Keep** the Ecole `Branching` loop for per-node bipartite features + strong
   branching (unchanged, already validated).
2. **Attach** a pyscipopt `Eventhdlr` to the same SCIP model
   (`env.model.as_pyscipopt()`) that logs EVERY processed node:
   `(node_id, parent_id, depth, lower_bound, n_children, outcome)`.
3. **Merge** post-hoc: branchable nodes carry full features; the event log
   supplies the complete tree. Then:
   - `is_leaf[i]` = node i has **no children in the full tree** (true terminal).
   - both-child topology is known from the full tree (even when a child was fathomed).
   - fathomed leaves are represented by (outcome, bound) — no graph needed, they
     never branch.

Rejected Option A (Python-owned B&B rewrite): correct but must re-implement
Ecole's exact features + LP management and is slower — high effort / high risk.

## Applied cuts (P0.5) — separate sub-stage
A pyscipopt `Sepa` (separator) that, at chosen nodes, generates the valid Gomory
pool, applies the selected cuts for real, reoptimizes the LP, and records the
**post-cut** state as the next observation. This is the unified branch-and-cut
data and is the larger piece; it lands after full-tree capture is validated.

## Schema additions (per trajectory)
```
full_node_ids     [M]   every processed node id (M >= recorded T)
full_parent_ids   [M]
full_depths       [M]
full_lower_bounds [M]
full_outcomes     [M]   0=internal 1=integer-feasible 2=infeasible 3=bound-pruned
true_next_is_leaf [T]   recomputed for recorded nodes from the full tree
```
Existing fields are unchanged, so the dataset stays backward compatible; when the
`full_*` arrays are present the dataset uses `true_next_is_leaf` and the full
tree for subtree sizes, else falls back to today's recorded-tree approximation.

## VALIDATION IS MANDATORY (cannot be run off-machine)
Which SCIP event types fire for which nodes (especially bound-pruned nodes) is
version-dependent. On Tyrone, for a few instances, check:
- every recorded branchable node id appears in `full_node_ids`;
- `sum(full_outcomes==1) >= 1` (at least one integer-feasible leaf) once optimal;
- `true_next_is_leaf` differs from the old proxy on some nodes (proves fathomed
  children are now captured);
- node count in `full_node_ids` ≈ SCIP's reported `getNTotalNodes()`.
If bound-pruned nodes are missing, widen the caught event mask (see the handler)
and re-check — this is the write-test-fix loop.

## Staged plan
- **Stage 1** — event handler + full-tree capture + `true_next_is_leaf`; validate on Tyrone.
- **Stage 2** — dataset consumes `full_*` (true leaves, full-tree subtree sizes).
- **Stage 3** — separator for applied cuts + post-cut states (P0.5 unified B&C).
- **Stage 4** — recollect once, validate with `scripts/validate_collection.py`.
