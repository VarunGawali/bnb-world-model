"""
labels.py — Derive training labels that were NOT explicitly collected but can
be recovered from data already present in the trajectory files.

Currently:
    subtree_size  — number of B&B nodes in the subtree rooted at each node.
                    Recovered from the `depths` array; requires no change to
                    the data-collection pipeline.

IMPORTANT — DFS assumption
--------------------------
Subtree size is recoverable from depths ONLY if the nodes were visited in
depth-first pre-order. In a DFS pre-order the depth sequence descends by
exactly +1 (visit a child) or jumps back to any shallower level (backtrack to
an ancestor), but never increases by more than 1.

SCIP's *default* node selection is a best-estimate hybrid, not pure DFS, so a
generic trajectory may violate this. `is_dfs_preorder` detects the violation;
`subtree_sizes_from_depths` returns None when the assumption does not hold so
the caller can fall back to size_weight=0 (pure value rollout) instead of
training on wrong labels.

If a future collection wants exact labels regardless of order, force DFS in
Ecole/SCIP with `nodeselection/dfs/stdpriority` set above the other selectors,
or record node/parent ids and reconstruct the tree directly.
"""

import numpy as np


def steps_to_go(n_steps: int):
    """
    Cost-to-go target (Gap 3): remaining B&B nodes after each step.

        steps_to_go(t) = n_steps - t          for t = 0 .. n_steps-1

    This is a Monte-Carlo return read straight off the trajectory and needs no
    DFS ordering, so it is valid on the collected non-DFS traces. Feed each
    per-node value as meta["steps_to_go"] to train the CostToGoHead.

    Returns:
        np.ndarray [n_steps] of remaining node counts (>= 1, last node = 1).
    """
    n = int(n_steps)
    return (n - np.arange(n, dtype=np.int64))


def is_dfs_preorder(depths) -> bool:
    """
    True if `depths` is consistent with a depth-first pre-order traversal.

    Valid transitions between consecutive visited nodes:
        depth[t+1] == depth[t] + 1   (descend to a child), or
        depth[t+1] <= depth[t]       (backtrack to some ancestor / sibling)
    An increase of more than +1 is impossible in a pre-order walk and proves
    the order is not DFS.
    """
    d = np.asarray(depths).astype(np.int64)
    if d.size <= 1:
        return True
    diff = d[1:] - d[:-1]
    return bool(np.all(diff <= 1))


def subtree_sizes_from_depths(depths):
    """
    Compute the subtree size (inclusive node count) for every node in a
    DFS pre-order trajectory.

    For node t, its subtree is the maximal run of subsequent nodes whose depth
    stays strictly greater than depth[t]; the size is that run length + 1 (the
    node itself). Implemented as a single stack pass in O(T).

    Args:
        depths : sequence of per-node B&B tree depths, in visitation order.

    Returns:
        np.ndarray [T] of subtree sizes (>= 1), or None if `depths` is not a
        valid DFS pre-order (caller should then skip subtree-size supervision).
    """
    d = np.asarray(depths).astype(np.int64)
    T = d.size
    if T == 0:
        return np.zeros(0, dtype=np.int64)
    if not is_dfs_preorder(d):
        return None

    sizes = np.ones(T, dtype=np.int64)
    stack = []  # indices of open ancestors on the current DFS path
    for t in range(T):
        # Pop ancestors that this node is not inside (depth <= their depth).
        while stack and d[t] <= d[stack[-1]]:
            stack.pop()
        # This node adds one to every still-open ancestor's subtree.
        for anc in stack:
            sizes[anc] += 1
        stack.append(t)
    return sizes


def subtree_sizes_from_tree(node_ids, parent_ids):
    """
    Exact recorded-subtree size per node from the true tree links — no DFS
    assumption, unlike `subtree_sizes_from_depths`.

    For each recorded node, returns the number of recorded nodes in the subtree
    rooted there (inclusive: itself + all recorded descendants). Because only
    branchable nodes are recorded, this COUNTS RECORDED NODES ONLY — it
    undercounts fathomed leaves the Ecole stream never yields — but it is
    tree-consistent and strictly better than the visitation-order `n_steps - t`
    proxy for the cost-to-go / subtree-size targets.

    Used for both the CostToGoHead target (nodes remaining to close this node's
    subtree) and the SubtreeSizeHead target, replacing the fabricated proxies.

    Args:
        node_ids, parent_ids : per-recorded-node SCIP ids (parent id may point to
                               an unrecorded fragment root; those links are ignored).
    Returns:
        np.ndarray [T] of inclusive recorded-subtree sizes (>= 1).
    """
    node_ids = np.asarray(node_ids, dtype=np.int64)
    parent_ids = np.asarray(parent_ids, dtype=np.int64)
    T = node_ids.size
    if T == 0:
        return np.zeros(0, dtype=np.int64)

    id2idx = {int(n): i for i, n in enumerate(node_ids)}   # last wins on dup
    children = [[] for _ in range(T)]
    parent_idx = [None] * T
    for i in range(T):
        pi = id2idx.get(int(parent_ids[i]))
        if pi is not None and pi != i:
            children[pi].append(i)
            parent_idx[i] = pi

    # Iterative post-order accumulation so a node's size is added to its parent
    # only after its own subtree is fully counted (safe for any tree shape).
    sizes = np.ones(T, dtype=np.int64)
    order, stack = [], [i for i in range(T) if parent_idx[i] is None]
    seen = set()
    while stack:
        u = stack.pop()
        if u in seen:                      # guard against malformed cycles
            continue
        seen.add(u)
        order.append(u)
        stack.extend(children[u])
    for u in reversed(order):              # children before parents
        if parent_idx[u] is not None:
            sizes[parent_idx[u]] += sizes[u]
    return sizes
