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
