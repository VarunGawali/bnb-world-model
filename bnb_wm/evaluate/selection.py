"""
selection.py — Checkpoint selection gate (P1.5).

The best model must be chosen by *what the solver actually needs*, in priority
order, NOT by policy Top-1 imitation accuracy (a proxy that a model can win while
solving worse). The gate is lexicographic:

    1. exactness   — fraction of instances solved to proven optimality  (higher)
    2. gap         — mean final optimality gap                          (lower)
    3. nodes       — mean B&B nodes                                     (lower)
    4. time        — mean wall-clock seconds                            (lower)
    5. cuts        — mean cuts added                                    (lower)
    6. memory      — mean peak memory                                   (lower)

A model that proves optimality on more instances always wins, regardless of how
fast a less-exact model is; among equally-exact models the tighter gap wins, and
so on down the list. This matches how a MILP practitioner ranks solvers.
"""

from __future__ import annotations

import math

# Priority order: (metric_key, higher_is_better). Earlier entries dominate.
_GATE = (
    ("exactness", True),
    ("gap",       False),
    ("nodes",     False),
    ("time",      False),
    ("cuts",      False),
    ("memory",    False),
)


def checkpoint_sort_key(metrics: dict) -> tuple:
    """
    Lexicographic sort key for one checkpoint's aggregated metrics; smaller is
    better (use directly with `min`/`sorted`). Missing keys sort worst: absent
    `exactness` -> 0, absent lower-is-better metrics -> +inf.
    """
    key = []
    for name, higher_is_better in _GATE:
        if higher_is_better:
            key.append(-float(metrics.get(name, 0.0)))   # more is better
        else:
            key.append(float(metrics.get(name, math.inf)))  # less is better
    return tuple(key)


def rank_checkpoints(records: dict) -> list:
    """
    Rank checkpoints best-first.

    Args:
        records : {checkpoint_id: aggregated_metrics_dict}
    Returns:
        list of checkpoint_ids ordered best -> worst by the P1.5 gate.
    """
    return sorted(records, key=lambda cid: checkpoint_sort_key(records[cid]))


def select_best_checkpoint(records: dict) -> str:
    """Return the single best checkpoint id under the P1.5 gate."""
    if not records:
        raise ValueError("select_best_checkpoint: no records provided")
    return min(records, key=lambda cid: checkpoint_sort_key(records[cid]))
