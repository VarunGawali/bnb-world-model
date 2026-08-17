"""Data loading for the BnB World Model."""

from .labels import (
    steps_to_go,
    is_dfs_preorder,
    subtree_sizes_from_depths,
)
from .datasets import (
    list_trajectory_files,
    split_files,
    compute_label_stats,
    build_pyg_data,
    gap_to_primal_norm,
    compute_feature_stats,
    TransitionDataset,
    transition_collate,
    SequenceDataset,
    make_sequence_collate,
)

# P0.10: `pyg_collate` is the historical name still referenced by
# scripts/evaluate.py and the tests; the canonical implementation is
# `transition_collate`. Export an alias so those imports resolve.
pyg_collate = transition_collate

__all__ = [
    "steps_to_go",
    "is_dfs_preorder",
    "subtree_sizes_from_depths",
    "list_trajectory_files",
    "split_files",
    "compute_label_stats",
    "build_pyg_data",
    "gap_to_primal_norm",
    "compute_feature_stats",
    "TransitionDataset",
    "transition_collate",
    "pyg_collate",
    "SequenceDataset",
    "make_sequence_collate",
]
