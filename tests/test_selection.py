"""Tests for the P1.5 checkpoint selection gate (pure Python, no torch)."""

from bnb_wm.evaluate.selection import (
    select_best_checkpoint, rank_checkpoints, checkpoint_sort_key,
)


def test_exactness_dominates_speed():
    # A faster, fewer-nodes model that proves fewer optima must LOSE.
    records = {
        "fast_inexact": {"exactness": 0.80, "gap": 0.05, "nodes": 100, "time": 1.0},
        "slow_exact":   {"exactness": 1.00, "gap": 0.00, "nodes": 500, "time": 9.0},
    }
    assert select_best_checkpoint(records) == "slow_exact"


def test_gap_breaks_exactness_ties():
    records = {
        "a": {"exactness": 1.0, "gap": 0.02, "nodes": 100, "time": 1.0},
        "b": {"exactness": 1.0, "gap": 0.01, "nodes": 200, "time": 2.0},
    }
    assert select_best_checkpoint(records) == "b"       # tighter gap wins


def test_nodes_break_gap_ties():
    records = {
        "a": {"exactness": 1.0, "gap": 0.0, "nodes": 300, "time": 1.0},
        "b": {"exactness": 1.0, "gap": 0.0, "nodes": 150, "time": 5.0},
    }
    assert select_best_checkpoint(records) == "b"       # fewer nodes wins


def test_missing_metrics_sort_worst():
    records = {
        "complete": {"exactness": 0.9, "gap": 0.1, "nodes": 100, "time": 1.0},
        "empty":    {},
    }
    assert select_best_checkpoint(records) == "complete"
    assert rank_checkpoints(records) == ["complete", "empty"]


def test_sort_key_is_min_better():
    better = checkpoint_sort_key({"exactness": 1.0, "gap": 0.0, "nodes": 10})
    worse  = checkpoint_sort_key({"exactness": 0.5, "gap": 0.2, "nodes": 99})
    assert better < worse
