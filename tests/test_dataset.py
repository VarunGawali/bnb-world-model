"""
test_dataset.py — Tests for dataset loading and collation.

Run with: pytest tests/test_dataset.py -v
"""

import numpy as np
import torch
import tempfile
from pathlib import Path
from torch.utils.data import DataLoader

from bnb_wm.data import TransitionDataset, SequenceDataset, pyg_collate
from bnb_wm.data.datasets import _root_to_leaf_paths


def _make_fake_npz(path, n_steps=5, n_vars=10, n_cons=6, n_edges=20,
                   node_ids=None, parent_ids=None):
    """Write a minimal fake trajectory .npz file for testing.

    Pass node_ids/parent_ids to exercise the P0.3 root->leaf path reconstruction;
    omit them to write a legacy file (visitation-order fallback).
    """
    extra = {}
    if node_ids is not None:
        extra["node_ids"] = np.asarray(node_ids, dtype=np.int64)
        extra["parent_ids"] = np.asarray(parent_ids, dtype=np.int64)
    np.savez_compressed(
        path,
        var_features   = np.random.randn(n_steps, n_vars, 19).astype(np.float32),
        con_features   = np.random.randn(n_steps, n_cons, 5).astype(np.float32),
        edge_indices   = np.tile(
            np.vstack([
                np.random.randint(0, n_cons, n_edges),
                np.random.randint(0, n_vars, n_edges),
            ]).astype(np.int64),
            (n_steps, 1, 1),
        ),
        edge_values    = np.random.randn(n_steps, n_edges).astype(np.float32),
        action_sets    = np.array(
            [np.random.choice(n_vars, 4, replace=False) for _ in range(n_steps)],
            dtype=object,
        ),
        local_branching_label = np.zeros(n_steps, dtype=np.int64),
        norm_dual_bounds      = np.random.rand(n_steps).astype(np.float32),
        next_is_leaf          = np.zeros(n_steps, dtype=np.int8),
        branching_vars        = np.random.randint(0, n_vars, n_steps).astype(np.int64),
        n_steps               = n_steps,
        **extra,
    )


class _DummyModel:
    """Stand-in for the encoder-bearing model in path-index tests.

    SequenceDataset builds its (file, path) index from node_ids/parent_ids
    alone — no encoder forward — so index-shape tests need only these attrs.
    """
    hidden_dim = 8

    class _Enc:
        def state_dict(self):
            return {}
    encoder = _Enc()


def test_root_to_leaf_paths_topology():
    # tree: root id=1 -> {id2, id3}; id2 -> {id4, id5}. parent 0 = unrecorded.
    node_ids   = [1, 2, 3, 4, 5]
    parent_ids = [0, 1, 1, 2, 2]
    paths = _root_to_leaf_paths(node_ids, parent_ids, max_path_len=64)
    assert sorted(paths) == sorted([[0, 2], [0, 1, 3], [0, 1, 4]])


def test_root_to_leaf_paths_caps_length_and_drops_singletons():
    node_ids, parent_ids = [1, 2, 3, 4, 5], [0, 1, 1, 2, 2]
    capped = _root_to_leaf_paths(node_ids, parent_ids, max_path_len=2)
    assert all(len(p) <= 2 for p in capped)          # keeps nearest ancestors
    assert _root_to_leaf_paths([5], [0], max_path_len=64) == []   # no transition


def test_root_to_leaf_paths_is_cycle_safe():
    # malformed self-referential parents must not hang or crash.
    assert _root_to_leaf_paths([1, 2], [2, 1], max_path_len=64) == []


def test_sequence_dataset_indexes_one_item_per_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "traj_00000.npz"
        _make_fake_npz(p, n_steps=5, node_ids=[1, 2, 3, 4, 5],
                       parent_ids=[0, 1, 1, 2, 2])
        ds = SequenceDataset([p], _DummyModel(), device="cpu")
        assert len(ds) == 3                          # three root->leaf paths


def test_sequence_dataset_legacy_files_fall_back_to_one_sequence():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "traj_00000.npz"
        _make_fake_npz(p, n_steps=5)                 # no node_ids/parent_ids
        ds = SequenceDataset([p], _DummyModel(), device="cpu")
        assert len(ds) == 1                          # whole visitation order


def test_transition_dataset_length():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "traj_00000.npz"
        _make_fake_npz(p, n_steps=5)

        ds = TransitionDataset([p])
        assert len(ds) == 5


def test_transition_dataset_item_shapes():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "traj_00000.npz"
        _make_fake_npz(p, n_steps=3, n_vars=10, n_cons=6)

        ds = TransitionDataset([p])
        graph, meta = ds[0]

        # x = [n_vars + n_cons, 19]
        assert graph.x.shape == (16, 19)
        assert graph.node_type.shape == (16,)
        assert "n_vars"      in meta
        assert "action_set"  in meta
        assert "local_label" in meta
        assert "norm_db"     in meta
        assert "is_leaf"     in meta


def test_pyg_collate_batch():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "traj_00000.npz"
        _make_fake_npz(p, n_steps=4, n_vars=8, n_cons=4)

        ds = TransitionDataset([p])
        loader = DataLoader(ds, batch_size=2, collate_fn=pyg_collate)
        batch, metas = next(iter(loader))

        assert batch.num_graphs == 2
        assert len(metas) == 2
