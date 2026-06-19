"""
test_dataset.py — Tests for dataset loading and collation.

Run with: pytest tests/test_dataset.py -v
"""

import numpy as np
import torch
import tempfile
from pathlib import Path
from torch.utils.data import DataLoader

from bnb_wm.data import TransitionDataset, pyg_collate


def _make_fake_npz(path, n_steps=5, n_vars=10, n_cons=6, n_edges=20):
    """Write a minimal fake trajectory .npz file for testing."""
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
    )


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
