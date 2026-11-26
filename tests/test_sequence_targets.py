"""
Phase-3 dynamics targets: the raw-graph path (RawSequenceDataset ->
raw_paths_to_transitions, used by --phase3_train_encoder and the latent cache)
must build exactly the transitions SequenceDataset builds, and must train the
per-variable transition and the reward head.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from bnb_wm.data.datasets import (
    RawSequenceDataset, SequenceDataset, make_raw_collate,
    raw_paths_to_transitions,
)
from bnb_wm.model.world_model import BnBWorldModel
from bnb_wm.training.trainer import Trainer

H, N_VARS, N_CONS, N_EDGES = 32, 10, 6, 20
# Tree: 1 -> {2 (up), 3 (down)}, 2 -> {4 (up), 5 (down)}.
NODE_IDS = [1, 2, 3, 4, 5]
PARENT_IDS = [0, 1, 1, 2, 2]
BRANCH_DIRS = [0, 1, -1, 1, -1]
DUAL_BOUNDS = [10.0, 11.0, 12.5, 11.5, 14.0]


def _write_traj(path):
    rng = np.random.default_rng(0)
    T = len(NODE_IDS)
    np.savez_compressed(
        path,
        n_steps=np.int64(T),
        var_features=rng.standard_normal((T, N_VARS, 19)).astype(np.float32),
        con_features=rng.standard_normal((T, N_CONS, 5)).astype(np.float32),
        edge_indices=np.stack([np.vstack([
            rng.integers(0, N_CONS, N_EDGES), rng.integers(0, N_VARS, N_EDGES),
        ]).astype(np.int64) for _ in range(T)]),
        edge_values=rng.standard_normal((T, N_EDGES)).astype(np.float32),
        branching_vars=np.array([3, 7, 1, 0, 5], dtype=np.int64),
        node_ids=np.array(NODE_IDS, dtype=np.int64),
        parent_ids=np.array(PARENT_IDS, dtype=np.int64),
        branch_dirs=np.array(BRANCH_DIRS, dtype=np.int8),
        dual_bounds=np.array(DUAL_BOUNDS, dtype=np.float64),
        root_bound=np.float64(10.0),
        primal_bound=np.float64(15.0),
        norm_dual_bounds=np.linspace(0, 1, T).astype(np.float32),
    )


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    model = BnBWorldModel(hidden_dim=H, n_gnn_layers=1, n_gnn_heads=4,
                          n_dyn_layers=1, n_dyn_heads=4).eval()
    tmp = tempfile.TemporaryDirectory()
    f = Path(tmp.name) / "traj_0.npz"
    _write_traj(f)
    seq = SequenceDataset([f], model, device="cpu", max_path_len=32)
    raw = RawSequenceDataset([f], max_path_len=32)
    batch = make_raw_collate()([raw[i] for i in range(len(raw))])
    yield model, seq, batch, [path for _, path in raw.index]
    tmp.cleanup()


def _raw_transitions(model, batch):
    gb = batch["batch_graphs"]
    with torch.no_grad():
        h_vars, z = model.encode(gb)
    return raw_paths_to_transitions(
        h_vars, z, gb.batch[gb.node_type == 0], batch["batch_sizes"],
        batch["branch_vars"], batch["dir_seq"], batch["bound_seq"],
        max_vars_recon=64)


def test_raw_path_matches_sequence_dataset(setup):
    model, seq, batch, _ = setup
    tr = _raw_transitions(model, batch)
    assert len(seq) == len(batch["batch_sizes"]) == 3      # three leaves
    for i in range(len(seq)):
        ref = seq[i]
        n = ref["valid_len"]
        assert tr["time_mask"][i].sum().item() == n
        for k in ("z_seq", "a_seq", "z_next_seq"):
            torch.testing.assert_close(tr[k][i, :n], ref[k], atol=1e-5, rtol=1e-5)
        for k in ("dir_seq", "bound_next_seq", "reward_seq"):
            torch.testing.assert_close(tr[k][i, :n], ref[k].float(),
                                       atol=1e-6, rtol=1e-6)
        # N_VARS < max_vars_recon: both keep every variable, in order.
        torch.testing.assert_close(tr["hv_seq"][i, :n, :N_VARS], ref["hv_seq"],
                                   atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(tr["hv_next_seq"][i, :n, :N_VARS],
                                   ref["hv_next_seq"], atol=1e-5, rtol=1e-5)
        assert tr["var_mask"][i, :n, :N_VARS].all()
        assert not tr["var_mask"][i, :n, N_VARS:].any()
        assert not tr["var_mask"][i, n:].any()


def test_transition_alignment_by_hand(setup):
    # Path root(1) -> 2 -> 4: transitions carry the CHILD's direction/bound.
    model, _, batch, paths = setup
    tr = _raw_transitions(model, batch)
    norm = (np.array(DUAL_BOUNDS) - 10.0) / 5.0
    i = paths.index([0, 1, 3])                # recorded steps of nodes 1, 2, 4
    assert not tr["time_mask"][i, 2:].any()   # leaf node 4 has no transition
    assert tr["dir_seq"][i, :2].tolist() == [1.0, 1.0]
    np.testing.assert_allclose(tr["bound_next_seq"][i, :2], norm[[1, 3]], atol=1e-6)
    np.testing.assert_allclose(tr["reward_seq"][i, :2],
                               [norm[1] - norm[0], norm[3] - norm[1]], atol=1e-6)


def test_online_path_trains_var_dynamics_and_reward(setup):
    model, _, batch, _ = setup
    with tempfile.TemporaryDirectory() as ck:
        trainer = Trainer(model, torch.device("cpu"), ck, amp=False)
        model.zero_grad(set_to_none=True)
        loss = trainer._dynamics_batch_loss(batch)
        loss.backward()
    vd = [p.grad for p in model.dynamics.var_dynamics.parameters()]
    assert all(g is not None and g.abs().sum() > 0 for g in vd)
    assert model.dyn_reward.weight.grad is not None
    assert model.dyn_reward.weight.grad.abs().sum() > 0
    assert model.dyn_bound.weight.grad.abs().sum() > 0
