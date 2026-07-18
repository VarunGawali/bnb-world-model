#!/usr/bin/env python
"""
smoke_test.py — Validate the whole pipeline on synthetic data (no real dataset).

Generates a handful of tiny trajectory .npz files that match the exact collected
schema, then exercises every moving part on CPU:

    1. data module        — TransitionDataset / SequenceDataset / collates
    2. model forward       — encoder + all heads
    3. all 5 training phases (1 epoch each, tiny model)
    4. the neural B&B solver on a random Set Cover instance
    5. the evaluation metrics

Run:
    python smoke_test.py                 # generate + run everything
    python smoke_test.py --gen-only DIR  # only write synthetic data to DIR
                                         # (then: python train.py --data_root DIR
                                         #        --max_files 8 --phases 1,2,3,4)

A green run here means the code is internally consistent. It does NOT validate
that your real .npz field layout matches (esp. edge_indices orientation) —
that's what the --max_files 8 smoke run on the real data checks.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Synthetic trajectory generation (matches the collected schema exactly)
# ---------------------------------------------------------------------------

def make_trajectory(rng, n_vars=12, n_cons=8, T=None):
    """Build one internally-consistent synthetic trajectory dict."""
    T = T or int(rng.integers(5, 10))

    var_features, con_features = [], []
    edge_indices, edge_values = [], []
    action_sets, branching_vars, local_labels = [], [], []
    depths, cut_features, cut_labels, cut_scores = [], [], [], []
    cut_lhs, cut_rhs, n_cuts = [], [], []

    # A fixed random covering matrix for the instance (edges reused per step).
    A = (rng.random((n_cons, n_vars)) < 0.35).astype(np.float32)
    A[np.arange(n_cons), rng.integers(0, n_vars, n_cons)] = 1.0  # no empty rows

    for t in range(T):
        vf = rng.random((n_vars, 19)).astype(np.float32)
        # index 14 = sol_frac; make a few variables fractional
        vf[:, 14] = 0.0
        frac_ids = rng.choice(n_vars, size=max(2, n_vars // 3), replace=False)
        vf[frac_ids, 14] = rng.uniform(0.1, 0.5, size=len(frac_ids))
        var_features.append(vf)
        con_features.append(rng.random((n_cons, 5)).astype(np.float32))

        ci, vi = np.where(A > 0.5)
        edge_indices.append(np.stack([ci, vi]).astype(np.int64))   # [2, E]
        edge_values.append(A[ci, vi].astype(np.float32))

        # action set = the fractional variables; pick one as the branch var
        aset = np.sort(frac_ids).astype(np.int32)
        action_sets.append(aset)
        lbl = int(rng.integers(0, len(aset)))
        local_labels.append(lbl)
        branching_vars.append(int(aset[lbl]))
        depths.append(int(rng.integers(0, 6)))

        # cuts: sometimes present, sometimes none (exercise both paths)
        nc = int(rng.integers(0, 4))
        n_cuts.append(nc)
        cut_features.append(rng.random((nc, 6)).astype(np.float32))
        cut_labels.append((rng.random(nc) < 0.3).astype(np.float32))
        cut_scores.append(rng.random(nc).astype(np.float32))
        cut_lhs.append((rng.random((nc, n_vars)) < 0.3).astype(np.float32))
        cut_rhs.append(np.ones(nc, dtype=np.float32))

    dual = np.sort(rng.random(T)).astype(np.float32)   # monotone-ish bound
    next_is_leaf = np.zeros(T, dtype=np.float32)
    next_is_leaf[-1] = 1.0
    ndb = (dual - dual.min()) / (dual.max() - dual.min() + 1e-8)

    def obj(a):
        return np.array(a, dtype=object)

    return dict(
        n_steps=np.array(T),
        var_features=obj(var_features), con_features=obj(con_features),
        edge_indices=obj(edge_indices), edge_values=obj(edge_values),
        action_sets=obj(action_sets),
        branching_vars=np.array(branching_vars, dtype=np.int32),
        local_branching_label=np.array(local_labels, dtype=np.int32),
        dual_bounds=dual, norm_dual_bounds=ndb.astype(np.float32),
        next_is_leaf=next_is_leaf, depths=np.array(depths, dtype=np.int32),
        cut_features=obj(cut_features), cut_labels=obj(cut_labels),
        cut_scores=obj(cut_scores), cut_lhs=obj(cut_lhs), cut_rhs=obj(cut_rhs),
        n_cuts=np.array(n_cuts, dtype=np.int32),
    )


def generate(out_dir, n=8, seed=0):
    rng = np.random.default_rng(seed)
    save_dir = Path(out_dir) / "set_cover"
    save_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        np.savez_compressed(save_dir / f"traj_instance_{i}.npz",
                            **make_trajectory(rng))
    print(f"  wrote {n} synthetic trajectories to {save_dir}")


# ---------------------------------------------------------------------------
# Full pipeline exercise
# ---------------------------------------------------------------------------

def run_pipeline(data_dir):
    import torch
    from torch.utils.data import DataLoader

    from bnb_wm.model.world_model import BnBWorldModel
    from bnb_wm.training.trainer import Trainer
    from bnb_wm.data import (
        list_trajectory_files, split_files,
        TransitionDataset, transition_collate,
        SequenceDataset, make_sequence_collate,
    )
    from bnb_wm.evaluate.metrics import topk_accuracy, compute_spearman

    device = torch.device("cpu")
    files = list_trajectory_files(data_dir)
    assert files, "no synthetic files found"
    tr, va, _ = split_files(files, 0.7, 0.3, 0.0)
    va = va or tr
    print(f"[data] {len(files)} files -> {len(tr)} train / {len(va)} val")

    def tloader(fl, cuts=False):
        ds = TransitionDataset(fl, with_cuts=cuts)
        return DataLoader(ds, batch_size=8, shuffle=True,
                          collate_fn=transition_collate)

    # tiny model for speed
    model = BnBWorldModel(hidden_dim=32, n_gnn_layers=2, n_gnn_heads=2,
                          n_dyn_layers=2, n_dyn_heads=2, max_seq=64).to(device)
    ckpt = Path(tempfile.mkdtemp())
    trainer = Trainer(model, device, ckpt, amp=False)

    stages = []

    def stage(name, fn):
        try:
            fn()
            print(f"[PASS] {name}")
            stages.append((name, True))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[FAIL] {name}: {e}")
            stages.append((name, False))

    stage("Phase 1 policy",
          lambda: trainer.train_policy(tloader(tr), tloader(va), epochs=1))
    stage("Phase 2 value",
          lambda: trainer.train_value(tloader(tr), tloader(va), epochs=1))

    def phase3():
        sc = make_sequence_collate(include_vars=True)
        def sl(fl):
            ds = SequenceDataset(fl, model, device, include_vars=True,
                                 max_vars_recon=8)
            return DataLoader(ds, batch_size=4, shuffle=True, collate_fn=sc)
        trainer.train_dynamics(sl(tr), sl(va), epochs=1, overshoot_depth=3)
    stage("Phase 3 dynamics (+overshoot)", phase3)

    stage("Phase 4 joint",
          lambda: trainer.train_joint(tloader(tr), tloader(va), epochs=1))
    stage("Phase 5 cuts",
          lambda: trainer.train_cuts(tloader(tr, True), tloader(va, True),
                                     epochs=1))

    stage("metric: topk_accuracy",
          lambda: topk_accuracy(model, tloader(va), device))
    stage("metric: spearman",
          lambda: compute_spearman(model, tloader(va), device))

    # Solver on a random Set Cover instance (scipy fallback, exercises the
    # rollout, cost-to-go node selection, tree rollout, reward return).
    def solver():
        from bnb_wm.solver.bnb_solver import BnBSolver
        rng = np.random.default_rng(1)
        n, m = 12, 8
        A = (rng.random((m, n)) < 0.4).astype(np.float64)
        A[np.arange(m), rng.integers(0, n, m)] = 1.0
        b = np.ones(m)
        c = rng.uniform(1, 5, n)
        solver = BnBSolver(model, device, time_limit=10, node_limit=50,
                           lookahead_k=3, lookahead_depth=2, branch_factor=2,
                           ctg_weight=1.0, node_selection="cost_to_go",
                           use_reward_return=True)
        res = solver.solve(A, b, c)
        print(f"  solver status={res.status} obj={res.objective:.3f} "
              f"nodes={res.n_nodes}")
        assert res.status in ("optimal", "feasible", "timeout")
    stage("neural B&B solver", solver)

    shutil.rmtree(ckpt, ignore_errors=True)

    print("\n" + "=" * 60)
    n_pass = sum(ok for _, ok in stages)
    for name, ok in stages:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"{n_pass}/{len(stages)} stages passed")
    return n_pass == len(stages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-only", metavar="DIR", default=None,
                    help="only generate synthetic data to DIR, then exit")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    if args.gen_only:
        generate(args.gen_only, n=args.n)
        return

    tmp = tempfile.mkdtemp(prefix="bnbwm_smoke_")
    try:
        print("Generating synthetic data...")
        generate(tmp, n=args.n)
        print("\nRunning pipeline...\n")
        ok = run_pipeline(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
