#!/usr/bin/env python
"""
train.py — End-to-end 5-phase training entry point for the BnB World Model.

Wires config -> data -> model -> trainer and runs the curriculum:

    Phase 1  policy      (imitation from strong branching)
    Phase 2  value       (encoder + policy frozen)
    Phase 3  dynamics    (encoder frozen; pre-encoded latent sequences)
    Phase 4  joint       (end-to-end; includes cost-to-go + value-consistency)
    Phase 5  cuts        (optional; requires --with_cuts and cut fields)

The same model object is carried across phases; after each phase its best
checkpoint is reloaded so the next phase starts from the best weights.

Usage
-----
    python train.py --config configs/default.yaml \
                    --data_root /path/to/data_with_cuts \
                    [--with_cuts] [--phases 1,2,3,4] [--max_files N]

Notes
-----
- Phase 3's SequenceDataset pre-encodes trajectories with the frozen encoder,
  so its loader uses num_workers=0 (the dataset holds the model).
- Run one small smoke pass first (--max_files 8 --config a tiny override) to
  confirm the .npz field layout matches build_pyg_data before the full run.
"""

import argparse
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

from bnb_wm.model.world_model import BnBWorldModel
from bnb_wm.training.trainer import Trainer
from bnb_wm.training.checkpoint import load_weights_only
from bnb_wm.data import (
    list_trajectory_files,
    split_files,
    compute_label_stats,
    TransitionDataset,
    transition_collate,
    SequenceDataset,
    make_sequence_collate,
)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg, device):
    m = cfg["model"]
    model = BnBWorldModel(
        hidden_dim=m["hidden_dim"],
        n_gnn_layers=m["n_gnn_layers"],
        n_gnn_heads=m["n_gnn_heads"],
        n_dyn_layers=m["n_dyn_layers"],
        n_dyn_heads=m["n_dyn_heads"],
        max_seq=m["max_seq"],
    )
    return model.to(device)


def reload_best(model, ckpt_dir, phase, device):
    """Reload a phase's best checkpoint into the model, if it exists."""
    best = ckpt_dir / f"phase{phase}_best.pt"
    if best.exists():
        load_weights_only(model, best, device=device)
        print(f"  Reloaded best weights from {best.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data_root", default=None,
                    help="override paths.data_root (dir of traj_*.npz)")
    ap.add_argument("--phases", default="1,2,3,4",
                    help="comma-separated phases to run, e.g. 1,2,3,4,5")
    ap.add_argument("--with_cuts", action="store_true",
                    help="load cut fields and enable Phase 5")
    ap.add_argument("--max_files", type=int, default=None,
                    help="cap number of trajectory files (fast experiments)")
    ap.add_argument("--num_workers", type=int, default=0,
                    help="DataLoader workers for the transition loaders "
                         "(parallel data loading; Phase 3 always uses 0)")
    ap.add_argument("--max_epochs", type=int, default=None,
                    help="cap every phase's epochs at this value (fast checks / "
                         "budget control); overrides the config caps when lower")
    ap.add_argument("--batch_size", type=int, default=None,
                    help="override training.batch_size (lower it to fit GPU "
                         "memory on large instances / a shared GPU)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    phases = [int(p) for p in args.phases.split(",") if p.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_root = args.data_root or cfg["paths"]["data_root"]
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    files = list_trajectory_files(data_root)
    if not files:
        raise SystemExit(f"No traj_*.npz files found under {data_root}")
    if args.max_files and args.max_files < len(files):
        # Sample a RANDOM subset, not the sorted prefix: the files sort by tier
        # (SC-easy < SC-hard < SC-medium), so a prefix would be one difficulty
        # only. A seeded shuffle keeps the subset representative across tiers.
        import numpy as np
        idx = np.random.default_rng(0).permutation(len(files))[: args.max_files]
        files = [files[i] for i in sorted(idx)]
    tr_files, va_files, _ = split_files(
        files,
        cfg["data"]["train_split"], cfg["data"]["val_split"],
        cfg["data"]["test_split"],
    )
    print(f"Trajectories: {len(files)} total | {len(tr_files)} train | "
          f"{len(va_files)} val")

    tcfg = cfg["training"]
    bs = args.batch_size or tcfg["batch_size"]

    def epochs_of(key):
        e = tcfg[key]
        return min(e, args.max_epochs) if args.max_epochs else e

    def transition_loader(file_list, shuffle):
        ds = TransitionDataset(file_list, with_cuts=args.with_cuts)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                          collate_fn=transition_collate,
                          num_workers=args.num_workers)

    # ---- class-imbalance correction (pos_weight) + early stopping ----
    patience = tcfg.get("patience")
    stats = compute_label_stats(tr_files, with_cuts=args.with_cuts)
    leaf_pw = (torch.tensor(float(stats["leaf_pos_weight"]), device=device)
               if stats["leaf_pos_weight"] else None)
    cut_pw = (torch.tensor(float(stats["cut_pos_weight"]), device=device)
              if stats["cut_pos_weight"] else None)
    print(f"pos_weight | leaf={stats['leaf_pos_weight']} "
          f"cut={stats['cut_pos_weight']} | early-stop patience={patience}")

    # ---- model + trainer ----
    model = build_model(cfg, device)
    trainer = Trainer(model, device, ckpt_dir, amp=tcfg.get("amp", True))

    # ---- Phase 1: policy ----
    if 1 in phases:
        print("\n=== Phase 1: Policy ===")
        trainer.train_policy(
            transition_loader(tr_files, True),
            transition_loader(va_files, False),
            epochs=epochs_of("epochs_phase1"), lr=tcfg["lr_phase1"],
            patience=patience,
        )
        reload_best(model, ckpt_dir, 1, device)

    # ---- Phase 2: value ----
    if 2 in phases:
        print("\n=== Phase 2: Value ===")
        trainer.train_value(
            transition_loader(tr_files, True),
            transition_loader(va_files, False),
            epochs=epochs_of("epochs_phase2"), lr=tcfg["lr_phase2"],
            patience=patience,
        )
        reload_best(model, ckpt_dir, 2, device)

    # ---- Phase 3: dynamics (pre-encoded sequences) ----
    if 3 in phases:
        print("\n=== Phase 3: Dynamics ===")
        seq_collate = make_sequence_collate(include_vars=True)

        def sequence_loader(file_list, shuffle):
            ds = SequenceDataset(file_list, model, device, include_vars=True)
            # num_workers=0: the dataset holds the (unpicklable) model.
            return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                              collate_fn=seq_collate, num_workers=0)

        trainer.train_dynamics(
            sequence_loader(tr_files, True),
            sequence_loader(va_files, False),
            epochs=epochs_of("epochs_phase3"), lr=tcfg["lr_phase3"],
            overshoot_depth=tcfg.get("overshoot_depth", 0),
            patience=patience,
        )
        reload_best(model, ckpt_dir, 3, device)

    # ---- Phase 4: joint fine-tune ----
    if 4 in phases:
        print("\n=== Phase 4: Joint ===")
        trainer.train_joint(
            transition_loader(tr_files, True),
            transition_loader(va_files, False),
            epochs=epochs_of("epochs_phase4"), lr=tcfg["lr_phase4"],
            pos_weight=leaf_pw, patience=patience,
        )
        reload_best(model, ckpt_dir, 4, device)

    # ---- Phase 5: cuts (optional) ----
    if 5 in phases:
        if not args.with_cuts:
            print("\n[skip] Phase 5 requested but --with_cuts not set.")
        else:
            print("\n=== Phase 5: Cut selection ===")
            trainer.train_cuts(
                transition_loader(tr_files, True),
                transition_loader(va_files, False),
                epochs=epochs_of("epochs_phase5"), lr=tcfg["lr_phase5"],
                pos_weight=cut_pw, patience=patience,
            )
            reload_best(model, ckpt_dir, 5, device)

    # ---- final save ----
    final = ckpt_dir / "model_final.pt"
    torch.save({"model": model.state_dict()}, final)
    print(f"\nDone. Final model saved to {final}")


if __name__ == "__main__":
    main()
