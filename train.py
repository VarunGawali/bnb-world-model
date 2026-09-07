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
import torch.multiprocessing as _mp
from torch.utils.data import DataLoader

# Avoid "received 0 items of ancdata" (file-descriptor exhaustion) when many
# DataLoader workers share tensors — common with --with_cuts and large batches.
# The file_system strategy shares via /dev/shm files instead of FDs.
try:
    _mp.set_sharing_strategy("file_system")
except Exception:
    pass

from bnb_wm.model.world_model import BnBWorldModel
from bnb_wm.training.trainer import Trainer
from bnb_wm.training.checkpoint import load_weights_only
from bnb_wm.training.repro import (
    seed_everything, write_provenance, dataset_fingerprint, file_md5,
)
from bnb_wm.data import (
    list_trajectory_files,
    split_files,
    compute_label_stats,
    compute_feature_stats,
    TransitionDataset,
    transition_collate,
    SequenceDataset,
    make_sequence_collate,
)
from bnb_wm.data.datasets import (
    ShardedBatchSampler,
    probe_edge_cost_per_node,
    RawSequenceDataset,
    make_raw_collate,
    CutTransitionDataset,
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
        # Multi-step-rollout stabilisers: residual latent prediction on by
        # default; heteroscedastic (Gaussian-NLL) transition off by default.
        dyn_residual=m.get("dyn_residual", True),
        dyn_heteroscedastic=m.get("dyn_heteroscedastic", False),
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
    ap.add_argument("--phase3_also_train", default="",
                    help="comma-separated module name-prefixes to unfreeze in Phase 3 "
                         "alongside dynamics/dyn_bound/dyn_reward. "
                         "e.g. 'encoder' to do end-to-end dynamics, "
                         "'encoder,policy,value' to also fine-tune those heads. "
                         "Empty (default) = frozen encoder, dynamics only.")
    ap.add_argument("--num_workers", type=int, default=0,
                    help="DataLoader workers for the transition loaders "
                         "(parallel data loading; Phase 3 always uses 0). "
                         "Set to the CPU core count; more oversubscribes.")
    ap.add_argument("--files_per_batch", type=int, default=4,
                    help="ShardedBatchSampler: distinct trajectory files per "
                         "transition batch. Lower = fewer np.load/decompress "
                         "calls (faster loading) but less instance diversity "
                         "per batch; 4 is a good balance. Set 0 to disable "
                         "sharded sampling and use plain shuffle.")
    ap.add_argument("--size_aware", type=int, default=1,
                    help="1 = size-aware (dynamic) batching: --batch_size is the "
                         "nominal count on a MEDIAN graph, and batches auto-grow "
                         "on easy instances / shrink on hard ones to hold a fixed "
                         "edge (memory) budget. Prevents OOM AND keeps throughput "
                         "high. 0 = fixed item count. Ignored if files_per_batch=0.")
    ap.add_argument("--max_epochs", type=int, default=None,
                    help="cap every phase's epochs at this value (fast checks / "
                         "budget control); overrides the config caps when lower")
    ap.add_argument("--batch_size", type=int, default=None,
                    help="override training.batch_size (lower it to fit GPU "
                         "memory on large instances / a shared GPU)")
    ap.add_argument("--seq_batch_size", type=int, default=None,
                    help="separate batch size for Phase 3 (dynamics sequences), "
                         "the memory-heavy phase; defaults to --batch_size. Set "
                         "this smaller so phases 1/2/4/5 can use a large batch.")
    ap.add_argument("--init_checkpoint", default=None,
                    help="warm-start: load these weights before the phase loop, "
                         "e.g. to re-run only --phases 3,4 on top of an existing "
                         "model (reuses phases 1,2). Architecture must match.")
    ap.add_argument("--seed", type=int, default=0,
                    help="base RNG seed for Python/NumPy/torch/CUDA + workers")
    ap.add_argument("--deterministic", action="store_true",
                    help="request deterministic algorithms (slower, exact repro)")
    # ---- Phase 3 encoder / cut-transition options ----
    ap.add_argument("--phase3_train_encoder", action="store_true",
                    help="unfreeze encoder in Phase 3 (joint dynamics+encoder). "
                         "Uses RawSequenceDataset (on-the-fly encoding, no stale "
                         "cache). Requires --encoder_lr_scale to control encoder LR.")
    ap.add_argument("--encoder_lr_scale", type=float, default=0.1,
                    help="encoder LR = lr_phase3 * encoder_lr_scale when "
                         "--phase3_train_encoder is set (default 0.1).")
    ap.add_argument("--encoder_warmup_epochs", type=int, default=5,
                    help="freeze encoder for first N epochs then ramp LR over "
                         "next N epochs; 0=no warmup (default 5).")
    ap.add_argument("--encode_cache_refresh_every", type=int, default=3,
                    help="re-encode full dataset every N epochs when "
                         "--phase3_train_encoder is set; 0=always encode "
                         "on-the-fly (slow). Default 3 gives ~3x speedup "
                         "with low latent staleness.")
    ap.add_argument("--cut_transitions_dir", default=None,
                    help="directory with *_cut.npz files from gen_cut_transitions.py; "
                         "enables cut-dynamics MSE loss during Phase 3.")
    ap.add_argument("--phase3_value_consist_weight", type=float, default=None,
                    help="weight for value-consistency Huber loss in Phase 3 "
                         "(default: config training.v_consist_weight or 0.1).")
    ap.add_argument("--phase3_cut_weight", type=float, default=None,
                    help="weight for cut-transition MSE in Phase 3 "
                         "(default: config training.cut_transition_weight or 0.1).")
    ap.add_argument("--phase3_cf_weight", type=float, default=None,
                    help="weight for counterfactual contrastive loss in Phase 3 "
                         "(default: config training.cf_contrastive_weight or 0.3).")
    ap.add_argument("--phase3_consist_weight", type=float, default=None,
                    help="weight for free-run consistency loss in Phase 3 "
                         "(default: config training.free_run_consist_weight or 0.3).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    phases = [int(p) for p in args.phases.split(",") if p.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # P2.3: seed everything and record provenance before any data touch.
    worker_init_fn = seed_everything(args.seed, deterministic=args.deterministic)
    print(f"Seed: {args.seed} (deterministic={args.deterministic})")

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
    # Provenance now that the file set is known — include the dataset fingerprint
    # (P2.3) so a run and its checkpoints are tied to the exact data + config.
    prov = write_provenance(ckpt_dir, {
        "seed": args.seed,
        "deterministic": args.deterministic,
        "config": args.config,
        "config_md5": file_md5(args.config),
        "data_root": str(data_root),
        "dataset": dataset_fingerprint(files),
        "phases": phases,
        "with_cuts": args.with_cuts,
    })
    print(f"Provenance written to {prov}")

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
        nw = args.num_workers
        common = dict(collate_fn=transition_collate, num_workers=nw,
                      worker_init_fn=worker_init_fn)
        # persistent_workers avoids re-spawning workers (and re-importing torch)
        # every epoch; prefetch/pin_memory overlap loading with GPU compute.
        if nw > 0:
            common.update(persistent_workers=True, prefetch_factor=4)
        if args.files_per_batch and args.files_per_batch > 0:
            # Sharded batching: each batch drawn from a few files, so workers
            # decompress far fewer trajectory files per batch (the loader was
            # GPU-starving otherwise). Randomness preserved via per-epoch
            # file/within-file shuffle in the sampler.
            file_cost = None
            if args.size_aware:
                # Cheap ZIP-directory probe (no decompression) -> edges/node,
                # so batches hold a fixed memory budget: big on easy instances,
                # small on hard ones. Prevents OOM without capping throughput.
                file_cost = probe_edge_cost_per_node(file_list)
            sampler = ShardedBatchSampler(
                [fi for (fi, _t) in ds.index], batch_size=bs,
                files_per_batch=args.files_per_batch, shuffle=shuffle,
                file_node_cost=file_cost)
            return DataLoader(ds, batch_sampler=sampler, **common)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, **common)

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
    if args.init_checkpoint:
        load_weights_only(model, args.init_checkpoint, device=device)
        print(f"Warm-started from {args.init_checkpoint}")
    else:
        # Input standardisation (prenorm): compute per-feature mean/std from the
        # training set and bake them into the encoder buffers before Phase 1.
        # Skipped on warm-start (those weights already carry their own stats).
        fs = compute_feature_stats(tr_files, max_files=200)
        if fs is not None:
            model.encoder.set_feature_stats(*fs)
            print(f"Feature standardisation set (var σ range "
                  f"{fs[1].min():.3g}–{fs[1].max():.3g})")
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

    # ---- Phase 3: dynamics ----
    if 3 in phases:
        print("\n=== Phase 3: Dynamics ===")
        seq_bs = args.seq_batch_size or bs

        train_encoder = args.phase3_train_encoder
        enc_lr_scale   = args.encoder_lr_scale

        if train_encoder:
            # Joint encoder+dynamics: RawSequenceDataset encodes on-the-fly with
            # the CURRENT encoder every step → no stale latent problem.
            print("  [encoder trainable] using RawSequenceDataset (on-the-fly encoding)")
            raw_collate = make_raw_collate()

            def sequence_loader(file_list, shuffle):
                ds = RawSequenceDataset(file_list)
                return DataLoader(ds, batch_size=seq_bs, shuffle=shuffle,
                                  collate_fn=raw_collate, num_workers=0)
        else:
            # Frozen encoder: encode once, cache to disk; later epochs load fast.
            seq_cache  = ckpt_dir / "seq_cache"
            seq_collate = make_sequence_collate(include_vars=True)

            def sequence_loader(file_list, shuffle):
                ds = SequenceDataset(file_list, model, device, include_vars=True,
                                     cache_dir=seq_cache)
                return DataLoader(ds, batch_size=seq_bs, shuffle=shuffle,
                                  collate_fn=seq_collate, num_workers=0)

        # Optional cut-transition MSE loss.
        # Split cut files by trajectory stem to avoid train/val contamination:
        # each *_cut.npz derives from the trajectory file with the same stem,
        # so we look up {stem}_cut.npz for each tr_file / va_file stem.
        cut_tr_loader = None
        cut_val_loader = None
        if args.cut_transitions_dir:
            cut_dir = Path(args.cut_transitions_dir)

            def _cut_files_for(traj_list):
                found = []
                for f in traj_list:
                    stem = Path(f).stem
                    candidate = cut_dir / f"{stem}_cut.npz"
                    if candidate.exists():
                        found.append(candidate)
                return found

            cut_tr_files = _cut_files_for(tr_files)
            cut_va_files = _cut_files_for(va_files)

            if cut_tr_files:
                cut_ds = CutTransitionDataset(cut_tr_files)
                cut_tr_loader = DataLoader(cut_ds, batch_size=seq_bs,
                                           shuffle=True, collate_fn=CutTransitionDataset.collate,
                                           num_workers=0)
                print(f"  Cut transitions train: {len(cut_tr_files)} files → loader ready")
            else:
                print(f"  [warn] --cut_transitions_dir={cut_dir}: no train cut files found; skipping")

            if cut_va_files:
                cut_va_ds = CutTransitionDataset(cut_va_files)
                cut_val_loader = DataLoader(cut_va_ds, batch_size=seq_bs,
                                            shuffle=False, collate_fn=CutTransitionDataset.collate,
                                            num_workers=0)
                print(f"  Cut transitions val:   {len(cut_va_files)} files → loader ready")

        # Resolve per-loss weights: CLI override > config > hard default.
        def _w(cli_val, cfg_key, default):
            if cli_val is not None:
                return cli_val
            return tcfg.get(cfg_key, default)

        also_train = tuple(
            s.strip() for s in args.phase3_also_train.split(",") if s.strip()
        )
        trainer.train_dynamics(
            sequence_loader(tr_files, True),
            sequence_loader(va_files, False),
            epochs=epochs_of("epochs_phase3"), lr=tcfg["lr_phase3"],
            overshoot_depth=tcfg.get("overshoot_depth", 0),
            patience=tcfg.get("patience_phase3", patience),
            also_train=also_train,
            also_train_encoder=train_encoder,
            encoder_lr_scale=enc_lr_scale,
            encoder_warmup_epochs=args.encoder_warmup_epochs if train_encoder else 0,
            encode_cache_refresh_every=args.encode_cache_refresh_every if train_encoder else 0,
            cut_loader=cut_tr_loader,
            cut_val_loader=cut_val_loader,
            v_consist_weight=_w(args.phase3_value_consist_weight, "v_consist_weight", 0.1),
            cut_weight=_w(args.phase3_cut_weight, "cut_transition_weight", 0.1),
            cf_weight=_w(args.phase3_cf_weight, "cf_contrastive_weight", 0.3),
            free_run_weight=_w(args.phase3_consist_weight, "free_run_consist_weight", 0.3),
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
