"""
train_highs.py — Orchestrate all training phases on HiGHS trajectory data.

Usage:
    python scripts/train_highs.py \
        --data_dir data/highs_trajectories \
        --ckpt_dir checkpoints/highs_retrain \
        [--warm_start checkpoints/phase3_best.pt] \
        [--phase 1 2 3 4] \
        [--cut_transitions_dir data/cut_transitions] \
        [--phase3_train_encoder]   # joint encoder+dynamics (recommended)

The script reads configs/default.yaml and allows per-field overrides via
--key value (dot-separated path, e.g. --training.lr_phase1 5e-4).

Warm-start: loads the GNN encoder + policy head weights, then re-initialises
dynamics + value_head + integrality_logit so Phase 3 trains a fresh dynamics
model while keeping strong policy priors.
"""

import argparse
import sys
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, random_split

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bnb_wm.model.world_model import BnBWorldModel
from bnb_wm.data.datasets import (
    TransitionDataset,
    SequenceDataset,
    RawSequenceDataset,
    CutTransitionDataset,
)
from bnb_wm.training.trainer import Trainer
from bnb_wm.training.checkpoint import load_weights_only


# ── helpers ───────────────────────────────────────────────────────────────────

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _apply_overrides(cfg: dict, overrides: list[str]):
    """Apply --key value pairs (key may be dot-separated) to cfg in-place."""
    i = 0
    while i < len(overrides):
        key = overrides[i].lstrip("-")
        val = overrides[i + 1]
        i += 2
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        # Try to parse as number/bool
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                if val.lower() in ("true", "false"):
                    val = val.lower() == "true"
        d[parts[-1]] = val


def _build_model(cfg: dict, device: torch.device) -> BnBWorldModel:
    mc = cfg["model"]
    return BnBWorldModel(
        hidden_dim=mc["hidden_dim"],
        n_gnn_layers=mc["n_gnn_layers"],
        n_gnn_heads=mc["n_gnn_heads"],
        n_dyn_layers=mc["n_dyn_layers"],
        n_dyn_heads=mc.get("n_dyn_heads", 4),
        max_seq=mc.get("max_seq", 512),
        dyn_residual=mc.get("dyn_residual", True),
        dyn_heteroscedastic=mc.get("dyn_heteroscedastic", False),
    ).to(device)


def _warm_start(model: BnBWorldModel, ckpt_path: Path, device: torch.device):
    """Load encoder + policy; reset dynamics, value_head, integrality weights."""
    print(f"[warm_start] loading from {ckpt_path}")
    load_weights_only(model, ckpt_path, device=device, strict=False)

    # Re-initialise components that need fresh training on HiGHS data.
    def _reset(module):
        for p in module.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    _reset(model.dynamics)
    _reset(model.value_head)
    if hasattr(model, "integrality_logit"):
        torch.nn.init.zeros_(model.integrality_logit)
    if hasattr(model, "integrality_head"):
        _reset(model.integrality_head)
    print("[warm_start] dynamics + value_head + integrality reset to random")


def _split_files(files: list[Path], train_frac: float, val_frac: float, seed: int):
    rng = random.Random(seed)
    files = list(files)
    rng.shuffle(files)
    n = len(files)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    return files[:n_train], files[n_train:n_train + n_val], files[n_train + n_val:]


def _transition_loaders(data_dirs: list[Path], cfg: dict, seed: int):
    """Build TransitionDataset train/val loaders from .npz files in one or more dirs."""
    files = []
    for d in data_dirs:
        files.extend(sorted(d.glob("**/*.npz")))
    files = [f for f in files if not f.name.endswith("_cut.npz")]
    if not files:
        raise FileNotFoundError(f"No .npz trajectory files found in {data_dirs}")
    print(f"Found {len(files)} trajectory files for TransitionDataset "
          f"(from {len(data_dirs)} dir(s))")

    dc   = cfg["data"]
    tc   = cfg["training"]
    tr_f, va_f, _ = _split_files(
        files, dc["train_split"], dc["val_split"], seed
    )

    tr_ds = TransitionDataset(tr_f)
    va_ds = TransitionDataset(va_f)
    bs    = tc["batch_size"]

    def _collate(batch):
        from torch_geometric.data import Batch
        graphs, metas = zip(*batch)
        return Batch.from_data_list(graphs), list(metas)

    tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True,
                           collate_fn=_collate, num_workers=2, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=bs, shuffle=False,
                           collate_fn=_collate, num_workers=2, pin_memory=True)
    return tr_loader, va_loader


def _sequence_loaders(data_dirs: list[Path], cfg: dict, seed: int,
                      raw: bool = False):
    """Build Sequence(Raw)Dataset train/val loaders."""
    files = []
    for d in data_dirs:
        files.extend(sorted(d.glob("**/*.npz")))
    files = [f for f in files if not f.name.endswith("_cut.npz")]
    if not files:
        raise FileNotFoundError(f"No .npz trajectory files found in {data_dirs}")

    dc = cfg["data"]
    tc = cfg["training"]
    tr_f, va_f, _ = _split_files(files, dc["train_split"], dc["val_split"], seed)

    DS = RawSequenceDataset if raw else SequenceDataset
    tr_ds = DS(tr_f)
    va_ds = DS(va_f)
    bs    = max(4, tc["batch_size"] // 4)  # sequences are longer; shrink batch

    def _collate_seq(batch):
        from torch.utils.data.dataloader import default_collate
        if isinstance(batch[0], dict):
            keys = batch[0].keys()
            out = {}
            for k in keys:
                vals = [b[k] for b in batch]
                try:
                    out[k] = torch.stack(vals) if isinstance(vals[0], torch.Tensor) else vals
                except Exception:
                    out[k] = vals
            return out
        return default_collate(batch)

    tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True,
                           collate_fn=_collate_seq, num_workers=2)
    va_loader = DataLoader(va_ds, batch_size=bs, shuffle=False,
                           collate_fn=_collate_seq, num_workers=2)
    return tr_loader, va_loader


def _cut_loader(cut_dir: Path, cfg: dict):
    if cut_dir is None or not cut_dir.exists():
        return None
    files = sorted(cut_dir.glob("**/*_cut.npz"))
    if not files:
        return None
    print(f"Found {len(files)} cut-transition files")
    ds = CutTransitionDataset(files)
    bs = max(4, cfg["training"]["batch_size"] // 4)
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=1)


# ── main phases ───────────────────────────────────────────────────────────────

def run(args, cfg, device):
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = _build_model(cfg, device)
    tc    = cfg["training"]

    if args.warm_start:
        _warm_start(model, Path(args.warm_start), device)

    trainer = Trainer(model, device, ckpt_dir, amp=tc["amp"])
    seed    = 42

    phases   = set(args.phase) if args.phase else {1, 2, 3, 4}
    data_dirs = [Path(d) for d in args.data_dir]

    # ── Phase 1: Policy ───────────────────────────────────────────────────────
    if 1 in phases:
        print("\n" + "=" * 60)
        print("PHASE 1 — Policy imitation (HiGHS strong branching labels)")
        print("=" * 60)
        tr_l, va_l = _transition_loaders(data_dirs, cfg, seed)
        trainer.train_policy(
            tr_l, va_l,
            epochs=tc["epochs_phase1"],
            lr=tc["lr_phase1"],
            patience=tc.get("patience", 6),
        )
        print("[Phase 1] Done. Best checkpoint: phase1_best.pt")

    # ── Phase 2: Value ────────────────────────────────────────────────────────
    if 2 in phases:
        print("\n" + "=" * 60)
        print("PHASE 2 — Value head (encoder frozen)")
        print("=" * 60)
        # Load best phase1 if it exists and we trained it this run.
        p1_ckpt = ckpt_dir / "phase1_best.pt"
        if p1_ckpt.exists() and 1 in phases:
            load_weights_only(model, p1_ckpt, device=device, strict=False)
            print(f"  Loaded {p1_ckpt}")
        tr_l, va_l = _transition_loaders(data_dirs, cfg, seed)
        trainer.train_value(
            tr_l, va_l,
            epochs=tc["epochs_phase2"],
            lr=tc["lr_phase2"],
            patience=tc.get("patience", 6),
        )
        print("[Phase 2] Done. Best checkpoint: phase2_best.pt")

    # ── Phase 3: Dynamics (joint encoder+dynamics when --phase3_train_encoder) ─
    if 3 in phases:
        print("\n" + "=" * 60)
        mode = "joint encoder+dynamics" if args.phase3_train_encoder else "dynamics (encoder frozen)"
        print(f"PHASE 3 — Dynamics Transformer ({mode})")
        print("=" * 60)
        p2_ckpt = ckpt_dir / "phase2_best.pt"
        if p2_ckpt.exists() and 2 in phases:
            load_weights_only(model, p2_ckpt, device=device, strict=False)
            print(f"  Loaded {p2_ckpt}")

        raw = args.phase3_train_encoder
        tr_l, va_l = _sequence_loaders(data_dirs, cfg, seed, raw=raw)
        cut_l = _cut_loader(
            Path(args.cut_transitions_dir) if args.cut_transitions_dir else None,
            cfg,
        )

        trainer.train_dynamics(
            tr_l, va_l,
            epochs=tc["epochs_phase3"],
            lr=tc["lr_phase3"],
            patience=tc.get("patience_phase3", tc.get("patience", 12)),
            overshoot_depth=tc.get("overshoot_depth", 3),
            cand_rank_weight=tc.get("cand_rank_weight", 0.5),
            cut_loader=cut_l,
            cut_weight=tc.get("cut_transition_weight", 0.1),
            v_consist_weight=tc.get("v_consist_weight", 0.1),
            also_train_encoder=raw,
            encoder_lr_scale=tc.get("encoder_lr_scale", 0.1),
        )
        print("[Phase 3] Done. Best checkpoint: phase3_best.pt")

    # ── Phase 4: Joint fine-tune ───────────────────────────────────────────────
    if 4 in phases:
        print("\n" + "=" * 60)
        print("PHASE 4 — Joint fine-tune (all parameters)")
        print("=" * 60)
        p3_ckpt = ckpt_dir / "phase3_best.pt"
        if p3_ckpt.exists() and 3 in phases:
            load_weights_only(model, p3_ckpt, device=device, strict=False)
            print(f"  Loaded {p3_ckpt}")
        tr_l, va_l = _transition_loaders(data_dirs, cfg, seed)
        trainer.train_joint(
            tr_l, va_l,
            epochs=tc["epochs_phase4"],
            lr=tc["lr_phase4"],
            patience=tc.get("patience", 6),
        )
        print("[Phase 4] Done. Best checkpoint: phase4_best.pt")

    print("\nAll selected phases complete.")
    print(f"Checkpoints saved to: {ckpt_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True, nargs="+",
                        help="One or more directories of HiGHS trajectory .npz files "
                             "(e.g. --data_dir data/highs_trajectories/easy "
                             "data/highs_trajectories/medium data/highs_trajectories/hard)")
    parser.add_argument("--ckpt_dir", default="checkpoints/highs_retrain",
                        help="Where to save checkpoints")
    parser.add_argument("--warm_start", default=None,
                        help="Path to existing checkpoint to warm-start from "
                             "(encoder + policy loaded; dynamics reset)")
    parser.add_argument("--phase", type=int, nargs="+", default=None,
                        help="Phases to run (default: 1 2 3 4)")
    parser.add_argument("--cut_transitions_dir", default=None,
                        help="Directory of *_cut.npz files for cut-dynamics training")
    parser.add_argument("--phase3_train_encoder", action="store_true",
                        help="Joint encoder+dynamics training in Phase 3 "
                             "(NextLat Theorem 3.2 requirement). Uses RawSequenceDataset.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"),
                        help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (auto-detected if omitted)")

    args, overrides = parser.parse_known_args()

    cfg = _load_config(Path(args.config))
    if overrides:
        _apply_overrides(cfg, overrides)

    _set_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run(args, cfg, device)


if __name__ == "__main__":
    main()
