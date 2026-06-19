"""
train.py — Main training entry point for all four phases.

Usage:
    python scripts/train.py --phase 1
    python scripts/train.py --phase 2
    python scripts/train.py --phase 3
    python scripts/train.py --phase 4
    python scripts/train.py --phase all   # run phases 1->4 sequentially
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Make sure the package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from bnb_wm.model import BnBWorldModel
from bnb_wm.data import TransitionDataset, SequenceDataset, pyg_collate
from bnb_wm.training import Trainer
from bnb_wm.training.checkpoint import load_weights_only


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path):
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        print("PyYAML not installed — using hardcoded defaults.")
        return {}


def make_loaders(data_root, problem, cfg):
    all_paths = sorted((Path(data_root) / problem).glob("*.npz"))
    random.seed(42)
    random.shuffle(all_paths)

    n = len(all_paths)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)

    train_paths = all_paths[:n_train]
    val_paths   = all_paths[n_train : n_train + n_val]
    test_paths  = all_paths[n_train + n_val :]

    max_files = cfg.get("data", {}).get("max_train_files", None)
    if max_files:
        train_paths = train_paths[:max_files]

    batch_size = cfg.get("training", {}).get("batch_size", 64)

    train_trans = TransitionDataset(train_paths)
    val_trans   = TransitionDataset(val_paths)
    train_seq   = SequenceDataset(train_paths)
    val_seq     = SequenceDataset(val_paths)

    train_loader = DataLoader(
        train_trans, batch_size=batch_size, shuffle=True,
        collate_fn=pyg_collate, num_workers=0
    )
    val_loader = DataLoader(
        val_trans, batch_size=batch_size, shuffle=False,
        collate_fn=pyg_collate, num_workers=0
    )
    seq_train_loader = DataLoader(
        train_seq, batch_size=1, shuffle=True, collate_fn=lambda x: x[0]
    )
    seq_val_loader = DataLoader(
        val_seq, batch_size=1, shuffle=False, collate_fn=lambda x: x[0]
    )

    print(f"Train transitions : {len(train_trans):,}")
    print(f"Val transitions   : {len(val_trans):,}")
    print(f"Test files        : {len(test_paths)}")

    # Compute pos_weight for integrality head
    n_pos = n_neg = 0
    for p in train_paths:
        d = np.load(p, allow_pickle=True)
        labels = d["next_is_leaf"]
        n_pos += int(labels.sum())
        n_neg += int(len(labels) - labels.sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    print(f"Integrality pos_weight = {pos_weight.item():.1f}")

    return train_loader, val_loader, seq_train_loader, seq_val_loader, pos_weight


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train BnB World Model")
    parser.add_argument("--phase",  type=str, default="1",
                        choices=["1", "2", "3", "4", "all"],
                        help="Training phase (1-4 or 'all')")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data",   type=str, default=None,
                        help="Override data root path")
    parser.add_argument("--ckpt",   type=str, default=None,
                        help="Override checkpoint directory")
    args = parser.parse_args()

    cfg = load_config(args.config)

    data_root = args.data or cfg.get("paths", {}).get("data_root", "data/")
    ckpt_dir  = Path(args.ckpt or cfg.get("paths", {}).get("checkpoint_dir", "checkpoints/"))
    problem   = cfg.get("data", {}).get("problem", "set_cover")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    m_cfg = cfg.get("model", {})
    model = BnBWorldModel(
        hidden_dim=m_cfg.get("hidden_dim", 128),
        n_gnn_layers=m_cfg.get("n_gnn_layers", 3),
    ).to(DEVICE)
    torch.backends.cudnn.benchmark = True

    t_cfg = cfg.get("training", {})
    trainer = Trainer(
        model, DEVICE, ckpt_dir,
        amp=t_cfg.get("amp", True),
    )

    train_loader, val_loader, seq_tl, seq_vl, pos_weight = make_loaders(
        data_root, problem, cfg
    )

    phases_to_run = ["1", "2", "3", "4"] if args.phase == "all" else [args.phase]

    for phase in phases_to_run:
        print(f"\n{'='*60}")
        print(f" PHASE {phase}")
        print(f"{'='*60}\n")

        if phase == "1":
            trainer.train_policy(
                train_loader, val_loader,
                epochs=t_cfg.get("epochs_phase1", 8),
                lr=t_cfg.get("lr_phase1", 1e-3),
            )

        elif phase == "2":
            best_p1 = ckpt_dir / "phase1_best.pt"
            if best_p1.exists():
                load_weights_only(model, best_p1, DEVICE)
                print(f"Loaded Phase 1 weights from {best_p1}")
            trainer.train_value(
                train_loader, val_loader,
                epochs=t_cfg.get("epochs_phase2", 8),
                lr=t_cfg.get("lr_phase2", 5e-4),
            )

        elif phase == "3":
            print("Phase 3 (dynamics) — use seq_train_loader / seq_val_loader")
            print("Implementation: add train_dynamics() to Trainer class.")

        elif phase == "4":
            best_p2 = ckpt_dir / "phase2_best.pt"
            if best_p2.exists():
                load_weights_only(model, best_p2, DEVICE)
                print(f"Loaded Phase 2 weights from {best_p2}")
            trainer.train_joint(
                train_loader, val_loader,
                epochs=t_cfg.get("epochs_phase4", 8),
                lr=t_cfg.get("lr_phase4", 1e-4),
                pos_weight=pos_weight,
            )


if __name__ == "__main__":
    main()
