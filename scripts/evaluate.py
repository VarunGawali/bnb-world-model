"""
evaluate.py — Evaluate a trained BnB World Model checkpoint.

Runs three evaluations:
    1. Policy top-k accuracy on the validation set
    2. Value head Spearman correlation on the validation set
    3. Macro benchmark (SCIP vs Random vs GNN) on held-out instances

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/phase4_best.pt
    python scripts/evaluate.py --checkpoint checkpoints/phase1_best.pt --no-benchmark
"""

import argparse
import sys
from pathlib import Path

import random
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from bnb_wm.model import BnBWorldModel
from bnb_wm.data import TransitionDataset, pyg_collate
from bnb_wm.training.checkpoint import load_weights_only
from bnb_wm.evaluate.metrics import topk_accuracy, rank_cdf, compute_spearman
from bnb_wm.evaluate.benchmark import run_macro_benchmark


def main():
    parser = argparse.ArgumentParser(description="Evaluate BnB World Model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data",  type=str, default="data/")
    parser.add_argument("--problem", type=str, default="set_cover")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="Skip the macro benchmark (faster)")
    parser.add_argument("--n-bench", type=int, default=10,
                        help="Number of instances for macro benchmark")
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}\n")

    # Load model
    model = BnBWorldModel(hidden_dim=128, n_gnn_layers=3).to(DEVICE)
    load_weights_only(model, args.checkpoint, DEVICE)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}\n")

    # Build val loader
    all_paths = sorted((Path(args.data) / args.problem).glob("*.npz"))
    random.seed(42)
    random.shuffle(all_paths)
    n = len(all_paths)
    val_paths = all_paths[int(0.8 * n) : int(0.9 * n)]

    val_loader = DataLoader(
        TransitionDataset(val_paths),
        batch_size=64, shuffle=False,
        collate_fn=pyg_collate, num_workers=0,
    )
    print(f"Val samples: {len(TransitionDataset(val_paths)):,}\n")

    # 1. Top-k accuracy
    print("--- Policy Top-k Accuracy ---")
    acc = topk_accuracy(model, val_loader, DEVICE, ks=(1, 3, 5))
    for k, v in acc.items():
        print(f"  Top-{k}: {v:.3f}")

    # 2. Rank CDF
    print("\n--- Rank CDF ---")
    rank_cdf(model, val_loader, DEVICE)

    # 3. Value head Spearman
    print("\n--- Value Head Spearman ---")
    compute_spearman(model, val_loader, DEVICE)

    # 4. Macro benchmark (optional)
    if not args.no_benchmark:
        print("\n--- Macro Benchmark ---")
        run_macro_benchmark(
            model, DEVICE,
            problem=args.problem,
            n_instances=args.n_bench,
        )


if __name__ == "__main__":
    main()
