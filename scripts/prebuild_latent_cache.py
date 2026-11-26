"""
prebuild_latent_cache.py — Pre-encode all trajectory sequences to disk.

Run once (or after each encoder update) before Phase 3 to avoid live GNN
encoding during training. Phase 3 then reads cached latents and only runs
the dynamics Transformer → epochs go from hours to minutes.

Workflow
--------
  # Step 1 — build cache from phase2 checkpoint (before first Phase 3 run):
  CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python scripts/prebuild_latent_cache.py \\
      --data_dir data/highs_trajectories/medium \\
      --checkpoint checkpoints/highs_retrain/phase2_best.pt \\
      --cache_dir  checkpoints/highs_retrain/latent_cache

  # Step 2 — train Phase 3 with cached latents (encoder frozen, fast):
  CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. torchrun --nproc_per_node=2 \\
      scripts/train_highs.py \\
      --data_dir data/highs_trajectories/medium \\
      --ckpt_dir checkpoints/highs_retrain \\
      --phase 3 4 \\
      --phase3_latent_cache checkpoints/highs_retrain/latent_cache \\
      --training.patience_phase3 5

  # Step 3 — refresh cache with latest encoder (run every ~5 epochs):
  python scripts/prebuild_latent_cache.py \\
      --data_dir data/highs_trajectories/medium \\
      --checkpoint checkpoints/highs_retrain/phase3_best.pt \\
      --cache_dir  checkpoints/highs_retrain/latent_cache

  # Step 4 — resume Phase 3 from phase3_best.pt (train_highs.py loads it automatically)
  # ... repeat until convergence

Cache format
------------
  <cache_dir>/chunk_NNNNN.pt
  Each file is a list of dicts, one per root→leaf path of T nodes, holding its
  T-1 parent→child transitions (bnb_wm.data.datasets.raw_paths_to_transitions,
  the same targets SequenceDataset builds):
    z_seq, a_seq, z_next_seq          [T-1, H]     (a_seq = action embedding)
    dir_seq, bound_next_seq, reward_seq, time_mask   [T-1]
    hv_seq, hv_next_seq               [T-1, K, H]  per-variable recon subset
    var_mask                          [T-1, K]     bool
    instance_weight                   float
  Caches built before this format (a_seq indices, unshifted bound/dir, no
  hv/reward) must be rebuilt.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bnb_wm.model.world_model import BnBWorldModel
from bnb_wm.data.datasets import (
    RawSequenceDataset, make_raw_collate, raw_paths_to_transitions,
)
from bnb_wm.training.checkpoint import load_weights_only
from torch.utils.data import DataLoader


def _file_hash(path: Path) -> str:
    return hashlib.md5(str(path).encode()).hexdigest()[:6]


def _cache_path(cache_dir: Path, traj_file: Path) -> Path:
    return cache_dir / f"{traj_file.stem}_{_file_hash(traj_file)}.pt"


@torch.no_grad()
def encode_file_group(model, file_paths, cache_dir, device, batch_size=1, max_path_len=32,
                      max_vars_recon=64):
    """Encode all root→leaf paths from a list of trajectory files, save to cache."""
    ds = RawSequenceDataset(file_paths, max_path_len=max_path_len)
    if len(ds) == 0:
        return 0

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_raw_collate(),
        num_workers=4,
        pin_memory=True,
    )

    # Accumulate encoded sequences per source file
    # ds.index: list of (file_idx, path, weight)
    file_sequences: dict[int, list[dict]] = {}

    for batch in loader:
        if "batch_graphs" not in batch:
            continue
        try:
            gb = batch["batch_graphs"].to(device)
            sizes = batch["batch_sizes"]           # [B] steps per path
            bvars = batch["branch_vars"]
            bound = batch["bound_seq"]             # [B, Tmax] per node
            dseq  = batch["dir_seq"]               # [B, Tmax] per node
            iw    = batch["instance_weight"]

            h_vars_all, z_all = model.encode(gb)   # [sum_vars, H], [sum_nodes, H]
            var_mask = gb.node_type == 0
            tr = raw_paths_to_transitions(
                h_vars_all, z_all, gb.batch[var_mask], sizes, bvars,
                dseq, bound, max_vars_recon=max_vars_recon)

            # Store each path's valid transitions (drop paths with none).
            for b_i, T_i in enumerate(sizes.tolist()):
                n_tr = int(T_i) - 1
                if n_tr < 1:
                    continue
                seq = {k: v[b_i, :n_tr].cpu() for k, v in tr.items()}
                seq["instance_weight"] = float(iw[b_i])
                file_sequences.setdefault(-1, []).append(seq)

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue

    return file_sequences.get(-1, [])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir",   required=True, nargs="+",
                        help="Directories of HiGHS trajectory .npz files")
    parser.add_argument("--checkpoint", required=True,
                        help="Model checkpoint to encode with (e.g. phase2_best.pt)")
    parser.add_argument("--cache_dir",  default="checkpoints/highs_retrain/latent_cache",
                        help="Output directory for cached latent .pt files")
    parser.add_argument("--config",     default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--max_path_len", type=int, default=32)
    parser.add_argument("--max_vars_recon", type=int, default=64,
                        help="Per-variable reconstruction subset size")
    parser.add_argument("--batch_size",   type=int, default=1,
                        help="Paths per encoding batch (reduce if OOM)")
    parser.add_argument("--device",     default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # Load model
    cfg = yaml.safe_load(open(args.config))
    mc  = cfg["model"]
    model = BnBWorldModel(
        hidden_dim=mc["hidden_dim"],
        n_gnn_layers=mc["n_gnn_layers"],
        n_gnn_heads=mc["n_gnn_heads"],
        n_dyn_layers=mc["n_dyn_layers"],
        n_dyn_heads=mc.get("n_dyn_heads", 4),
        max_seq=mc.get("max_seq", 512),
        dyn_residual=mc.get("dyn_residual", True),
        dyn_heteroscedastic=mc.get("dyn_heteroscedastic", False),
    ).to(device).eval()
    load_weights_only(model, Path(args.checkpoint), device=device, strict=False)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Collect trajectory files
    files = []
    for d in args.data_dir:
        files.extend(sorted(Path(d).glob("**/*.npz")))
    files = [f for f in files if not f.name.endswith("_cut.npz")]
    print(f"Found {len(files)} trajectory files")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Process all files as one dataset, save in chunks of ~50 files
    chunk_size = 50
    total_seqs = 0
    for i in tqdm(range(0, len(files), chunk_size), desc="Encoding chunks"):
        chunk = files[i:i + chunk_size]
        # Cache key: one .pt per chunk (by first file's hash)
        cache_key = cache_dir / f"chunk_{i:05d}.pt"
        if cache_key.exists():
            # Check if any source file is newer than cache
            cache_mtime = cache_key.stat().st_mtime
            if all(f.stat().st_mtime < cache_mtime for f in chunk):
                # Load to count sequences
                seqs = torch.load(cache_key, map_location="cpu", weights_only=False)
                total_seqs += len(seqs)
                continue  # already up to date

        seqs = encode_file_group(
            model, chunk, cache_dir, device,
            batch_size=args.batch_size,
            max_path_len=args.max_path_len,
            max_vars_recon=args.max_vars_recon,
        )
        if seqs:
            torch.save(seqs, cache_key)
            total_seqs += len(seqs)

    print(f"\nCache built: {total_seqs} sequences in {cache_dir}")
    print(f"Now run Phase 3 with: --phase3_latent_cache {cache_dir}")


if __name__ == "__main__":
    main()
