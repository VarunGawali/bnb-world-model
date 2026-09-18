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
  <cache_dir>/<file_stem>_<hash6>.pt
  Each file is a list of dicts, one per root→leaf path:
    z_seq          [T, H]   latent at each node
    z_next_seq     [T, H]   latent at each child node
    a_seq          [T]      branching variable index
    dir_seq        [T]      branch direction (+1/-1/0)
    bound_seq      [T]      normalised dual bound targets
    instance_weight float
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
from bnb_wm.data.datasets import RawSequenceDataset, make_raw_collate
from bnb_wm.training.checkpoint import load_weights_only
from torch.utils.data import DataLoader


def _file_hash(path: Path) -> str:
    return hashlib.md5(str(path).encode()).hexdigest()[:6]


def _cache_path(cache_dir: Path, traj_file: Path) -> Path:
    return cache_dir / f"{traj_file.stem}_{_file_hash(traj_file)}.pt"


@torch.no_grad()
def encode_file_group(model, file_paths, cache_dir, device, batch_size=1, max_path_len=32):
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
            bvars = batch["branch_vars"].to(device)
            bound = batch["bound_seq"].to(device) if "bound_seq" in batch else None
            dseq  = batch["dir_seq"].to(device)   if "dir_seq"   in batch else None
            tmask = batch["time_mask"].to(device)  if "time_mask" in batch else None
            iw    = batch["instance_weight"]

            h_vars_all, z_all = model.encode(gb)   # [sum_nodes, H], [sum_paths, H]

            # Split z_all back into per-path sequences using sizes
            B = len(sizes)
            H = z_all.size(-1)
            Tmax = int(sizes.max().item())

            # Rebuild per-path z_seq / z_next_seq from the flat encode
            # (same logic as _online_encode_raw_batch in trainer.py)
            z_seq_batch      = torch.zeros(B, Tmax, H, device=device)
            z_next_seq_batch = torch.zeros(B, Tmax, H, device=device)
            a_seq_batch      = torch.zeros(B, Tmax, dtype=torch.long, device=device)

            offset = 0
            for b_i, T_i in enumerate(sizes.tolist()):
                T_i = int(T_i)
                z_i = z_all[offset:offset + T_i]      # [T_i, H]
                z_seq_batch[b_i, :T_i]      = z_i
                z_next_seq_batch[b_i, :T_i - 1] = z_i[1:]  # child = next in path
                offset += T_i

                # branching vars for this path
                bv_start = int(sizes[:b_i].sum()) if b_i > 0 else 0
                a_seq_batch[b_i, :T_i] = bvars[bv_start:bv_start + T_i]

            # Store each sequence dict
            for b_i, T_i in enumerate(sizes.tolist()):
                T_i = int(T_i)
                seq = {
                    "z_seq":          z_seq_batch[b_i, :T_i].cpu(),
                    "z_next_seq":     z_next_seq_batch[b_i, :T_i].cpu(),
                    "a_seq":          a_seq_batch[b_i, :T_i].cpu(),
                    "instance_weight": float(iw[b_i]),
                }
                if dseq is not None:
                    seq["dir_seq"] = dseq[b_i, :T_i].cpu()
                if bound is not None:
                    seq["bound_seq"] = bound[b_i, :T_i].cpu()
                if tmask is not None:
                    seq["time_mask"] = tmask[b_i, :T_i].cpu()

                # ds.index[b_i] gives (file_idx, path_indices, weight)
                # We store all sequences for a file together
                # Since batch may mix files, group by file_idx isn't trivial here.
                # Use a flat list per batch instead; save per batch.
                fi = 0  # placeholder — we save all per batch below
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
        )
        if seqs:
            torch.save(seqs, cache_key)
            total_seqs += len(seqs)

    print(f"\nCache built: {total_seqs} sequences in {cache_dir}")
    print(f"Now run Phase 3 with: --phase3_latent_cache {cache_dir}")


if __name__ == "__main__":
    main()
