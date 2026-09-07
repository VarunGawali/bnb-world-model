"""
Pre-build the SequenceDataset latent cache for all trajectory files.
Run ONCE before Phase-3 training. Fast because there's no training loop overhead.

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/prebuild_seq_cache.py [--workers N]
"""
import argparse, time, torch, yaml
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data_root",  default="data/trajectories")
parser.add_argument("--cache_dir",  default="checkpoints/seq_cache")
parser.add_argument("--config",     default="configs/default.yaml")
parser.add_argument("--checkpoint", default="checkpoints/model_rl_best.pt")
parser.add_argument("--max_files",  type=int, default=None)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

cfg = yaml.safe_load(open(args.config))
from bnb_wm.model.world_model import BnBWorldModel
model = BnBWorldModel(**cfg["model"]).to(device).eval()
sd = torch.load(args.checkpoint, map_location="cpu")
model.load_state_dict(sd["model"], strict=False)
print("Model loaded.")

from bnb_wm.data.datasets import list_trajectory_files, SequenceDataset

files = list_trajectory_files(args.data_root)
if args.max_files:
    files = files[:args.max_files]
print(f"Trajectories: {len(files)}")

cache_dir = Path(args.cache_dir)

# Build the dataset — this triggers cache building for all files that aren't
# cached yet. SequenceDataset.__init__ just builds the index (fast); the cache
# is written during __getitem__. We iterate every item once to warm the cache.
print("Building index...")
ds = SequenceDataset(files, model, device, include_vars=True, cache_dir=cache_dir)
print(f"Dataset items: {len(ds)} paths across {len(files)} files")

print("Warming cache (one pass through dataset)...")
t0 = time.perf_counter()
skipped = 0
for i in tqdm(range(len(ds))):
    try:
        _ = ds[i]
    except Exception as e:
        skipped += 1
dt = time.perf_counter() - t0
print(f"Done in {dt:.1f}s  ({skipped} skipped)")
print(f"Cache dir: {cache_dir}")
