"""
Measure time-to-first-batch vs time-for-dynamics-loss.
Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/time_first_batch.py
"""
import time, torch, yaml
from pathlib import Path

CFG   = yaml.safe_load(open("configs/default.yaml"))
CKPT  = "checkpoints/model_rl_best.pt"
DEVICE = torch.device("cuda")

# ── cache file count ─────────────────────────────────────────────────────────
cache_dir = Path("checkpoints/seq_cache")
cache_files = list(cache_dir.rglob("*.pt")) + list(cache_dir.rglob("*.pkl")) + \
              list(cache_dir.rglob("*.npz"))
print(f"seq_cache files: {len(cache_files)}  (in {cache_dir})")
for d in sorted(cache_dir.iterdir()):
    n = len(list(d.iterdir())) if d.is_dir() else 0
    print(f"  {d.name}: {n} files")

# ── model ────────────────────────────────────────────────────────────────────
from bnb_wm.model.world_model import BnBWorldModel
model = BnBWorldModel(**CFG["model"]).to(DEVICE).eval()
sd = torch.load(CKPT, map_location="cpu")
model.load_state_dict(sd["model"], strict=False)
print("Model loaded.\n")

# ── build loader (same as train.py frozen path) ───────────────────────────────
from bnb_wm.data import list_trajectory_files, SequenceDataset, make_sequence_collate
from torch.utils.data import DataLoader

seq_cache  = Path("checkpoints/seq_cache")
seq_collate = make_sequence_collate(include_vars=True)

files = list_trajectory_files("data/trajectories")
tr_files = files[:100]   # small subset — just to measure time-to-first-batch
print(f"Using {len(tr_files)} train files for timing test.")

print("Building SequenceDataset index...")
t0 = time.perf_counter()
ds = SequenceDataset(tr_files, model, DEVICE, include_vars=True, cache_dir=seq_cache)
print(f"  index built in {time.perf_counter()-t0:.2f}s  ({len(ds)} paths)")

loader = DataLoader(ds, batch_size=4, shuffle=False,
                    collate_fn=seq_collate, num_workers=0)

# ── time-to-first-batch ───────────────────────────────────────────────────────
print("\nTiming first batch (this is the key measurement)...")
t0 = time.perf_counter()
batch = next(iter(loader))
t_batch = time.perf_counter() - t0
print(f"TIME TO GET FIRST BATCH: {t_batch:.2f}s")

# ── time for dynamics forward + loss ─────────────────────────────────────────
print("\nTiming dynamics forward + loss...")
from bnb_wm.training.trainer import Trainer
import tempfile
trainer = Trainer(model, DEVICE, ckpt_dir=tempfile.mkdtemp())

# Time each loss component separately
model.train()

# 1. Just the encode step
t0 = time.perf_counter()
try:
    # simulate what _dynamics_batch_loss does: encode the batch
    from bnb_wm.data.datasets import build_pyg_data
    from torch_geometric.data import Batch as PygBatch
    if "batch_graphs" in batch:
        gb = batch["batch_graphs"].to(DEVICE)
        with torch.no_grad():
            h_vars_all, z_all = model.encode(gb)
        t_enc = time.perf_counter() - t0
        print(f"TIME TO ENCODE BATCH: {t_enc:.2f}s")
    else:
        print("batch has no batch_graphs; keys:", list(batch.keys()))
        t_enc = 0
except Exception as e:
    print(f"Encode timing failed: {e}")
    t_enc = 0

# 2. Full dynamics loss
t0 = time.perf_counter()
try:
    loss, comps = trainer._dynamics_batch_loss(batch, return_components=True)
    t_loss = time.perf_counter() - t0
    print(f"TIME FOR DYNAMICS LOSS (full): {t_loss:.2f}s")
    print(f"  components: { {k: f'{v:.4f}' for k,v in comps.items()} }")
except Exception as e:
    t_loss = time.perf_counter() - t0
    print(f"Loss timing failed after {t_loss:.2f}s: {e}")

print("\n=== VERDICT ===")
print(f"  data loading:    {t_batch:.1f}s")
try:
    print(f"  dynamics loss:   {t_loss:.1f}s")
    print(f"  total expected:  {t_batch + t_loss:.1f}s  (vs observed 1021s)")
except:
    pass
