"""
Definitive Phase-3 iteration timing: sequence batch + cut batch + backward.
Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/time_full_iter.py
"""
import time, torch, yaml, tempfile
from pathlib import Path
from torch.cuda.amp import GradScaler
from contextlib import contextmanager

CFG    = yaml.safe_load(open("configs/default.yaml"))
CKPT   = "checkpoints/model_rl_best.pt"
DEVICE = torch.device("cuda")

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def lap(label, t0):
    sync()
    dt = time.perf_counter() - t0
    print(f"  {label}: {dt:.3f}s")
    return time.perf_counter()

# ── model ────────────────────────────────────────────────────────────────────
from bnb_wm.model.world_model import BnBWorldModel
model = BnBWorldModel(**CFG["model"]).to(DEVICE).train()
sd = torch.load(CKPT, map_location="cpu")
model.load_state_dict(sd["model"], strict=False)
print("Model loaded.\n")

# ── sequence loader (frozen encoder, cached latents) ─────────────────────────
from bnb_wm.data import list_trajectory_files, SequenceDataset, make_sequence_collate
from torch.utils.data import DataLoader

seq_collate = make_sequence_collate(include_vars=True)
files = list_trajectory_files("data/trajectories")
tr_files = files[:200]
ds = SequenceDataset(tr_files, model, DEVICE, include_vars=True,
                     cache_dir=Path("checkpoints/seq_cache"))
seq_loader = DataLoader(ds, batch_size=4, shuffle=True,
                        collate_fn=seq_collate, num_workers=0)
seq_iter = iter(seq_loader)
print(f"Sequence dataset: {len(ds)} paths")

# ── cut loader ────────────────────────────────────────────────────────────────
from bnb_wm.data.datasets import CutTransitionDataset
cut_files = list(Path("data/cut_transitions").rglob("*.npz"))
print(f"Cut files: {len(cut_files)}")
if cut_files:
    cut_ds = CutTransitionDataset(cut_files)
    cut_loader = DataLoader(cut_ds, batch_size=4, shuffle=True, num_workers=0)
    cut_iter = iter(cut_loader)
else:
    cut_iter = None
    print("WARNING: no cut files found")

# ── optimizer + scaler ────────────────────────────────────────────────────────
trainable = [p for p in model.parameters() if p.requires_grad]
# make dynamics params trainable for this test
for name, p in model.named_parameters():
    p.requires_grad = "dynamics" in name or "cut_action_embed" in name
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable, lr=5e-4)
scaler = GradScaler()

from bnb_wm.training.trainer import Trainer
trainer = Trainer(model, DEVICE, ckpt_dir=tempfile.mkdtemp())
trainer.cut_transition_weight = 0.2
trainer.v_consist_weight = 0.0
trainer.cf_contrastive_weight = 0.0
trainer.free_run_consist_weight = 0.0
trainer.overshoot_depth = 0
trainer._also_train_encoder = False

print("\n=== Timing one full training iteration ===")
print("(with CUDA sync at each boundary)\n")

# ── ITERATION ────────────────────────────────────────────────────────────────
sync(); t_start = time.perf_counter()

# 1. zero grad
optimizer.zero_grad(set_to_none=True)
t = lap("zero_grad", t_start)

# 2. get sequence batch
batch = next(seq_iter)
batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
         for k, v in batch.items()}
t = lap("seq batch fetch+move", t)

# 3. get cut batch and attach
if cut_iter is not None:
    try:
        cut_batch = next(cut_iter)
    except StopIteration:
        cut_iter = iter(cut_loader)
        cut_batch = next(cut_iter)
    # show cut batch keys and sizes
    print(f"  cut_batch keys: {list(cut_batch.keys())}")
    for k, v in cut_batch.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k}: {tuple(v.shape)}")
    # attach to sequence batch so _dynamics_batch_loss picks them up
    batch["cut_batch"] = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                          for k, v in cut_batch.items()}
t = lap("cut batch fetch+move", t)

# 4. forward + loss (with AMP)
with torch.autocast("cuda", enabled=True):
    loss, comps = trainer._dynamics_batch_loss(batch, return_components=True)
t = lap("forward + loss (AMP)", t)
print(f"  loss={loss.item():.4f}  comps={  {k: f'{v:.4f}' for k,v in comps.items()}  }")

# 5. backward
scaler.scale(loss).backward()
t = lap("backward", t)

# 6. optimizer step
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(trainable, 1.0)
scaler.step(optimizer)
scaler.update()
t = lap("optimizer step", t)

sync()
total = time.perf_counter() - t_start
print(f"\nTOTAL: {total:.3f}s  (vs observed 1021s in training)")

# ── also check cut_transition_weight propagation ─────────────────────────────
print(f"\ncut_transition_weight on trainer: {trainer.cut_transition_weight}")
print(f"'cut_transition' in comps: {'cut_transition' in comps}")
if 'cut_transition' in comps:
    print(f"  cut_transition loss value: {comps['cut_transition']:.6f}")
