"""
Profile Phase-3 bottlenecks: data loading, GNN encode, cut encode,
dynamics forward, overshoot/free-run. Run with:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/profile_phase3.py
"""
import time, torch, numpy as np, yaml
from pathlib import Path
from torch_geometric.data import Batch

CFG    = yaml.safe_load(open("configs/default.yaml"))
CKPT   = "checkpoints/model_rl_best.pt"
DEVICE = torch.device("cuda")

from bnb_wm.model.world_model import BnBWorldModel
model = BnBWorldModel(**CFG["model"]).to(DEVICE).eval()
sd = torch.load(CKPT, map_location="cpu")
model.load_state_dict(sd["model"], strict=False)
print("Model loaded.\n")

from bnb_wm.data.datasets import build_pyg_data

def tick():
    torch.cuda.synchronize()
    return time.perf_counter()

def tock(label, t0):
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"  [{label}] {dt:.3f}s")
    return dt

# ── find biggest trajectory files ───────────────────────────────────────────
all_files = sorted(Path("data/trajectories").rglob("*.npz"),
                   key=lambda f: f.stat().st_size, reverse=True)
biggest = [f for f in all_files if "n_steps" in np.load(f, allow_pickle=True).files][:5]
print("=== Largest trajectory files ===")
for f in biggest:
    d = np.load(f, allow_pickle=True)
    T = int(d["n_steps"])
    vf = np.asarray(d["var_features"][0])
    cf = np.asarray(d["con_features"][0])
    ei = np.asarray(d["edge_indices"][0])
    print(f"  {f.name}  T={T}  vars={vf.shape[0]}  cons={cf.shape[0]}  edges={ei.shape[1]}  size={f.stat().st_size//1024}KB")

# ── A: load biggest file ─────────────────────────────────────────────────────
print("\n=== A: disk load (largest file) ===")
f_big = biggest[0]
t0 = tick()
d = np.load(f_big, allow_pickle=True)
_ = {k: d[k] for k in d.files}   # force all arrays into memory
tock(f"load {f_big.name}", t0)

# ── B: build all PyG graphs for that trajectory ─────────────────────────────
T = int(d["n_steps"])
print(f"\n=== B: build PyG graphs ({T} steps, largest traj) ===")
t0 = tick()
graphs = [build_pyg_data(d["var_features"][t], d["con_features"][t],
                         d["edge_indices"][t], d["edge_values"][t])
          for t in range(T)]
tock(f"build_pyg_data x{T}", t0)

# ── C: GNN encode all steps (no_grad) ───────────────────────────────────────
print(f"\n=== C: GNN encode all {T} steps (no_grad) ===")
gb = Batch.from_data_list(graphs).to(DEVICE)
print(f"  batched: nodes={gb.x.shape[0]}  edges={gb.edge_index.shape[1]}")
with torch.no_grad():
    t0 = tick()
    h_vars, z, _ = model.encoder(gb.x, gb.edge_index, gb.node_type, gb.batch,
                                  edge_attr=getattr(gb, "edge_attr", None))
    tock(f"encode {T}-step trajectory", t0)

# ── D: SequenceDataset cache-miss path (encode + disk write) ─────────────────
print("\n=== D: seq_cache write (simulate cache miss) ===")
import tempfile, os
cache_dir = Path(tempfile.mkdtemp())
from bnb_wm.data.datasets import SequenceDataset
t0 = tick()
ds = SequenceDataset([f_big], model, DEVICE, include_vars=True, cache_dir=cache_dir)
tock(f"SequenceDataset init (encode+write) for 1 file", t0)
print(f"  cache files written: {list(cache_dir.iterdir())}")

# ── E: SequenceDataset __getitem__ (cache hit) ───────────────────────────────
print("\n=== E: SequenceDataset __getitem__ (cache HIT) ===")
if len(ds) > 0:
    t0 = tick()
    item = ds[0]
    tock("__getitem__ (cache hit, idx=0)", t0)
    print(f"  item keys: {list(item.keys()) if isinstance(item, dict) else type(item)}")

# ── F: cut transition encode ─────────────────────────────────────────────────
print("\n=== F: cut transition encode ===")
cut_files = list(Path("data/cut_transitions").rglob("*.npz"))
print(f"  cut files: {len(cut_files)}")
if cut_files:
    cf = np.load(cut_files[0], allow_pickle=True)
    print(f"  cut file keys: {list(cf.keys())}")
    # find graph keys
    vf_key = next((k for k in cf.keys() if "var_features" in k), None)
    if vf_key:
        prefix = vf_key.replace("var_features", "")
        g_cut = build_pyg_data(cf[f"var_features{prefix}"], cf[f"con_features{prefix}"],
                               cf[f"edge_indices{prefix}"], cf[f"edge_values{prefix}"])
        gb_cut = Batch.from_data_list([g_cut]).to(DEVICE)
        with torch.no_grad():
            t0 = tick()
            model.encode(gb_cut)
            tock("encode 1 cut graph", t0)

# ── G: dynamics forward ──────────────────────────────────────────────────────
print("\n=== G: dynamics forward (1 step) ===")
z1 = z[:1]
h_v = h_vars[:10]
a   = h_v[:1]
with torch.no_grad():
    t0 = tick()
    z2 = model.dynamics(z1, a, h_v, d=torch.ones(1, device=DEVICE))
    tock("dynamics 1 step", t0)

# ── H: overshoot (depth=3) ───────────────────────────────────────────────────
print("\n=== H: overshoot rollout (depth=3) ===")
with torch.no_grad():
    z_s = z1.clone()
    t0 = tick()
    for _ in range(3):
        z_s = model.dynamics(z_s, a, h_v, d=torch.ones(1, device=DEVICE))
    tock("3-step rollout", t0)

print("\n=== DONE — paste output to identify bottleneck ===")
