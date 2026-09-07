"""
Profile Phase-3 bottlenecks: data loading, GNN encode, cut encode,
dynamics forward, overshoot/free-run. Run with:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scratchpad/profile_phase3.py
"""
import time, torch, numpy as np, yaml
from pathlib import Path
from torch_geometric.data import Batch

# ── config / model ──────────────────────────────────────────────────────────
CFG   = yaml.safe_load(open("configs/default.yaml"))
CKPT  = "checkpoints/model_rl_best.pt"
DEVICE = torch.device("cuda")

from bnb_wm.model.world_model import BnBWorldModel
model = BnBWorldModel(**CFG["model"]).to(DEVICE).eval()
sd = torch.load(CKPT, map_location="cpu")
model.load_state_dict(sd["model"], strict=False)
print("Model loaded.")

# ── helpers ──────────────────────────────────────────────────────────────────
def tick(label):
    torch.cuda.synchronize()
    t = time.perf_counter()
    return label, t

def tock(label, t0):
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"  [{label}] {dt:.3f}s")
    return dt

# ── A: disk / npz loading ───────────────────────────────────────────────────
print("\n=== A: disk loading (5 traj files) ===")
traj_files = sorted(Path("data/trajectories").rglob("traj_sc_*.npz"))[:5]
_, t0 = tick("load")
dicts = []
for f in traj_files:
    d = np.load(f, allow_pickle=True)
    dicts.append({k: d[k] for k in d.files})
tock("load 5 traj npz", t0)

# ── B: PyG graph build for one step ─────────────────────────────────────────
print("\n=== B: build PyG graph (one B&B step) ===")
from bnb_wm.data.datasets import build_pyg_data
d0 = dicts[0]
_, t0 = tick("pyg")
g = build_pyg_data(d0["var_features"][0], d0["con_features"][0],
                   d0["edge_indices"][0], d0["edge_values"][0])
tock("build_pyg_data (1 step)", t0)
print(f"  nodes={g.x.shape[0]}  edges={g.edge_index.shape[1]}")

# ── C: GNN encode (one graph, no grad) ──────────────────────────────────────
print("\n=== C: GNN encode (1 graph, no_grad) ===")
gb = Batch.from_data_list([g]).to(DEVICE)
with torch.no_grad():
    _, t0 = tick("enc1")
    h_vars, z, _ = model.encoder(gb.x, gb.edge_index, gb.node_type, gb.batch,
                                  edge_attr=gb.edge_attr if hasattr(gb,"edge_attr") else None)
    tock("encode 1 graph", t0)

# ── D: GNN encode (full trajectory, no grad) ─────────────────────────────────
print("\n=== D: GNN encode (full traj, all steps, no_grad) ===")
T = int(d0["n_steps"])
print(f"  trajectory steps = {T}")
graphs = [build_pyg_data(d0["var_features"][t], d0["con_features"][t],
                         d0["edge_indices"][t], d0["edge_values"][t])
          for t in range(T)]
gb_all = Batch.from_data_list(graphs).to(DEVICE)
with torch.no_grad():
    _, t0 = tick("enc_traj")
    h_vars_all, z_all, _ = model.encoder(gb_all.x, gb_all.edge_index,
                                          gb_all.node_type, gb_all.batch,
                                          edge_attr=gb_all.edge_attr if hasattr(gb_all,"edge_attr") else None)
    tock(f"encode full traj ({T} steps)", t0)

# ── E: cut transition encode ─────────────────────────────────────────────────
print("\n=== E: cut transition GNN encode ===")
cut_files = sorted(Path("data/cut_transitions").rglob("*_cut.npz"))[:3]
if cut_files:
    cf = np.load(cut_files[0], allow_pickle=True)
    from bnb_wm.data.datasets import build_pyg_data
    gb_cut = Batch.from_data_list([
        build_pyg_data(cf["var_features_before"], cf["con_features_before"],
                       cf["edge_indices_before"], cf["edge_values_before"])
    ]).to(DEVICE)
    with torch.no_grad():
        _, t0 = tick("cut_enc")
        model.encode(gb_cut)
        tock("encode cut graph_before (1 cut file)", t0)
else:
    print("  no cut files found")

# ── F: dynamics forward (one step) ──────────────────────────────────────────
print("\n=== F: dynamics Transformer (one step, no rollout) ===")
z_s = z_all[:1]   # single state
h_v = h_vars_all[:5]   # first 5 vars (fake candidates)
a   = h_v[:1]
with torch.no_grad():
    _, t0 = tick("dyn1")
    z_next = model.dynamics(z_s, a, h_v, d=torch.ones(1,device=DEVICE))
    tock("dynamics 1 step", t0)

# ── G: overshoot rollout (depth=3) ──────────────────────────────────────────
print("\n=== G: overshoot rollout (depth=3, 1 candidate) ===")
with torch.no_grad():
    _, t0 = tick("overshoot")
    for _ in range(3):
        z_s = model.dynamics(z_s, a, h_v, d=torch.ones(1,device=DEVICE))
    tock("3-step rollout (sequential)", t0)

print("\n=== SUMMARY ===")
print("Run the timings above to identify which letter is your 1021s bottleneck.")
