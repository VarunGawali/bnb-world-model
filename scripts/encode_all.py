"""
Offline Step 1: encode all trajectory files with the frozen encoder.

Output: data/encoded/<stem>.pt per trajectory, each containing:
    z_seq    [T, H]        graph latents
    a_seq    [T, H]        action (branching-var) embeddings
    hv_seq   [T, K, H]     per-variable embeddings (top K vars)
    bound_seq [T]          normalised dual bounds
    dir_seq  [T]           branch direction (+1 / -1 / 0)
    branch   [T]           integer branching-var indices

Run once before Phase-3 training:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/encode_all.py
"""
import argparse, time, hashlib, torch, yaml, numpy as np
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Batch

parser = argparse.ArgumentParser()
parser.add_argument("--data_root",  default="data/trajectories")
parser.add_argument("--out_dir",    default="data/encoded")
parser.add_argument("--config",     default="configs/default.yaml")
parser.add_argument("--checkpoint", default="checkpoints/model_rl_best.pt")
parser.add_argument("--max_vars",   type=int, default=64,
                    help="top-K variable embeddings to store per step")
parser.add_argument("--chunk",      type=int, default=8,
                    help="steps per GNN forward (reduce if OOM)")
parser.add_argument("--max_files",  type=int, default=None)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

cfg = yaml.safe_load(open(args.config))
from bnb_wm.model.world_model import BnBWorldModel
model = BnBWorldModel(**cfg["model"]).to(device).eval()
sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
model.load_state_dict(sd["model"], strict=False)
print("Encoder loaded (frozen).")

from bnb_wm.data.datasets import list_trajectory_files, build_pyg_data
from bnb_wm.data.datasets import gap_to_primal_norm

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

files = list_trajectory_files(args.data_root)
if args.max_files:
    files = files[:args.max_files]
print(f"Trajectories to encode: {len(files)}")

skipped = errors = 0
t0 = time.perf_counter()

for f in tqdm(files, desc="encoding"):
    out = out_dir / (f.stem + ".pt")
    if out.exists():
        continue  # already encoded — skip

    try:
        d = np.load(f, allow_pickle=True)
        if "n_steps" not in d:
            skipped += 1
            continue
        T = int(d["n_steps"])
        if T < 2:
            skipped += 1
            continue

        datas = [
            build_pyg_data(d["var_features"][t], d["con_features"][t],
                           d["edge_indices"][t], d["edge_values"][t])
            for t in range(T)
        ]
        branch = np.asarray(d["branching_vars"]).astype(np.int64)

        # Encode in chunks to bound peak GPU memory.
        z_parts, hv_parts, vb_parts = [], [], []
        step_off = 0
        with torch.no_grad():
            for s in range(0, T, args.chunk):
                cb = Batch.from_data_list(datas[s:s + args.chunk]).to(device)
                h_v, z_c = model.encode(cb)
                z_parts.append(z_c.cpu())
                vm = cb.node_type == 0
                vb_parts.append(cb.batch[vm].cpu() + step_off)
                hv_parts.append(h_v.cpu())
                step_off += (s + args.chunk <= T and args.chunk or T - s)
                del cb, h_v, z_c

        z_seq = torch.cat(z_parts, dim=0)           # [T, H]
        h_all = torch.cat(hv_parts, dim=0)          # [sumV, H]
        var_batch = torch.cat(vb_parts, dim=0)      # [sumV]

        K = args.max_vars
        H = z_seq.size(1)
        a_list, hv_list = [], []
        for t in range(T):
            ht = h_all[var_batch == t]              # [n_vars_t, H]
            bv = int(branch[t])
            bv = bv if 0 <= bv < ht.size(0) else 0
            a_list.append(ht[bv])
            k = min(K, ht.size(0))
            pad = torch.zeros(K, H)
            pad[:k] = ht[:k]
            hv_list.append(pad)

        bound_seq = torch.as_tensor(gap_to_primal_norm(d), dtype=torch.float32)
        if "branch_dirs" in d:
            dir_seq = torch.as_tensor(
                np.asarray(d["branch_dirs"], dtype=np.float32))
        else:
            dir_seq = torch.zeros(T, dtype=torch.float32)

        bundle = {
            "z_seq":     z_seq,                             # [T, H]
            "a_seq":     torch.stack(a_list, dim=0),        # [T, H]
            "hv_seq":    torch.stack(hv_list, dim=0),       # [T, K, H]
            "bound_seq": bound_seq,                         # [T]
            "dir_seq":   dir_seq,                           # [T]
            "branch":    torch.as_tensor(branch, dtype=torch.long),  # [T]
        }
        torch.save(bundle, out)

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        errors += 1
        tqdm.write(f"  OOM: {f.name} — skipped")
    except Exception as e:
        errors += 1
        tqdm.write(f"  ERROR: {f.name}: {e}")

dt = time.perf_counter() - t0
encoded = len(list(out_dir.glob("*.pt")))
print(f"\nDone in {dt:.0f}s  |  {encoded} files in {out_dir}  "
      f"|  {skipped} skipped  |  {errors} errors")
