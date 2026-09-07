"""
Offline Step 2: encode all cut-transition files with the frozen encoder.

Output: data/encoded_cuts/<stem>.pt per transition, each containing:
    z_before  [H]    latent of LP state before cut
    z_after   [H]    latent of LP state after cut
    cut_feats [6]    cut feature vector
    delta_lb  scalar improvement in lower bound

Run once before Phase-3 training:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/encode_cuts.py
"""
import argparse, time, torch, yaml, numpy as np
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Batch

parser = argparse.ArgumentParser()
parser.add_argument("--cut_dir",    default="data/cut_transitions")
parser.add_argument("--out_dir",    default="data/encoded_cuts")
parser.add_argument("--config",     default="configs/default.yaml")
parser.add_argument("--checkpoint", default="checkpoints/model_rl_best.pt")
parser.add_argument("--batch_size", type=int, default=8,
                    help="cut graphs to encode together (reduce if OOM)")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

cfg = yaml.safe_load(open(args.config))
from bnb_wm.model.world_model import BnBWorldModel
model = BnBWorldModel(**cfg["model"]).to(device).eval()
sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
model.load_state_dict(sd["model"], strict=False)
print("Encoder loaded (frozen).")

from bnb_wm.data.datasets import build_pyg_data

cut_files = sorted(Path(args.cut_dir).rglob("*.npz"))
print(f"Cut transition files: {len(cut_files)}")

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

skipped = errors = 0
t0 = time.perf_counter()

for f in tqdm(cut_files, desc="encoding cuts"):
    out = out_dir / (f.stem + ".pt")
    if out.exists():
        continue

    try:
        d = np.load(f, allow_pickle=True)

        # Support both key naming conventions.
        vf_b = d["vf_before"] if "vf_before" in d else d["var_features_before"]
        cf_b = d["cf_before"] if "cf_before" in d else d["con_features_before"]
        ei_b = d["ei_before"] if "ei_before" in d else d["edge_indices_before"]
        ev_b = d["ev_before"] if "ev_before" in d else d["edge_values_before"]
        vf_a = d["vf_after"]  if "vf_after"  in d else d["var_features_after"]
        cf_a = d["cf_after"]  if "cf_after"  in d else d["con_features_after"]
        ei_a = d["ei_after"]  if "ei_after"  in d else d["edge_indices_after"]
        ev_a = d["ev_after"]  if "ev_after"  in d else d["edge_values_after"]

        g_before = build_pyg_data(vf_b, cf_b, ei_b, ev_b)
        g_after  = build_pyg_data(vf_a, cf_a, ei_a, ev_a)

        with torch.no_grad():
            gb = Batch.from_data_list([g_before]).to(device)
            ga = Batch.from_data_list([g_after]).to(device)
            _, z_before = model.encode(gb)   # [1, H]
            _, z_after  = model.encode(ga)   # [1, H]

        cut_feats = torch.as_tensor(
            np.asarray(d["cut_feats"], dtype=np.float32))  # [6]
        delta_lb = float(d["delta_lb"]) if "delta_lb" in d else 0.0

        bundle = {
            "z_before":  z_before.squeeze(0).cpu(),  # [H]
            "z_after":   z_after.squeeze(0).cpu(),   # [H]
            "cut_feats": cut_feats,                   # [6]
            "delta_lb":  torch.tensor(delta_lb),      # scalar
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
