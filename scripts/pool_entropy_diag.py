"""
pool_entropy_diag.py — Measure CrossAttentionPool attention entropy.

Prints mean/median/min V_eff per graph (effective number of attended
variables). V_eff → 1: peaked, instance-specific z. V_eff → V: diffuse,
z ≈ mean pool.

Usage:
    python scripts/pool_entropy_diag.py \
        --checkpoint checkpoints/best.pt \
        --data_root data/ \
        --n_instances 200
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from bnb_wm.model.world_model import BnBWorldModel
from bnb_wm.data.dataset import BnBDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",   required=True)
    ap.add_argument("--data_root",    default="data/")
    ap.add_argument("--n_instances",  type=int, default=200)
    ap.add_argument("--batch_size",   type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt   = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    cfg = ckpt.get("config", {})
    mcfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    model = BnBWorldModel(
        hidden_dim   = mcfg.get("hidden_dim",   128),
        n_gnn_layers = mcfg.get("n_gnn_layers", 3),
        n_gnn_heads  = mcfg.get("n_gnn_heads",  4),
        n_dyn_layers = mcfg.get("n_dyn_layers", 4),
        n_dyn_heads  = mcfg.get("n_dyn_heads",  4),
        max_seq      = mcfg.get("max_seq",      512),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()

    dataset = BnBDataset(args.data_root, split="test",
                         max_files=args.n_instances)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    all_veff, all_V = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, v_eff = model.pool_entropy(batch)
            # number of variables per graph
            var_mask = batch.node_type == 0
            V_per = torch.zeros(batch.num_graphs, device=device)
            V_per.scatter_add_(0, batch.batch[var_mask],
                               torch.ones(var_mask.sum(), device=device))
            all_veff.append(v_eff.cpu())
            all_V.append(V_per.cpu())

    v_eff_all = torch.cat(all_veff).numpy()
    V_all     = torch.cat(all_V).numpy()
    frac      = v_eff_all / np.maximum(V_all, 1)   # V_eff / V  (0=peaked, 1=diffuse)

    print(f"\n=== CrossAttentionPool entropy diagnostic ({len(v_eff_all)} graphs) ===")
    print(f"V_eff  mean={v_eff_all.mean():.1f}  median={np.median(v_eff_all):.1f}"
          f"  min={v_eff_all.min():.1f}  max={v_eff_all.max():.1f}")
    print(f"V      mean={V_all.mean():.0f}  median={np.median(V_all):.0f}")
    print(f"V_eff/V mean={frac.mean():.3f}  median={np.median(frac):.3f}")
    print()
    print("Interpretation:")
    print(f"  V_eff/V = {frac.mean():.3f} — ", end="")
    if frac.mean() > 0.5:
        print("DIFFUSE attention. z ≈ mean pool. Consider auxiliary loss or"
              " temperature scaling on pool logits.")
    elif frac.mean() > 0.1:
        print("moderately peaked. Acceptable for small graphs; monitor on hard instances.")
    else:
        print("PEAKED. z is instance-specific — pool is working as intended.")


if __name__ == "__main__":
    main()
