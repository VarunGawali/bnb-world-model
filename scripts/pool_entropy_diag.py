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
from torch.utils.data import Dataset as TorchDataset

from bnb_wm.model.world_model import BnBWorldModel
from bnb_wm.data.datasets import (
    list_trajectory_files, split_files, TransitionDataset,
)


class _GraphOnlyDataset(TorchDataset):
    """Wraps TransitionDataset and returns only the PyG Data (drops meta)."""
    def __init__(self, inner):
        self._inner = inner
    def __len__(self):
        return len(self._inner)
    def __getitem__(self, i):
        data, _ = self._inner[i]
        return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",   required=True)
    ap.add_argument("--data_root",    default="data/")
    ap.add_argument("--n_instances",  type=int, default=200)
    ap.add_argument("--batch_size",   type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Checkpoints may use different key names depending on the saving code.
    state_dict_key = next(
        (k for k in ("model_state_dict", "state_dict", "model") if k in ckpt),
        None,
    )
    if state_dict_key is None:
        # The checkpoint IS the state dict (plain torch.save(model.state_dict()))
        state_dict = ckpt
        cfg = {}
    else:
        state_dict = ckpt[state_dict_key]
        cfg = ckpt.get("config", ckpt.get("cfg", {}))

    print(f"Checkpoint keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")
    print(f"Using state_dict key: {state_dict_key!r}")

    mcfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    model = BnBWorldModel(
        hidden_dim   = mcfg.get("hidden_dim",   128),
        n_gnn_layers = mcfg.get("n_gnn_layers", 3),
        n_gnn_heads  = mcfg.get("n_gnn_heads",  4),
        n_dyn_layers = mcfg.get("n_dyn_layers", 4),
        n_dyn_heads  = mcfg.get("n_dyn_heads",  4),
        max_seq      = mcfg.get("max_seq",      512),
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    all_files = list_trajectory_files(args.data_root)
    _, _, test_files = split_files(all_files)
    test_files = test_files[:args.n_instances]
    dataset = _GraphOnlyDataset(TransitionDataset(test_files))
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
