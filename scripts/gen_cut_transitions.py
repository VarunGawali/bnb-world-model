"""
gen_cut_transitions.py — Generate cut transition pairs for dynamics training.

For each trajectory file (at the ROOT node, t=0):
  1. Load LP matrices (A, b, c) and GNN features.
  2. Encode the root node → z_before [H].
  3. Generate one Gomory cut via generate_cg_cuts().
  4. Inject the cut, re-solve LP (HiGHS warm-start from LP basis).
  5. Re-encode → z_after [H].
  6. Store (z_before, cut_feats [6], z_after) as .npz in data/cut_transitions/.

Run ONCE before Phase 3 retraining:
    uv run python scripts/gen_cut_transitions.py \\
        --data_dir data/trajectories \\
        --out_dir  data/cut_transitions \\
        --checkpoint checkpoints/phase2_best.pt

The trainer's _dynamics_batch_loss picks up these files automatically when
a CutTransitionDataset is added to the Phase 3 loader.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


def _build_cut_features(cut, x_lp, c, A, b):
    """6-dim Gomory cut feature vector — mirrors collect_with_cuts_v2."""
    coeffs  = np.asarray(cut["coeffs"], dtype=np.float64)
    rhs     = float(cut["rhs"])
    n       = len(x_lp)

    violation     = float(np.dot(coeffs, x_lp) - rhs)
    norm_viol     = violation / (np.linalg.norm(coeffs) + 1e-8)
    density       = float((coeffs != 0).mean())
    obj_dot       = float(np.dot(coeffs, c))
    obj_norm      = obj_dot / (np.linalg.norm(coeffs) * np.linalg.norm(c) + 1e-8)
    active        = coeffs != 0
    frac          = np.abs(x_lp[active] - np.round(x_lp[active]))
    frac_support  = float(frac.mean()) if active.any() else 0.0
    gain_proxy    = float(violation * (np.linalg.norm(coeffs) + 1e-8))

    return np.array([violation, norm_viol, density, obj_norm,
                     gain_proxy, frac_support], dtype=np.float32)


def _inject_cut_and_resolve(A, b, c, cut, x_lp_basis):
    """Append cut row and warm-start re-solve. Returns (x_new, A_new, b_new)."""
    try:
        import highspy
    except ImportError:
        raise ImportError("highspy required: uv pip install highspy")

    coeffs = np.asarray(cut["coeffs"], dtype=np.float64)
    rhs    = float(cut["rhs"])

    A_new = np.vstack([A, coeffs[np.newaxis, :]])
    b_new = np.append(b, rhs)

    n = A_new.shape[1]
    m = A_new.shape[0]

    h = highspy.Highs()
    h.silent()

    inf = highspy.kHighsInf
    h.addVars(n, np.zeros(n), np.full(n, inf))
    h.changeColsCost(n, np.arange(n, dtype=np.int32), c.astype(np.float64))

    for i in range(m):
        row = A_new[i]
        nz  = row.nonzero()[0]
        h.addRow(-inf, b_new[i], len(nz),
                 nz.astype(np.int32), row[nz].astype(np.float64))

    h.run()
    sol = h.getInfoValue("primal_solution_status")[1]
    if sol != 2:   # not optimal
        return None, A_new, b_new

    x_new = np.array(h.getInfoValue("sol")[1] if False else
                     [h.getInfoValue(f"col_value_{i}")[1] for i in range(n)],
                     dtype=np.float64)
    # Simpler: use getSolution
    sol_obj = h.getSolution()
    x_new = np.asarray(sol_obj.col_value, dtype=np.float64)
    return x_new, A_new, b_new


def _encode_node(model, var_feats, con_feats, edge_idx, edge_vals, device):
    """GNN encode a single B&B node, return z [H]."""
    from torch_geometric.data import Data, Batch
    from bnb_wm.data.graph_utils import build_pyg_data

    data = build_pyg_data(var_feats, con_feats, edge_idx, edge_vals)
    batch = Batch.from_data_list([data]).to(device)
    with torch.no_grad():
        h_vars, z, _ = model.encoder(
            batch.x, batch.edge_index, batch.node_type, batch.batch,
            edge_attr=getattr(batch, "edge_attr", None),
        )
    return z[0].cpu()


def process_file(fpath, model, device, out_dir):
    d = np.load(fpath, allow_pickle=True)
    T = int(d["n_steps"])
    if T == 0:
        return False

    # Use root node (t=0).
    var_feats  = d["var_features"][0]
    con_feats  = d["con_features"][0]
    edge_idx   = d["edge_indices"][0]
    edge_vals  = d["edge_values"][0]

    # LP matrices — present in files collected with collect_with_cuts_v2.
    if "A" not in d or "b" not in d or "c" not in d:
        return False

    A  = np.asarray(d["A"],  dtype=np.float64)
    b  = np.asarray(d["b"],  dtype=np.float64)
    c  = np.asarray(d["c"],  dtype=np.float64)
    x_lp = np.asarray(d.get("x_lp", np.zeros(c.shape)), dtype=np.float64)

    # Encode before cut.
    z_before = _encode_node(model, var_feats, con_feats, edge_idx, edge_vals, device)

    # Generate one Gomory cut.
    try:
        from bnb_wm.cuts.cg_cuts import generate_cg_cuts
        cuts = generate_cg_cuts(A, b, c, x_lp=x_lp, max_cuts=1)
    except Exception:
        return False

    if not cuts:
        return False
    cut = cuts[0]

    cut_feats = _build_cut_features(cut, x_lp, c, A, b)

    # Inject cut and re-solve.
    x_new, A_new, b_new = _inject_cut_and_resolve(A, b, c, cut, x_lp)
    if x_new is None:
        return False

    # Re-encode after cut (we can't rebuild full GNN features without SCIP,
    # so we approximate: re-use the structural features but update LP solution
    # columns with the new x_lp. Feature columns 0..8 = LP solution values;
    # column 14 = fractionality. This is an approximation — a full re-encode
    # from SCIP would be more accurate but requires an LP solve + feature rebuild).
    var_feats_new = var_feats.copy().astype(np.float32)
    n_vars = var_feats_new.shape[0]
    n_use  = min(n_vars, len(x_new))
    # Update LP value column (col 0) and fractionality (col 14).
    var_feats_new[:n_use, 0]  = x_new[:n_use].astype(np.float32)
    frac = np.abs(x_new[:n_use] - np.round(x_new[:n_use])).astype(np.float32)
    var_feats_new[:n_use, 14] = frac

    z_after = _encode_node(model, var_feats_new, con_feats, edge_idx, edge_vals, device)

    if torch.allclose(z_before, z_after, atol=1e-5):
        return False   # cut had no effect on the latent — skip

    # Save.
    stem = Path(fpath).stem
    out_path = out_dir / f"{stem}_cut.npz"
    np.savez_compressed(
        out_path,
        z_before=z_before.numpy().astype(np.float32),
        cut_feats=cut_feats,
        z_after=z_after.numpy().astype(np.float32),
    )
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",   required=True)
    ap.add_argument("--out_dir",    required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device",     default="cpu")
    ap.add_argument("--max_files",  type=int, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model.
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from bnb_wm.model.world_model import BnBWorldModel
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = BnBWorldModel()
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()

    files = sorted(Path(args.data_dir).rglob("*.npz"))
    if args.max_files:
        files = files[:args.max_files]

    ok = fail = 0
    for f in files:
        try:
            if process_file(f, model, device, out_dir):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"  skip {f.name}: {e}")
            fail += 1

    print(f"\nDone: {ok} cut transitions written, {fail} skipped → {out_dir}")


if __name__ == "__main__":
    main()
