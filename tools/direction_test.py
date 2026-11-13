"""
direction_test.py — Does dynamics_step_full distinguish d_t=+1 vs d_t=-1?

If the dynamics model is direction-blind, child latent states z(+1) and z(-1)
will be nearly identical (relative split << 1e-2), bound predictions will be
equal for both children, and pseudo-cost scoring cannot usefully rank variables.

Usage
-----
    PYTHONPATH=. python tools/direction_test.py \
        --checkpoint checkpoints/highs_retrain/phase4_final.pt \
        --n_rows 100 --n_cols 200 --seed 1 --n_vars 10 --n_instances 5
"""

from __future__ import annotations

import argparse
import numpy as np
import torch


def gen_instance(n_rows, n_cols, rng, density=0.05):
    A = (rng.random((n_rows, n_cols)) < density).astype(np.float64)
    for i in range(n_rows):
        if A[i].sum() == 0:
            A[i, rng.integers(n_cols)] = 1.0
    c = rng.uniform(1.0, 10.0, n_cols)
    b = np.ones(n_rows, dtype=np.float64)
    return A, b, c


def encode_instance(model, device, A, b, c):
    """Encode a root-LP instance; return (h_vars, z, x_lp, frac_idx) or None."""
    import highspy
    from bnb_wm.features import var_features, con_features, edge_arrays
    from bnb_wm.features import sol_from_highs

    m_rows, n_cols = A.shape

    # solve root LP
    hs = highspy.Highs()
    hs.setOptionValue("output_flag", False)
    hs.setOptionValue("presolve", "off")
    lb = np.zeros(n_cols)
    ub = np.ones(n_cols)
    hs.addVars(n_cols, lb, ub)
    col_idx = np.arange(n_cols, dtype=np.int32)
    hs.changeColsCost(n_cols, col_idx, c.astype(np.float64))
    inf = highspy.kHighsInf
    for i in range(m_rows):
        idx = np.where(np.abs(A[i]) > 1e-12)[0]
        if len(idx) == 0:
            continue
        hs.addRow(float(b[i]), inf, len(idx),
                  idx.astype(np.int32), A[i, idx].astype(np.float64))
    hs.run()

    sol = sol_from_highs(highspy, hs, n_cols, m_rows)
    x_lp = sol["x"]
    frac = np.abs(x_lp - np.round(x_lp))
    frac_idx = np.where(frac > 1e-4)[0]
    if len(frac_idx) == 0:
        return None

    # build graph features (same path as NeuralBnBSolver._structure + _encode)
    ei, ev = edge_arrays(A)
    rhs_src = b[ei[0]].astype(np.float32)
    edge_attr_np = np.stack(
        [ev, ev / (np.abs(rhs_src) + 1e-8), np.sign(ev)], axis=1
    ).astype(np.float32)
    con_to_var = np.vstack([ei[0] + n_cols, ei[1]])
    var_to_con = np.vstack([ei[1], ei[0] + n_cols])
    edge_index_np = np.hstack([con_to_var, var_to_con]).astype(np.int64)
    edge_attr_np  = np.concatenate([edge_attr_np, edge_attr_np], axis=0)

    n_rows_per_var = (A != 0).sum(axis=0).astype(np.float32)
    vf = var_features(A, b, c, sol, n_rows_per_var)
    cf = con_features(A, b, c, sol)

    x_np = np.zeros((n_cols + m_rows, 19), dtype=np.float32)
    x_np[:n_cols] = vf
    x_np[n_cols:, :5] = cf

    x          = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    edge_index = torch.as_tensor(edge_index_np, dtype=torch.long, device=device)
    edge_attr  = torch.as_tensor(edge_attr_np, dtype=torch.float32, device=device)
    node_type  = torch.cat([
        torch.zeros(n_cols, dtype=torch.long, device=device),
        torch.ones(m_rows,  dtype=torch.long, device=device),
    ])
    batch = torch.zeros(n_cols + m_rows, dtype=torch.long, device=device)

    with torch.no_grad():
        h_vars, z = model.encoder(x, edge_index, node_type, batch,
                                  edge_attr=edge_attr)

    return h_vars, z, x_lp, frac_idx, frac


def test_direction(model, device, A, b, c, n_test=10):
    enc = encode_instance(model, device, A, b, c)
    if enc is None:
        return None
    h_vars, z, x_lp, frac_idx, frac = enc

    results = []
    for vi in frac_idx[:n_test]:
        a_t = h_vars[vi].unsqueeze(0)   # [1, H]
        h_t = h_vars.unsqueeze(0)       # [1, V, H]
        with torch.no_grad():
            z_up,   _, _ = model.dynamics_step_full(z, a_t, h_t, None, d_t=+1.0)
            z_down, _, _ = model.dynamics_step_full(z, a_t, h_t, None, d_t=-1.0)

        diff     = (z_up - z_down).norm().item()
        rel_split = diff / (z.norm().item() + 1e-12)

        b_up   = model.dynamics_bound_pred(z_up).item()
        b_down = model.dynamics_bound_pred(z_down).item()
        b_base = model.dynamics_bound_pred(z).item()

        results.append({
            "var": int(vi),
            "frac": float(frac[vi]),
            "rel_split": rel_split,
            "delta_up": b_up   - b_base,
            "delta_down": b_down - b_base,
            "asym": abs((b_up - b_base) - (b_down - b_base)),
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n_rows",      type=int, default=100)
    ap.add_argument("--n_cols",      type=int, default=200)
    ap.add_argument("--seed",        type=int, default=1)
    ap.add_argument("--n_instances", type=int, default=5)
    ap.add_argument("--n_vars",      type=int, default=10,
                    help="Fractional variables to test per instance.")
    ap.add_argument("--device",      default="cuda")
    args = ap.parse_args()

    import torch
    from bnb_wm.model.world_model import BnBWorldModel
    from bnb_wm.training.checkpoint import load_weights_only

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = BnBWorldModel().to(device)
    load_weights_only(model, args.checkpoint, device=device)
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint} on {device}\n")

    rng = np.random.default_rng(args.seed)
    all_splits     = []
    all_delta_up   = []
    all_delta_down = []

    for inst_i in range(args.n_instances):
        A, b, c = gen_instance(args.n_rows, args.n_cols, rng)
        res = test_direction(model, device, A, b, c, n_test=args.n_vars)
        if res is None:
            print(f"Instance {inst_i}: root LP is integral — skipped\n")
            continue

        print(f"Instance {inst_i}  ({len(res)} fractional vars tested):")
        print(f"  {'var':>5}  {'frac':>6}  {'rel_split':>10}  "
              f"{'Δbnd_up':>9}  {'Δbnd_dn':>9}  {'|up-dn|':>8}")
        for r in res:
            print(f"  {r['var']:>5}  {r['frac']:>6.3f}  {r['rel_split']:>10.4e}  "
                  f"  {r['delta_up']:>8.4f}  {r['delta_down']:>8.4f}  "
                  f"  {r['asym']:>7.4f}")
            all_splits.append(r['rel_split'])
            all_delta_up.append(r['delta_up'])
            all_delta_down.append(r['delta_down'])
        print()

    if not all_splits:
        print("No fractional LP instances found.")
        return

    n = len(all_splits)
    arr_splits = np.array(all_splits)
    asym_arr   = np.abs(np.array(all_delta_up) - np.array(all_delta_down))

    print(f"{'='*65}")
    print(f"Summary over {n} (instance, var) pairs:")
    print(f"  rel_split   mean={np.mean(arr_splits):.3e}  "
          f"median={np.median(arr_splits):.3e}  "
          f"max={np.max(arr_splits):.3e}")
    print(f"  Δbound_up   mean={np.mean(all_delta_up):.4f}  "
          f"std={np.std(all_delta_up):.4f}")
    print(f"  Δbound_dn   mean={np.mean(all_delta_down):.4f}  "
          f"std={np.std(all_delta_down):.4f}")
    print(f"  |up−down|   mean={np.mean(asym_arr):.4f}  "
          f"max={np.max(asym_arr):.4f}")
    print()

    med = np.median(arr_splits)
    if med < 1e-3:
        print("VERDICT: DIRECTION-BLIND  (median rel_split < 1e-3)")
        print("  The dynamics model produces nearly identical latent states for")
        print("  d_t=+1 and d_t=-1.  Every dyn_pseudo result is uninformative.")
    elif med < 0.05:
        print("VERDICT: WEAKLY DIRECTION-AWARE  (1e-3 ≤ median rel_split < 0.05)")
        print("  Small but non-trivial signal; dyn_pseudo may be underpowered")
        print("  rather than fundamentally broken.")
    else:
        print("VERDICT: DIRECTION-AWARE  (median rel_split ≥ 0.05)")
        print("  dyn_pseudo failure must have another cause (calibration, scale).")


if __name__ == "__main__":
    main()
