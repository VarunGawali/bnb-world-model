"""
collect_dagger.py — Lean DAgger collection to fix the policy's distribution shift.

The policy was trained on strong-branching's OWN trajectory, but at deploy it
visits its own (worse) states it never saw. DAgger closes that gap: run the
CURRENT policy in the B&B search, and at every node it actually visits, query the
strong-branching expert for the correct decision. Aggregating these
policy-state / expert-label pairs into the training set and retraining removes
the compounding error that blows up trees on hard instances.

This is the LEAN variant — deliberately ~10x cheaper than collect_with_cuts_v2:
  * no cut generation, no separator, no HiGHS
  * no independent optimal-solve (value target falls back to per-file min-max)
  * one policy forward pass per node (NOT the expensive lookahead rollout)
It still records the full strong-branching scores over the candidate set, so the
retrain can use soft/ranking imitation (policy_loss_soft), not just the argmax.

Output matches the trajectory schema (traj_*.npz) so the existing
TransitionDataset / train.py Phase-1 pipeline consumes it directly, and can be
aggregated with the original data (mixed hard/soft labels are handled).

Usage
-----
    PYTHONPATH=. python scripts/collect_dagger.py \
        --checkpoint checkpoints/model_final.pt \
        --n_easy 150 --n_medium 150 --n_hard 100 \
        --out_dir data/trajectories --beta 0.0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import ecole
import yaml
from bnb_wm.model.world_model import BnBWorldModel
from bnb_wm.training.checkpoint import load_weights_only
from bnb_wm.evaluate.benchmark import _format_obs
from scripts.collect_with_cuts_v2 import (
    _node_bipartite_arrays, NodeIdentity, AbsoluteDualBound,
)

# Per-tier nominal (n_rows, n_cols, density) — matches generate_instances.py.
TIERS = {
    "SC-easy":   dict(n_rows=200,  n_cols=400,  density=0.05),
    "SC-medium": dict(n_rows=500,  n_cols=1000, density=0.05),
    "SC-hard":   dict(n_rows=1000, n_cols=2000, density=0.05),
}


def _jitter(base, rng):
    return dict(
        n_rows=int(round(base["n_rows"] * rng.uniform(0.85, 1.15))),
        n_cols=int(round(base["n_cols"] * rng.uniform(0.85, 1.15))),
        density=float(np.clip(base["density"] * rng.uniform(0.8, 1.2), 0.02, 0.2)),
    )


def _make_env(time_limit):
    params = {
        "limits/time": time_limit,
        "separating/maxrounds": 0, "separating/maxroundsroot": 0,
        "presolving/maxrounds": 0,
        "heuristics/trysol/freq": -1, "heuristics/feaspump/freq": -1,
        "heuristics/rens/freq": -1, "heuristics/rins/freq": -1,
    }
    return ecole.environment.Branching(
        observation_function=ecole.observation.NodeBipartite(),
        information_function={
            "scores": ecole.observation.StrongBranchingScores(),
            "dual_bound": AbsoluteDualBound(),
            "node": NodeIdentity(),
        },
        scip_params=params,
    )


@torch.no_grad()
def _policy_action(model, obs, action_set, device):
    """One-pass policy argmax over the candidate set (the DEPLOYED one-pass
    policy — no lookahead, so collection stays lean and samples exactly the
    states that policy visits)."""
    batch = _format_obs(obs, device)
    h_vars, z = model.encode(batch)
    var_mask = batch.node_type == 0
    scores = model.policy(h_vars, z[batch.batch[var_mask]])
    aset = torch.as_tensor(np.asarray(action_set, dtype=np.int64), device=device)
    masked = torch.full_like(scores, -1e4)
    masked[aset] = scores[aset]
    return int(masked.argmax())


def _record_instance(model, env, ecole_model, device, max_steps, beta, rng):
    try:
        obs, action_set, _, done, info = env.reset(ecole_model)
    except Exception as e:
        print(f"  reset failed: {e}", flush=True)
        return None
    if done or action_set is None or not len(action_set):
        return None

    keys = ("var_features", "con_features", "edge_indices", "edge_values",
            "action_sets", "branching_vars", "local_branching_label",
            "sb_scores", "dual_bounds", "depths", "node_ids", "parent_ids",
            "branch_dirs")
    buf = {k: [] for k in keys}

    step = 0
    while not done and action_set is not None and len(action_set) and step < max_steps:
        scores = info.get("scores")
        if scores is None:
            break
        scores = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=-1e30)
        aset = np.asarray(action_set, dtype=np.int64)
        sb_local = int(scores[aset].argmax())               # EXPERT label

        # DAgger: execute the POLICY (so we sample the policy's own states);
        # with prob beta fall back to the expert for stability.
        if rng.random() < beta:
            chosen = int(aset[sb_local])
        else:
            try:
                pa = _policy_action(model, obs, action_set, device)
            except Exception:
                pa = int(aset[sb_local])
            chosen = pa

        vf, cf, ei, ev = _node_bipartite_arrays(obs)
        nid, pid, bdir, depth, local_lb = info.get(
            "node", (-1, -1, 0, -1, float("nan")))
        if not np.isfinite(local_lb):
            local_lb = float(info.get("dual_bound", 0.0))

        buf["var_features"].append(vf); buf["con_features"].append(cf)
        buf["edge_indices"].append(ei); buf["edge_values"].append(ev)
        buf["action_sets"].append(aset.astype(np.int32))
        buf["branching_vars"].append(chosen)
        buf["local_branching_label"].append(sb_local)       # target = expert
        buf["sb_scores"].append(scores[aset].astype(np.float32))
        buf["dual_bounds"].append(float(local_lb))
        buf["depths"].append(int(depth))
        buf["node_ids"].append(int(nid)); buf["parent_ids"].append(int(pid))
        buf["branch_dirs"].append(int(bdir))

        try:
            obs, action_set, _, done, info = env.step(chosen)
        except Exception as e:
            print(f"  step failed: {e}", flush=True)
            break
        step += 1

    n = len(buf["branching_vars"])
    if n == 0:
        return None

    db = np.asarray(buf["dual_bounds"], dtype=np.float32)
    # next_is_leaf proxy: a node is a "leaf" for training if it was the last
    # recorded step (no recorded child). Coarse but only used for the aux head.
    next_is_leaf = np.zeros(n, dtype=np.float32); next_is_leaf[-1] = 1.0
    return {
        "n_steps": np.asarray(n),
        "var_features": np.asarray(buf["var_features"], dtype=object),
        "con_features": np.asarray(buf["con_features"], dtype=object),
        "edge_indices": np.asarray(buf["edge_indices"], dtype=object),
        "edge_values": np.asarray(buf["edge_values"], dtype=object),
        "action_sets": np.asarray(buf["action_sets"], dtype=object),
        "branching_vars": np.asarray(buf["branching_vars"], dtype=np.int32),
        "local_branching_label": np.asarray(buf["local_branching_label"], dtype=np.int32),
        "sb_scores": np.asarray(buf["sb_scores"], dtype=object),
        "dual_bounds": db,
        "norm_dual_bounds": ((db - db.min()) / (np.ptp(db) + 1e-8)).astype(np.float32),
        "next_is_leaf": next_is_leaf,
        "depths": np.asarray(buf["depths"], dtype=np.int32),
        "node_ids": np.asarray(buf["node_ids"], dtype=np.int64),
        "parent_ids": np.asarray(buf["parent_ids"], dtype=np.int64),
        "branch_dirs": np.asarray(buf["branch_dirs"], dtype=np.int32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n_easy", type=int, default=150)
    ap.add_argument("--n_medium", type=int, default=150)
    ap.add_argument("--n_hard", type=int, default=100)
    ap.add_argument("--out_dir", default="data/trajectories",
                    help="written under <out_dir>/<tier>/ so it aggregates with "
                         "the existing data (list_trajectory_files picks it up)")
    ap.add_argument("--beta", type=float, default=0.0,
                    help="prob of executing the EXPERT instead of the policy "
                         "(0 = pure policy states, the standard DAgger target)")
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--time_limit", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(args.config))["model"]
    model = BnBWorldModel(
        hidden_dim=cfg["hidden_dim"], n_gnn_layers=cfg["n_gnn_layers"],
        n_gnn_heads=cfg["n_gnn_heads"], n_dyn_layers=cfg["n_dyn_layers"],
        n_dyn_heads=cfg["n_dyn_heads"], max_seq=cfg["max_seq"],
        dyn_residual=cfg.get("dyn_residual", True),
        dyn_heteroscedastic=cfg.get("dyn_heteroscedastic", False),
    ).to(device)
    load_weights_only(model, args.checkpoint, device=device)
    model.eval()
    print(f"Loaded {args.checkpoint} on {device} | beta={args.beta}", flush=True)

    counts = {"SC-easy": args.n_easy, "SC-medium": args.n_medium,
              "SC-hard": args.n_hard}
    env = _make_env(args.time_limit)
    total_nodes = 0
    for tier, n in counts.items():
        if n <= 0:
            continue
        out = Path(args.out_dir) / tier
        out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(args.seed + hash(tier) % 10_000)
        gen = ecole.instance.SetCoverGenerator(**_jitter(TIERS[tier], rng))
        got = 0
        for i in range(n):
            try:
                gen.seed(args.seed + i)
                instance = next(gen)
                traj = _record_instance(
                    model, env, instance, device,
                    args.max_steps, args.beta, rng)
            except Exception as e:
                print(f"  [{tier} {i}] error: {e}", flush=True)
                continue
            if traj is None:
                continue
            path = out / f"traj_dagger_{tier}_{args.seed + i}.npz"
            np.savez_compressed(path, **traj)
            got += 1
            total_nodes += int(traj["n_steps"])
            if (i + 1) % 10 == 0:
                print(f"  [{tier}] {i + 1}/{n} | {got} saved | "
                      f"last steps={int(traj['n_steps'])}", flush=True)
        print(f"{tier}: {got}/{n} trajectories saved -> {out}", flush=True)

    print(f"\nDone. {total_nodes} total nodes collected. Retrain Phase 1:", flush=True)
    print("  python train.py --config configs/default.yaml "
          f"--data_root {args.out_dir} --phases 1 --with_cuts ...", flush=True)


if __name__ == "__main__":
    main()
