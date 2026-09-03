"""
rl_finetune.py — RL fine-tuning of the branching policy (past the imitation ceiling).

Imitation (SB / DAgger) is provably capped here: pushing harder on the SB-argmax
proxy stops helping and even regresses, because SB-match != node-optimality. RL
optimizes the TRUE objective directly: minimize the B&B tree size.

Formulation (episodic policy gradient / REINFORCE, tree-MDP flavour):
  * State  = a B&B node (bipartite graph); action = branch on a candidate var.
  * The policy SAMPLES a candidate from its softmax over the action set.
  * An episode = solving one instance; its length N = number of branchings.
  * Reward   = -log(1 + N)  (fewer nodes -> higher return). log compresses the
    order-of-magnitude spread in tree sizes so a few huge instances don't
    dominate the gradient.
  * Advantage = R - baseline, standardized across the batch of episodes
    (baseline = mean return; standardization is the variance-reduction).
  * Update: for every recorded step, grad of  -advantage * logpi(a|s), plus an
    entropy bonus to keep exploration alive. Warm-started from the DAgger policy,
    small LR -> this is *fine-tuning*, not training from scratch.

Memory: rollouts are collected WITHOUT grad (store the per-node graph, the
candidate set, the sampled action, and the episode advantage), then the policy
gradient is applied in shuffled minibatches that RE-RUN the policy with grad —
so peak memory is one minibatch, not a whole tree of retained graphs.

Warning: episodic REINFORCE on B&B trees is high variance. Use many episodes per
iteration, keep LR small, and watch the greedy-eval node count — if it degrades,
lower the LR or raise entropy. This is the honest RL attempt; a per-node
subtree-credit (true tree MDP) would reduce variance further but needs in-episode
subtree tracking Ecole does not expose here.

Usage
-----
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/rl_finetune.py \
        --checkpoint checkpoints/model_dagger_r1.pt \
        --iterations 40 --episodes_per_iter 16 --lr 5e-5 --out_dir checkpoints
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).parent.parent))

import ecole
import yaml
from torch_geometric.data import Data, Batch
from bnb_wm.model.world_model import BnBWorldModel
from bnb_wm.training.checkpoint import load_weights_only, save_checkpoint
from bnb_wm.data.datasets import FEATURE_CLIP as _FC

TIERS = {
    "SC-easy":   dict(n_rows=200,  n_cols=400,  density=0.05),
    "SC-medium": dict(n_rows=500,  n_cols=1000, density=0.05),
    "SC-hard":   dict(n_rows=1000, n_cols=2000, density=0.05),
}


def _obs_to_data(obs):
    """Ecole NodeBipartite obs -> a single PyG Data (CPU), matching _format_obs /
    build_pyg_data (bidirectional edges, 3-dim edge_attr, clipped features)."""
    vf_raw = (obs.variable_features if hasattr(obs, "variable_features")
              else obs.column_features)
    cf_raw = (obs.constraint_features if hasattr(obs, "constraint_features")
              else obs.row_features)
    vf = np.clip(np.nan_to_num(np.asarray(vf_raw, np.float32),
                               nan=0.0, posinf=_FC, neginf=-_FC), -_FC, _FC)
    cf = np.clip(np.nan_to_num(np.asarray(cf_raw, np.float32),
                               nan=0.0, posinf=_FC, neginf=-_FC), -_FC, _FC)
    ei = np.asarray(obs.edge_features.indices, np.int64)
    ev = np.asarray(obs.edge_features.values, np.float32)
    if ev.ndim == 2:
        ev = ev[:, 0]
    ev = np.nan_to_num(ev.flatten(), nan=0.0, posinf=1e6, neginf=-1e6)
    rhs = cf[ei[0], 1] if cf.shape[1] > 1 else np.ones(len(ei[0]), np.float32)
    ea = np.stack([ev, ev / (np.abs(rhs) + 1e-8), np.sign(ev)], 1).astype(np.float32)

    nv, nc = vf.shape[0], cf.shape[0]
    x = torch.cat([torch.tensor(vf), F.pad(torch.tensor(cf), (0, 14))], 0)
    node_type = torch.cat([torch.zeros(nv, dtype=torch.long),
                           torch.ones(nc, dtype=torch.long)])
    ei_t = torch.tensor(ei, dtype=torch.long)
    ea_t = torch.tensor(ea)
    c2v = torch.stack([ei_t[0] + nv, ei_t[1]], 0)
    v2c = torch.stack([ei_t[1], ei_t[0] + nv], 0)
    edge_index = torch.cat([c2v, v2c], 1)
    edge_attr = torch.cat([ea_t, ea_t], 0)
    return Data(x=x, edge_index=edge_index, node_type=node_type,
                edge_attr=edge_attr)


def _policy_logits(model, batch, action_set_local):
    """Return policy logits over a single graph's candidate set (local indices)."""
    h_vars, z = model.encode(batch)
    var_mask = batch.node_type == 0
    scores = model.policy(h_vars, z[batch.batch[var_mask]])
    return scores[action_set_local]


@torch.no_grad()
def rollout(model, instance, device, max_steps, greedy):
    """Run one episode. Returns (steps, n_nodes) where steps is a list of
    (Data_cpu, action_set_tensor, sampled_local_action)."""
    env = ecole.environment.Branching(
        observation_function=ecole.observation.NodeBipartite(),
        scip_params={"limits/time": 30, "separating/maxrounds": 0,
                     "presolving/maxrounds": 0},
    )
    try:
        obs, action_set, _, done, _ = env.reset(instance)
    except Exception:
        return [], max_steps
    steps = []
    n = 0
    while not done and action_set is not None and len(action_set) and n < max_steps:
        aset = np.asarray(action_set, dtype=np.int64)
        data = _obs_to_data(obs)
        batch = Batch.from_data_list([data]).to(device)
        aset_t = torch.as_tensor(aset, dtype=torch.long, device=device)
        logits = _policy_logits(model, batch, aset_t)
        if greedy:
            a_local = int(logits.argmax())
        else:
            a_local = int(Categorical(logits=logits).sample())
        steps.append((data, aset, a_local))
        try:
            obs, action_set, _, done, _ = env.step(int(aset[a_local]))
        except Exception:
            break
        n += 1
    return steps, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--episodes_per_iter", type=int, default=16)
    ap.add_argument("--minibatch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--unfreeze_encoder", action="store_true")
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--eval_instances", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="checkpoints")
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
    print(f"Loaded {args.checkpoint} on {device}", flush=True)

    for name, p in model.named_parameters():
        p.requires_grad = ("policy" in name) or (
            args.unfreeze_encoder and "encoder" in name)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=args.lr)
    print(f"Trainable params: {sum(p.numel() for p in trainable):,} "
          f"(encoder {'unfrozen' if args.unfreeze_encoder else 'frozen'})",
          flush=True)

    rng = np.random.default_rng(args.seed)
    tiers = list(TIERS)
    generators = {t: ecole.instance.SetCoverGenerator(**TIERS[t]) for t in tiers}
    for t in tiers:
        generators[t].seed(args.seed)

    # fixed eval instances (greedy node count -> the metric we actually track)
    eval_insts = []
    for i in range(args.eval_instances):
        t = tiers[i % len(tiers)]
        eval_insts.append(next(generators[t]))

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    best_eval = float("inf")

    for it in range(1, args.iterations + 1):
        # ---- rollout phase (no grad): sample episodes, collect steps ----
        model.eval()
        all_steps, ep_nodes, ep_returns = [], [], []
        for e in range(args.episodes_per_iter):
            t = tiers[e % len(tiers)]
            inst = next(generators[t])
            steps, n = rollout(model, inst, device, args.max_steps, greedy=False)
            if not steps:
                continue
            R = -np.log1p(n)                    # fewer nodes -> higher return
            ep_nodes.append(n)
            ep_returns.append(R)
            all_steps.append((steps, R))

        if not all_steps:
            print(f"iter {it}: no episodes collected", flush=True)
            continue
        # standardize advantages across episodes (baseline = mean return)
        R_arr = np.array([R for _, R in all_steps], dtype=np.float64)
        adv_mean, adv_std = R_arr.mean(), R_arr.std() + 1e-8
        flat = []   # (data, aset, a_local, advantage)
        for (steps, R) in all_steps:
            adv = float((R - adv_mean) / adv_std)
            for (data, aset, a_local) in steps:
                flat.append((data, aset, a_local, adv))

        # ---- update phase (grad, minibatched policy gradient) ----
        model.train()
        rng.shuffle(flat)
        pg_loss_sum = ent_sum = m = 0
        for s in range(0, len(flat), args.minibatch):
            mb = flat[s:s + args.minibatch]
            opt.zero_grad(set_to_none=True)
            losses, ents = [], []
            try:
                for (data, aset, a_local, adv) in mb:
                    batch = Batch.from_data_list([data]).to(device)
                    aset_t = torch.as_tensor(aset, dtype=torch.long, device=device)
                    logits = _policy_logits(model, batch, aset_t)
                    dist = Categorical(logits=logits)
                    logp = dist.log_prob(torch.tensor(a_local, device=device))
                    losses.append(-adv * logp)
                    ents.append(dist.entropy())
                if not losses:
                    continue
                pg = torch.stack(losses).mean()
                ent = torch.stack(ents).mean()
                loss = pg - args.entropy_coef * ent
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                pg_loss_sum += float(pg.detach())
                ent_sum += float(ent.detach())
                m += 1
            except RuntimeError as ex:
                if "out of memory" in str(ex).lower():
                    opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                    continue
                raise

        mean_nodes = float(np.mean(ep_nodes)) if ep_nodes else float("nan")
        print(f"iter {it:03d} | sampled mean nodes={mean_nodes:.1f} "
              f"(min {min(ep_nodes)}, max {max(ep_nodes)}) | "
              f"pg_loss={pg_loss_sum/max(1,m):.4f} ent={ent_sum/max(1,m):.3f}",
              flush=True)

        # ---- greedy eval on fixed instances ----
        if it % args.eval_every == 0 or it == args.iterations:
            model.eval()
            ev = [rollout(model, inst, device, args.max_steps, greedy=True)[1]
                  for inst in eval_insts]
            mn = float(np.mean(ev))
            print(f"  [eval] greedy mean nodes={mn:.1f} over {len(ev)} instances",
                  flush=True)
            if mn < best_eval:
                best_eval = mn
                save_checkpoint(model, opt, it, {"eval_nodes": mn},
                                Path(args.out_dir) / "model_rl_best.pt")
                print(f"    saved model_rl_best.pt (eval nodes {mn:.1f})", flush=True)

    print(f"\nDone. Best greedy eval mean nodes: {best_eval:.1f}", flush=True)
    print("Benchmark model_rl_best.pt against model_dagger_r1.pt with ablation.py.",
          flush=True)


if __name__ == "__main__":
    main()
