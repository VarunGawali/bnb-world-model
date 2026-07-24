"""
ablation.py — Node-count benchmark with an ablation sweep and paired significance.

Evaluates a trained model as the branching rule inside SCIP (via Ecole) against
SCIP's default branching, on a common held-out set of instances, and sweeps the
planning components so each contributes one row of the paper's results table.

For every instance we record the number of B&B nodes SCIP explores under each
policy; because the same instances are used for every method, the per-instance
node counts are paired, and we assess each method against SCIP with a Wilcoxon
signed-rank test (reproducibility checklist items 4.10-4.12).

Ablation configurations (each is the branching rule; SCIP handles the rest):
    scip           SCIP default (pseudocost) branching        [baseline]
    policy_only    argmax of the learned policy, no rollout
    value_rollout  latent rollout scored by value only
    cost_to_go     rollout minus predicted cost-to-go
    tree_rollout   cost-to-go rollout with branching factor 2
    reward_return  MuZero-style return (per-step reward + value bootstrap)

Usage:
    python -m bnb_wm.evaluate.ablation --checkpoint checkpoints/model_final.pt \
        --n_instances 100 --n_rows 500 --n_cols 1000 --time_limit 60 \
        --out results/ablation.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

try:
    import ecole
    from pyscipopt import Model as SCIPModel
except ImportError:
    ecole = None
    SCIPModel = None

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

from bnb_wm.evaluate.benchmark import _format_obs


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------

# Each config is the set of rollout parameters used to pick a branching variable.
# mode "policy" ignores the rollout; mode "rollout" calls model.rollout_candidate.
ABLATIONS = {
    "policy_only":   dict(mode="policy"),
    "value_rollout": dict(mode="rollout", depth=3, gamma=0.95, k=5,
                          ctg_weight=0.0, branch_factor=1, use_reward_return=False),
    "cost_to_go":    dict(mode="rollout", depth=3, gamma=0.95, k=5,
                          ctg_weight=1.0, branch_factor=1, use_reward_return=False),
    "tree_rollout":  dict(mode="rollout", depth=3, gamma=0.95, k=5,
                          ctg_weight=1.0, branch_factor=2, use_reward_return=False),
    "reward_return": dict(mode="rollout", depth=3, gamma=0.95, k=5,
                          ctg_weight=1.0, branch_factor=2, use_reward_return=True),
}

# Classical (non-learned) branching baselines, for a fair comparison spectrum:
# SCIP reliability branching is the strong upper baseline, these are the weak ones.
BASELINES = {
    "random":          dict(mode="random"),
    "most_fractional": dict(mode="most_fractional"),
}
_LEAF_SKIP = 0.8


# ---------------------------------------------------------------------------
# Parameterized branching-variable selection
# ---------------------------------------------------------------------------

def _pick_action(model, batch, action_set, device, cfg, past_tokens):
    """Pick a branching variable under one ablation config; returns (action, tokens)."""
    mode = cfg["mode"]

    # --- classical baselines: no model needed, short-circuit before encoding ---
    if mode == "random":
        return int(np.random.choice(action_set)), past_tokens
    if mode == "most_fractional":
        var_mask = batch.node_type == 0
        vf = batch.x[var_mask]                       # [n_vars, 19]
        # Ecole layout: column 14 = sol_frac = |x - round(x)| in [0, 0.5];
        # most-fractional = largest sol_frac among the candidates.
        frac = vf[:, 14] if vf.size(1) > 14 else torch.zeros(vf.size(0), device=device)
        aset_t = torch.tensor(action_set, dtype=torch.long, device=device)
        best = int(aset_t[int(frac[aset_t].argmax())])
        return best, past_tokens

    # --- learned policy / rollout ---
    h_vars, z = model.encode(batch)
    var_mask  = batch.node_type == 0
    var_batch = batch.batch[var_mask]

    scores_all = model.policy_scores(h_vars, z, var_batch)
    aset_t = torch.tensor(action_set, dtype=torch.long, device=device)
    masked = torch.full_like(scores_all, -1e4)
    masked[aset_t] = scores_all[aset_t]

    # Policy-only, or near-leaf shortcut: take the top policy score.
    if cfg["mode"] == "policy":
        return int(masked.argmax()), past_tokens

    leaf_prob = torch.sigmoid(model.integrality_logit(z)).item()
    if leaf_prob > _LEAF_SKIP:
        return int(masked.argmax()), past_tokens

    k = min(cfg["k"], len(action_set))
    top_k = masked.topk(k).indices
    valid_mask = torch.zeros(scores_all.size(0), dtype=torch.bool, device=device)
    valid_mask[aset_t] = True

    best_action, best_ret = int(top_k[0]), -float("inf")
    for cand in top_k:
        r = model.rollout_candidate(
            z, h_vars, int(cand),
            depth=cfg["depth"], gamma=cfg["gamma"],
            valid_mask=valid_mask, past_tokens=past_tokens,
            size_weight=0.0, ctg_weight=cfg["ctg_weight"],
            branch_factor=cfg["branch_factor"],
            use_reward_return=cfg["use_reward_return"],
        )
        if r > best_ret:
            best_ret, best_action = r, int(cand)

    a_emb = h_vars[best_action].unsqueeze(0)
    _, past_tokens = model.dynamics_step(z, a_emb, past_tokens)
    return best_action, past_tokens


def _scip_node_count(env):
    """SCIP node count for the just-finished Ecole episode (fallback: None)."""
    try:
        return int(env.model.as_pyscipopt().getNNodes())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run(model, device, configs, n_instances, generator_kwargs,
        time_limit=60, seed=0):
    """
    Returns a dict: method -> list of per-instance node counts (aligned by index).
    "scip" is always included as the baseline.
    """
    if ecole is None or SCIPModel is None:
        raise ImportError("Ecole and PySCIPOpt are required for the benchmark.")

    gkw = generator_kwargs
    generator = ecole.instance.SetCoverGenerator(
        n_rows=gkw.get("n_rows", 500),
        n_cols=gkw.get("n_cols", 1000),
        density=gkw.get("density", 0.05),
    )
    generator.seed(seed)
    np.random.seed(seed)   # reproducible random-branching baseline

    scip_params = {
        "limits/time":              time_limit,
        "separating/maxrounds":     0,
        "presolving/maxrounds":     0,
    }
    env = ecole.environment.Branching(
        observation_function=ecole.observation.NodeBipartite(),
        scip_params=scip_params,
    )

    methods = ["scip"] + list(configs.keys())
    nodes = {m: [] for m in methods}
    model.eval()

    print(f"Evaluating {n_instances} instances | methods: {methods}\n")
    for i in range(n_instances):
        instance = next(generator)

        # ---- SCIP default (pseudocost) ----
        m = instance.copy_orig().as_pyscipopt()
        m.hideOutput()
        m.setParam("limits/time", time_limit)
        m.setParam("separating/maxrounds", 0)
        m.setParam("presolving/maxrounds", 0)
        m.optimize()
        nodes["scip"].append(int(m.getNNodes()))

        # ---- each learned config ----
        for name, cfg in configs.items():
            obs, action_set, _, done, _ = env.reset(instance.copy_orig())
            steps, past = 0, None
            with torch.no_grad():
                while not done and action_set is not None and len(action_set) > 0:
                    batch = _format_obs(obs, device)
                    action, past = _pick_action(
                        model, batch, action_set, device, cfg, past
                    )
                    obs, action_set, _, done, _ = env.step(action)
                    steps += 1
            n = _scip_node_count(env)
            nodes[name].append(n if n is not None else steps)

        row = " | ".join(f"{m}:{nodes[m][-1]}" for m in methods)
        print(f"  [{i+1:3d}/{n_instances}] {row}")

    return nodes


def summarize(nodes):
    """Print and return a per-method summary vs. SCIP with Wilcoxon significance."""
    scip = np.asarray(nodes["scip"], dtype=float)
    rows = []
    print("\n" + "=" * 72)
    print(f"{'Method':<16}{'nodes(mean±std)':<22}{'median':<10}"
          f"{'vs SCIP':<10}{'p':<10}")
    print("-" * 72)
    for m, vals in nodes.items():
        v = np.asarray(vals, dtype=float)
        mean, std, med = v.mean(), v.std(), np.median(v)
        if m == "scip":
            red, p = 0.0, None
        else:
            red = 100.0 * (scip.mean() - v.mean()) / max(scip.mean(), 1e-9)
            p = None
            if wilcoxon is not None and np.any(v != scip):
                try:
                    p = float(wilcoxon(scip, v).pvalue)
                except Exception:
                    p = None
        rows.append(dict(method=m, mean=mean, std=std, median=med,
                         reduction_pct=red, wilcoxon_p=p))
        pstr = f"{p:.2e}" if p is not None else "--"
        print(f"{m:<16}{mean:8.1f} ± {std:6.1f}     {med:<10.0f}"
              f"{red:>6.1f}%   {pstr:<10}")
    print("=" * 72)
    print("Reduction = mean node reduction vs SCIP (higher is better). "
          "p = Wilcoxon signed-rank on paired per-instance counts.")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n_instances", type=int, default=100)
    ap.add_argument("--n_rows", type=int, default=500)
    ap.add_argument("--n_cols", type=int, default=1000)
    ap.add_argument("--density", type=float, default=0.05)
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ablation.json")
    args = ap.parse_args()

    import yaml
    from bnb_wm.model.world_model import BnBWorldModel
    from bnb_wm.training.checkpoint import load_weights_only

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(args.config))["model"]
    model = BnBWorldModel(
        hidden_dim=cfg["hidden_dim"], n_gnn_layers=cfg["n_gnn_layers"],
        n_gnn_heads=cfg["n_gnn_heads"], n_dyn_layers=cfg["n_dyn_layers"],
        n_dyn_heads=cfg["n_dyn_heads"], max_seq=cfg["max_seq"],
    ).to(device)
    load_weights_only(model, args.checkpoint, device=device)
    print(f"Loaded {args.checkpoint} on {device}")

    nodes = run(
        model, device, {**BASELINES, **ABLATIONS},
        n_instances=args.n_instances,
        generator_kwargs=dict(n_rows=args.n_rows, n_cols=args.n_cols,
                              density=args.density),
        time_limit=args.time_limit, seed=args.seed,
    )
    summary = summarize(nodes)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"per_instance": nodes, "summary": summary,
               "config": vars(args)}, open(out, "w"), indent=2)
    print(f"\nSaved raw counts + summary to {out}")


if __name__ == "__main__":
    main()
