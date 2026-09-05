"""
benchmark.py — Macro-level solver benchmark.

Compares three branching strategies on the same set of instances:
    1. SCIP default   (pseudocost branching)
    2. Random branching
    3. BnB-WM         (full model: policy + dynamics lookahead)

The GNN branching loop now uses the complete model at inference:
    - PolicyHead (Pointer Network) for branching scores
    - DynamicsTransformer for 1-step latent lookahead over top-k candidates
    - IntegralityHead to detect near-leaf nodes and reduce lookahead cost
    - Edge features (edge_attr) passed to the encoder

Metrics reported per instance and on average:
    - Nodes explored
    - Wall-clock time (seconds)
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from pathlib import Path

try:
    import ecole
    from pyscipopt import Model as SCIPModel
except ImportError:
    ecole = None
    SCIPModel = None

# Number of top-k candidates to evaluate with dynamics lookahead
_LOOKAHEAD_K = 5
# Number of latent steps to unroll per candidate
_LOOKAHEAD_DEPTH = 3
# Discount factor per lookahead step
_LOOKAHEAD_GAMMA = 0.95
# Weight on the predicted-subtree-size penalty in the rollout score.
# Gated off (0.0): traces use SCIP's non-DFS node order, so subtree_size labels
# are not derivable and the SubtreeSizeHead is untrained. Value rollout only.
_SIZE_WEIGHT = 0.0
# Weight on the predicted cost-to-go (remaining nodes) in the rollout score
# (Gap 3). Trainable on the non-DFS traces; set 0 for the pure-value ablation.
_CTG_WEIGHT = 1.0
# Rollout branching factor (Gap 4): 1 = single greedy path, >1 = predicted tree.
_BRANCH_FACTOR = 2
# MuZero-style return (Fix 3): sum gamma^t r_t + gamma^k V(leaf). False = value
# summed at every step (the ablation baseline).
_USE_REWARD_RETURN = True
# Integrality probability threshold above which lookahead is skipped
_LEAF_PROB_SKIP = 0.8
# Confidence gate: if softmax(policy)[top-1 candidate] >= this threshold,
# trust the policy and skip the expensive rollout entirely. None = disabled.
# Mirrors the skip_confident parameter in ablation.py.
_SKIP_CONFIDENT: float | None = None
# Adaptive rollout depth + candidate count (items 10+11).
# High-confidence decisions (p_top >= conf_high) use k=1, depth=1.
# Medium-confidence (p_top >= conf_mid) use k=min(K,2), depth=min(D,2).
# None = disabled (full budget always).
_ADAPTIVE_CONF_HIGH: float | None = None
_ADAPTIVE_CONF_MID:  float | None = None
# Direction-spread uncertainty penalty (0.0 = disabled). See solver config.
_UNCERTAINTY_WEIGHT: float = 0.0


def apply_config(cfg: dict | None):
    """
    P2.7: override the rollout/lookahead constants from a loaded YAML config so
    the benchmark honours `configs/*.yaml` instead of these hardcoded defaults.

    Reads the `benchmark:` section (falling back to `solver:` for shared knobs).
    Unknown/missing keys keep their module default. Call once before benchmarking.
    """
    if not cfg:
        return
    global _LOOKAHEAD_K, _LOOKAHEAD_DEPTH, _LOOKAHEAD_GAMMA, _SIZE_WEIGHT
    global _CTG_WEIGHT, _BRANCH_FACTOR, _USE_REWARD_RETURN, _SKIP_CONFIDENT
    global _ADAPTIVE_CONF_HIGH, _ADAPTIVE_CONF_MID, _UNCERTAINTY_WEIGHT
    b = {**cfg.get("solver", {}), **cfg.get("benchmark", {})}   # benchmark wins
    _LOOKAHEAD_K       = int(b.get("lookahead_k", _LOOKAHEAD_K))
    _LOOKAHEAD_DEPTH   = int(b.get("lookahead_depth", _LOOKAHEAD_DEPTH))
    _LOOKAHEAD_GAMMA   = float(b.get("lookahead_gamma", _LOOKAHEAD_GAMMA))
    _SIZE_WEIGHT       = float(b.get("size_weight", _SIZE_WEIGHT))
    _CTG_WEIGHT        = float(b.get("ctg_weight", _CTG_WEIGHT))
    _BRANCH_FACTOR     = int(b.get("branch_factor", _BRANCH_FACTOR))
    _USE_REWARD_RETURN = bool(b.get("use_reward_return", _USE_REWARD_RETURN))
    if "skip_confident" in b:
        _SKIP_CONFIDENT = float(b["skip_confident"])
    if "adaptive_conf_high" in b:
        _ADAPTIVE_CONF_HIGH = float(b["adaptive_conf_high"])
    if "adaptive_conf_mid" in b:
        _ADAPTIVE_CONF_MID = float(b["adaptive_conf_mid"])
    _UNCERTAINTY_WEIGHT = float(b.get("uncertainty_weight", _UNCERTAINTY_WEIGHT))


def _format_obs(obs, device):
    """Convert an Ecole NodeBipartite observation to a PyG Batch with edge_attr."""
    vf_raw = (obs.variable_features if hasattr(obs, "variable_features")
              else obs.column_features)
    cf_raw = (obs.constraint_features if hasattr(obs, "constraint_features")
              else obs.row_features)

    # Clip to FEATURE_CLIP (±1e4), matching build_pyg_data so the encoder's
    # fixed input standardisation sees the same range at train and deploy.
    from bnb_wm.data.datasets import FEATURE_CLIP as _FC
    vf = np.clip(np.nan_to_num(
        np.array(vf_raw, dtype=np.float32), nan=0.0, posinf=_FC, neginf=-_FC
    ), -_FC, _FC)
    cf = np.clip(np.nan_to_num(
        np.array(cf_raw, dtype=np.float32), nan=0.0, posinf=_FC, neginf=-_FC
    ), -_FC, _FC)
    ei = np.array(obs.edge_features.indices, dtype=np.int64)   # [2, E]

    # Edge values: constraint coefficients from Ecole
    ev_raw = np.array(obs.edge_features.values, dtype=np.float32)
    if ev_raw.ndim == 2:
        ev_raw = ev_raw[:, 0]
    ev_raw = np.nan_to_num(ev_raw.flatten(), nan=0.0, posinf=1e6, neginf=-1e6)

    # 3-dim edge features: [coeff, norm_coeff, sign]
    # Ecole constraint feature layout (5-dim):
    #   [0] obj_cosine_similarity  [1] bias/RHS  [2] is_tight
    #   [3] dual_value             [4] age
    # Normalise coefficient by constraint RHS (index 1 = bias).
    con_src = ei[0]  # constraint indices (before node offset)
    rhs_src = cf[con_src, 1] if cf.shape[1] > 1 else np.ones(len(con_src))
    norm_ev  = ev_raw / (np.abs(rhs_src) + 1e-8)
    sign_ev  = np.sign(ev_raw)
    edge_attr_np = np.stack([ev_raw, norm_ev, sign_ev], axis=1).astype(np.float32)

    vf_t  = torch.tensor(vf, dtype=torch.float32, device=device)
    cf_t  = torch.tensor(cf, dtype=torch.float32, device=device)
    ei_t  = torch.tensor(ei, dtype=torch.long,    device=device)
    ea_t  = torch.tensor(edge_attr_np, dtype=torch.float32, device=device)

    n_vars = vf_t.size(0)
    n_cons = cf_t.size(0)

    cf_pad = F.pad(cf_t, (0, 14))   # pad to 19 dims
    x      = torch.cat([vf_t, cf_pad], dim=0)

    node_type  = torch.cat([
        torch.zeros(n_vars, dtype=torch.long, device=device),
        torch.ones(n_cons,  dtype=torch.long, device=device),
    ])
    # P0.2: bidirectional edges (must match build_pyg_data). Reverse edges let
    # the encoder's variable->constraint pass see edges; duplicate edge_attr.
    con_to_var = torch.stack([ei_t[0] + n_vars, ei_t[1]], dim=0)
    var_to_con = torch.stack([ei_t[1], ei_t[0] + n_vars], dim=0)
    edge_index = torch.cat([con_to_var, var_to_con], dim=1)
    edge_attr  = torch.cat([ea_t, ea_t], dim=0)

    data = Data(
        x=x, edge_index=edge_index, node_type=node_type, edge_attr=edge_attr
    )
    return Batch.from_data_list([data])


def _gnn_pick_action(model, batch, action_set, device, past_tokens=None, depth=0):
    """
    Pick the best branching variable using the full model at inference.

    Steps:
        1. Encode node → h_vars, z
        2. IntegralityHead → skip lookahead for near-leaf nodes
        3. PolicyHead (Pointer Network) → baseline scores
        4. DynamicsTransformer multi-step lookahead over top-k candidates:
               for each candidate a in top-k:
                   unroll dynamics for LOOKAHEAD_DEPTH steps
                   accumulate discounted value estimates
               pick a with highest discounted return
        5. Advance token buffer with chosen action and return

    Returns:
        action      : int
        past_tokens : updated token buffer for dynamics Transformer
    """
    h_vars, z = model.encode(batch)
    var_mask  = batch.node_type == 0
    var_batch = batch.batch[var_mask]
    x_var     = batch.x[var_mask]

    # --- integrality check: skip lookahead for near-leaf nodes ---
    # Pass real depth + n_frac so IntegralityHead sees the same inputs it was
    # trained on (matching ablation.py / trainer.py behaviour).
    n_frac_val = float((x_var[:, 14] > 0.05).sum()) if x_var.size(1) > 14 else 0.0
    depth_t = torch.tensor([float(depth)], device=device)
    nfrac_t = torch.tensor([n_frac_val], device=device)
    leaf_prob = torch.sigmoid(
        model.integrality_logit(z, depth_t, nfrac_t)
    ).item()

    # --- policy scores ---
    scores_all = model.policy_scores(h_vars, z, var_batch)

    aset_t = torch.tensor(action_set, dtype=torch.long, device=device)
    masked = torch.full_like(scores_all, -1e4)
    masked[aset_t] = scores_all[aset_t]

    if leaf_prob > _LEAF_PROB_SKIP:
        best_action = int(masked.argmax())
        return best_action, past_tokens

    # Confidence gate + adaptive compute (items 10+11).
    # Compute p_top once; reuse for both the skip gate and adaptive scaling.
    p_top = float(torch.softmax(scores_all[aset_t], dim=0).max())
    if _SKIP_CONFIDENT is not None and p_top >= _SKIP_CONFIDENT:
        return int(masked.argmax()), past_tokens

    # Adaptive k and depth: shrink the rollout budget on high/medium-confidence
    # decisions so the expensive lookahead is reserved for genuinely hard choices.
    eff_k     = _LOOKAHEAD_K
    eff_depth = _LOOKAHEAD_DEPTH
    if _ADAPTIVE_CONF_HIGH is not None and p_top >= _ADAPTIVE_CONF_HIGH:
        eff_k, eff_depth = 1, 1
    elif _ADAPTIVE_CONF_MID is not None and p_top >= _ADAPTIVE_CONF_MID:
        eff_k     = min(_LOOKAHEAD_K, 2)
        eff_depth = min(_LOOKAHEAD_DEPTH, 2)

    # --- real multi-step latent rollout over top-k candidates ---
    k            = min(eff_k, len(action_set))
    top_k_global = masked.topk(k).indices

    valid_mask = torch.zeros(scores_all.size(0), dtype=torch.bool, device=device)
    valid_mask[aset_t] = True

    scores = model.rollout_top_k_batched(
        z, h_vars, top_k_global,
        depth=eff_depth,
        gamma=_LOOKAHEAD_GAMMA,
        valid_mask=valid_mask,
        past_tokens=past_tokens,
        size_weight=_SIZE_WEIGHT,
        ctg_weight=_CTG_WEIGHT,
        branch_factor=_BRANCH_FACTOR,
        use_reward_return=_USE_REWARD_RETURN,
        uncertainty_weight=_UNCERTAINTY_WEIGHT,
    )
    best_action = int(top_k_global[int(scores.argmax())])

    # Advance the token buffer with the chosen action
    a_emb_chosen = h_vars[best_action].unsqueeze(0)
    _, past_tokens = model.dynamics_step(z, a_emb_chosen, past_tokens)

    return best_action, past_tokens


def run_macro_benchmark(
    model,
    device,
    problem: str = "set_cover",
    n_instances: int = 10,
    time_limit: int = 60,
    generator_kwargs: dict = None,
    config: dict = None,
):
    """
    Run macro benchmark: SCIP vs Random vs GNN (full model).

    Args:
        model          : BnBWorldModel (loaded, eval mode)
        device         : torch.device
        problem        : problem type string
        n_instances    : number of instances to test
        time_limit     : per-instance time limit in seconds
        generator_kwargs : passed to ecole generator

    Returns:
        results : dict with keys "scip", "random", "gnn"
                  each a list of (n_nodes, time_sec) tuples
    """
    if ecole is None or SCIPModel is None:
        raise ImportError("Ecole and PySCIPOpt are required for benchmarking.")

    apply_config(config)      # P2.7: honour YAML lookahead/rollout knobs

    gkw = generator_kwargs or {}

    if problem == "set_cover":
        generator = ecole.instance.SetCoverGenerator(
            n_rows=gkw.get("n_rows", 500),
            n_cols=gkw.get("n_cols", 1000),
            density=gkw.get("density", 0.05),
        )
    else:
        raise ValueError(f"Unsupported problem type: {problem}")

    scip_params = {
        "limits/time":                   time_limit,
        "separating/maxrounds":          0,
        "presolving/maxrounds":          0,
        "branching/relpscost/priority":  100000,
    }

    env = ecole.environment.Branching(
        observation_function=ecole.observation.NodeBipartite(),
        scip_params=scip_params,
    )

    results = {"scip": [], "random": [], "gnn": []}
    model.eval()
    print(f"Running macro benchmark on {n_instances} instances ({problem})...\n")

    for i in range(n_instances):
        instance = next(generator)

        # ---- 1. SCIP default ----
        m = instance.copy_orig().as_pyscipopt()
        m.hideOutput()
        m.setParam("limits/time", time_limit)
        m.setParam("separating/maxrounds", 0)
        m.setParam("presolving/maxrounds", 0)
        t0 = time.perf_counter()
        m.optimize()
        scip_time  = time.perf_counter() - t0
        scip_nodes = m.getNNodes()
        results["scip"].append((scip_nodes, scip_time))

        # ---- 2. Random branching ----
        obs, action_set, _, done, _ = env.reset(instance.copy_orig())
        t0 = time.perf_counter()
        rand_nodes = 0
        while not done and action_set is not None and len(action_set) > 0:
            action = int(np.random.choice(action_set))
            obs, action_set, _, done, _ = env.step(action)
            rand_nodes += 1
        results["random"].append((rand_nodes, time.perf_counter() - t0))

        # ---- 3. GNN branching (full model) ----
        obs, action_set, _, done, _ = env.reset(instance.copy_orig())
        t0 = time.perf_counter()
        gnn_nodes   = 0
        past_tokens = None

        with torch.no_grad():
            while not done and action_set is not None and len(action_set) > 0:
                batch = _format_obs(obs, device)
                action, past_tokens = _gnn_pick_action(
                    model, batch, action_set, device, past_tokens, depth=gnn_nodes
                )
                obs, action_set, _, done, _ = env.step(action)
                gnn_nodes += 1

        results["gnn"].append((gnn_nodes, time.perf_counter() - t0))

        print(
            f"Instance {i+1:2d}/{n_instances} | "
            f"SCIP: {scip_nodes:4d} nodes {scip_time:5.2f}s | "
            f"Random: {results['random'][-1][0]:4d} nodes "
            f"{results['random'][-1][1]:5.2f}s | "
            f"GNN: {gnn_nodes:4d} nodes {results['gnn'][-1][1]:5.2f}s"
        )

    print("\n" + "=" * 64)
    print("AVERAGES:")
    for method, res in results.items():
        avg_nodes = np.mean([r[0] for r in res])
        avg_time  = np.mean([r[1] for r in res])
        print(f"  {method.upper():8s} -> {avg_nodes:6.1f} nodes | {avg_time:5.2f}s")

    return results
