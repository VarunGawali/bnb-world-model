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
    global _CTG_WEIGHT, _BRANCH_FACTOR, _USE_REWARD_RETURN
    b = {**cfg.get("solver", {}), **cfg.get("benchmark", {})}   # benchmark wins
    _LOOKAHEAD_K       = int(b.get("lookahead_k", _LOOKAHEAD_K))
    _LOOKAHEAD_DEPTH   = int(b.get("lookahead_depth", _LOOKAHEAD_DEPTH))
    _LOOKAHEAD_GAMMA   = float(b.get("lookahead_gamma", _LOOKAHEAD_GAMMA))
    _SIZE_WEIGHT       = float(b.get("size_weight", _SIZE_WEIGHT))
    _CTG_WEIGHT        = float(b.get("ctg_weight", _CTG_WEIGHT))
    _BRANCH_FACTOR     = int(b.get("branch_factor", _BRANCH_FACTOR))
    _USE_REWARD_RETURN = bool(b.get("use_reward_return", _USE_REWARD_RETURN))


def _format_obs(obs, device):
    """Convert an Ecole NodeBipartite observation to a PyG Batch with edge_attr.

    Remaps Ecole's raw NodeBipartite variable features to the 19-dim layout
    used by collect_highs.py._var_features() so eval features match training:

        Training layout (collect_highs.py):
         0  obj_coef_norm     c / max|c|
         1  has_lb            CONSTANT 1
         2  has_ub            CONSTANT 1
         3  sol_is_at_lb      LP value <= lb + eps  (lb=0 for binary)
         4  sol_is_at_ub      LP value >= ub - eps  (ub=1 for binary, ~0 at root)
         5  basis_status      0=lower 1=basic 2=upper
         6  reduced_cost_norm rc / max|rc|
         7  (zero)            CONSTANT 0
         8  n_rows_norm       degree_j / n_cons  (computed from edge_index)
         9-12 (zeros)         CONSTANT 0
        13  sol_val           LP solution value
        14  sol_frac          |sol_val - round(sol_val)|  <- CRITICAL
        15  lp_obj_norm       CONSTANT 1 (for set-cover minimisation)
        16  n_rows_tight_norm #tight_constraints_for_j / n_cons
        17  lb                CONSTANT 0
        18  ub                CONSTANT 1

        Ecole NodeBipartite empirical column layout (identified via
        check_feature_layout.py; basis one-hot group confirmed at 15,16,17):
         0  obj_coef          normalised by Ecole (different scale from training)
         1  type_binary       CONSTANT 1
         2-4 other type flags CONSTANT 0
         5  has_lb            CONSTANT 1
         6  has_ub            CONSTANT 1
         7  reduced_cost      signed, Ecole-normalised
         8  sol_val           LP solution value
         9  incumbent_val     = sol_val at root when no incumbent found
        10  sol_is_at_lb      binary flag
        11  (other flag)
        12  (other feature)
        13  (other flag)
        14  lp_obj or similar (NOT sol_frac — max > 0.5 observed)
        15  basis_lower       1 when at lower bound (non-basic)
        16  basis_basic       1 when basic
        17  basis_upper       1 when at upper bound (non-basic)
        18  age / constant    CONSTANT 0
    """
    vf_raw = np.array(
        obs.variable_features if hasattr(obs, "variable_features")
        else obs.column_features, dtype=np.float32)
    cf_ecole = np.array(
        obs.constraint_features if hasattr(obs, "constraint_features")
        else obs.row_features, dtype=np.float32)
    ei = np.array(obs.edge_features.indices, dtype=np.int64)   # [2, E]
    ev_raw = np.array(obs.edge_features.values, dtype=np.float32)
    if ev_raw.ndim == 2:
        ev_raw = ev_raw[:, 0]
    ev_raw = np.nan_to_num(ev_raw.flatten(), nan=0.0, posinf=1e6, neginf=-1e6)

    vf_raw = np.nan_to_num(vf_raw, nan=0.0, posinf=1e4, neginf=-1e4)
    cf_ecole = np.nan_to_num(cf_ecole, nan=0.0, posinf=1e4, neginf=-1e4)

    n_vars = vf_raw.shape[0]
    n_cons = cf_ecole.shape[0]

    # --- Extract features from Ecole columns (empirically identified) ---

    # obj_coef: Ecole col 0 normalised by Ecole's own scale; re-normalise to
    # match training collector (c / max|c|) so scale matches learned weights.
    obj_raw = vf_raw[:, 0]
    obj_max = float(np.abs(obj_raw).max()) + 1e-8
    obj_coef_norm = (obj_raw / obj_max).astype(np.float32)

    # sol_val: Ecole col 8 (continuous, range [0,1], #unique ≈ #fractional_vars)
    sol_val = np.clip(vf_raw[:, 8], 0.0, 1.0)

    # sol_frac: Ecole does not store this separately in this version — compute.
    sol_frac = np.abs(sol_val - np.round(sol_val)).astype(np.float32)

    # basis status: Ecole one-hot group confirmed at (15=lower, 16=basic, 17=upper)
    basis_lower = vf_raw[:, 15]
    basis_basic = vf_raw[:, 16]
    basis_upper = vf_raw[:, 17]
    # encode: 0=lower, 1=basic, 2=upper  (matches collect_highs.py)
    basis_status = (basis_basic * 1.0 + basis_upper * 2.0).astype(np.float32)

    # sol_is_at_lb: use basis_lower (= 1 when non-basic at lb=0, same as at_lb)
    sol_is_at_lb = basis_lower.astype(np.float32)

    # reduced_cost: Ecole col 7; re-normalise to match collect_highs.py rc/max|rc|
    rc_raw = vf_raw[:, 7]
    rc_max = float(np.abs(rc_raw).max()) + 1e-8
    rc_norm = (rc_raw / rc_max).astype(np.float32)

    # n_rows_norm: degree of each variable / n_cons  (from bipartite edge_index)
    # ei[0] = constraint index, ei[1] = variable index
    n_rows_per_var = np.bincount(ei[1], minlength=n_vars).astype(np.float32)
    n_rows_norm = n_rows_per_var / max(n_cons, 1)

    # n_rows_tight_norm: for each variable j, count tight constraints it appears in
    # Ecole constraint features col 2 = is_tight (confirmed from fingerprint)
    is_tight = cf_ecole[:, 2] if cf_ecole.shape[1] > 2 else np.zeros(n_cons, np.float32)
    tight_per_var = np.zeros(n_vars, dtype=np.float32)
    np.add.at(tight_per_var, ei[1], is_tight[ei[0]])
    n_rows_tight_norm = tight_per_var / max(n_cons, 1)

    # --- Assemble 19-dim training-layout feature matrix ---
    vf = np.zeros((n_vars, 19), dtype=np.float32)
    vf[:, 0]  = obj_coef_norm          # obj_coef_norm
    vf[:, 1]  = 1.0                    # has_lb  = 1
    vf[:, 2]  = 1.0                    # has_ub  = 1
    vf[:, 3]  = sol_is_at_lb           # sol_is_at_lb
    # col 4 sol_is_at_ub ≈ 0 at root for set-cover (left as 0)
    vf[:, 5]  = basis_status           # basis_status 0/1/2
    vf[:, 6]  = rc_norm                # reduced_cost_norm
    # col 7 zero (left as 0)
    vf[:, 8]  = n_rows_norm            # n_rows_norm
    # cols 9-12 zeros (left as 0)
    vf[:, 13] = sol_val                # sol_val
    vf[:, 14] = sol_frac               # sol_frac  <- CRITICAL
    vf[:, 15] = 1.0                    # lp_obj_norm = 1 (set-cover minimisation)
    vf[:, 16] = n_rows_tight_norm      # n_rows_tight_norm
    # col 17 lb = 0 (left as 0)
    vf[:, 18] = 1.0                    # ub = 1

    # --- Constraint features: remap to training layout ---
    # Training layout (collect_highs.py._con_features):
    #   0: obj_cos  1: rhs (raw, =1 for set-cover)  2: is_tight
    #   3: dual_value_norm   4: n_vars_norm
    # Ecole layout:
    #   0: obj_cos (Ecole-normalised)  1: normalised_RHS (≠ raw b)
    #   2: is_tight  3: dual_value  4: n_vars_per_row / n_vars
    cf = np.zeros((n_cons, 5), dtype=np.float32)
    if cf_ecole.shape[1] >= 1:
        cf[:, 0] = cf_ecole[:, 0]          # obj_cos (Ecole and training both normalise)
    cf[:, 1] = 1.0                          # raw RHS = 1 for set-cover (CONSTANT)
    if cf_ecole.shape[1] >= 3:
        cf[:, 2] = cf_ecole[:, 2]           # is_tight (same in both)
    if cf_ecole.shape[1] >= 4:
        # Ecole dual values are small-magnitude; re-normalise to match training
        y_raw = cf_ecole[:, 3]
        y_max = float(np.abs(y_raw).max()) + 1e-8
        cf[:, 3] = (y_raw / y_max).astype(np.float32)
    if cf_ecole.shape[1] >= 5:
        cf[:, 4] = cf_ecole[:, 4]           # n_vars_norm

    from bnb_wm.data.datasets import FEATURE_CLIP as _FC
    vf = np.clip(vf, -_FC, _FC)
    cf = np.clip(cf, -_FC, _FC)

    # --- Edge features: [coeff, coeff/|RHS|, sign(coeff)] ---
    # Training collector normalises by raw RHS = 1 → norm_ev = ev_raw / 1 = ev_raw.
    con_src = ei[0]
    # cf[:, 1] = 1.0 so norm_ev = ev_raw (consistent with training).
    rhs_src = cf[con_src, 1]
    norm_ev = ev_raw / (np.abs(rhs_src) + 1e-8)
    sign_ev = np.sign(ev_raw)
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


def _gnn_pick_action(model, batch, action_set, device, past_tokens=None):
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

    # --- integrality check: skip lookahead for near-leaf nodes ---
    leaf_prob = torch.sigmoid(
        model.integrality_logit(z)
    ).item()

    # --- policy scores ---
    scores_all = model.policy_scores(h_vars, z, var_batch)

    aset_t = torch.tensor(action_set, dtype=torch.long, device=device)
    masked = torch.full_like(scores_all, -1e4)
    masked[aset_t] = scores_all[aset_t]

    if leaf_prob > _LEAF_PROB_SKIP:
        best_action = int(masked.argmax())
        return best_action, past_tokens

    # --- real multi-step latent rollout over top-k candidates ---
    # For each candidate the model predicts BOTH z_{t+1} and h_vars_{t+1},
    # re-runs the policy on the predicted state to pick the next action, and
    # accumulates discounted value estimates — a genuine branching-sequence
    # simulation rather than replaying the same variable.
    k            = min(_LOOKAHEAD_K, len(action_set))
    top_k_global = masked.topk(k).indices

    valid_mask = torch.zeros(scores_all.size(0), dtype=torch.bool, device=device)
    valid_mask[aset_t] = True

    best_action = int(top_k_global[0])
    best_return = -float("inf")

    for cand_idx in top_k_global:
        discounted_return = model.rollout_candidate(
            z, h_vars, int(cand_idx),
            depth=_LOOKAHEAD_DEPTH,
            gamma=_LOOKAHEAD_GAMMA,
            valid_mask=valid_mask,
            past_tokens=past_tokens,
            size_weight=_SIZE_WEIGHT,
            ctg_weight=_CTG_WEIGHT,
            branch_factor=_BRANCH_FACTOR,
            use_reward_return=_USE_REWARD_RETURN,
        )
        if discounted_return > best_return:
            best_return = discounted_return
            best_action = int(cand_idx)

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
                    model, batch, action_set, device, past_tokens
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
