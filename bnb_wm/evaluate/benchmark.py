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
# Integrality probability threshold above which lookahead is skipped
_LEAF_PROB_SKIP = 0.8


def _format_obs(obs, device):
    """Convert an Ecole NodeBipartite observation to a PyG Batch with edge_attr."""
    vf_raw = (obs.variable_features if hasattr(obs, "variable_features")
              else obs.column_features)
    cf_raw = (obs.constraint_features if hasattr(obs, "constraint_features")
              else obs.row_features)

    vf = np.nan_to_num(
        np.array(vf_raw, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6
    )
    cf = np.nan_to_num(
        np.array(cf_raw, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6
    )
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
    edge_index = torch.stack([ei_t[0] + n_vars, ei_t[1]], dim=0)

    data = Data(
        x=x, edge_index=edge_index, node_type=node_type, edge_attr=ea_t
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

    # --- multi-step dynamics lookahead over top-k candidates ---
    k            = min(_LOOKAHEAD_K, len(action_set))
    top_k_global = masked.topk(k).indices
    bvec1        = torch.zeros(1, dtype=torch.long, device=device)

    best_action = int(top_k_global[0])
    best_return = -float("inf")

    for cand_idx in top_k_global:
        a_emb = h_vars[cand_idx].unsqueeze(0)   # [1, H]

        z_cur      = z
        tokens_cur = past_tokens
        discounted_return = 0.0
        gamma = 1.0

        for _ in range(_LOOKAHEAD_DEPTH):
            z_cur, tokens_cur = model.dynamics_step(z_cur, a_emb, tokens_cur)
            v = model.value_pred(
                z_cur, z_cur, bvec1, frac_mask=None
            ).item()
            discounted_return += gamma * v
            gamma *= _LOOKAHEAD_GAMMA

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
