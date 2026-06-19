"""
benchmark.py — Macro-level solver benchmark.

Compares three branching strategies on the same set of instances:
    1. SCIP default       (pseudocost branching)
    2. Random branching
    3. BnB-WM (our GNN policy)

Metrics reported per instance and on average:
    - Nodes explored
    - Total wall-clock time (seconds)
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


def _format_obs(obs, device):
    """Convert an Ecole NodeBipartite observation to a PyG Batch."""
    vf_raw = (obs.variable_features if hasattr(obs, "variable_features")
              else obs.column_features)
    cf_raw = (obs.constraint_features if hasattr(obs, "constraint_features")
              else obs.row_features)

    vf = np.nan_to_num(np.array(vf_raw, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    cf = np.nan_to_num(np.array(cf_raw, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    ei = np.array(obs.edge_features.indices, dtype=np.int64)

    vf_t  = torch.tensor(vf, dtype=torch.float32, device=device)
    cf_t  = torch.tensor(cf, dtype=torch.float32, device=device)
    ei_t  = torch.tensor(ei, dtype=torch.long,    device=device)

    n_vars = vf_t.size(0)
    n_cons = cf_t.size(0)

    cf_pad = F.pad(cf_t, (0, 14))
    x = torch.cat([vf_t, cf_pad], dim=0)

    node_type = torch.cat([
        torch.zeros(n_vars, dtype=torch.long, device=device),
        torch.ones(n_cons,  dtype=torch.long, device=device),
    ])

    edge_index = torch.stack([ei_t[0] + n_vars, ei_t[1]], dim=0)
    return Batch.from_data_list([Data(x=x, edge_index=edge_index, node_type=node_type)])


def run_macro_benchmark(
    model,
    device,
    problem: str = "set_cover",
    n_instances: int = 10,
    time_limit: int = 60,
    generator_kwargs: dict = None,
):
    """
    Run a macro benchmark comparing SCIP, Random, and GNN branching.

    Args:
        model          : BnBWorldModel (loaded, eval mode)
        device         : torch.device
        problem        : problem type string
        n_instances    : number of instances to test
        time_limit     : per-instance time limit in seconds
        generator_kwargs : passed to ecole generator

    Returns:
        results : dict with keys "scip", "random", "gnn"
                  each mapping to list of (n_nodes, time_sec) tuples
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
        raise ValueError(f"Unsupported problem type for benchmark: {problem}")

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
        rand_time = time.perf_counter() - t0
        results["random"].append((rand_nodes, rand_time))

        # ---- 3. GNN branching ----
        obs, action_set, _, done, _ = env.reset(instance.copy_orig())
        t0 = time.perf_counter()
        gnn_nodes = 0
        with torch.no_grad():
            while not done and action_set is not None and len(action_set) > 0:
                batch = _format_obs(obs, device)
                h_vars, _ = model.encode(batch)
                scores = model.policy_scores(h_vars)

                aset_t = torch.tensor(action_set, dtype=torch.long, device=device)
                masked = torch.full_like(scores, -1e4)
                masked[aset_t] = scores[aset_t]
                best_action = int(masked.argmax())

                obs, action_set, _, done, _ = env.step(best_action)
                gnn_nodes += 1
        gnn_time = time.perf_counter() - t0
        results["gnn"].append((gnn_nodes, gnn_time))

        print(
            f"Instance {i+1:2d}/{n_instances} | "
            f"SCIP: {scip_nodes:4d} nodes {scip_time:5.2f}s | "
            f"Random: {rand_nodes:4d} nodes {rand_time:5.2f}s | "
            f"GNN: {gnn_nodes:4d} nodes {gnn_time:5.2f}s"
        )

    print("\n" + "=" * 60)
    print("AVERAGES:")
    for method, res in results.items():
        avg_nodes = np.mean([r[0] for r in res])
        avg_time  = np.mean([r[1] for r in res])
        print(f"  {method.upper():8s} -> {avg_nodes:6.1f} nodes | {avg_time:5.2f}s")

    return results
