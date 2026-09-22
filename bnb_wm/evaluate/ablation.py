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
from collections import defaultdict
from pathlib import Path
from time import perf_counter

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
from bnb_wm.solver.bnb_solver import BnBSolver


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------

# Each config is the set of rollout parameters used to pick a branching variable.
# mode "policy" ignores the rollout; mode "rollout" calls model.rollout_candidate.
ABLATIONS = {
    "policy_only":   dict(mode="policy"),
    # random_topk: takes policy's top-k candidates, picks one uniformly at random.
    # This isolates whether the dynamics rollout adds value BEYOND the policy's
    # top-k shortlist. If reward_return beats random_topk, dynamics matters.
    # If not, the policy's ranking is sufficient and rollout adds only overhead.
    "random_topk":   dict(mode="random_topk", k=3),
    # depth=2, k=3 (was 3/5): ~2-3x faster per node so instances actually finish
    # within the time budget, making node-count comparisons valid (not timeouts).
    "value_rollout": dict(mode="rollout", depth=2, gamma=0.95, k=3,
                          ctg_weight=0.0, branch_factor=1, use_reward_return=False),
    "cost_to_go":    dict(mode="rollout", depth=2, gamma=0.95, k=3,
                          ctg_weight=1.0, branch_factor=1, use_reward_return=False),
    "tree_rollout":  dict(mode="rollout", depth=2, gamma=0.95, k=3,
                          ctg_weight=1.0, branch_factor=2, use_reward_return=False),
    "reward_return": dict(mode="rollout", depth=2, gamma=0.95, k=3,
                          ctg_weight=1.0, branch_factor=2, use_reward_return=True),
}

# Classical (non-learned) branching baselines, for a fair comparison spectrum:
# SCIP reliability branching is the strong upper baseline, these are the weak ones.
BASELINES = {
    "random":          dict(mode="random"),
    "most_fractional": dict(mode="most_fractional"),
}
_LEAF_SKIP = 0.8

# Instance size tiers.  --tiers medium hard  runs both in one invocation.
TIERS = {
    "medium": dict(n_rows=500,  n_cols=1000, time_limit=60),
    "hard":   dict(n_rows=1000, n_cols=2000, time_limit=120),
}


# ---------------------------------------------------------------------------
# Parameterized branching-variable selection
# ---------------------------------------------------------------------------

class _NodeDepth:
    """Ecole information function exposing the current node's B&B depth, so the
    IntegralityHead gets the SAME depth input at inference that it saw in training
    (Phase 4). Without it, depth defaulted to 0 -> the leaf-probability gate that
    decides whether to skip the rollout was systematically wrong."""
    def before_reset(self, model):
        pass

    def extract(self, model, done):
        try:
            return int(model.as_pyscipopt().getDepth())
        except Exception:
            return 0


def _pick_action(model, batch, action_set, device, cfg, past_tokens, depth=0,
                 timing_acc=None, rollout_preds=None):
    """Pick a branching variable under one ablation config.

    Args:
        timing_acc   : dict accumulator for component times (seconds); mutated in place.
                       Keys: 'encode', 'policy', 'rollout'.  Pass None to skip timing.
        rollout_preds: list accumulator for (predicted_score, chosen_var_idx) tuples
                       used later to compare against actual child LP bounds.

    Returns:
        (action, past_tokens)
    """
    mode = cfg["mode"]

    # --- classical baselines: no model needed, short-circuit before encoding ---
    if mode == "random":
        return int(np.random.choice(action_set)), past_tokens
    if mode == "most_fractional":
        var_mask = batch.node_type == 0
        vf = batch.x[var_mask]
        frac = vf[:, 14] if vf.size(1) > 14 else torch.zeros(vf.size(0), device=device)
        aset_t = torch.tensor(action_set, dtype=torch.long, device=device)
        best = int(aset_t[int(frac[aset_t].argmax())])
        return best, past_tokens

    # --- learned policy / rollout ---
    # Priority 1: time GNN encode
    if timing_acc is not None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = perf_counter()

    h_vars, z = model.encode(batch)

    if timing_acc is not None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timing_acc["encode"] += perf_counter() - t0

    var_mask  = batch.node_type == 0
    var_batch = batch.batch[var_mask]

    # Priority 1: time policy head
    if timing_acc is not None:
        t0 = perf_counter()

    scores_all = model.policy_scores(h_vars, z, var_batch)
    aset_t = torch.tensor(action_set, dtype=torch.long, device=device)
    masked = torch.full_like(scores_all, -1e4)
    masked[aset_t] = scores_all[aset_t]

    if timing_acc is not None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timing_acc["policy"] += perf_counter() - t0

    if cfg["mode"] == "policy":
        return int(masked.argmax()), past_tokens

    # random_topk: policy shortlists top-k, then picks uniformly at random.
    # Isolates whether dynamics rollout adds value beyond the policy shortlist.
    if cfg["mode"] == "random_topk":
        k = min(cfg.get("k", 3), len(action_set))
        top_k = masked.topk(k).indices
        return int(top_k[np.random.randint(k)]), past_tokens

    # Integrality gate
    x_var = batch.x[var_mask]
    n_frac_val = float((x_var[:, 14] > 0.05).sum()) if x_var.size(1) > 14 else 0.0
    depth_t = torch.tensor([float(depth)], device=device)
    nfrac_t = torch.tensor([n_frac_val], device=device)
    leaf_prob = torch.sigmoid(model.integrality_logit(z, depth_t, nfrac_t)).item()
    if leaf_prob > _LEAF_SKIP:
        return int(masked.argmax()), past_tokens

    # Confidence gate
    conf = cfg.get("skip_confident")
    if conf is not None:
        p_top = float(torch.softmax(scores_all[aset_t], dim=0).max())
        if p_top >= conf:
            return int(masked.argmax()), past_tokens

    # Priority 1: time rollout
    if timing_acc is not None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = perf_counter()

    k = min(cfg["k"], len(action_set))
    top_k = masked.topk(k).indices
    valid_mask = torch.zeros(scores_all.size(0), dtype=torch.bool, device=device)
    valid_mask[aset_t] = True

    rets = []
    for cand in top_k:
        rets.append(model.rollout_candidate_batched(
            z, h_vars, int(cand),
            depth=cfg["depth"], gamma=cfg["gamma"],
            valid_mask=valid_mask, past_tokens=past_tokens,
            size_weight=0.0, ctg_weight=cfg["ctg_weight"],
            branch_factor=cfg["branch_factor"],
            use_reward_return=cfg["use_reward_return"],
        ))

    if timing_acc is not None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timing_acc["rollout"] += perf_counter() - t0

    lam = cfg.get("anchor_lambda")
    if lam is None:
        best_idx = int(np.argmax(rets))
        best_action = int(top_k[best_idx])
    else:
        pol = np.array([masked[int(c)].item() for c in top_k], dtype=float)
        ret = np.array(rets, dtype=float)

        def _z(x):
            s = x.std()
            return (x - x.mean()) / s if s > 1e-8 else np.zeros_like(x)

        final = _z(pol) + lam * _z(ret)
        best_idx = int(np.argmax(final))
        best_action = int(top_k[best_idx])

    # Priority 2: record predicted rollout score for chosen action
    if rollout_preds is not None:
        rollout_preds.append({
            "chosen_var": best_action,
            "predicted_score": float(rets[best_idx]),
            "n_candidates": k,
        })

    a_emb = h_vars[best_action].unsqueeze(0)
    _, past_tokens = model.dynamics_step(z, a_emb, past_tokens)
    return best_action, past_tokens


def _scip_metrics(scip, fallback_nodes):
    """(nodes, solved, time_s, cuts, obj, dual, gap) from a pyscipopt Model."""
    try:
        n = int(scip.getNNodes())
    except Exception:
        n = fallback_nodes
    try:
        solved = (scip.getStatus() == "optimal")
    except Exception:
        solved = False
    try:
        t = float(scip.getSolvingTime())
    except Exception:
        t = float("nan")
    try:
        c = int(scip.getNCutsApplied())
    except Exception:
        c = -1
    try:
        obj = float(scip.getObjVal())
    except Exception:
        obj = float("nan")
    try:
        dual = float(scip.getDualbound())
    except Exception:
        dual = float("nan")
    if not (np.isnan(obj) or np.isnan(dual)):
        denom = max(abs(obj), abs(dual), 1e-8)
        gap = abs(obj - dual) / denom
    else:
        gap = float("nan")
    return n, solved, t, c, obj, dual, gap


def _root_dual(env):
    """Read the LP relaxation bound right after env.reset(), before branching."""
    try:
        return float(env.model.as_pyscipopt().getDualbound())
    except Exception:
        return float("nan")


def _instance_to_abc(instance):
    """Extract (A, b, c) numpy arrays from an Ecole instance (pyscipopt Model).

    Returns the LP relaxation in the form:
        min  c^T x   s.t.  A x >= b,   x in {0,1}^n
    which matches BnBSolver.solve()'s expected format.

    Uses pyscipopt's getLPRowsData / getColsData to read constraint coefficients.
    Falls back to None on any error so the caller can skip the HiGHS baseline.
    """
    try:
        scip = instance.copy_orig().as_pyscipopt()
        scip.hideOutput()
        scip.setParam("presolving/maxrounds", 0)
        scip.setParam("separating/maxrounds", 0)
        # Solve LP relaxation only to access row/column data
        scip.optimize()

        vars_ = scip.getVars(transformed=False)
        n = len(vars_)
        c = np.array([v.getObj() for v in vars_], dtype=np.float64)

        rows = scip.getLPRowsData()
        if rows is None or len(rows) == 0:
            # Fallback: read from constraints before LP solve
            rows = scip.getConss()

        m_rows = len(rows)
        A = np.zeros((m_rows, n), dtype=np.float64)
        b = np.zeros(m_rows, dtype=np.float64)
        var_idx = {v.name: i for i, v in enumerate(vars_)}

        for ri, row in enumerate(rows):
            try:
                lhs = row.getLhs()
                rhs = row.getRhs()
                # Use LHS (>=) side; if both finite use LHS
                b[ri] = lhs if lhs > -1e19 else -rhs
                cols, vals = scip.getRowVarsAndCoefs(row)
                for v, coef in zip(cols, vals):
                    j = var_idx.get(v.name, -1)
                    if j >= 0:
                        A[ri, j] = coef if lhs > -1e19 else -coef
            except Exception:
                pass

        return A, b, c
    except Exception:
        return None


def _episode_stats(env, fallback_steps):
    """(nodes, solved, time_s, cuts, obj, dual, gap) for the just-finished episode."""
    try:
        scip = env.model.as_pyscipopt()
        return _scip_metrics(scip, fallback_steps)
    except Exception:
        return fallback_steps, False, float("nan"), -1, float("nan"), float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run(model, device, configs, n_instances, generator_kwargs,
        time_limit=60, seed=0, separate=False, strong_branching=False,
        pseudocost=False, highs_baseline=False):
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

    # separate=False disables cutting planes to isolate BRANCHING quality (the
    # node-count comparison). separate=True leaves SCIP's default separators on
    # so the "cuts" metric is meaningful (true branch-and-cut). maxrounds=-1 is
    # SCIP's "unlimited default rounds"; 0 disables separation entirely.
    sep_rounds = -1 if separate else 0
    scip_params = {
        "limits/time":              time_limit,
        "separating/maxrounds":     sep_rounds,
        "presolving/maxrounds":     0,
    }
    env = ecole.environment.Branching(
        observation_function=ecole.observation.NodeBipartite(),
        information_function={"depth": _NodeDepth()},
        scip_params=scip_params,
    )

    extra = (["strong_branching"] if strong_branching else []) \
            + (["pseudocost"] if pseudocost else []) \
            + (["highs_mf", "highs_policy", "highs_full_wm"] if highs_baseline else [])
    methods = ["scip"] + extra + list(configs.keys())
    nodes    = {m: [] for m in methods}
    solved   = {m: [] for m in methods}   # solved-to-optimality flags
    times    = {m: [] for m in methods}   # SCIP solving time (seconds)
    cuts     = {m: [] for m in methods}   # cutting planes applied
    gaps     = {m: [] for m in methods}   # primal gap at termination
    obj_vals = {m: [] for m in methods}   # best incumbent value
    dual_vals= {m: [] for m in methods}   # best dual bound at termination
    root_lps = {m: [] for m in methods}   # LP relaxation at root (before branching)
    lp_solves= {m: [] for m in methods}   # total LP solves per instance (HiGHS methods only)
    lp_times = {m: [] for m in methods}   # cumulative LP time per instance (HiGHS methods only)

    # Priority 1: per-episode component timing (learned methods only)
    # Each entry is a dict {encode, policy, rollout} in seconds for one instance.
    timings  = {m: [] for m in methods if m not in ("scip", "strong_branching",
                                                      "pseudocost", "random",
                                                      "most_fractional")}

    # Priority 2: rollout prediction accuracy
    # List of dicts per instance: predicted_score, actual_child_dual, error
    rollout_acc = {m: [] for m in timings}

    model.eval()

    print(f"Evaluating {n_instances} instances | methods: {methods}\n")
    try:
      for i in range(n_instances):
        instance = next(generator)

        def _append(key, n, opt, t, c, obj, dual, gap, root_lp=float("nan"),
                    lp_s=float("nan"), lp_t=float("nan")):
            nodes[key].append(n)
            solved[key].append(opt)
            times[key].append(t)
            cuts[key].append(c)
            gaps[key].append(gap)
            obj_vals[key].append(obj)
            dual_vals[key].append(dual)
            root_lps[key].append(root_lp)
            lp_solves[key].append(lp_s)
            lp_times[key].append(lp_t)

        # ---- SCIP default (pseudocost) ----
        m = instance.copy_orig().as_pyscipopt()
        m.hideOutput()
        m.setParam("limits/time", time_limit)
        m.setParam("separating/maxrounds", sep_rounds)
        m.setParam("presolving/maxrounds", 0)
        m.optimize()
        _append("scip", *_scip_metrics(m, 0))

        # ---- full strong branching (the oracle the policy imitates) ----
        if strong_branching:
            ms = instance.copy_orig().as_pyscipopt()
            ms.hideOutput()
            ms.setParam("limits/time", time_limit)
            ms.setParam("separating/maxrounds", sep_rounds)
            ms.setParam("presolving/maxrounds", 0)
            ms.setParam("branching/fullstrong/priority", 536870911)
            ms.optimize()
            _append("strong_branching", *_scip_metrics(ms, 0))

        # ---- pure pseudocost branching (classical standard rule) ----
        if pseudocost:
            mp = instance.copy_orig().as_pyscipopt()
            mp.hideOutput()
            mp.setParam("limits/time", time_limit)
            mp.setParam("separating/maxrounds", sep_rounds)
            mp.setParam("presolving/maxrounds", 0)
            mp.setParam("branching/pscost/priority", 536870911)
            mp.optimize()
            _append("pseudocost", *_scip_metrics(mp, 0))

        # ---- HiGHS fair baselines (same Python+LP overhead as model) ----
        # Three variants sharing the same BnBSolver Python loop:
        #   highs_mf       : most-fractional branching (no model)  -- dumb baseline
        #   highs_policy   : learned policy, no rollout             -- policy contribution
        #   highs_full_wm  : full world model (policy + rollout)    -- full contribution
        # All use cut_mode="none" to isolate branching; cuts are a separate axis.
        if highs_baseline:
            abc = _instance_to_abc(instance)
            for hname, hmode in (("highs_mf",      "most_fractional"),
                                  ("highs_policy",  "policy"),
                                  ("highs_full_wm", "rollout")):
                if abc is not None:
                    A_h, b_h, c_h = abc
                    hs = BnBSolver(
                        model, device,
                        time_limit=time_limit,
                        node_limit=500_000,
                        cut_mode="none",
                    )
                    hs.branch_mode = hmode
                    try:
                        res = hs.solve(A_h, b_h, c_h)
                        h_opt = (res.status == "optimal")
                        h_obj = float(res.objective)
                        h_dual = h_obj if h_opt else float("nan")
                        _append(hname, res.n_nodes, h_opt, res.solve_time, -1,
                                h_obj, h_dual, float(res.optimality_gap),
                                lp_s=res.lp_solves, lp_t=res.lp_time)
                    except Exception as e:
                        print(f"  [{hname}] instance {i+1} failed: {e}")
                        _append(hname, 0, False, float("nan"), -1,
                                float("nan"), float("nan"), float("nan"))
                else:
                    _append(hname, 0, False, float("nan"), -1,
                            float("nan"), float("nan"), float("nan"))

        # ---- each learned config ----
        for name, cfg in configs.items():
            obs, action_set, _, done, info = env.reset(instance.copy_orig())
            root_lp = _root_dual(env)   # LP relaxation before first branch

            # Priority 1: initialise per-episode timing accumulator
            ep_timing = {"encode": 0.0, "policy": 0.0, "rollout": 0.0}
            use_timing = name in timings

            # Priority 2: rollout prediction store
            ep_rollout_preds = [] if name in rollout_acc else None
            ep_rollout_pairs = []   # (predicted_score, actual_child_dual)

            steps, past = 0, None
            with torch.no_grad():
                while not done and action_set is not None and len(action_set) > 0:
                    batch = _format_obs(obs, device)
                    depth = int(info.get("depth", 0)) if isinstance(info, dict) else 0

                    # Pre-step dual bound (= current LP bound before branching)
                    pre_dual = float("nan")
                    if ep_rollout_preds is not None and cfg.get("mode") == "rollout":
                        try:
                            pre_dual = float(env.model.as_pyscipopt().getDualbound())
                        except Exception:
                            pass

                    action, past = _pick_action(
                        model, batch, action_set, device, cfg, past, depth=depth,
                        timing_acc=ep_timing if use_timing else None,
                        rollout_preds=ep_rollout_preds,
                    )
                    obs, action_set, _, done, info = env.step(action)
                    steps += 1

                    # Priority 2: post-step dual = actual child LP bound
                    if ep_rollout_preds and not np.isnan(pre_dual):
                        pred = ep_rollout_preds[-1]
                        try:
                            post_dual = float(env.model.as_pyscipopt().getDualbound())
                            actual_delta = post_dual - pre_dual
                            ep_rollout_pairs.append({
                                "predicted_score": pred["predicted_score"],
                                "actual_delta_lb": actual_delta,
                                "n_candidates": pred["n_candidates"],
                            })
                        except Exception:
                            pass

            _append(name, *_episode_stats(env, steps), root_lp=root_lp)

            # Priority 1: store per-episode timing summary
            if use_timing:
                timings[name].append(ep_timing)

            # Priority 2: store rollout accuracy pairs for this instance
            if name in rollout_acc:
                rollout_acc[name].append(ep_rollout_pairs)

        row = " | ".join(
            f"{m}:{nodes[m][-1]}{'' if solved[m][-1] else '*'}" for m in methods
        )
        print(f"  [{i+1:3d}/{n_instances}] {row}")
    except KeyboardInterrupt:
        print("\nInterrupted -- saving completed instances only.")

    # Truncate every method to the number of fully-completed instances so a
    # mid-instance interrupt leaves aligned, valid arrays.
    done_n = min(len(nodes[m]) for m in methods)
    for m in methods:
        nodes[m]     = nodes[m][:done_n]
        solved[m]    = solved[m][:done_n]
        times[m]     = times[m][:done_n]
        cuts[m]      = cuts[m][:done_n]
        gaps[m]      = gaps[m][:done_n]
        obj_vals[m]  = obj_vals[m][:done_n]
        dual_vals[m] = dual_vals[m][:done_n]
        root_lps[m]  = root_lps[m][:done_n]
        lp_solves[m] = lp_solves[m][:done_n]
        lp_times[m]  = lp_times[m][:done_n]
    for m in timings:
        timings[m] = timings[m][:done_n]
    for m in rollout_acc:
        rollout_acc[m] = rollout_acc[m][:done_n]
    print(f"  (* = hit time/node limit, NOT solved to optimality) "
          f"[{done_n} instances completed]")
    return nodes, solved, times, cuts, gaps, obj_vals, dual_vals, root_lps, timings, rollout_acc, lp_solves, lp_times


def _nanmean(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(x.mean()) if x.size else float("nan")


def _summarize_timings(timings):
    """Print component timing breakdown for learned methods."""
    if not timings:
        return
    print("\n--- Component timing breakdown (learned methods) ---")
    print(f"{'Method':<20}{'encode(s)':<12}{'policy(s)':<12}{'rollout(s)':<12}"
          f"{'model_total(s)':<16}{'encode%':<10}{'rollout%':<10}")
    print("-" * 90)
    for name, eps in timings.items():
        if not eps:
            continue
        enc = np.mean([e["encode"]  for e in eps])
        pol = np.mean([e["policy"]  for e in eps])
        rol = np.mean([e["rollout"] for e in eps])
        tot = enc + pol + rol
        enc_pct = 100 * enc / tot if tot > 0 else 0
        rol_pct = 100 * rol / tot if tot > 0 else 0
        print(f"{name:<20}{enc:<12.3f}{pol:<12.3f}{rol:<12.3f}"
              f"{tot:<16.3f}{enc_pct:<10.1f}{rol_pct:<10.1f}")
    print("(times are mean per-instance totals across all branching nodes)")


def _summarize_rollout_accuracy(rollout_acc):
    """Print world-model prediction accuracy vs actual child LP bound."""
    if not rollout_acc:
        return
    print("\n--- Rollout prediction accuracy (predicted score vs actual ΔLB) ---")
    print(f"{'Method':<20}{'n_pairs':<10}{'mean|err|':<12}{'spearman':<12}"
          f"{'sign_agree':<12}")
    print("-" * 65)

    try:
        from scipy.stats import spearmanr
    except ImportError:
        spearmanr = None

    for name, instances in rollout_acc.items():
        pairs = [p for ep in instances for p in ep]
        if not pairs:
            print(f"{name:<20}{'no data'}")
            continue
        preds  = np.array([p["predicted_score"] for p in pairs])
        actual = np.array([p["actual_delta_lb"]  for p in pairs])
        mae = float(np.mean(np.abs(preds - actual)))

        rho = float("nan")
        if spearmanr is not None and len(preds) >= 4:
            try:
                rho = float(spearmanr(preds, actual).statistic)
            except Exception:
                pass

        # top-1 rate: among pairs where n_candidates > 1, was the chosen
        # (max-score) candidate also the one with the highest actual ΔLB?
        # We only have the chosen candidate per node, so instead check if
        # predicted_score > 0 correlates with actual_delta_lb > 0.
        pos_agree = int(np.sum((preds > 0) == (actual > 0)))
        sign_agree = pos_agree / len(preds) if len(preds) > 0 else float("nan")

        print(f"{name:<20}{len(preds):<10}{mae:<12.4f}{rho:<12.4f}{sign_agree:<12.3f}")
    print("(mae = mean absolute error between predicted rollout score and actual ΔLB)")
    print("(sign_agree = fraction where sign(predicted_score) == sign(actual ΔLB))")


def summarize(nodes, solved=None, times=None, cuts=None,
              gaps=None, obj_vals=None, dual_vals=None, root_lps=None,
              timings=None, rollout_acc=None, lp_solves=None, lp_times=None):
    """Print and return a per-method summary vs. SCIP with Wilcoxon significance.

    PRIMARY metric: nodes on instances solved to optimality by BOTH this method
    and SCIP (the only unconfounded node comparison; timeouts skew all-instance
    means because a slow-per-node method can appear to use fewer nodes by timing
    out before the tree is built).

    Also reports: all-instance median nodes, solved %, timeout %, gap@end,
    gap_closed vs root LP, solve time, cuts.

    NOTE: do NOT use this function to select which configuration to report in
    the paper. The evaluation set must be held out from model selection.
    """
    scip = np.asarray(nodes["scip"], dtype=float)
    n_inst = len(scip)
    scip_opt = np.asarray(solved["scip"], dtype=bool) if solved is not None else None

    # ---- PRIMARY: both-solved node comparison ----
    rows = []
    print("\n" + "=" * 110)
    print("PRIMARY METRIC — nodes on instances solved by BOTH method and SCIP "
          "(unconfounded by timeouts)")
    print(f"{'Method':<16}{'n_both':>7}{'SCIP_nodes':>12}{'Meth_nodes':>12}"
          f"{'reduction':>11}{'p(Wilcoxon)':>13}{'%solved':>9}{'%timeout':>10}")
    print("-" * 110)
    for m, vals in nodes.items():
        v = np.asarray(vals, dtype=float)
        pct_solved  = 100.0 * float(np.mean(solved[m])) if solved is not None else float("nan")
        pct_timeout = 100.0 * float(np.mean(~np.asarray(solved[m], dtype=bool))) if solved is not None else float("nan")

        if m == "scip" or scip_opt is None:
            nb, red, p, sv_mean, mv_mean = 0, 0.0, None, float("nan"), float("nan")
        else:
            mask = scip_opt & np.asarray(solved[m], dtype=bool)
            nb   = int(mask.sum())
            sv   = scip[mask]
            mv   = v[mask]
            sv_mean = float(sv.mean()) if nb > 0 else float("nan")
            mv_mean = float(mv.mean()) if nb > 0 else float("nan")
            red = 100.0 * (sv_mean - mv_mean) / max(sv_mean, 1e-9) if nb > 0 else float("nan")
            p = None
            if nb >= 2 and wilcoxon is not None and np.any(sv != mv):
                try:
                    p = float(wilcoxon(sv, mv).pvalue)
                except Exception:
                    pass

        pstr = f"{p:.2e}" if p is not None else "--"
        rstr = f"{red:+.1f}%" if not np.isnan(red) else "--"
        print(f"{m:<16}{nb:>7}{sv_mean:>12.1f}{mv_mean:>12.1f}"
              f"{rstr:>11}{pstr:>13}{pct_solved:>8.0f}%{pct_timeout:>9.0f}%")

        # Also compute per-method summary stats for the rows dict
        mean_time = _nanmean(times[m]) if times is not None else None
        mean_cuts = _nanmean(cuts[m]) if cuts is not None else None
        mean_gap  = _nanmean(gaps[m]) if gaps is not None else None
        mean_gap_closed = None
        if gaps is not None and root_lps is not None and obj_vals is not None:
            gc_vals = []
            for obj, root_lp, g in zip(obj_vals[m], root_lps[m], gaps[m]):
                if np.isnan(obj) or np.isnan(root_lp) or np.isnan(g):
                    continue
                denom = max(abs(obj), 1e-8)
                root_gap = abs(obj - root_lp) / denom
                if root_gap > 1e-8:
                    gc_vals.append((root_gap - g) / root_gap)
            mean_gap_closed = float(np.mean(gc_vals)) if gc_vals else float("nan")
        mean_lp_solves = _nanmean(lp_solves[m]) if lp_solves is not None else None
        mean_lp_time   = _nanmean(lp_times[m])  if lp_times  is not None else None
        rows.append(dict(
            method=m,
            n_both_solved=nb,
            scip_nodes_both=sv_mean, method_nodes_both=mv_mean,
            reduction_pct_both=red, wilcoxon_p_both=p,
            pct_solved=pct_solved, pct_timeout=pct_timeout,
            all_mean=float(v.mean()), all_std=float(v.std()), all_median=float(np.median(v)),
            mean_gap=mean_gap, mean_gap_closed=mean_gap_closed,
            mean_time=mean_time, mean_cuts=mean_cuts,
            mean_lp_solves=mean_lp_solves, mean_lp_time=mean_lp_time,
        ))
    print("=" * 110)
    print(f"(N={n_inst} instances; p = Wilcoxon signed-rank on paired both-solved counts)")
    print("(reduction > 0 means fewer nodes = better branching efficiency)")

    # ---- SECONDARY: all-instance overview ----
    print("\nSECONDARY — all instances (includes timeouts; interpret with caution)")
    print(f"{'Method':<16}{'mean±std':>22}{'median':>9}"
          f"{'gap@end':>10}{'gap_closed':>12}{'time(s)':>10}{'cuts':>8}")
    print("-" * 90)
    for r in rows:
        m = r["method"]
        gstr  = f"{r['mean_gap']*100:.2f}%" if r.get("mean_gap") is not None and not np.isnan(r["mean_gap"]) else "--"
        gcstr = f"{r['mean_gap_closed']*100:.1f}%" if r.get("mean_gap_closed") is not None and not np.isnan(r["mean_gap_closed"]) else "--"
        tstr  = f"{r['mean_time']:.2f}" if r.get("mean_time") is not None else "--"
        cstr  = f"{r['mean_cuts']:.1f}" if r.get("mean_cuts") is not None else "--"
        print(f"{m:<16}{r['all_mean']:8.1f} ± {r['all_std']:6.1f}    {r['all_median']:<9.0f}"
              f"{gstr:>10}{gcstr:>12}{tstr:>10}{cstr:>8}")

    if timings:
        _summarize_timings(timings)
    if rollout_acc:
        _summarize_rollout_accuracy(rollout_acc)

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tiers", nargs="+", default=None,
                    choices=list(TIERS), metavar="TIER",
                    help="run one or more named instance tiers in a single invocation "
                         f"({', '.join(TIERS)}). Each tier saves its own JSON and "
                         "prints its own summary. Overrides --n_rows/--n_cols/--time_limit.")
    ap.add_argument("--n_instances", type=int, default=50)
    ap.add_argument("--n_rows", type=int, default=500)
    ap.add_argument("--n_cols", type=int, default=1000)
    ap.add_argument("--density", type=float, default=0.05)
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument("--separate", action="store_true",
                    help="leave SCIP's cutting planes ON (true branch-and-cut) "
                         "so the 'cuts' metric is meaningful. Default off, which "
                         "isolates branching quality for the node comparison.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=None,
                    help="override rollout depth for all rollout configs "
                         "(e.g. 1 for a shallow one-step latent lookahead)")
    ap.add_argument("--k", type=int, default=None,
                    help="override rollout candidate count (e.g. 2 to let the "
                         "rollout only re-rank the policy's top-2)")
    ap.add_argument("--anchor_lambda", type=float, default=None,
                    help="policy-anchored selection: blend standardized policy "
                         "prior with lambda*rollout return (0 = pure policy, "
                         "large = pure rollout). Try 0.3-1.0.")
    ap.add_argument("--skip_confident", type=float, default=None,
                    help="skip the rollout when the top candidate's softmax "
                         "probability exceeds this (e.g. 0.5): big speedup, "
                         "runs the lookahead only on genuinely close decisions.")
    ap.add_argument("--strong_branching", action="store_true",
                    help="add a full-strong-branching baseline (fewest nodes, "
                         "but very slow -- the oracle the policy imitates).")
    ap.add_argument("--pseudocost", action="store_true",
                    help="add a pure pseudocost-branching baseline (the standard "
                         "cheap classical rule).")
    ap.add_argument("--highs_baseline", action="store_true",
                    help="add a HiGHS most-fractional baseline: same Python B&B loop "
                         "and LP overhead as our model, but dumb branching. This is "
                         "the FAIR comparison (vs SCIP which uses near-zero C++ overhead).")
    ap.add_argument("--methods", default=None,
                    help="comma-separated subset of methods to run (e.g. "
                         "'reward_return' for a fast final-model-vs-SCIP head-to-"
                         "head, or 'policy_only,reward_return'). SCIP is always "
                         "included. Default: all baselines + ablations.")
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

    all_configs = {**BASELINES, **ABLATIONS}
    if args.methods:
        want = [m.strip() for m in args.methods.split(",") if m.strip()]
        missing = [m for m in want if m not in all_configs]
        if missing:
            raise SystemExit(f"Unknown method(s) {missing}. "
                             f"Choose from {list(all_configs)}")
        configs = {m: all_configs[m] for m in want}
    else:
        configs = all_configs

    # Inference-time overrides (no retraining): shallow one-step lookahead
    # (--depth 1) over a small candidate set (--k 2) keeps the latent dynamics
    # in the loop but avoids compounding error and over-riding the policy.
    if any(x is not None for x in
           (args.depth, args.k, args.anchor_lambda, args.skip_confident)):
        for name, cfg in configs.items():
            if cfg.get("mode") == "rollout":
                if args.depth is not None:
                    cfg["depth"] = args.depth
                if args.k is not None:
                    cfg["k"] = args.k
                if args.anchor_lambda is not None:
                    cfg["anchor_lambda"] = args.anchor_lambda
                if args.skip_confident is not None:
                    cfg["skip_confident"] = args.skip_confident

    # Build tier list: explicit --tiers wins; otherwise a single tier from
    # --n_rows/--n_cols/--time_limit (named "custom").
    if args.tiers:
        tier_cfgs = {t: dict(TIERS[t], density=args.density) for t in args.tiers}
    else:
        tier_cfgs = {"custom": dict(n_rows=args.n_rows, n_cols=args.n_cols,
                                    time_limit=args.time_limit, density=args.density)}

    out_base = Path(args.out)
    all_results = {}

    for tier_name, tier_kw in tier_cfgs.items():
        t_limit    = tier_kw.pop("time_limit", args.time_limit)
        t_density  = tier_kw.pop("density", args.density)
        gkw        = dict(n_rows=tier_kw["n_rows"], n_cols=tier_kw["n_cols"],
                          density=t_density)

        print(f"\n{'='*60}")
        print(f"TIER: {tier_name.upper()}  "
              f"({gkw['n_rows']}×{gkw['n_cols']}, time_limit={t_limit}s, "
              f"n={args.n_instances})")
        print(f"{'='*60}")

        nodes, solved, times, cuts, gaps, obj_vals, dual_vals, root_lps, \
            timings, rollout_acc, lp_solves, lp_times = run(
            model, device, configs,
            n_instances=args.n_instances,
            generator_kwargs=gkw,
            time_limit=t_limit, seed=args.seed, separate=args.separate,
            strong_branching=args.strong_branching, pseudocost=args.pseudocost,
            highs_baseline=args.highs_baseline,
        )
        summary = summarize(nodes, solved, times, cuts, gaps, obj_vals, dual_vals, root_lps,
                            timings=timings, rollout_acc=rollout_acc,
                            lp_solves=lp_solves, lp_times=lp_times)

        # Save per-tier JSON alongside the base output path.
        # e.g. --out results/final.json → results/final_medium.json
        if tier_name == "custom":
            out_path = out_base
        else:
            out_path = out_base.with_stem(out_base.stem + f"_{tier_name}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tier_record = {
            "tier": tier_name, "generator": gkw, "time_limit": t_limit,
            "per_instance": nodes, "solved": solved, "times": times,
            "cuts": cuts, "gaps": gaps, "obj_vals": obj_vals,
            "dual_vals": dual_vals, "root_lps": root_lps,
            "timings": timings, "rollout_acc": rollout_acc,
            "lp_solves": lp_solves, "lp_times": lp_times,
            "summary": summary, "config": vars(args),
        }
        json.dump(tier_record, open(out_path, "w"), indent=2)
        print(f"\nSaved {tier_name} results → {out_path}")
        all_results[tier_name] = tier_record

    if len(tier_cfgs) > 1:
        combined_path = out_base.with_stem(out_base.stem + "_all_tiers")
        json.dump(all_results, open(combined_path, "w"), indent=2)
        print(f"\nCombined all-tier results → {combined_path}")


if __name__ == "__main__":
    main()
