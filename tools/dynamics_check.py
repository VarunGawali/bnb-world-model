"""
dynamics_check.py — does the learned dynamics predict real B&C transitions?

Needs only a checkpoint; states come from generated instances, so no
trajectory files are required. For each instance the tool dives from the root
along the policy's choices (random branch direction), solving every node LP
for real. At each visited node it takes the policy's top-k fractional
variables plus a few random ones and, for both branch directions, solves the
real child LP, encodes the real child, and compares it with the model's
one-step prediction from the parent:

  one-step error     ||z_hat - z'|| / ||z'|| and mean_i ||h_hat_i - h'_i|| / ||h'_i||,
                     against the "nothing changes" baseline (z, h_i in place of
                     the prediction). A transition no better than this
                     baseline has not learned the dynamics.
  bound/reward       Spearman correlation of the predicted bound change and
                     reward with the real child-LP bound change.
  ranking            agreement of the policy, predicted-bound product and
                     rollout scores (K = 1, 2, 3) with real strong branching
                     (product of child-LP bound gains): Spearman, top-1 hit
                     rate, and the strong-branching score of each method's
                     pick relative to the best candidate.
  multi-step         along each dive, the model is rolled open-loop for
                     k = 1, 2, 3 steps with the real actions and compared with
                     the real states k steps later, against the same baseline.

    PYTHONPATH=. python tools/dynamics_check.py --checkpoint CKPT \
        --tier medium --seeds 0 --n_instances 10 --out results/rebuttal/dynamics_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_benchmark import TIERS, instance  # noqa: E402


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def rel(x, y):
    return float(torch.linalg.vector_norm(x - y) / torch.clamp(torch.linalg.vector_norm(y), min=1e-8))


def delta_cos(pred, base, true):
    """Cosine between the predicted change and the real change."""
    dp, dt = (pred - base).reshape(-1), (true - base).reshape(-1)
    if float(torch.linalg.vector_norm(dt)) < 1e-8:
        return None
    return float(torch.nn.functional.cosine_similarity(dp.unsqueeze(0), dt.unsqueeze(0)))


def per_var_delta_cos(h_hat, h_base, h_true, min_change=1e-3):
    """Mean over variables whose embedding actually changed."""
    dp, dt = h_hat - h_base, h_true - h_base
    moved = torch.linalg.vector_norm(dt, dim=-1) > min_change * torch.clamp(
        torch.linalg.vector_norm(h_base, dim=-1), min=1e-8)
    if not bool(moved.any()):
        return None
    return float(torch.nn.functional.cosine_similarity(dp[moved], dt[moved], dim=-1).mean())


def per_var_rel(h_hat, h_true):
    num = torch.linalg.vector_norm(h_hat - h_true, dim=-1)
    den = torch.clamp(torch.linalg.vector_norm(h_true, dim=-1), min=1e-8)
    return float((num / den).mean())


class Env:
    """Solver internals reused for exact LP solves and encodings."""

    def __init__(self, model, device, A, b, c):
        from bnb_wm.solver.config import SolverConfig
        from bnb_wm.solver.neural_bnb import NeuralBnBSolver
        s = NeuralBnBSolver(model, device, SolverConfig(branch_mode="policy", cut_mode="none"))
        s._t0 = time.perf_counter()
        s._A, s._b, s._c = A, b, c
        s._m0, s._n = A.shape
        s._reset_instance_state()
        s._lp_init()
        self.s = s
        self.model, self.device = model, device

    def node(self, vlb, vub):
        lp = self.s._lp_solve(vlb, vub, None)
        if not lp.feasible:
            return None
        h, z = self.s._encode(lp.sol, vlb, vub)
        frac = np.flatnonzero((lp.x > 1e-4) & (lp.x < 1 - 1e-4))
        return {"lp": lp, "h": h, "z": z, "frac": frac, "vlb": vlb, "vub": vub}

    def policy(self, st):
        bvec = torch.zeros(st["h"].size(0), dtype=torch.long, device=self.device)
        with torch.no_grad():
            return self.model.policy_scores(st["h"], st["z"], bvec).float().cpu().numpy()

    def child_tokens(self, st, var, past):
        return self.s._child_tokens(st["z"], st["h"], var, past)


def child_bounds(vlb, vub, var, d):
    vlb, vub = vlb.copy(), vub.copy()
    if d > 0:
        vlb[var] = 1.0
    else:
        vub[var] = 0.0
    return vlb, vub


def check_instance(model, device, A, b, c, rng, args, out):
    env = Env(model, device, A, b, c)
    n = A.shape[1]
    root = env.node(np.zeros(n), np.ones(n))
    if root is None or len(root["frac"]) == 0:
        return
    path_states, path_actions = [root], []
    st, past = root, None
    big = 10.0 * float(np.max(np.abs(c)))

    for depth in range(args.dive_depth):
        if st is None or len(st["frac"]) < 2:
            break
        scores = env.policy(st)
        frac = st["frac"]
        order = frac[np.argsort(-scores[frac])]
        top = list(order[: args.k])
        rest = [v for v in frac if v not in top]
        rand = list(rng.choice(rest, size=min(args.n_random, len(rest)), replace=False)) if rest else []
        cands = [int(v) for v in top + rand]
        parent_obj = float(st["lp"].obj)
        with torch.no_grad():
            b_parent = float(model.dynamics_bound_pred(st["z"]))

        gains, pred_gains, pred_rew = {}, {}, {}
        for v in cands:
            for d in (1.0, -1.0):
                vlb, vub = child_bounds(st["vlb"], st["vub"], v, d)
                ch = env.node(vlb, vub)
                with torch.no_grad():
                    z_hat, h_hat, _ = model.dynamics_step_full(
                        st["z"], st["h"][v].unsqueeze(0), st["h"], past, d)
                    pb = float(model.dynamics_bound_pred(z_hat))
                    pr = float(model.dynamics_reward_pred(z_hat))
                pred_gains[(v, d)], pred_rew[(v, d)] = pb - b_parent, pr
                if ch is None:
                    gains[(v, d)] = big
                    continue
                gains[(v, d)] = max(float(ch["lp"].obj) - parent_obj, 0.0)
                h_hat = h_hat.reshape(ch["h"].shape)
                out["one_step"].append({
                    "depth": depth, "in_topk": v in top,
                    "z_err": rel(z_hat.reshape(-1), ch["z"].reshape(-1)),
                    "z_copy": rel(st["z"].reshape(-1), ch["z"].reshape(-1)),
                    "z_cos": float(torch.nn.functional.cosine_similarity(z_hat.reshape(1, -1), ch["z"].reshape(1, -1))),
                    "h_err": per_var_rel(h_hat, ch["h"]),
                    "h_copy": per_var_rel(st["h"], ch["h"]),
                    "z_dcos": delta_cos(z_hat, st["z"], ch["z"]),
                    "h_dcos": per_var_delta_cos(h_hat, st["h"], ch["h"]),
                    "real_gain": gains[(v, d)], "pred_bound_gain": pb - b_parent, "pred_reward": pr,
                })

        eps = 1e-6
        sb = {v: max(gains[(v, 1.0)], eps) * max(gains[(v, -1.0)], eps) for v in cands}
        methods = {
            "policy": {v: float(scores[v]) for v in cands},
            "pred_bound_product": {v: max(pred_gains[(v, 1.0)], eps) * max(pred_gains[(v, -1.0)], eps) for v in cands},
            "pred_reward_sum": {v: pred_rew[(v, 1.0)] + pred_rew[(v, -1.0)] for v in cands},
        }
        valid = torch.zeros(st["h"].size(0), dtype=torch.bool, device=device)
        valid[torch.as_tensor(frac, device=device)] = True
        for K in args.depths:
            with torch.no_grad():
                methods[f"rollout_K{K}"] = {v: float(model.rollout_candidate_batched(
                    st["z"], st["h"], v, depth=K, gamma=args.gamma, valid_mask=valid,
                    past_tokens=past, size_weight=0.0, ctg_weight=0.0,
                    branch_factor=2, use_reward_return=True)) for v in cands}
        best = max(sb.values())
        for name, sc in methods.items():
            for subset, vs in (("topk", [v for v in cands if v in top]), ("all", cands)):
                if len(vs) < 2:
                    continue
                pick = max(vs, key=lambda v: sc[v])
                best_sub = max(sb[v] for v in vs)
                out["ranking"].append({
                    "method": name, "subset": subset, "depth": depth,
                    "spearman": spearman([sc[v] for v in vs], [sb[v] for v in vs]),
                    "top1": bool(sb[pick] >= best_sub),
                    "sb_ratio": float(sb[pick] / best_sub) if best_sub > 0 else None,
                })

        # follow the policy's choice down a random direction
        v = int(order[0])
        d = float(rng.choice([1.0, -1.0]))
        toks = env.child_tokens(st, v, past)
        vlb, vub = child_bounds(st["vlb"], st["vub"], v, d)
        nxt = env.node(vlb, vub)
        if nxt is None:
            break
        path_actions.append((v, d, past))
        past = toks[d]
        st = nxt
        path_states.append(st)

    # open-loop multi-step prediction along the real path
    for t in range(len(path_actions)):
        z, h = path_states[t]["z"], path_states[t]["h"]
        past = path_actions[t][2]
        for k in range(1, min(args.max_k, len(path_actions) - t) + 1):
            v, d, _ = path_actions[t + k - 1]
            with torch.no_grad():
                z, h, past = model.dynamics_step_full(z, h[v].unsqueeze(0), h, past, d)
            h = h.reshape(path_states[t + k]["h"].shape)
            real = path_states[t + k]
            out["multi_step"].append({
                "k": k,
                "z_err": rel(z.reshape(-1), real["z"].reshape(-1)),
                "z_copy": rel(path_states[t]["z"].reshape(-1), real["z"].reshape(-1)),
                "h_err": per_var_rel(h, real["h"]),
                "h_copy": per_var_rel(path_states[t]["h"], real["h"]),
                "z_dcos": delta_cos(z, path_states[t]["z"], real["z"]),
                "h_dcos": per_var_delta_cos(h, path_states[t]["h"], real["h"]),
            })


def summarize(out):
    def mean(xs):
        xs = [x for x in xs if x is not None and np.isfinite(x)]
        return (float(np.mean(xs)), len(xs)) if xs else (float("nan"), 0)

    lines = []
    os_ = out["one_step"]
    if os_:
        lines += ["## One-step prediction (real child vs prediction from parent)", "",
                  "| | model | nothing-changes baseline | model better in |",
                  "|---|---|---|---|"]
        for key in ("z", "h"):
            m, n = mean([r[f"{key}_err"] for r in os_])
            c0, _ = mean([r[f"{key}_copy"] for r in os_])
            wins = np.mean([r[f"{key}_err"] < r[f"{key}_copy"] for r in os_])
            name = "graph latent z (rel. L2)" if key == "z" else "per-variable h (mean rel. L2)"
            lines.append(f"| {name} | {m:.3f} | {c0:.3f} | {100 * wins:.0f}% of {n} |")
        lines += ["", "Direction of the predicted change vs the real change (cosine; about 0 for an "
                  "untrained transition): z "
                  f"{mean([r['z_dcos'] for r in os_])[0]:.3f}, per-variable h {mean([r['h_dcos'] for r in os_])[0]:.3f}"]
        g = [r["real_gain"] for r in os_]
        lines += ["", f"Spearman with real child-bound gain: predicted bound change "
                  f"{spearman([r['pred_bound_gain'] for r in os_], g)}, predicted reward "
                  f"{spearman([r['pred_reward'] for r in os_], g)} (n={len(os_)})"]
    ms = out["multi_step"]
    if ms:
        lines += ["", "## Open-loop multi-step prediction (real actions)", "",
                  "| k | n | z model | z baseline | z change cos | h model | h baseline | h change cos |",
                  "|---|---|---|---|---|---|---|---|"]
        for k in sorted({r["k"] for r in ms}):
            rs = [r for r in ms if r["k"] == k]
            lines.append(f"| {k} | {len(rs)} | {mean([r['z_err'] for r in rs])[0]:.3f} | "
                         f"{mean([r['z_copy'] for r in rs])[0]:.3f} | {mean([r['z_dcos'] for r in rs])[0]:.3f} | "
                         f"{mean([r['h_err'] for r in rs])[0]:.3f} | {mean([r['h_copy'] for r in rs])[0]:.3f} | "
                         f"{mean([r['h_dcos'] for r in rs])[0]:.3f} |")
    rk = out["ranking"]
    if rk:
        lines += ["", "## Agreement with real strong branching", "",
                  "| candidates | method | nodes | Spearman | top-1 hit | SB score of pick / best |",
                  "|---|---|---|---|---|---|"]
        for subset in ("topk", "all"):
            for mth in dict.fromkeys(r["method"] for r in rk):
                rs = [r for r in rk if r["method"] == mth and r["subset"] == subset]
                if not rs:
                    continue
                lines.append(f"| {subset} | {mth} | {len(rs)} | {mean([r['spearman'] for r in rs])[0]:.3f} | "
                             f"{100 * np.mean([r['top1'] for r in rs]):.0f}% | {mean([r['sb_ratio'] for r in rs])[0]:.3f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tier", choices=TIERS, default="medium")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n_instances", type=int, default=10)
    ap.add_argument("--dive_depth", type=int, default=10)
    ap.add_argument("--k", type=int, default=5, help="policy top-k candidates per node")
    ap.add_argument("--n_random", type=int, default=3, help="extra random fractional candidates")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--max_k", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from bnb_wm.model.world_model import BnBWorldModel
    from bnb_wm.training.checkpoint import load_weights_only
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = BnBWorldModel().to(device)
    load_weights_only(model, args.checkpoint, device=device)
    model.eval()

    out = {"args": vars(args), "checkpoint_load": getattr(model, "_load_report", {}),
           "one_step": [], "multi_step": [], "ranking": []}
    rng = np.random.default_rng(12345)
    t0 = time.time()
    for s in args.seeds:
        for i in range(args.n_instances):
            A, b, c = instance(args.tier, s, i)
            check_instance(model, device, A, b, c, rng, args, out)
            print(f"seed {s} instance {i}: {len(out['one_step'])} transitions, "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
    text = summarize(out)
    print("\n" + text)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    Path(args.out).with_suffix(".md").write_text(text + "\n")


if __name__ == "__main__":
    main()
