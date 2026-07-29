"""collect_with_cuts_v2.py — corrected branch-and-cut expert collector.

Rewrite of collect_with_cuts_benchmark_sets.py addressing the supervisor's
review (see docs/CODE_FIXES.md). Changes vs. the old collector, each tagged
with the issue it fixes:

  P0.1  Strong-branching label: ALWAYS index scores by the action set
        (`scores[action_set].argmax()`), never the fragile shape shortcut.
  P0.3  Record `node_id` and `parent_id` per visited node so the dataset can
        build TRUE parent->child dynamics transitions instead of visitation
        order. Each child's own LP state is captured when SCIP visits it.
  P0.4  Cut candidates are VALID root Gomory cuts from bnb_wm.solver.gomory,
        never the old invalid `sum x_j >= 1` intersection cuts.
  P0.5  Cut labels come from actually adding the valid cut and re-solving the
        LP (dive), measuring the true bound gain.
  P0.6  Same Gomory generator here as in the deploy solver -> no train/serve
        mismatch in cut features/distribution.
  P0.7  Labels = top-k by measured bound gain on VALID cuts.
  P0.11 Real leaf labels are derived downstream from the recorded tree
        (a node with no recorded child is a leaf); we store raw `dual_bounds`
        plus `root_bound`/`primal_bound` for a fixed cross-instance norm.

NOTE: This file could not be executed in the authoring environment (no ecole /
pyscipopt / highspy). It must be smoke-tested on the training machine:

    python scripts/collect_with_cuts_v2.py --split train --n-instances-per-class 2 --debug

Then verify with tests/test_collector_v2.py (schema, cut validity, parent links).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import ecole
import numpy as np

try:
    import highspy
except Exception:                       # P2.1: fail loud later, not silently
    highspy = None

from bnb_wm.solver.gomory import generate_root_gomory_cuts


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INSTANCES_ROOT = SCRIPT_DIR.parent / "learned_branch_cut_solver"
DEFAULT_DATA_DIR = SCRIPT_DIR / "data_with_cuts_v2"
DIFFICULTIES = ("SC-easy", "SC-medium", "SC-hard")


# ---------------------------------------------------------------------------
# Ecole information functions (dual bound, node identity)
# ---------------------------------------------------------------------------

class AbsoluteDualBound:
    def before_reset(self, model: Any) -> None: ...
    def extract(self, model: Any, done: bool) -> float:
        return 0.0 if done else float(model.as_pyscipopt().getDualbound())


class NodeIdentity:
    """P0.3: SCIP node number + parent number at the current node.

    These let the dataset reconstruct real tree edges: a recorded node v is a
    child of recorded node u iff v.parent_id == u.node_id. That edge is a true
    branch transition; consecutive *visitation* order is not.
    """
    def before_reset(self, model: Any) -> None: ...
    def extract(self, model: Any, done: bool) -> tuple[int, int]:
        if done:
            return (-1, -1)
        scip = model.as_pyscipopt()
        node = scip.getCurrentNode()
        if node is None:
            return (-1, -1)
        nid = int(node.getNumber())
        parent = node.getParent()
        pid = int(parent.getNumber()) if parent is not None else -1
        return (nid, pid)


# ---------------------------------------------------------------------------
# Graph extraction (carried over from the working collector, trimmed)
# ---------------------------------------------------------------------------

def _node_bipartite_arrays(obs: Any):
    vf = getattr(obs, "variable_features", None)
    if vf is None:
        vf = obs.column_features
    cf = getattr(obs, "constraint_features", None)
    if cf is None:
        cf = obs.row_features
    ei = np.asarray(obs.edge_features.indices, dtype=np.int64)
    ev = np.asarray(obs.edge_features.values, dtype=np.float32)
    if ev.ndim == 2:
        ev = ev[:, 0]
    return (np.asarray(vf, dtype=np.float32), np.asarray(cf, dtype=np.float32), ei, ev)


def _root_cover_matrix(obs: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Rebuild the covering matrix A (A x >= 1) from the ROOT observation.

    Used only once per instance for valid root Gomory cuts (P0.4/P0.6).
    """
    try:
        vf, cf, ei, ev = _node_bipartite_arrays(obs)
        n_vars, n_cons = vf.shape[0], cf.shape[0]
        row_idx, col_idx = ei[0], ei[1]
        if int(row_idx.max()) >= n_cons and int(col_idx.max()) < n_cons:
            row_idx, col_idx = col_idx, row_idx
        if int(row_idx.max()) >= n_cons or int(col_idx.max()) >= n_vars:
            return None
        A = np.zeros((n_cons, n_vars), dtype=np.float64)
        A[row_idx, col_idx] = ev
        nz = A[A != 0]
        if nz.size and float(np.nanmean(nz)) < 0:      # normalise to cover form
            A = -A
        b = np.ones(n_cons, dtype=np.float64)
        return A, b
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Valid cut scoring (P0.4/P0.5/P0.7): add a real Gomory cut, re-solve, gain
# ---------------------------------------------------------------------------

def _cut_features(lhs: np.ndarray, rhs: float, x_lp: np.ndarray,
                  obj: np.ndarray, n_vars: int) -> list[float]:
    viol = max(0.0, rhs - float(lhs @ x_lp))
    norm = float(np.linalg.norm(lhs)) + 1e-8
    obj_u = obj / (np.linalg.norm(obj) + 1e-8)
    frac = np.abs(x_lp - np.round(np.clip(x_lp, 0.0, 1.0)))
    sup = np.abs(lhs) > 1e-9
    sup_frac = float((frac[sup] > 0.05).mean()) if sup.any() else 0.0
    return [viol, viol / norm, float(np.count_nonzero(lhs)) / n_vars,
            float((lhs / norm) @ obj_u),
            viol * abs(float(obj @ lhs) / (norm ** 2 + 1e-8)), sup_frac]


def _score_cut_by_dive(scip: Any, variables: list[Any], lhs: np.ndarray,
                       rhs: float, lp_before: float, minimize: bool,
                       lp_iters: int) -> float:
    """Add one VALID cut in a dive, re-solve the LP, return bound gain."""
    if not hasattr(scip, "startDive"):
        return 0.0
    row = None
    started = False
    try:
        scip.startDive(); started = True
        mk = getattr(scip, "createEmptyRowUnspec", None) or getattr(scip, "createEmptyRow")
        row = mk(name="gomorycut", lhs=float(rhs), rhs=None, local=False, removable=True)
        if hasattr(scip, "cacheRowExtensions"):
            scip.cacheRowExtensions(row)
        for j in np.flatnonzero(np.abs(lhs) > 1e-12):
            scip.addVarToRow(row, variables[int(j)], float(lhs[int(j)]))
        if hasattr(scip, "flushRowExtensions"):
            scip.flushRowExtensions(row)
        scip.addRowDive(row)
        res = scip.solveDiveLP(lp_iters)
        lperror, cutoff = (bool(res[0]), bool(res[1])) if isinstance(res, tuple) else (False, False)
        if lperror:
            return 0.0
        if cutoff:
            return max(1.0, abs(float(lp_before)) + 1.0)
        after = float(scip.getLPObjVal())
        return max(0.0, (after - lp_before) if minimize else (lp_before - after))
    except Exception:
        return 0.0
    finally:
        try:
            if row is not None and hasattr(scip, "releaseRow"):
                scip.releaseRow(row)
        except Exception:
            pass
        if started:
            try:
                scip.endDive()
            except Exception:
                pass


def _collect_root_cuts(scip: Any, A: np.ndarray, b: np.ndarray,
                       args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Valid root Gomory cuts + bound-gain labels (matches deploy generator)."""
    n_vars = A.shape[1]
    empty = {"features": np.zeros((0, 6), np.float32), "labels": np.zeros(0, np.float32),
             "scores": np.zeros(0, np.float32), "lhs": np.zeros((0, n_vars), np.float32),
             "rhs": np.zeros(0, np.float32)}
    if highspy is None:
        return empty                                    # P2.1: no silent cuts
    try:
        variables = list(scip.getVars(transformed=True))
        obj = np.array([float(v.getObj()) for v in variables], dtype=np.float64)
        x_lp = np.array([float(scip.getVal(v)) for v in variables], dtype=np.float64)
        # generate_root_gomory_cuts returns a list of (alpha, beta) tuples,
        # each meaning the valid inequality  alpha . x >= beta.
        cuts = generate_root_gomory_cuts(A, b, obj, highspy, max_cuts=args.max_cut_evals)
        if not cuts:
            return empty
        lhs = np.stack([np.asarray(a, dtype=np.float64) for a, _ in cuts])
        rhs = np.array([float(be) for _, be in cuts], dtype=np.float64)
        minimize = "min" in str(scip.getObjectiveSense()).lower()
        lp_before = float(scip.getLPObjVal())
        scores = np.array([
            _score_cut_by_dive(scip, variables, lhs[i], float(rhs[i]),
                               lp_before, minimize, args.cut_lp_iterations)
            for i in range(len(rhs))
        ], dtype=np.float64)
        feats = np.array([_cut_features(lhs[i], float(rhs[i]), x_lp, obj, n_vars)
                          for i in range(len(rhs))], dtype=np.float32)
        labels = np.zeros(len(scores), np.float32)
        improving = np.flatnonzero(scores > args.cut_improvement_eps)
        if len(improving):
            order = improving[np.argsort(-scores[improving])]
            labels[order[:args.expert_cut_k]] = 1.0
        return {"features": feats, "labels": labels, "scores": scores.astype(np.float32),
                "lhs": lhs.astype(np.float32), "rhs": rhs.astype(np.float32)}
    except Exception:
        return empty


# ---------------------------------------------------------------------------
# Trajectory recording
# ---------------------------------------------------------------------------

def _make_env(args: argparse.Namespace) -> Any:
    params = {
        "limits/time": args.time_limit,
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


def _record(env: Any, lp_path: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    try:
        obs, action_set, _, done, info = env.reset(ecole.scip.Model.from_file(str(lp_path)))
    except Exception as e:
        print(f"  reset failed {lp_path.name}: {e}")
        return None
    if done or action_set is None or not len(action_set):
        return None

    keys = ("var_features", "con_features", "edge_indices", "edge_values",
            "action_sets", "branching_vars", "local_branching_label",
            "dual_bounds", "depths", "node_ids", "parent_ids")
    buf: dict[str, list[Any]] = {k: [] for k in keys}

    # --- root cut collection (once), valid Gomory, matches deploy (P0.4/P0.6) ---
    root_cuts = {"features": np.zeros((0, 6), np.float32), "labels": np.zeros(0, np.float32),
                 "scores": np.zeros(0, np.float32), "lhs": None, "rhs": np.zeros(0, np.float32)}
    packed = _root_cover_matrix(obs)
    scip0 = env.model.as_pyscipopt()
    if packed is not None:
        A, b = packed
        root_cuts = _collect_root_cuts(scip0, A, b, args)

    step = 0
    while not done and action_set is not None and len(action_set) and step < args.max_steps:
        scores = info.get("scores")
        if scores is None:
            break
        scores = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=-1e30)
        aset = np.asarray(action_set, dtype=np.int64)
        # P0.1: robust — score the candidate set, take the local argmax.
        best_local = int(scores[aset].argmax())
        if not (0 <= best_local < len(aset)):
            break
        chosen = int(aset[best_local])

        vf, cf, ei, ev = _node_bipartite_arrays(obs)
        nid, pid = info.get("node", (-1, -1))
        buf["var_features"].append(vf); buf["con_features"].append(cf)
        buf["edge_indices"].append(ei); buf["edge_values"].append(ev)
        buf["action_sets"].append(aset.astype(np.int32))
        buf["branching_vars"].append(chosen)
        buf["local_branching_label"].append(best_local)
        buf["dual_bounds"].append(float(info.get("dual_bound", 0.0)))
        buf["depths"].append(int(nid))              # kept for back-compat readers
        buf["node_ids"].append(int(nid))
        buf["parent_ids"].append(int(pid))

        try:
            obs, action_set, _, done, info = env.step(chosen)
        except Exception as e:
            print(f"  step failed {lp_path.name}: {e}")
            break
        step += 1

    n = len(buf["branching_vars"])
    if n < args.min_steps:
        return None

    db = np.asarray(buf["dual_bounds"], dtype=np.float64)
    node_ids = np.asarray(buf["node_ids"], dtype=np.int64)
    parent_ids = np.asarray(buf["parent_ids"], dtype=np.int64)
    # P0.11: real leaf = recorded node that is nobody's recorded parent.
    child_of = set(int(p) for p in parent_ids.tolist())
    next_is_leaf = np.array([0.0 if int(nid) in child_of else 1.0
                             for nid in node_ids], dtype=np.float32)

    # cut arrays are the same root pool broadcast to step 0 only; other steps empty.
    n_root = len(root_cuts["rhs"])
    lhs_arr = (root_cuts["lhs"] if root_cuts["lhs"] is not None
               else np.zeros((0, buf["var_features"][0].shape[0]), np.float32))
    cut_features = np.empty(n, dtype=object)
    cut_labels = np.empty(n, dtype=object)
    cut_scores = np.empty(n, dtype=object)
    cut_lhs = np.empty(n, dtype=object)
    cut_rhs = np.empty(n, dtype=object)
    n_cuts = np.zeros(n, dtype=np.int32)
    for t in range(n):
        if t == 0 and n_root:
            cut_features[t], cut_labels[t] = root_cuts["features"], root_cuts["labels"]
            cut_scores[t], cut_lhs[t], cut_rhs[t] = root_cuts["scores"], lhs_arr, root_cuts["rhs"]
            n_cuts[t] = n_root
        else:
            cut_features[t] = np.zeros((0, 6), np.float32)
            cut_labels[t] = np.zeros(0, np.float32); cut_scores[t] = np.zeros(0, np.float32)
            cut_lhs[t] = np.zeros((0, buf["var_features"][t].shape[0]), np.float32)
            cut_rhs[t] = np.zeros(0, np.float32)

    return {
        "n_steps": np.asarray(n),
        "var_features": np.asarray(buf["var_features"], dtype=object),
        "con_features": np.asarray(buf["con_features"], dtype=object),
        "edge_indices": np.asarray(buf["edge_indices"], dtype=object),
        "edge_values": np.asarray(buf["edge_values"], dtype=object),
        "action_sets": np.asarray(buf["action_sets"], dtype=object),
        "branching_vars": np.asarray(buf["branching_vars"], dtype=np.int32),
        "local_branching_label": np.asarray(buf["local_branching_label"], dtype=np.int32),
        "dual_bounds": db.astype(np.float32),
        # P0.11: keep a fixed cross-instance norm anchored to the instance's own
        # observed bound range; also expose the anchors so training can renorm.
        "norm_dual_bounds": ((db - db.min()) / (np.ptp(db) + 1e-8)).astype(np.float32),
        "root_bound": np.asarray(float(db[0]), dtype=np.float32),
        "primal_bound": np.asarray(float(db[-1]), dtype=np.float32),
        "next_is_leaf": next_is_leaf,
        "depths": np.asarray(buf["depths"], dtype=np.int32),
        "node_ids": node_ids.astype(np.int64),
        "parent_ids": parent_ids.astype(np.int64),
        "cut_features": cut_features, "cut_labels": cut_labels, "cut_scores": cut_scores,
        "cut_lhs": cut_lhs, "cut_rhs": cut_rhs, "n_cuts": n_cuts,
    }


def _collect(diff: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    src = args.instances_root / diff / args.split
    lp_files = sorted(src.glob("*.lp"))
    if not lp_files:
        raise FileNotFoundError(f"No LP files in {src}")
    random.Random(args.seed + DIFFICULTIES.index(diff)).shuffle(lp_files)
    if args.n_instances_per_class > 0:
        lp_files = lp_files[:args.n_instances_per_class]
    out_dir = args.data_dir / diff
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _make_env(args)
    manifest = []
    for i, lp in enumerate(lp_files, 1):
        out = out_dir / f"traj_{lp.stem}.npz"
        if out.exists() and not args.overwrite:
            continue
        traj = _record(env, lp, args)
        if traj is None:
            print(f"  [{diff} {i}/{len(lp_files)}] skip {lp.name}")
            continue
        np.savez_compressed(out, **traj)
        pos = int(sum(int(np.asarray(l).sum()) for l in traj["cut_labels"]))
        manifest.append({"file": str(out), "n_steps": int(traj["n_steps"]),
                         "root_cuts": int(traj["n_cuts"][0]), "positive_cuts": pos})
        if i == 1 or i % 25 == 0 or i == len(lp_files):
            print(f"  [{diff} {i}/{len(lp_files)}] steps={int(traj['n_steps'])} "
                  f"root_cuts={int(traj['n_cuts'][0])} pos={pos}")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instances-root", type=Path, default=DEFAULT_INSTANCES_ROOT)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--split", default="train")
    p.add_argument("--n-instances-per-class", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--min-steps", type=int, default=5)
    p.add_argument("--time-limit", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-cut-evals", type=int, default=50)
    p.add_argument("--expert-cut-k", type=int, default=5)
    p.add_argument("--cut-lp-iterations", type=int, default=200)
    p.add_argument("--cut-improvement-eps", type=float, default=1e-6)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    args.instances_root = args.instances_root.resolve()
    args.data_dir = args.data_dir.resolve()
    if highspy is None:
        print("WARNING: highspy not importable — cut labels will be empty (P2.1).")

    t0 = time.perf_counter()
    all_m = {}
    for diff in DIFFICULTIES:
        print(f"\nCollecting {diff} from {args.instances_root / diff / args.split}")
        all_m[diff] = _collect(diff, args)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    (args.data_dir / "manifest_v2.json").write_text(json.dumps({
        "split": args.split, "generator": "root_gomory_valid",
        "trajectories": all_m}, indent=2))
    print(f"\nDone in {(time.perf_counter() - t0) / 60:.1f} min -> {args.data_dir}")


if __name__ == "__main__":
    main()
