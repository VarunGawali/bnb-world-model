"""
run_benchmark.py — parallel, resumable benchmark with per-instance records.

Each job is one (seed, instance, row) solve. Results are appended to a JSONL
file as they finish, so an interrupted run resumes where it stopped and can be
spread over several invocations. Rows are defined in configs/benchmark_rows.yaml.

Instances are generated exactly as in tools/benchmark_solvers.py: instance i of
seed s is the i-th draw of gen_instance from np.random.default_rng(s), so
"seed=1/i" matches the instances behind the submitted tables.

Examples
--------
    # count jobs and estimate worst-case wall time, run nothing
    python tools/run_benchmark.py --tier medium --seeds 0 2 3 --n_instances 10 \
        --group main --dry_run

    # reference run (long SCIP) first, so other rows can check cut validity
    python tools/run_benchmark.py --tier medium --seeds 0 2 3 --n_instances 10 \
        --group reference --workers 4 --out results/v2/medium_ref.jsonl

    python tools/run_benchmark.py --tier medium --seeds 0 2 3 --n_instances 10 \
        --group main --workers 4 --checkpoint checkpoints/highs_retrain/phase4_final.pt \
        --reference results/v2/medium_ref.jsonl --out results/v2/medium.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TIERS = {
    "easy":   (200, 400, 0.05),
    "medium": (500, 1000, 0.05),
    "hard":   (1000, 2000, 0.05),
}

# --------------------------------------------------------------------------
# Rows and instances
# --------------------------------------------------------------------------

def load_rows(path: Path):
    import yaml
    spec = yaml.safe_load(path.read_text())
    return spec["common"], spec["rows"], spec["groups"]


def resolve_row(name, common, rows):
    row = dict(rows[name])
    kind = row.pop("solver")
    if kind == "neural":
        return kind, {**common, **row}
    return kind, row


def instance(tier, seed, idx):
    from tools.benchmark_solvers import gen_instance
    m, n, dens = TIERS[tier]
    rng = np.random.default_rng(seed)
    for _ in range(idx + 1):
        A, b, c = gen_instance(m, n, rng, dens)
    return A, b, c


def instance_hash(A, b, c):
    h = hashlib.sha256()
    for arr in (A, b, c):
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------
# Worker state (one per process)
# --------------------------------------------------------------------------

_W: dict = {}


def _init_worker(args_dict):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _W["args"] = args_dict
    _W["model"] = None
    _W["inst"] = {}
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def _model():
    if _W["model"] is None:
        import torch
        from bnb_wm.model.world_model import BnBWorldModel
        from bnb_wm.training.checkpoint import load_weights_only
        a = _W["args"]
        dev = torch.device(a["device"] if (a["device"] == "cpu" or torch.cuda.is_available()) else "cpu")
        model = BnBWorldModel().to(dev)
        load_weights_only(model, a["checkpoint"], device=dev)
        model.eval()
        _W["model"], _W["device"] = model, dev
        _W["load_report"] = getattr(model, "_load_report", {})
    return _W["model"], _W["device"]


def _get_instance(tier, seed, idx):
    key = (tier, seed, idx)
    if key not in _W["inst"]:
        _W["inst"] = {key: instance(tier, seed, idx)}  # keep one in memory
    return _W["inst"][key]


# --------------------------------------------------------------------------
# Solvers
# --------------------------------------------------------------------------

def _cut_check(cuts, solutions):
    """Count committed cuts violated by known integer-feasible solutions.
    A globally valid cut is satisfied by every integer-feasible point."""
    out = {}
    for name, x in solutions.items():
        if x is None or not cuts:
            continue
        viol = [c.rhs - float(c.lhs @ x) for c in cuts]
        out[name] = {"n_cuts": len(cuts),
                     "n_violated": int(sum(v > 1e-6 for v in viol)),
                     "max_violation": float(max(0.0, max(viol)))}
    return out


def _run_ours(kind, cfg_dict, A, b, c, tl, nl, ref_x):
    from bnb_wm.solver.config import SolverConfig
    cfg_dict = dict(cfg_dict)
    sb_init = cfg_dict.pop("sb_init", None)
    eta = cfg_dict.pop("eta", None)
    tl = cfg_dict.pop("time_limit", tl)
    if kind == "classical":
        from bnb_wm.solver.classical_bnb import ClassicalBnBSolver
        cfg = SolverConfig(branch_mode="most_fractional", cut_mode="none",
                           ors_cascade=False, katz_weight=0.0,
                           node_selection="bound", primal_heuristic=True,
                           time_limit=tl, node_limit=nl, exact=True)
        solver = ClassicalBnBSolver(config=cfg, sb_init=sb_init, eta=eta)
        model_time, decision_lps = 0.0, None
    else:
        from bnb_wm.solver.neural_bnb import NeuralBnBSolver
        model, dev = _model()
        cfg = SolverConfig(**{**cfg_dict, "time_limit": tl, "node_limit": nl})
        solver = NeuralBnBSolver(model, dev, cfg)
    t0 = time.perf_counter()
    r = solver.solve(A, b, c)
    wall = time.perf_counter() - t0
    d = dict(r.diagnostics or {})
    cuts = list(solver._lp.committed_cuts) if getattr(solver, "_lp", None) else []
    sol = None if r.solution is None else np.asarray(r.solution, dtype=np.float64)
    rec = {
        "status": r.status,
        "obj": float(r.objective) if np.isfinite(r.objective) else None,
        "bound": d.pop("global_lb", None),
        "gap": float(r.optimality_gap) if np.isfinite(r.optimality_gap) else None,
        "nodes": int(r.n_nodes),
        "wall_time": wall,
        "lp_time": float(r.lp_time),
        "lp_solves": int(getattr(r, "lp_solves", 0) or getattr(r, "tree_lps", 0)),
        "decision_lps": int(getattr(r, "decision_lps", 0) or 0),
        "model_time": float(getattr(r, "model_time", 0.0) or 0.0),
        "cuts_added": int(r.cuts_added),
        "cut_validity": _cut_check(cuts, {"own_incumbent": sol, "reference": ref_x}),
        "trace": d.pop("trace", []),
        "resolved_config": d.pop("config", None) or {**cfg_dict, "sb_init": sb_init, "eta": eta},
        "diagnostics": d,
    }
    if sol is not None:
        rec["solution_ones"] = np.flatnonzero(sol > 0.5).tolist() if _W["args"]["save_solutions"] else None
    return rec


def _run_scip(row, A, b, c, tl):
    from pyscipopt import Model
    tl = row.get("time_limit", tl)
    variant = row.get("variant", "default")
    m_rows, n_cols = A.shape
    m = Model()
    m.hideOutput(True)
    m.setParam("limits/time", float(tl))
    m.setParam("parallel/maxnthreads", 1)
    if variant in ("pseudocost", "fullstrong"):
        m.setParam("presolving/maxrounds", 0)
        m.setParam("separating/maxrounds", 0)
        m.setParam("separating/maxroundsroot", 0)
    if variant == "pseudocost":
        m.setParam("branching/pscost/priority", 536870911)
    elif variant == "fullstrong":
        m.setParam("branching/fullstrong/priority", 536870911)
    xs = [m.addVar(f"x{j}", vtype="B", obj=float(c[j])) for j in range(n_cols)]
    for i in range(m_rows):
        nz = np.flatnonzero(A[i] > 0)
        m.addCons(sum(xs[j] for j in nz) >= float(b[i]))
    m.setMinimize()
    t0 = time.perf_counter()
    m.optimize()
    wall = time.perf_counter() - t0
    status = m.getStatus()
    obj = bound = None
    sol_ones = None
    if m.getNSols() > 0:
        obj = float(m.getObjVal())
        if row.get("save_solution"):
            best = m.getBestSol()
            sol_ones = [j for j in range(n_cols) if m.getSolVal(best, xs[j]) > 0.5]
    try:
        bound = float(m.getDualbound())
    except Exception:
        pass
    gap = (obj - bound) / max(abs(obj), 1e-10) if obj is not None and bound is not None else None
    return {"status": "optimal" if status == "optimal" else ("timeout" if status == "timelimit" else status),
            "obj": obj, "bound": bound, "gap": gap, "nodes": int(m.getNNodes()),
            "wall_time": wall, "resolved_config": row, "solution_ones": sol_ones}


def run_job(job):
    a = _W["args"]
    tier, seed, idx, name = job["tier"], job["seed"], job["idx"], job["row"]
    A, b, c = _get_instance(tier, seed, idx)
    common, rows, _ = load_rows(Path(a["rows"]))
    kind, cfg = resolve_row(name, common, rows)
    ref = a["reference"].get(f"{tier}/{seed}/{idx}")
    ref_x = None
    if ref and ref.get("solution_ones") is not None:
        ref_x = np.zeros(A.shape[1])
        ref_x[ref["solution_ones"]] = 1.0
    t_start = time.time()
    try:
        if kind == "scip":
            rec = _run_scip(cfg, A, b, c, a["time_limit"])
        else:
            rec = _run_ours(kind, cfg, A, b, c, a["time_limit"], a["node_limit"], ref_x)
        err = None
    except Exception as e:  # record and keep going; never silently drop a job
        import traceback
        rec, err = {"status": "crash"}, traceback.format_exc()
    return {"tier": tier, "seed": seed, "idx": idx, "row": name, "kind": kind,
            "instance_sha": instance_hash(A, b, c), "shape": list(A.shape),
            "time_limit": cfg.get("time_limit", a["time_limit"]),
            "node_limit": a["node_limit"], "started": t_start, "error": err,
            "host": a["host"], "checkpoint_load": _W.get("load_report"), **rec}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def _jsonable(o):
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def _provenance(args):
    def sh(cmd):
        try:
            return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    info = {"commit": sh(["git", "rev-parse", "HEAD"]),
            "dirty": bool(sh(["git", "status", "--porcelain", "--untracked-files=no"])),
            "python": platform.python_version(), "platform": platform.platform(),
            "cpu": platform.processor() or platform.machine(), "n_cpu": os.cpu_count()}
    for mod in ("numpy", "torch", "highspy", "pyscipopt"):
        try:
            info[mod] = getattr(__import__(mod), "__version__", "installed")
        except ImportError:
            info[mod] = None
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def _load_reference(paths):
    ref = {}
    for p in paths or []:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            if r.get("solution_ones") is not None:
                key = f"{r['tier']}/{r['seed']}/{r['idx']}"
                if key not in ref or (r["obj"] is not None and r["obj"] < ref[key]["obj"]):
                    ref[key] = {"obj": r["obj"], "solution_ones": r["solution_ones"]}
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=TIERS, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--n_instances", type=int, required=True)
    ap.add_argument("--first_instance", type=int, default=0)
    ap.add_argument("--group", default=None)
    ap.add_argument("--rows_include", nargs="+", default=None, help="explicit row names")
    ap.add_argument("--rows", default=str(ROOT / "configs/benchmark_rows.yaml"))
    ap.add_argument("--time_limit", type=float, default=None,
                    help="default: 300 s (easy, medium) or 600 s (hard)")
    ap.add_argument("--node_limit", type=int, default=100_000)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--reference", nargs="*", default=None,
                    help="JSONL file(s) with scip_ref solutions, for cut-validity checks")
    ap.add_argument("--save_solutions", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if args.time_limit is None:
        args.time_limit = 600.0 if args.tier == "hard" else 300.0
    common, rows, groups = load_rows(Path(args.rows))
    names = list(args.rows_include or []) + (groups[args.group] if args.group else [])
    names = list(dict.fromkeys(names))
    unknown = [n for n in names if n not in rows]
    if not names or unknown:
        sys.exit(f"no rows selected or unknown rows {unknown}; groups: {list(groups)}")
    needs_model = any(rows[n]["solver"] == "neural" for n in names)
    if needs_model and not args.checkpoint and not args.dry_run:
        sys.exit("--checkpoint is required for neural rows")

    jobs = [{"tier": args.tier, "seed": s, "idx": i, "row": n}
            for s in args.seeds
            for i in range(args.first_instance, args.first_instance + args.n_instances)
            for n in names]

    def limit_of(n):
        return float(rows[n].get("time_limit", args.time_limit))
    worst = sum(limit_of(j["row"]) for j in jobs)
    print(f"{len(jobs)} jobs ({len(names)} rows x {len(args.seeds)} seeds x {args.n_instances} instances); "
          f"worst case {worst/3600:.1f} h serial, {worst/3600/max(1, args.workers):.1f} h on {args.workers} workers")
    if args.dry_run:
        return
    if not args.out:
        sys.exit("--out is required")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") != "crash":
                done.add((r["tier"], r["seed"], r["idx"], r["row"]))
    todo = [j for j in jobs if (j["tier"], j["seed"], j["idx"], j["row"]) not in done]
    print(f"{len(done)} already done, {len(todo)} to run")

    prov = _provenance(args)
    meta = out.with_suffix(".meta.json")
    meta.write_text(json.dumps({"args": vars(args), "provenance": prov,
                                "rows": {n: resolve_row(n, common, rows) for n in names}}, indent=2))
    if prov.get("dirty"):
        print("WARNING: uncommitted changes in the working tree; results may not be reproducible")

    wargs = {**vars(args), "reference": _load_reference(args.reference),
             "host": platform.node()}
    t0 = time.time()
    with open(out, "a") as fh:
        def write(rec, k):
            fh.write(json.dumps(rec, default=_jsonable) + "\n")
            fh.flush()
            obj = "-" if rec.get("obj") is None else f"{rec['obj']:.3f}"
            gap = "-" if rec.get("gap") is None else f"{100*rec['gap']:.2f}%"
            print(f"[{k}/{len(todo)} {(time.time()-t0)/60:6.1f} min] {rec['tier']} s{rec['seed']}/{rec['idx']} "
                  f"{rec['row']:<22} {rec['status']:<9} obj={obj} gap={gap} "
                  f"nodes={rec.get('nodes', '-')} t={rec.get('wall_time', 0):.0f}s", flush=True)
        if args.workers <= 1:
            _init_worker(wargs)
            for k, j in enumerate(todo, 1):
                write(run_job(j), k)
        else:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            with ctx.Pool(args.workers, initializer=_init_worker, initargs=(wargs,),
                          maxtasksperchild=50) as pool:
                for k, rec in enumerate(pool.imap_unordered(run_job, todo), 1):
                    write(rec, k)


if __name__ == "__main__":
    main()
