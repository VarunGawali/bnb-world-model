"""
analyze_benchmark.py — tables and paired statistics from run_benchmark.py output.

For every instance, the best known objective is the best feasible objective
over all rows (including scip_ref), and the best proven bound is the largest
lower bound any row certified. Rows are compared on:

  * complete         share of instances closed (status "optimal")
  * SGM nodes        all instances, and on the instances every compared row closed
  * SGM time         wall clock (shift 1 s)
  * primal gap       (obj - best known) / |best known|, at the end of the run
  * proven gap       the row's own (obj - bound) / |obj| at the end
  * primal integral  time-average of the primal gap over the run (Berthold),
                     gap = 1 before the first incumbent
  * ms / node        encode, branch, rollout and LP time per node

Paired comparisons against --baseline use the Wilcoxon signed-rank test and a
bootstrap 95% CI for the ratio of shifted geometric means.

    python tools/analyze_benchmark.py results/v2/medium*.jsonl --baseline paper_policy \
        --md results/v2/medium.md --csv results/v2/medium_instances.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(paths):
    recs = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            key = (r["tier"], r["seed"], r["idx"], r["row"])
            if r.get("status") == "crash" and key in recs:
                continue
            recs[key] = r
    return list(recs.values())


def sgm(xs, shift):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    if not xs:
        return float("nan")
    return float(np.exp(np.mean(np.log(np.maximum(xs, 0.0) + shift))) - shift)


def primal_integral(trace, best, t_limit):
    """Normalised primal integral in [0, 1] from [t, nodes, ub, lb, open] points."""
    if not trace or best is None or not t_limit:
        return None

    def gap(ub):
        if ub is None:
            return 1.0
        den = max(abs(ub), abs(best))
        return 0.0 if den == 0 else min(1.0, max(0.0, (ub - best) / den))

    pts = sorted((p[0], p[2]) for p in trace)
    total, prev_t, prev_g = 0.0, 0.0, 1.0
    for t, ub in pts:
        t = min(t, t_limit)
        total += prev_g * (t - prev_t)
        prev_t, prev_g = t, gap(ub)
    total += prev_g * max(0.0, t_limit - prev_t)
    return total / t_limit


def per_instance(recs):
    by_inst = defaultdict(dict)
    for r in recs:
        by_inst[(r["tier"], r["seed"], r["idx"])][r["row"]] = r
    ref = {}
    for key, rows in by_inst.items():
        objs = [r["obj"] for r in rows.values()
                if r.get("obj") is not None and r.get("status") != "crash"
                and r.get("diagnostics", {}).get("incumbent_feasible", True) is not False]
        bounds = [r["bound"] for r in rows.values() if r.get("bound") is not None
                  and r.get("status") not in ("crash", "lp_error")]
        best = min(objs) if objs else None
        lb = max(bounds) if bounds else None
        if best is not None and lb is not None:
            lb = min(lb, best)
        certified = (best is not None and lb is not None
                     and best - lb <= 1e-6 * max(1.0, abs(best)))
        ref[key] = {"best": best, "bound": lb, "certified": certified}
    return by_inst, ref


def wilcoxon(a, b):
    try:
        from scipy.stats import wilcoxon as w
    except ImportError:
        return None
    d = np.asarray(a, float) - np.asarray(b, float)
    if len(d) < 1 or np.all(d == 0):
        return 1.0
    return float(w(a, b, zero_method="wilcox").pvalue)


def boot_ratio(a, b, shift, n=10000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    la, lb = np.log(a + shift), np.log(b + shift)
    ra = np.exp(la[idx].mean(1)) - shift
    rb = np.exp(lb[idx].mean(1)) - shift
    ratio = ra / np.maximum(rb, 1e-12)
    return float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))


def pct(x):
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:.2f}%"


def num(x, fmt="{:,.0f}"):
    return "—" if x is None or not np.isfinite(x) else fmt.format(x)


def analyze(recs, rows_order, baseline):
    out = []
    by_inst, ref = per_instance(recs)
    tiers = sorted({k[0] for k in by_inst})
    for tier in tiers:
        insts = sorted(k for k in by_inst if k[0] == tier)
        present = [r for r in rows_order if any(r in by_inst[k] for k in insts)]
        n_cert = sum(ref[k]["certified"] for k in insts)
        out.append(f"\n## {tier}: {len(insts)} instances, seeds "
                   f"{sorted({k[1] for k in insts})}; {n_cert} certified optimal by some run\n")

        # failures and integrity checks
        issues = []
        for k in insts:
            for row, r in by_inst[k].items():
                d = r.get("diagnostics", {})
                cv = r.get("cut_validity", {})
                bad = []
                if r.get("status") in ("crash", "lp_error", "incomplete"):
                    bad.append(r["status"])
                if d.get("incumbent_feasible") is False:
                    bad.append("infeasible incumbent")
                if d.get("lp_unresolved"):
                    bad.append(f"{d['lp_unresolved']} unresolved LPs")
                for src, v in cv.items():
                    if v.get("n_violated"):
                        bad.append(f"{v['n_violated']} cuts violated by {src}")
                best = ref[k]["best"]
                if r.get("status") == "optimal" and r.get("obj") is not None and best is not None \
                        and r["obj"] > best + 1e-6 * max(1.0, abs(best)):
                    bad.append(f"claims optimal at {r['obj']:.4f} but best known is {best:.4f}")
                if bad:
                    issues.append(f"- s{k[1]}/{k[2]} {row}: " + "; ".join(bad))
        out.append("Integrity issues: " + ("none" if not issues else f"{len(issues)}\n" + "\n".join(issues)))

        common = [k for k in insts if all(by_inst[k].get(r, {}).get("status") == "optimal" for r in present)]
        hdr = ("| Row | n | Complete | SGM nodes | SGM nodes (all closed, n={}) | SGM time (s) | "
               "Mean primal gap | Mean proven gap | Primal integral | ms/node enc · pol · roll · LP |").format(len(common))
        out += ["", hdr, "|" + "---|" * 10]
        for row in present:
            rs = [by_inst[k][row] for k in insts if row in by_inst[k] and by_inst[k][row].get("status") != "crash"]
            if not rs:
                continue
            nodes = [r.get("nodes") for r in rs]
            pg = [(r["obj"] - ref[(r["tier"], r["seed"], r["idx"])]["best"]) /
                  max(abs(ref[(r["tier"], r["seed"], r["idx"])]["best"]), 1e-10)
                  for r in rs if r.get("obj") is not None and ref[(r["tier"], r["seed"], r["idx"])]["best"]]
            prov = [r["gap"] for r in rs if r.get("gap") is not None]
            pis = [primal_integral(r.get("trace"), ref[(r["tier"], r["seed"], r["idx"])]["best"], r.get("time_limit"))
                   for r in rs]
            pis = [p for p in pis if p is not None]
            tm = defaultdict(list)
            for r in rs:
                t = dict(r.get("diagnostics", {}).get("timing", {}))
                if "lp" not in t and r.get("lp_time"):
                    t["lp"] = r["lp_time"]
                n = max(1, r.get("nodes") or 1)
                for name in ("encode", "branch", "rollout", "lp"):
                    if name in t:
                        tm[name].append(1000 * t[name] / n)
            ms = " · ".join(num(np.mean(tm[n]), "{:.1f}") if tm[n] else "—" for n in ("encode", "branch", "rollout", "lp"))
            out.append(f"| {row} | {len(rs)} | {sum(r['status'] == 'optimal' for r in rs)}/{len(rs)} | "
                       f"{num(sgm(nodes, 10))} | {num(sgm([by_inst[k][row]['nodes'] for k in common], 10)) if common else '—'} | "
                       f"{num(sgm([r['wall_time'] for r in rs], 1), '{:.1f}')} | {pct(np.mean(pg)) if pg else '—'} | "
                       f"{pct(np.mean(prov)) if prov else '—'} | {num(np.mean(pis), '{:.3f}') if pis else '—'} | {ms} |")

        if baseline in present:
            out += ["", f"Paired comparison with `{baseline}` (Wilcoxon signed-rank; bootstrap 95% CI of SGM ratio)", "",
                    "| Row | metric | n pairs | SGM ratio | 95% CI | wins/losses | p |", "|---|---|---|---|---|---|---|"]
            for row in present:
                if row == baseline:
                    continue
                for metric, shift, closed_only in (("nodes", 10, True), ("primal_gap", 1e-4, False),
                                                   ("proven_gap", 1e-4, False)):
                    a, b = [], []
                    for k in insts:
                        r1, r0 = by_inst[k].get(row), by_inst[k].get(baseline)
                        if not r1 or not r0 or "crash" in (r1.get("status"), r0.get("status")):
                            continue
                        if closed_only and not (r1["status"] == r0["status"] == "optimal"):
                            continue
                        if metric == "nodes":
                            v1, v0 = r1.get("nodes"), r0.get("nodes")
                        elif metric == "primal_gap":
                            best = ref[k]["best"]
                            if best is None or r1.get("obj") is None or r0.get("obj") is None:
                                continue
                            v1, v0 = (r1["obj"] - best) / abs(best), (r0["obj"] - best) / abs(best)
                        else:
                            v1, v0 = r1.get("gap"), r0.get("gap")
                        if v1 is None or v0 is None:
                            continue
                        a.append(v1)
                        b.append(v0)
                    if not a:
                        continue
                    ratio = (sgm(a, shift) + shift) / (sgm(b, shift) + shift) if metric != "nodes" else sgm(a, shift) / max(sgm(b, shift), 1e-12)
                    lo, hi = boot_ratio(a, b, shift)
                    p = wilcoxon(a, b)
                    wins = sum(x < y for x, y in zip(a, b))
                    losses = sum(x > y for x, y in zip(a, b))
                    out.append(f"| {row} | {metric}{' (both closed)' if closed_only else ''} | {len(a)} | "
                               f"{ratio:.3f} | [{lo:.3f}, {hi:.3f}] | {wins}/{losses} | "
                               f"{'—' if p is None else f'{p:.4f}'} |")
    return "\n".join(out), by_inst, ref


def write_csv(path, by_inst, ref):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tier", "seed", "idx", "instance_sha", "row", "status", "obj", "bound", "proven_gap",
                    "best_known", "best_bound", "certified", "nodes", "wall_time", "lp_time",
                    "lp_status_counts", "lp_unresolved", "incumbent_feasible", "cuts_added",
                    "cut_violations_vs_reference"])
        for k in sorted(by_inst):
            for row, r in sorted(by_inst[k].items()):
                d = r.get("diagnostics", {})
                w.writerow([*k, r.get("instance_sha"), row, r.get("status"), r.get("obj"), r.get("bound"),
                            r.get("gap"), ref[k]["best"], ref[k]["bound"], ref[k]["certified"],
                            r.get("nodes"), r.get("wall_time"), r.get("lp_time"),
                            json.dumps(d.get("lp_status_counts")), d.get("lp_unresolved"),
                            d.get("incumbent_feasible"), r.get("cuts_added"),
                            (r.get("cut_validity", {}).get("reference") or {}).get("n_violated")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--baseline", default="paper_policy")
    ap.add_argument("--rows", nargs="+", default=None, help="row order (default: as found)")
    ap.add_argument("--md", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    recs = load(args.results)
    if not recs:
        sys.exit("no records")
    order = args.rows or list(dict.fromkeys(r["row"] for r in recs))
    text, by_inst, ref = analyze(recs, order, args.baseline)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n")
    if args.csv:
        write_csv(args.csv, by_inst, ref)


if __name__ == "__main__":
    main()
