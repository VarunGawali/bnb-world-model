"""
Report figures from the corrected benchmark (tools/run_benchmark.py output).

    python paper/make_report_plots.py results/rebuttal/medium.jsonl \
        results/rebuttal/medium_ref.jsonl results/rebuttal/hard.jsonl \
        results/rebuttal/hard_ref.jsonl --out paper/

Writes, as .pdf and .png:
  plot1_gap_vs_nodes     proven gap at the time limit vs nodes explored, per run
  plot2_gap_by_method    mean primal and proven gap per method
  plot3_anytime_gap      mean primal gap over time (from the anytime traces)

The best known objective of an instance is the best feasible objective over
all runs passed in, including SCIP.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, FuncFormatter

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10.5,
    "axes.titlesize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8.5, "axes.linewidth": 0.8, "savefig.dpi": 400,
})

# row name -> (label, colour, marker); order = legend and bar order
ROWS = {
    "scip_default":       ("SCIP default",            "#000000", "X"),
    "scip_pscost":        ("SCIP, branching only",    "#666666", "X"),
    "mf":                 ("Most-fractional",         "#888888", "o"),
    "sb1":                ("SB-1",                    "#bbbbbb", "o"),
    "sb4":                ("SB-4",                    "#555555", "o"),
    "sb8":                ("SB-8",                    "#222222", "o"),
    "paper_policy":       ("Policy",                  "#4477AA", "s"),
    "paper_mf_blend":     ("MF-blend",                "#EE6677", "s"),
    "paper_policy_cuts":  ("Policy + cuts",           "#CCBB44", "s"),
    "paper_rollout":      ("Rollout (K=3)",           "#228833", "^"),
    "paper_rollout_size": ("Rollout + size (K=1)",    "#66CCEE", "^"),
    "paper_neural_best":  ("Neural-Best",             "#AA3377", "D"),
}
TIER_TITLE = {"easy": "Easy (200×400)", "medium": "Medium (500×1000)", "hard": "Hard (1000×2000)"}


def load(paths):
    recs = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            if r.get("status") == "crash" or r["row"] not in ROWS:
                continue
            recs[(r["tier"], r["seed"], r["idx"], r["row"])] = r
    by_tier = defaultdict(list)
    for r in recs.values():
        by_tier[r["tier"]].append(r)
    best = {}
    for r in recs.values():
        k = (r["tier"], r["seed"], r["idx"])
        if r.get("obj") is not None and r.get("diagnostics", {}).get("incumbent_feasible", True) is not False:
            best[k] = min(best.get(k, np.inf), r["obj"])
    return by_tier, best


def primal_gap(obj, z):
    return 100.0 * (obj - z) / max(1.0, abs(z))


def tiers_in(by_tier):
    return [t for t in ("easy", "medium", "hard") if t in by_tier]


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / name}.pdf/.png")


def plot_gap_vs_nodes(by_tier, out):
    tiers = tiers_in(by_tier)
    fig, axes = plt.subplots(1, len(tiers), figsize=(5.2 * len(tiers), 4), squeeze=False)
    for ax, t in zip(axes[0], tiers):
        for row, (lab, col, mk) in ROWS.items():
            rs = [r for r in by_tier[t] if r["row"] == row and r.get("gap") is not None]
            if rs:
                ax.scatter([r["nodes"] for r in rs], [100 * r["gap"] for r in rs],
                           s=22, c=col, marker=mk, alpha=0.8, edgecolors="none", label=lab)
        ax.set_xscale("log")
        xs = [r["nodes"] for r in by_tier[t] if r.get("nodes")]
        ticks = [v for v in (200, 300, 500, 1000, 2000, 3000, 5000, 10000, 20000)
                 if min(xs) * 0.8 <= v <= max(xs) * 1.25]
        ax.set_xticks(ticks)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_xlabel("Nodes explored within the time limit (log scale)")
        ax.set_ylabel("Proven gap at the time limit (%)")
        ax.set_title(TIER_TITLE[t])
        ax.grid(alpha=0.3, lw=0.5)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.text(0.5, -0.17, "One point per run. SCIP node counts come from a different solver and are not directly comparable.",
             ha="center", fontsize=8)
    save(fig, out, "plot1_gap_vs_nodes")


def plot_gap_by_method(by_tier, best, out):
    tiers = tiers_in(by_tier)
    fig, axes = plt.subplots(1, len(tiers), figsize=(6.2 * len(tiers), 4.8), squeeze=False)
    fig.subplots_adjust(wspace=0.6)
    for ax, t in zip(axes[0], tiers):
        names, prim, prov, cols = [], [], [], []
        for row, (lab, col, _) in ROWS.items():
            rs = [r for r in by_tier[t] if r["row"] == row]
            pg = [primal_gap(r["obj"], best[(t, r["seed"], r["idx"])]) for r in rs
                  if r.get("obj") is not None and (t, r["seed"], r["idx"]) in best]
            gv = [100 * r["gap"] for r in rs if r.get("gap") is not None]
            if pg and gv:
                names.append(lab); prim.append(np.mean(pg)); prov.append(np.mean(gv)); cols.append(col)
        y = np.arange(len(names))[::-1]
        ax.barh(y + 0.2, prov, height=0.38, color=cols, alpha=0.45, label="Proven gap")
        ax.barh(y - 0.2, prim, height=0.38, color=cols, label="Primal gap")
        for yi, a, b in zip(y, prov, prim):
            ax.text(a, yi + 0.2, f" {a:.1f}", va="center", fontsize=7.5)
            ax.text(b, yi - 0.2, f" {b:.2f}", va="center", fontsize=7.5)
        ax.set_yticks(y); ax.set_yticklabels(names)
        ax.set_xlabel("Mean gap at the time limit (%)")
        ax.set_title(TIER_TITLE[t])
        ax.grid(axis="x", alpha=0.3, lw=0.5)
        ax.set_xlim(0, max(prov) * 1.18)
    fig.text(0.5, -0.04, "Light bars: proven gap (own bound). Dark bars: primal gap (distance to best known solution). Lower is better.",
             ha="center", fontsize=8)
    save(fig, out, "plot2_gap_by_method")


def plot_anytime(by_tier, best, out, cap=10.0):
    tiers = tiers_in(by_tier)
    fig, axes = plt.subplots(1, len(tiers), figsize=(5.2 * len(tiers), 4), squeeze=False)
    for ax, t in zip(axes[0], tiers):
        limit = max(r.get("time_limit") or 0 for r in by_tier[t])
        grid = np.linspace(0, limit, 300)
        for row, (lab, col, _) in ROWS.items():
            rs = [r for r in by_tier[t] if r["row"] == row and r.get("trace")]
            if not rs:
                continue
            curves = []
            for r in rs:
                z = best.get((t, r["seed"], r["idx"]))
                if z is None:
                    continue
                pts = sorted((p[0], p[2]) for p in r["trace"])
                g = np.full_like(grid, cap)
                for tp, ub in pts:
                    if ub is not None:
                        g[grid >= tp] = min(cap, primal_gap(ub, z))
                curves.append(g)
            if curves:
                ax.plot(grid, np.mean(curves, axis=0), color=col, lw=1.6, label=lab)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Mean primal gap (%), capped at {cap:g}%")
        ax.set_title(TIER_TITLE[t])
        ax.grid(alpha=0.3, lw=0.5)
        ax.set_xlim(0, limit)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.text(0.5, -0.17, "Before a run's first solution its gap is counted at the cap. SCIP is not shown: it records no trace.",
             ha="center", fontsize=8)
    save(fig, out, "plot3_anytime_gap")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", default="paper")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    by_tier, best = load(args.results)
    plot_gap_vs_nodes(by_tier, out)
    plot_gap_by_method(by_tier, best, out)
    plot_anytime(by_tier, best, out)


if __name__ == "__main__":
    main()
