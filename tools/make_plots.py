#!/usr/bin/env python
"""
make_plots.py — Publication figures for the CutWorld paper.

Turns the evaluation JSONs into vector PDFs for \\includegraphics, in the style
of a results figure with several panels:

    fig_ablation_nodes.pdf  median explored nodes per method (log scale),
                            SCIP highlighted -- the branching comparison.
    fig_solved.pdf          %solved-to-optimality per method (the validity gate).
    fig_cuts.pdf            cut-selection ablation: nodes for none/heuristic/
                            learned (from the cut_ablation printout).

Inputs are the JSON written by bnb_wm.evaluate.ablation (per_instance / solved /
times / cuts). The cut panel takes numbers on the command line since
cut_ablation.py prints rather than dumps JSON.

Run (any env with matplotlib):
    python tools/make_plots.py \
        --ablation results/ablation_warm.json \
        --cut_none 9.73 --cut_heur 8.60 --cut_learn 8.47 \
        --outdir paper/figures

Then in the paper add, e.g.:
    \\includegraphics[width=\\linewidth]{figures/fig_ablation_nodes.pdf}
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# neutral, print-friendly styling
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})
LEARNED_C, SCIP_C, BASE_C = "#3b6fb0", "#444444", "#b0b0b0"


def _label(m):
    return {
        "scip": "SCIP", "strong_branching": "strong", "pseudocost": "pscost", "random": "random", "most_fractional": "most-frac",
        "policy_only": "policy", "value_rollout": "+value",
        "cost_to_go": "+ctg", "tree_rollout": "+tree", "reward_return": "+reward",
    }.get(m, m)


def plot_nodes(data, outdir):
    nodes = data["per_instance"]
    order = [m for m in ["strong_branching", "scip", "pseudocost", "random",
                         "most_fractional", "policy_only", "value_rollout",
                         "cost_to_go", "tree_rollout", "reward_return"]
             if m in nodes]
    meds = [np.median(np.asarray(nodes[m], float)) for m in order]
    colors = [SCIP_C if m in ("scip", "strong_branching")
              else BASE_C if m in ("random", "most_fractional", "pseudocost")
              else LEARNED_C for m in order]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.bar([_label(m) for m in order], meds, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("median nodes (log)")
    ax.set_title("Explored nodes by branching rule")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    p = os.path.join(outdir, "fig_ablation_nodes.pdf")
    fig.savefig(p); plt.close(fig); print("wrote", p)


def plot_solved(data, outdir):
    solved = data.get("solved")
    if not solved:
        print("no 'solved' field; skipping %solved plot"); return
    order = [m for m in ["strong_branching", "scip", "pseudocost", "random",
                         "most_fractional", "policy_only", "value_rollout",
                         "cost_to_go", "tree_rollout", "reward_return"]
             if m in solved]
    pct = [100.0 * np.mean(np.asarray(solved[m], float)) for m in order]
    colors = [SCIP_C if m in ("scip", "strong_branching")
              else BASE_C if m in ("random", "most_fractional", "pseudocost")
              else LEARNED_C for m in order]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.bar([_label(m) for m in order], pct, color=colors)
    ax.set_ylabel("% solved to optimality")
    ax.set_ylim(0, 105)
    ax.set_title("Optimality rate within time limit")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    p = os.path.join(outdir, "fig_solved.pdf")
    fig.savefig(p); plt.close(fig); print("wrote", p)


def plot_time(data, outdir):
    times = data.get("times")
    if not times:
        print("no 'times' field; skipping time plot"); return
    order = [m for m in ["strong_branching", "scip", "pseudocost", "random",
                         "most_fractional", "policy_only", "value_rollout",
                         "cost_to_go", "tree_rollout", "reward_return"]
             if m in times]
    means = [float(np.mean(np.asarray(times[m], float))) for m in order]
    colors = [SCIP_C if m in ("scip", "strong_branching")
              else BASE_C if m in ("random", "most_fractional", "pseudocost")
              else LEARNED_C for m in order]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.bar([_label(m) for m in order], means, color=colors)
    ax.set_ylabel("mean solve time (s)")
    ax.set_title("Solving time by branching rule")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    p = os.path.join(outdir, "fig_time.pdf")
    fig.savefig(p); plt.close(fig); print("wrote", p)


def plot_tradeoff(data, outdir, tag="", title=None):
    """Nodes-vs-time scatter: one labeled point per method (both log axes)."""
    nodes = data["per_instance"]; times = data.get("times")
    if not times:
        print("no 'times' field; skipping tradeoff plot"); return
    order = [m for m in ["strong_branching", "scip", "pseudocost", "random",
                         "most_fractional", "policy_only", "value_rollout",
                         "cost_to_go", "tree_rollout", "reward_return"]
             if m in nodes and m in times]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for m in order:
        x = float(np.median(np.asarray(nodes[m], float)))
        y = float(np.mean(np.asarray(times[m], float)))
        if m == "reward_return":
            c, mk, sz = LEARNED_C, "*", 200        # highlight our full model
        elif m in ("scip", "strong_branching"):
            c, mk, sz = SCIP_C, "s", 60
        elif m in ("random", "most_fractional", "pseudocost"):
            c, mk, sz = BASE_C, "^", 60
        else:
            c, mk, sz = LEARNED_C, "o", 55
        ax.scatter(x, y, c=c, marker=mk, s=sz, zorder=3)
        ax.annotate(_label(m), (x, y), textcoords="offset points",
                    xytext=(5, 4), fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("median nodes (log)")
    ax.set_ylabel("mean time, s (log)")
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    fig.tight_layout()
    p = os.path.join(outdir, f"fig_tradeoff{tag}.pdf")
    fig.savefig(p); plt.close(fig); print("wrote", p)


def plot_cuts(vals, outdir):
    names = ["no cuts", "max-violation", "learned"]
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    colors = [BASE_C, BASE_C, LEARNED_C]
    ax.bar(names, vals, color=colors)
    ax.set_ylabel("mean nodes")
    ax.set_title("Cut selection (1 cut / node)")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    p = os.path.join(outdir, "fig_cuts.pdf")
    fig.savefig(p); plt.close(fig); print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", default=None, help="ablation JSON path")
    ap.add_argument("--cut_none", type=float, default=None)
    ap.add_argument("--cut_heur", type=float, default=None)
    ap.add_argument("--cut_learn", type=float, default=None)
    ap.add_argument("--outdir", default="paper/figures")
    ap.add_argument("--tag", default="",
                    help="suffix for the tradeoff filename, e.g. _medium, so a "
                         "per-tier run does not overwrite (fig_tradeoff_medium.pdf)")
    ap.add_argument("--title", default=None, help="title for the tradeoff plot")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.ablation:
        data = json.load(open(args.ablation))
        plot_nodes(data, args.outdir)
        plot_solved(data, args.outdir)
        plot_time(data, args.outdir)
        plot_tradeoff(data, args.outdir, tag=args.tag, title=args.title)
    if None not in (args.cut_none, args.cut_heur, args.cut_learn):
        plot_cuts([args.cut_none, args.cut_heur, args.cut_learn], args.outdir)
    if not args.ablation and args.cut_none is None:
        print("Nothing to plot: pass --ablation and/or --cut_* values.")


if __name__ == "__main__":
    main()
