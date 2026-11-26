"""
Summary figures from the corrected benchmark tables (report Tables 1 and 2).

These use the per-method means reported in the tables, so they need no raw
result files. Per-run figures (scatter of every run, anytime curves) still
need the JSONL results and paper/make_report_plots.py.

    python paper/make_summary_plots.py --out paper/

Writes fig_primal_gap_by_method.png/.pdf and fig_nodes_vs_proven_gap.png/.pdf.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# group -> (colour, marker); categorical slots 1-3, validated all-pairs
GROUPS = {
    "SCIP (reference solver)": ("#00b894", "s"),
    "Classical branching (our harness)": ("#257de0", "o"),
    "Neural (CutWorld)": ("#f47230", "^"),
}
SCIP, CLS, NEU = GROUPS

# (method, group, SGM nodes, primal gap %, proven gap %)
MEDIUM = [
    ("SCIP default", SCIP, 2260, 0.73, 17.61),
    ("SCIP, branching only", SCIP, 5289, 1.00, 17.05),
    ("Most-fractional", CLS, 4645, 3.20, 17.87),
    ("SB-1", CLS, 4660, 3.51, 18.04),
    ("SB-4", CLS, 3492, 3.48, 17.59),
    ("SB-8", CLS, 3140, 3.81, 17.76),
    ("Policy (imitation)", NEU, 2711, 6.11, 23.25),
    ("+ MF-blend", NEU, 1960, 4.47, 18.72),
    ("+ Cuts (attn)", NEU, 2201, 6.44, 23.62),
    ("+ Rollout (K=3)", NEU, 1209, 6.70, 24.01),
    ("+ Rollout+Size (K=1)", NEU, 1867, 6.12, 23.13),
    ("Neural-Best", NEU, 1612, 4.41, 19.05),
]
HARD = [
    ("SCIP default", SCIP, 1343, 2.82, 34.31),
    ("SCIP, branching only", SCIP, 970, 2.07, 34.09),
    ("Most-fractional", CLS, 2307, 0.81, 31.64),
    ("SB-1", CLS, 1830, 1.79, 32.46),
    ("SB-4", CLS, 588, 2.49, 33.79),
    ("SB-8", CLS, 465, 2.86, 34.09),
    ("Policy (imitation)", NEU, 2711, 3.38, 35.47),
    ("+ MF-blend", NEU, 1566, 1.95, 31.91),
    ("+ Cuts (attn)", NEU, 2297, 3.36, 35.53),
    ("+ Rollout (K=3)", NEU, 1675, 3.49, 35.58),
    ("+ Rollout+Size (K=1)", NEU, 2191, 3.82, 35.70),
    ("Neural-Best", NEU, 1509, 1.69, 31.73),
]
TIERS = [("Medium tier: 500×1000, 30 instances, 300 s", MEDIUM),
         ("Hard tier: 1000×2000, 15 instances, 600 s", HARD)]

INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK,
    "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
})


def legend(fig, markers=False):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    if markers:
        h = [Line2D([], [], ls="none", marker=m, ms=9, mfc=c, mec="white", label=g)
             for g, (c, m) in GROUPS.items()]
    else:
        h = [Patch(fc=c, label=g) for g, (c, _) in GROUPS.items()]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=11, labelcolor=INK)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out / name}.png/.pdf")


def primal_gap_bars(out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.subplots_adjust(wspace=0.55, top=0.84)
    for ax, (title, rows) in zip(axes, TIERS):
        names = [r[0] for r in rows][::-1]
        vals = [r[3] for r in rows][::-1]
        cols = [GROUPS[r[1]][0] for r in rows][::-1]
        y = range(len(rows))
        ax.barh(y, vals, height=0.72, color=cols, edgecolor="white", linewidth=2)
        for yi, v in zip(y, vals):
            ax.text(v + 0.08, yi, f"{v:.2f}%", va="center", fontsize=10, color=INK)
        ax.set_yticks(list(y))
        ax.set_yticklabels(names)
        for lab in ax.get_yticklabels():
            if lab.get_text() == "Neural-Best":
                lab.set_fontweight("bold")
        ax.set_xlim(0, max(vals) * 1.22)
        ax.set_xlabel("Mean primal gap at the time limit (%), lower is better")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.tick_params(axis="y", length=0)
        style(ax)
    legend(fig)
    save(fig, out, "fig_primal_gap_by_method")


# label offsets (points) per method, chosen to avoid collisions
LABELS = {
    "medium": {"SCIP default": (10, -16), "SCIP, branching only": (-14, -18),
               "Most-fractional": (-34, 30), "SB-1": (12, 16), "Policy (imitation)": (12, -12),
               "+ MF-blend": (12, -4), "+ Rollout (K=3)": (10, 8), "Neural-Best": (-14, 12)},
    "hard": {"SCIP default": (10, 8), "SB-8": (10, 14), "Most-fractional": (-12, -16),
             "Policy (imitation)": (-12, -18), "+ MF-blend": (10, 12),
             "+ Rollout (K=3)": (-12, -16), "Neural-Best": (-14, -16)},
}


def nodes_vs_gap(out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.subplots_adjust(wspace=0.28, top=0.84)
    for ax, (title, rows), key in zip(axes, TIERS, ("medium", "hard")):
        for name, g, nodes, _, prov in rows:
            c, m = GROUPS[g]
            best = name == "Neural-Best"
            ax.scatter(nodes, prov, s=150 if best else 80, marker=m, c=c,
                       edgecolors=INK if best else "white", linewidths=1.6 if best else 2,
                       zorder=3)
            if name in LABELS[key]:
                dx, dy = LABELS[key][name]
                ax.annotate(name, (nodes, prov), xytext=(dx, dy), textcoords="offset points",
                            ha="left" if dx > 0 else "right", va="center", fontsize=9.5,
                            color=INK, fontweight="bold" if best else "normal",
                            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8,
                                            shrinkA=0, shrinkB=5))
        ax.set_xlabel("SGM nodes processed within the time limit")
        ax.set_ylabel("Mean proven gap at the time limit (%)")
        ax.set_title(title, loc="left")
        ax.grid(color=GRID, lw=0.8)
        ax.margins(x=0.12, y=0.12)
        style(ax)
    legend(fig, markers=True)
    fig.text(0.5, -0.03, "Method means from the report's Tables 1-2. Every run reached the time limit. "
             "SCIP node counts come from a different solver and are not directly comparable.",
             ha="center", fontsize=10, color=INK2)
    save(fig, out, "fig_nodes_vs_proven_gap")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    primal_gap_bars(out)
    nodes_vs_gap(out)


if __name__ == "__main__":
    main()
