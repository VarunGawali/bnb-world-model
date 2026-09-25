"""
CutWorld paper figures  —  run with:  python make_plots.py
Outputs: plot1_nodes_vs_time.pdf/png
         plot2_neural_ablation.pdf/png
         plot3_performance_profiles.pdf/png
         plot4_rollout_vs_policy.pdf/png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.labelsize":    10.5,
    "axes.titlesize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   8.5,
    "axes.linewidth":    0.8,
    "figure.dpi":        150,
    "savefig.dpi":       400,
})

# ─────────────────────────────────────────────────────────────
# PALETTE  (colour-blind safe: Blue / Red / Green / Purple /
#           Orange / Teal / Gray family)
# ─────────────────────────────────────────────────────────────

C = {
    "mf":               "#888888",
    "sb1":              "#bbbbbb",
    "sb4":              "#555555",
    "sb8":              "#222222",
    "policy":           "#4477AA",
    "mf_blend_20":      "#EE6677",
    "policy_cuts_attn": "#CCBB44",
    "rollout_d1":       "#228833",
    "rollout_d1_size1": "#66CCEE",
    "neural_best":      "#AA3377",
    "neural_best_d3":   "#EE8844",
}

LABELS = {
    "mf":               "MF",
    "sb1":              "SB-1",
    "sb4":              "SB-4",
    "sb8":              "SB-8",
    "policy":           "Policy",
    "mf_blend_20":      "MF-Blend-20",
    "policy_cuts_attn": "Policy+Cuts",
    "rollout_d1":       "Rollout-D1",
    "rollout_d1_size1": "Rollout-D1-Size",
    "neural_best":      "Neural-Best",
    "neural_best_d3":   "Neural-Best-D3",
}

# ─────────────────────────────────────────────────────────────
# PER-INSTANCE DATA
# ─────────────────────────────────────────────────────────────

MED = {   # 500×1000, 10 instances, 300 s limit
    "mf":               {"nodes": [6084,3470,4495,3970,5588,5229,5151,4850,5192,5286],
                         "time":  [301.7,301.5,301.0,300.6,301.4,301.8,301.4,301.8,301.8,301.3],
                         "solved": False},
    "sb1":              {"nodes": [4490,4450,3770,5169,5151,4400,4266,4552,4448,5621],
                         "time":  [301.3,301.4,302.0,301.4,301.7,301.2,301.0,301.2,301.4,301.0],
                         "solved": False},
    "sb4":              {"nodes": [2191,3434,1212,3516,3598,3666,3028,3049,3323,4909],
                         "time":  [300.9,301.5,300.6,300.9,300.9,301.0,300.9,301.1,300.9,301.2],
                         "solved": False},
    "sb8":              {"nodes": [1465,2346,1036,2980,4050,3006,3299,3133,3238,4916],
                         "time":  [300.6,300.6,300.5,301.3,301.0,300.9,300.9,300.7,300.7,301.0],
                         "solved": False},
    "policy":           {"nodes": [5875,6077,5247,5349,6075,5393,5295,5869,5921,6349],
                         "time":  [238.7,242.3,231.3,233.5,244.7,239.7,234.8,237.8,235.7,238.2],
                         "solved": True},
    "mf_blend_20":      {"nodes": [4005,4239,4045,4143,4625,4111,4063,4179,4131,4145],
                         "time":  [219.6,221.9,216.5,215.7,224.8,217.1,218.3,213.0,216.4,218.1],
                         "solved": True},
    "policy_cuts_attn": {"nodes": [5019,5401,4647,4935,5719,5383,5525,5785,5695,6009],
                         "time":  [241.0,245.3,235.2,236.8,248.8,244.2,241.1,237.0,234.7,240.5],
                         "solved": True},
    "rollout_d1":       {"nodes": [3011,3343,2959,2913,3179,2901,3115,3051,2911,5041],
                         "time":  [271.3,273.3,267.3,269.1,278.2,272.8,271.6,272.1,269.0,269.5],
                         "solved": True},
    "rollout_d1_size1": {"nodes": [4101,4383,3763,3759,4221,3971,4269,4427,4419,6357],
                         "time":  [261.5,264.2,254.5,258.7,267.3,257.5,257.7,257.8,261.6,247.8],
                         "solved": True},
    "neural_best":      {"nodes": [2927,3135,2883,2855,3437,3197,3247,3055,3289,4631],
                         "time":  [242.3,247.3,239.4,240.3,251.0,244.5,243.5,240.6,240.3,235.1],
                         "solved": True},
}

HARD = {  # 1000×2000, 5 instances, 600 s limit
    "mf":               {"nodes": [2644,3490,3338,3410,3966],
                         "time":  [601.1,600.9,601.0,600.9,601.1],
                         "solved": False},
    "sb1":              {"nodes": [1994,2825,2614,2678,3194],
                         "time":  [600.9,600.9,600.8,601.0,601.0],
                         "solved": False},
    "sb4":              {"nodes": [576,1222,1118,1078,1676],
                         "time":  [600.2,600.4,600.5,600.4,600.5],
                         "solved": False},
    "sb8":              {"nodes": [791,1186,967,1098,1586],
                         "time":  [600.2,600.4,600.4,600.9,600.5],
                         "solved": False},
    "policy":           {"nodes": [4721,4351,4423,4459,5269],
                         "time":  [390.8,381.5,385.4,388.0,401.4],
                         "solved": True},
    "mf_blend_20":      {"nodes": [2615,2793,2759,2537,2999],
                         "time":  [352.4,353.7,351.1,348.6,360.0],
                         "solved": True},
    "policy_cuts_attn": {"nodes": [4405,4113,4329,4429,4849],
                         "time":  [395.7,388.7,392.3,393.5,406.1],
                         "solved": True},
    "rollout_d1":       {"nodes": [3917,3595,3699,3945,5027],
                         "time":  [423.9,418.4,418.5,422.8,430.6],
                         "solved": True},
    "rollout_d1_size1": {"nodes": [4261,3863,4251,4123,5559],
                         "time":  [398.5,389.9,393.1,397.1,403.1],
                         "solved": True},
    "neural_best":      {"nodes": [2335,2589,2339,2705,3123],
                         "time":  [364.0,366.0,362.8,363.2,368.3],
                         "solved": True},
    "neural_best_d3":   {"nodes": [2283,2369,2203,2625,2975],
                         "time":  [387.6,391.7,388.8,386.8,397.4],
                         "solved": True},
}

# ─────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────

def sgm(vals, shift=10):
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    return np.exp(np.mean(np.log(v + shift))) - shift


# ═════════════════════════════════════════════════════════════
# PLOT 1 — nodes vs. wall-clock time with survival-style
#           censoring markers
# ═════════════════════════════════════════════════════════════

CLASSICAL = ["mf", "sb1", "sb4", "sb8"]
NEURAL_ORDER = [
    "policy", "mf_blend_20", "policy_cuts_attn",
    "rollout_d1", "rollout_d1_size1", "neural_best", "neural_best_d3",
]
NEURAL_MARKER = {
    "policy":           "o",
    "mf_blend_20":      "s",
    "policy_cuts_attn": "^",
    "rollout_d1":       "D",
    "rollout_d1_size1": "P",
    "neural_best":      "*",
    "neural_best_d3":   "X",
}

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))

for ax, data, tier_name, wall, xmax in [
    (axes[0], MED,  "Medium  (500×1000, 300 s)",  300, 310),
    (axes[1], HARD, "Hard  (1000×2000, 600 s)",   600, 620),
]:
    # --- classical: censored at the time-limit wall ---
    for m in CLASSICAL:
        nodes = np.asarray(data[m]["nodes"])
        x = np.full(len(nodes), wall)
        ax.scatter(x, nodes, s=45,
                   facecolors="white", edgecolors=C[m],
                   linewidths=1.4, marker="o", zorder=4)
        ax.scatter(x, nodes, s=25, color=C[m], marker=">", zorder=5)

    # --- neural: solved, plotted at actual time ---
    for m in NEURAL_ORDER:
        if m not in data:
            continue
        times = np.asarray(data[m]["time"])
        nodes = np.asarray(data[m]["nodes"])
        ax.scatter(times, nodes,
                   s=70 if m == "neural_best" else 50,
                   marker=NEURAL_MARKER[m],
                   color=C[m],
                   edgecolors="black", linewidths=0.45,
                   alpha=0.9, zorder=3)

    # --- time-limit wall ---
    ax.axvline(wall, color="black", linestyle="--", linewidth=1.1, alpha=0.75)
    yhi = ax.get_ylim()[1]
    ax.text(wall - 2, yhi * 0.97, f"{wall} s limit",
            rotation=90, ha="right", va="top", fontsize=8.5)

    ax.set_xlim(0, xmax)
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("B&B nodes at termination")
    ax.set_title(tier_name, loc="left", fontweight="bold")
    ax.grid(True, alpha=0.18, linewidth=0.7)

# legend
legend_handles = (
    [Line2D([0],[0], marker="o", ls="", markerfacecolor="white",
             markeredgecolor=C[m], markersize=7, label=LABELS[m])
     for m in CLASSICAL]
    + [Line2D([0],[0], marker=NEURAL_MARKER[m], ls="", color=C[m],
               markersize=7, label=LABELS[m])
       for m in NEURAL_ORDER if m in HARD]
    + [Line2D([0],[0], marker="o", ls="", markerfacecolor="white",
               markeredgecolor="black", markersize=7, label="Censored at limit"),
       Line2D([0],[0], marker="o", ls="", color="black",
               markersize=7, label="Solved")]
)
axes[1].legend(handles=legend_handles, loc="upper left",
               frameon=False, fontsize=7.5, ncol=2)

fig.text(0.5, 0.01,
         "Classical runs are shown at the censoring wall (○→); "
         "neural runs are shown at their observed solve time.",
         ha="center", fontsize=8.5)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("plot1_nodes_vs_time.pdf", bbox_inches="tight")
plt.savefig("plot1_nodes_vs_time.png", dpi=400, bbox_inches="tight")
print("✓ plot1_nodes_vs_time")
plt.close()


# ═════════════════════════════════════════════════════════════
# PLOT 2 — neural ablation horizontal bar
#          one color family; tier = lightness
# ═════════════════════════════════════════════════════════════

NEURAL_BAR = [
    "policy", "mf_blend_20", "policy_cuts_attn",
    "rollout_d1", "rollout_d1_size1", "neural_best", "neural_best_d3",
]

sgm_med  = {m: sgm(MED[m]["nodes"])  for m in NEURAL_BAR if m in MED}
sgm_hard = {m: sgm(HARD[m]["nodes"]) for m in NEURAL_BAR if m in HARD}

y  = np.arange(len(NEURAL_BAR))
h  = 0.34

policy_med  = sgm_med["policy"]
policy_hard = sgm_hard["policy"]

fig, ax = plt.subplots(figsize=(8.5, 4.7))

medium_vals = [sgm_med.get(m, np.nan) for m in NEURAL_BAR]
hard_vals   = [sgm_hard.get(m, np.nan) for m in NEURAL_BAR]

ax.barh(y + h/2, medium_vals, height=h,
        color="#9AA9B8", edgecolor="black", linewidth=0.5,
        alpha=0.65, label="Medium")
ax.barh(y - h/2, hard_vals,   height=h,
        color="#4477AA", edgecolor="black", linewidth=0.5,
        alpha=0.90, label="Hard")

# value labels + % relative to Policy
for i, m in enumerate(NEURAL_BAR):
    if not np.isnan(medium_vals[i]):
        delta = 100 * (medium_vals[i] - policy_med) / policy_med
        ax.text(medium_vals[i] + 60, y[i] + h/2,
                f"{medium_vals[i]:.0f}  ({delta:+.0f}%)",
                va="center", fontsize=7.5)
    if not np.isnan(hard_vals[i]):
        delta = 100 * (hard_vals[i] - policy_hard) / policy_hard
        ax.text(hard_vals[i] + 60, y[i] - h/2,
                f"{hard_vals[i]:.0f}  ({delta:+.0f}%)",
                va="center", fontsize=7.5)

# Policy reference line
ax.axvline(policy_hard, linestyle="--", color="black",
           linewidth=1.0, alpha=0.55, label="Policy (hard) baseline")

ax.set_yticks(y)
ax.set_yticklabels([LABELS[m] for m in NEURAL_BAR])
ax.invert_yaxis()
ax.set_xlabel("Shifted geometric mean of B&B nodes")
ax.set_title("Neural ablation: search effort at termination",
             loc="left", fontweight="bold")
ax.legend(frameon=False, ncol=3, loc="lower right")
ax.grid(axis="x", alpha=0.18)

plt.tight_layout()
plt.savefig("plot2_neural_ablation.pdf", bbox_inches="tight")
plt.savefig("plot2_neural_ablation.png", dpi=400, bbox_inches="tight")
print("✓ plot2_neural_ablation")
plt.close()


# ═════════════════════════════════════════════════════════════
# PLOT 3 — performance profiles, Medium / Hard side-by-side
# ═════════════════════════════════════════════════════════════

PROFILE_METHODS = [
    "mf", "sb1", "sb4", "sb8",
    "policy", "mf_blend_20", "policy_cuts_attn",
    "rollout_d1", "rollout_d1_size1", "neural_best",
]

def effective_times(data, method):
    if not data[method]["solved"]:
        return np.full(len(data[method]["nodes"]), np.inf)
    return np.asarray(data[method]["time"], dtype=float)

def performance_profile(data, methods, tau_max=1.4, n_tau=600):
    T    = np.vstack([effective_times(data, m) for m in methods]).T
    best = np.min(np.where(np.isfinite(T), T, np.inf), axis=1)
    tau  = np.linspace(1.0, tau_max, n_tau)
    return tau, {
        m: np.array([np.mean(T[:, j] <= t * best + 1e-9) for t in tau])
        for j, m in enumerate(methods)
    }

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)

for ax, data, title in [
    (axes[0], MED,  "Medium: 500×1000"),
    (axes[1], HARD, "Hard: 1000×2000"),
]:
    tau, profiles = performance_profile(data, PROFILE_METHODS)
    for m in PROFILE_METHODS:
        ls = "--" if not data[m]["solved"] else "-"
        ax.plot(tau, profiles[m], color=C[m], lw=2.0,
                linestyle=ls, label=LABELS[m])

    # annotate flat classical lines
    ax.text(1.39, 0.03,
            "Classical: all instances\nunsolved (τ = ∞)",
            ha="right", va="bottom", fontsize=7.5,
            color="#555555", style="italic")

    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(r"$\tau$  (factor over per-instance best time)")
    ax.set_xlim(1.0, 1.4)
    ax.set_xticks([1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40])
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.18)

axes[0].set_ylabel("Fraction of instances solved within τ × best")
axes[1].legend(frameon=False, fontsize=7.5, loc="lower right")
fig.suptitle("Performance profiles by problem tier", fontsize=12)

fig.text(0.5, 0.01,
         "Unsolved runs are assigned infinite effective time. "
         "Profiles are computed independently for each tier.",
         ha="center", fontsize=8.0)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("plot3_performance_profiles.pdf", bbox_inches="tight")
plt.savefig("plot3_performance_profiles.png", dpi=400, bbox_inches="tight")
print("✓ plot3_performance_profiles")
plt.close()


# ═════════════════════════════════════════════════════════════
# PLOT 4 — rollout vs. policy paired scatter
# ═════════════════════════════════════════════════════════════

pol_med  = np.asarray(MED["policy"]["nodes"])
rol_med  = np.asarray(MED["rollout_d1"]["nodes"])
pol_hard = np.asarray(HARD["policy"]["nodes"])
rol_hard = np.asarray(HARD["rollout_d1"]["nodes"])

pol_all = np.concatenate([pol_med, pol_hard])
rol_all = np.concatenate([rol_med, rol_hard])

reduction      = 100 * (1 - rol_all / pol_all)
med_reduction  = 100 * (1 - rol_med  / pol_med)
hard_reduction = 100 * (1 - rol_hard / pol_hard)
mean_r         = np.mean(reduction)
mean_med       = np.mean(med_reduction)
mean_hard      = np.mean(hard_reduction)
n_improved     = int(np.sum(rol_all < pol_all))
n_total        = len(pol_all)

print(f"\nPlot 4 — node reduction summary")
print(f"  Overall mean:  {mean_r:.1f}%")
print(f"  Medium mean:   {mean_med:.1f}%")
print(f"  Hard mean:     {mean_hard:.1f}%")
print(f"  Min:           {reduction.min():.1f}%")
print(f"  Max:           {reduction.max():.1f}%")
print(f"  Improved:      {n_improved}/{n_total}")
print(f"  Per-instance (medium): {np.round(med_reduction,1)}")
print(f"  Per-instance (hard):   {np.round(hard_reduction,1)}")

lo = 2500
hi = 6900

fig, ax = plt.subplots(figsize=(6.1, 5.8))

ax.plot([lo, hi], [lo, hi], color="black", linestyle="--",
        linewidth=1.1, alpha=0.7, label="y = x  (parity)")

ax.scatter(pol_med, rol_med,
           s=58, marker="o", color="#4477AA",
           edgecolors="black", linewidths=0.5,
           alpha=0.85, label="Medium", zorder=3)
ax.scatter(pol_hard, rol_hard,
           s=68, marker="^", color="#EE6677",
           edgecolors="black", linewidths=0.5,
           alpha=0.85, label="Hard", zorder=3)

# instance labels
for i, (x, y_) in enumerate(zip(pol_med, rol_med)):
    ax.annotate(f"M{i}", (x, y_), xytext=(5, 4),
                textcoords="offset points", fontsize=7.5)
for i, (x, y_) in enumerate(zip(pol_hard, rol_hard)):
    ax.annotate(f"H{i}", (x, y_), xytext=(5, 4),
                textcoords="offset points", fontsize=7.5)

# summary box — tier-split reductions
summary = (
    f"Medium mean reduction: {mean_med:.1f}%\n"
    f"Hard mean reduction:   {mean_hard:.1f}%\n"
    f"Overall mean:          {mean_r:.1f}%\n"
    f"Instances reduced:     {n_improved}/{n_total}"
)
ax.text(0.04, 0.96, summary, transform=ax.transAxes,
        va="top", ha="left", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="0.7", alpha=0.95))

ax.set_xlabel("Policy B&B nodes")
ax.set_ylabel("Rollout-D1 B&B nodes")
ax.set_title("Per-instance B&B node count: rollout vs. policy",
             loc="left", fontweight="bold")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.18)
ax.legend(frameon=False, loc="lower right")

plt.tight_layout()
plt.savefig("plot4_rollout_vs_policy.pdf", bbox_inches="tight")
plt.savefig("plot4_rollout_vs_policy.png", dpi=400, bbox_inches="tight")
print("✓ plot4_rollout_vs_policy")
plt.close()

print("\nAll figures saved.")
