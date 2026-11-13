#!/usr/bin/env python
"""
plot_rollout_accuracy.py — Diagnose rollout prediction quality.

Reads rollout_acc data from the ablation JSON and produces:
  1. Scatter plot: predicted_score vs actual_delta_lb (per tier)
  2. Histogram of |actual_delta_lb| — shows whether sign flips are noise
     (near-zero ΔLB) or real prediction failures (large ΔLB with wrong sign)
  3. Prints summary statistics

Usage:
    python tools/plot_rollout_accuracy.py results/final_all_tiers.json
    python tools/plot_rollout_accuracy.py results/final_medium.json results/final_hard.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[warn] matplotlib not available — printing stats only")


def load_rollout_pairs(path):
    """Returns list of {predicted_score, actual_delta_lb, tier} dicts."""
    with open(path) as f:
        data = json.load(f)

    pairs = []
    # Handle both per-tier top-level keys and flat structure
    if "tiers" in data:
        for tier_name, tier_data in data["tiers"].items():
            ra = tier_data.get("rollout_acc", {})
            for method, eps in ra.items():
                for ep in eps:
                    for p in ep:
                        pairs.append({
                            "predicted": p["predicted_score"],
                            "actual":    p["actual_delta_lb"],
                            "tier":      tier_name,
                            "method":    method,
                        })
    else:
        # Flat: rollout_acc key at top level
        ra = data.get("rollout_acc", {})
        tier_name = data.get("tier", Path(path).stem)
        for method, eps in ra.items():
            for ep in eps:
                for p in ep:
                    pairs.append({
                        "predicted": p["predicted_score"],
                        "actual":    p["actual_delta_lb"],
                        "tier":      tier_name,
                        "method":    method,
                    })
    return pairs


def stats(pairs, label=""):
    pred   = np.array([p["predicted"] for p in pairs])
    actual = np.array([p["actual"]    for p in pairs])

    # Spearman
    from scipy.stats import spearmanr
    rho, pval = spearmanr(pred, actual)

    sign_agree = np.mean((pred > 0) == (actual > 0))

    # Split by |actual_delta_lb| magnitude
    thresh = np.percentile(np.abs(actual), 50)  # median magnitude
    big_mask = np.abs(actual) > thresh
    sign_big = np.mean((pred[big_mask] > 0) == (actual[big_mask] > 0)) if big_mask.sum() > 0 else float("nan")
    sign_small = np.mean((pred[~big_mask] > 0) == (actual[~big_mask] > 0)) if (~big_mask).sum() > 0 else float("nan")

    print(f"\n{'='*60}")
    print(f"  {label}  (n={len(pairs):,})")
    print(f"{'='*60}")
    print(f"  Spearman ρ        : {rho:+.4f}  (p={pval:.2e})")
    print(f"  Sign agreement    : {sign_agree:.3f}  ({sign_agree*100:.1f}%)")
    print(f"  |ΔLB| > median    : sign_agree = {sign_big:.3f}  ({sign_big*100:.1f}%)  [n={big_mask.sum()}]")
    print(f"  |ΔLB| <= median   : sign_agree = {sign_small:.3f}  ({sign_small*100:.1f}%)  [n={(~big_mask).sum()}]")
    print(f"  median |ΔLB|      : {thresh:.5f}")
    print(f"  % pairs |ΔLB|<1e-4: {100*np.mean(np.abs(actual)<1e-4):.1f}%  (near-zero = sign is noise)")
    print(f"  pred  range       : [{pred.min():.4f}, {pred.max():.4f}]")
    print(f"  actual range      : [{actual.min():.4f}, {actual.max():.4f}]")

    # Key diagnostic: if |actual| is tiny everywhere, sign_agree is meaningless
    near_zero_pct = 100 * np.mean(np.abs(actual) < 1e-4)
    if near_zero_pct > 60:
        print(f"\n  >> DIAGNOSIS: {near_zero_pct:.0f}% of actual ΔLB are near-zero (<1e-4).")
        print(f"     Sign flips are NOISE — both branches give nearly identical LP bounds.")
        print(f"     Low sign_agree does NOT indicate model failure; it indicates degenerate instances.")
    elif sign_big < 0.45:
        print(f"\n  >> DIAGNOSIS: sign_agree on large-ΔLB pairs = {sign_big*100:.0f}% (below 50%).")
        print(f"     Model is anti-correlated with the true better branch.")
        print(f"     This IS a real prediction failure — dynamics model not learning transitions.")
    elif sign_big > 0.55:
        print(f"\n  >> DIAGNOSIS: sign_agree on large-ΔLB pairs = {sign_big*100:.0f}% (above 50%).")
        print(f"     Model is directionally correct on meaningful decisions.")
        print(f"     Low overall sign_agree is driven by near-zero ΔLB noise — acceptable.")

    return pred, actual, thresh


def plot(all_pairs, out_path):
    tiers = sorted(set(p["tier"] for p in all_pairs))
    n_cols = len(tiers)
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 10))
    if n_cols == 1:
        axes = [[axes[0]], [axes[1]]]

    for col, tier in enumerate(tiers):
        ps = [p for p in all_pairs if p["tier"] == tier]
        pred   = np.array([p["predicted"] for p in ps])
        actual = np.array([p["actual"]    for p in ps])

        # ---- top row: scatter ----
        ax = axes[0][col]
        near_zero = np.abs(actual) < 1e-4
        ax.scatter(actual[~near_zero], pred[~near_zero], alpha=0.3, s=8,
                   color="#2563eb", label=f"large |ΔLB| (n={np.sum(~near_zero)})")
        ax.scatter(actual[near_zero],  pred[near_zero],  alpha=0.15, s=4,
                   color="#94a3b8", label=f"near-zero |ΔLB| (n={np.sum(near_zero)})")
        # perfect-correlation line
        lo = min(actual.min(), pred.min())
        hi = max(actual.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="y=x")
        ax.axhline(0, color="gray", lw=0.5, ls=":")
        ax.axvline(0, color="gray", lw=0.5, ls=":")
        from scipy.stats import spearmanr
        rho, _ = spearmanr(pred, actual)
        sign_ag = np.mean((pred > 0) == (actual > 0))
        ax.set_title(f"{tier}  |  ρ={rho:+.3f}  sign={sign_ag:.2f}", fontsize=11)
        ax.set_xlabel("actual ΔLB (LP bound improvement)")
        ax.set_ylabel("predicted rollout score")
        ax.legend(fontsize=7, markerscale=2)

        # ---- bottom row: |ΔLB| histogram with sign-correct coloring ----
        ax2 = axes[1][col]
        correct_mask = (pred > 0) == (actual > 0)
        ax2.hist(np.abs(actual[correct_mask]),  bins=60, alpha=0.6, color="#16a34a",
                 label=f"sign correct ({correct_mask.sum()})", density=True)
        ax2.hist(np.abs(actual[~correct_mask]), bins=60, alpha=0.6, color="#dc2626",
                 label=f"sign wrong ({(~correct_mask).sum()})", density=True)
        ax2.axvline(1e-4, color="k", lw=1, ls="--", label="|ΔLB|=1e-4 (near-zero)")
        ax2.set_xlabel("|actual ΔLB|")
        ax2.set_ylabel("density")
        ax2.set_title(f"{tier} — sign-correct vs wrong by |ΔLB| magnitude")
        ax2.legend(fontsize=7)
        ax2.set_xscale("symlog", linthresh=1e-5)

    fig.suptitle("Rollout Prediction Accuracy Diagnosis", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+", help="one or more ablation JSON files")
    ap.add_argument("--out", default="results/rollout_accuracy_diagnosis.png")
    args = ap.parse_args()

    all_pairs = []
    for path in args.jsons:
        pairs = load_rollout_pairs(path)
        if not pairs:
            print(f"[warn] No rollout_acc data found in {path}")
            continue
        all_pairs.extend(pairs)
        tier_methods = {}
        for p in pairs:
            tier_methods.setdefault(p["tier"], set()).add(p["method"])
        for tier, methods in tier_methods.items():
            subset = [p for p in pairs if p["tier"] == tier]
            stats(subset, label=f"{Path(path).name} / {tier}")

    if not all_pairs:
        print("No rollout data found in any JSON.")
        sys.exit(1)

    if HAS_MPL:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        plot(all_pairs, args.out)
    else:
        print("\n[info] Install matplotlib to generate the plot: pip install matplotlib")


if __name__ == "__main__":
    main()
