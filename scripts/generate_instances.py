"""generate_instances.py — generate fresh Set Cover instances to disk.

Writes NEW, reproducible Set Cover instances as .lp files in the tier/split
layout the collector (collect_with_cuts_v2.py) expects:

    <out-root>/SC-easy/{train,val,test}/sc_<i>.lp
    <out-root>/SC-medium/{train,val,test}/sc_<i>.lp
    <out-root>/SC-hard/{train,val,test}/sc_<i>.lp

Variety: instances are drawn from ecole.instance.SetCoverGenerator seeded from
--base-seed. Use a base seed DIFFERENT from any previous collection so the new
instances do not duplicate the old dataset. With --jitter, each instance's size
and density are perturbed within a per-tier band, widening the distribution the
model sees (better generalization) instead of a single fixed shape per tier.

Reproducibility: the (tier, split, index) triple maps to a deterministic seed,
so re-running reproduces byte-identical instances; train/val/test never share
seeds, so splits are disjoint.

Example:
    python scripts/generate_instances.py --out-root data_instances \
        --n-train 800 --n-val 100 --n-test 100 --base-seed 20260731 --jitter

NOTE: requires ecole; could not be run in the authoring environment. Smoke-test:
    python scripts/generate_instances.py --out-root /tmp/sc --n-train 2 --n-val 1 --n-test 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import ecole


# Per-tier nominal (n_rows, n_cols, density). Match the paper's tiers.
TIERS = {
    "SC-easy":   dict(n_rows=200,  n_cols=400,  density=0.05),
    "SC-medium": dict(n_rows=500,  n_cols=1000, density=0.05),
    "SC-hard":   dict(n_rows=1000, n_cols=2000, density=0.05),
}
SPLITS = ("train", "val", "test")
# Distinct offset per split so seeds (hence instances) never overlap across splits.
_SPLIT_OFFSET = {"train": 0, "val": 10_000_000, "test": 20_000_000}
_TIER_OFFSET = {"SC-easy": 0, "SC-medium": 100_000_000, "SC-hard": 200_000_000}


def _seed_for(tier: str, split: str, i: int, base: int) -> int:
    """Deterministic, collision-free seed for one instance."""
    return (base + _TIER_OFFSET[tier] + _SPLIT_OFFSET[split] + i) % (2**31 - 1)


def _jittered_params(tier: str, rng: np.random.Generator) -> dict:
    """Perturb size/density within a per-tier band for distribution variety."""
    base = TIERS[tier]
    rows = int(round(base["n_rows"] * rng.uniform(0.85, 1.15)))
    cols = int(round(base["n_cols"] * rng.uniform(0.85, 1.15)))
    dens = float(np.clip(base["density"] * rng.uniform(0.8, 1.2), 0.02, 0.2))
    return dict(n_rows=rows, n_cols=cols, density=dens)


def _generate_one(tier: str, split: str, i: int, args) -> "ecole.scip.Model":
    seed = _seed_for(tier, split, i, args.base_seed)
    if args.jitter:
        params = _jittered_params(tier, np.random.default_rng(seed))
    else:
        params = dict(TIERS[tier])
    gen = ecole.instance.SetCoverGenerator(**params)
    gen.seed(seed)
    return next(gen)


def _write_split(tier: str, split: str, n: int, args) -> int:
    out_dir = args.out_root / tier / split
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for i in range(n):
        path = out_dir / f"sc_{i:05d}.lp"
        if path.exists() and not args.overwrite:
            continue
        model = _generate_one(tier, split, i, args)
        # Ecole model -> .lp on disk (pyscipopt fallback across versions).
        try:
            model.write_problem(str(path))
        except AttributeError:
            model.as_pyscipopt().writeProblem(str(path))
        written += 1
        if i == 0 or (i + 1) % 100 == 0 or i == n - 1:
            print(f"  [{tier}/{split}] {i + 1}/{n} -> {path.name}")
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", type=Path, required=True,
                   help="Root dir; tier/split subfolders are created under it.")
    p.add_argument("--tiers", nargs="+", default=list(TIERS),
                   choices=list(TIERS))
    p.add_argument("--n-train", type=int, default=800)
    p.add_argument("--n-val", type=int, default=100)
    p.add_argument("--n-test", type=int, default=100)
    p.add_argument("--base-seed", type=int, default=20260731,
                   help="Use a NEW value to get instances distinct from prior runs.")
    p.add_argument("--jitter", action="store_true",
                   help="Perturb size/density per instance for more variety.")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    args.out_root = args.out_root.resolve()

    counts = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    total = 0
    for tier in args.tiers:
        for split in SPLITS:
            n = counts[split]
            if n <= 0:
                continue
            print(f"\nGenerating {tier}/{split} ({n} instances, "
                  f"{'jittered' if args.jitter else 'fixed'} size)")
            total += _write_split(tier, split, n, args)
    print(f"\nDone. Wrote {total} new .lp instances under {args.out_root}")
    print("Point the collector at it:  --instances-root", args.out_root)


if __name__ == "__main__":
    main()
