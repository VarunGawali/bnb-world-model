"""
merge_checkpoints.py — Graft the policy head from one checkpoint onto another.

Phase 4 joint training degraded the policy head while training the dynamics/
world-model components. This script copies policy.* weights from a donor
checkpoint (typically phase1_best.pt) into a base checkpoint (phase4_best.pt),
writing a merged checkpoint that has trained dynamics + working policy.

Usage
-----
    python tools/merge_checkpoints.py \
        --base   checkpoints/phase4_best.pt \
        --donor  checkpoints/phase1_best.pt \
        --out    checkpoints/merged_p4_policy1.pt \
        --heads  policy

    # Also graft the encoder (if phase 4 degraded it too):
    python tools/merge_checkpoints.py \
        --base   checkpoints/phase4_best.pt \
        --donor  checkpoints/phase1_best.pt \
        --out    checkpoints/merged_p4_enc1_policy1.pt \
        --heads  policy encoder
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _clean(state: dict) -> dict:
    return {k.replace("_orig_mod.", ""): v for k, v in state.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",  required=True, help="checkpoint to graft INTO")
    ap.add_argument("--donor", required=True, help="checkpoint to take heads FROM")
    ap.add_argument("--out",   required=True, help="output path")
    ap.add_argument("--heads", nargs="+", default=["policy"],
                    help="module prefixes to copy from donor (e.g. policy encoder)")
    ap.add_argument("--dry_run", action="store_true",
                    help="print which keys would be copied, then exit")
    args = ap.parse_args()

    base_ckpt  = torch.load(args.base,  map_location="cpu", weights_only=False)
    donor_ckpt = torch.load(args.donor, map_location="cpu", weights_only=False)

    base_state  = _clean(base_ckpt.get("model", base_ckpt))
    donor_state = _clean(donor_ckpt.get("model", donor_ckpt))

    prefixes = tuple(f"{h}." for h in args.heads)
    donor_keys = [k for k in donor_state if k.startswith(prefixes)]
    base_keys  = [k for k in base_state  if k.startswith(prefixes)]

    print(f"base  checkpoint: {args.base}  ({len(base_state)} keys)")
    print(f"donor checkpoint: {args.donor}  ({len(donor_state)} keys)")
    print(f"prefixes to graft: {args.heads}")
    print(f"  donor keys matching: {len(donor_keys)}")
    print(f"  base  keys matching: {len(base_keys)}")

    only_in_donor = set(donor_keys) - set(base_keys)
    only_in_base  = set(base_keys)  - set(donor_keys)
    if only_in_donor:
        print(f"  WARNING: keys only in donor (will be added): {sorted(only_in_donor)}")
    if only_in_base:
        print(f"  WARNING: keys only in base  (will be removed): {sorted(only_in_base)}")

    if args.dry_run:
        print("\nDRY RUN — keys that would be copied from donor:")
        for k in sorted(donor_keys):
            shape_b = tuple(base_state[k].shape) if k in base_state else "ABSENT"
            shape_d = tuple(donor_state[k].shape)
            match = "✓" if shape_b == shape_d else f"SHAPE MISMATCH {shape_b} vs {shape_d}"
            print(f"  {k:<60} {match}")
        return

    # Shape check
    mismatches = []
    for k in donor_keys:
        if k in base_state and base_state[k].shape != donor_state[k].shape:
            mismatches.append(k)
    if mismatches:
        raise SystemExit(f"Shape mismatches for keys: {mismatches} — aborting.")

    # Graft
    merged = dict(base_state)
    for k in donor_keys:
        merged[k] = donor_state[k]

    # Build merged checkpoint (copy metadata from base, update model state)
    out_ckpt = dict(base_ckpt)
    out_ckpt["model"] = merged
    out_ckpt["merged_from"] = {
        "base":  str(args.base),
        "donor": str(args.donor),
        "heads": args.heads,
    }
    # Keep base epoch/metrics; note the graft in a separate key
    out_ckpt["donor_epoch"]   = donor_ckpt.get("epoch", "?")
    out_ckpt["donor_metrics"] = donor_ckpt.get("metrics", {})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, out_path)
    print(f"\nMerged checkpoint written to {out_path}")
    print(f"  {len(donor_keys)} keys grafted from donor")
    print(f"  {len(merged) - len(donor_keys)} keys kept from base")


if __name__ == "__main__":
    main()
