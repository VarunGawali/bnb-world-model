"""
generate_data.py — Generate B&B trajectory datasets.

Usage:
    python scripts/generate_data.py --problem set_cover --n 1000
    python scripts/generate_data.py --problem knapsack  --n 500
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bnb_wm.data.generate import generate_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate BnB trajectory datasets")
    parser.add_argument("--problem",  type=str,  default="set_cover",
                        choices=["set_cover", "knapsack", "cauctions"])
    parser.add_argument("--n",        type=int,  default=1000,
                        help="Number of trajectories to generate")
    parser.add_argument("--out",      type=str,  default="data/",
                        help="Output directory")
    parser.add_argument("--n_rows",   type=int,  default=500)
    parser.add_argument("--n_cols",   type=int,  default=1000)
    parser.add_argument("--density",  type=float, default=0.05)
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Max B&B steps per instance")
    args = parser.parse_args()

    generate_dataset(
        problem=args.problem,
        n_instances=args.n,
        out_dir=args.out,
        max_steps=args.max_steps,
        n_rows=args.n_rows,
        n_cols=args.n_cols,
        density=args.density,
    )


if __name__ == "__main__":
    main()
