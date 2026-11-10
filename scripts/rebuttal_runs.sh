#!/usr/bin/env bash
# Rebuttal experiments: 30 medium + 15 hard instances on the untouched
# generator seeds 0, 2, 3, plus the old seed-1 instances for comparison.
# Every step is resumable: re-run the script and finished jobs are skipped.
# Worst-case wall time on 4 workers is about 16 h; check with --dry_run.
#
#   CKPT=checkpoints/highs_retrain/phase4_final.pt WORKERS=4 bash scripts/rebuttal_runs.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=${CKPT:?set CKPT to the checkpoint behind the submitted tables}
WORKERS=${WORKERS:-4}
OUT=${OUT:-results/rebuttal}
DEVICE=${DEVICE:-cuda}
mkdir -p "$OUT"
export PYTHONPATH=.

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Commit or stash local changes first, so results map to one commit." >&2
  exit 1
fi

run() {  # tier, seeds, n, out-name, extra args...
  local tier=$1 seeds=$2 n=$3 name=$4; shift 4
  python tools/run_benchmark.py --tier "$tier" --seeds $seeds --n_instances "$n" \
      --workers "$WORKERS" --device "$DEVICE" --checkpoint "$CKPT" \
      --out "$OUT/$name.jsonl" "$@"
}

# 1. Cause of the old gaps (~15 min): the paper's seed-1 medium instances with
#    the fixed solver. Before the fix these "completed" with positive gaps.
run medium 1 10 medium_seed1_check --rows_include paper_policy paper_neural_best scip_default

# 2. Reference solutions (SCIP default at the tier budget), needed by step 3's
#    cut-validity check.
run hard   "0 2 3" 5  hard_ref   --group reference
run medium "0 2 3" 10 medium_ref --group reference

# 3. Main tables. Hard first: longest jobs and the weakest tier in the paper.
run hard   "0 2 3" 5  hard   --group tables --reference "$OUT/hard_ref.jsonl"
run medium "0 2 3" 10 medium --group tables --reference "$OUT/medium_ref.jsonl"

# 4. Ablations on 10 medium instances (seed 0): rollout depth and cut selection.
run medium 0 10 medium_ablations --rows_include rollout_k1 rollout_k2 rollout_k3 \
    cuts_random cuts_violation cuts_attention --reference "$OUT/medium_ref.jsonl"

# 5. Analysis.
for t in medium hard; do
  python tools/analyze_benchmark.py "$OUT/${t}_ref.jsonl" "$OUT/$t.jsonl" \
      --baseline paper_policy --md "$OUT/$t.md" --csv "$OUT/${t}_instances.csv"
done
python tools/analyze_benchmark.py "$OUT/medium_ref.jsonl" "$OUT/medium.jsonl" "$OUT/medium_ablations.jsonl" \
    --rows paper_policy rollout_k1 rollout_k2 rollout_k3 cuts_random cuts_violation cuts_attention \
    --baseline paper_policy --md "$OUT/medium_ablations.md"
python tools/analyze_benchmark.py "$OUT/medium_seed1_check.jsonl" --baseline paper_policy \
    --md "$OUT/medium_seed1_check.md"
echo "Done. Tables in $OUT/*.md, per-instance data in $OUT/*_instances.csv"
