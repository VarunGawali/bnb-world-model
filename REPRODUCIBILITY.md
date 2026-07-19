# AAAI Reproducibility Checklist — draft responses

Answers mapped to this project. Items marked **[TODO]** require an experiment or
write-up action before the paper deadline; the three under §4 (runs, variation,
significance) drive changes to how the benchmark is run — see the note at the end.

## 1. General Paper Structure
- 1.1 Conceptual outline / pseudocode of methods: **yes** (rollout + curriculum pseudocode)
- 1.2 Delineates opinion/hypothesis vs. facts: **yes**
- 1.3 Pedagogical references for background: **yes**

## 2. Theoretical Contributions
- 2.1 Makes theoretical contributions: **no** — the contribution is empirical/methodological.
  (2.2–2.8 therefore **NA**. Optimality preservation is argued informally: the model
  affects only search order and heuristic choices, never bounds/pruning correctness.)

## 3. Dataset Usage
- 3.1 Relies on datasets: **yes**
- 3.2 Motivation for chosen datasets: **yes** — Set Cover is a standard learn-to-branch
  benchmark; easy/medium/hard tiers test size generalization.
- 3.3 Novel datasets in a data appendix: **partial** — we release the trajectory-collection
  notebook/script that generates the traces (`notebooks/collect_with_cuts.ipynb`).
- 3.4 Novel datasets public upon publication: **yes**
- 3.5 Existing-literature datasets cited: **yes** — instances via Ecole `SetCoverGenerator` (cite Ecole/Gasse et al.).
- 3.6 Existing datasets publicly available: **yes**
- 3.7 Non-public datasets described: **NA**

## 4. Computational Experiments
- 4.1 Includes computational experiments: **yes**
- 4.2 Hyperparameter ranges + selection criterion: **[TODO]** — report ranges tried
  (hidden_dim, layers, lr, batch, lookahead_depth, branch_factor, gamma) and that final
  values were chosen by val metric per phase. Values live in `configs/default.yaml`.
- 4.3 Preprocessing code in appendix: **yes** (`bnb_wm/data/`)
- 4.4 Source code in a code appendix: **yes** (full repo)
- 4.5 Source code public upon publication: **yes**
- 4.6 Code comments referencing paper steps: **yes** (heads/rollout/losses documented)
- 4.7 Seed method described: **yes** — fixed seeds for the file split (`split_files`, seed=0),
  subset sampling, and per-variable subsampling; state the seed set used.
- 4.8 Computing infrastructure: **yes** —
    GPU: 2× NVIDIA Quadro RTX 5000 (16 GB); Driver 595.71.05; CUDA 13.2
    OS: Ubuntu; Python 3.10 (conda)
    Libraries: PyTorch 2.x, PyTorch Geometric 2.x, NumPy, SciPy, Ecole, PySCIPOpt
    (fill exact versions with `pip freeze` on the training machine).
- 4.9 Metrics described + motivated: **yes** — primary metric is **explored node count**
  (the solver's cost); also wall-clock. Motivation: branching quality = tree size.
- 4.10 Number of runs per reported result: **[TODO]** — run **multiple seeds** (≥3) and/or
  report over a large held-out instance set; state the count.
- 4.11 Beyond single-number summaries (variation/confidence): **[TODO]** — report
  **mean ± std** (and/or CIs) across seeds/instances, not just the mean.
- 4.12 Significance tests: **[TODO]** — the benchmark evaluates SCIP and the model on the
  **same** instances (paired), so apply a **Wilcoxon signed-rank test** on per-instance node
  counts; report p-values.
- 4.13 Final hyperparameters listed: **yes** (`configs/default.yaml`).

---

## What this changes in our experiments (the actionable part)

Items **4.10 / 4.11 / 4.12** mean single-number results are not enough. Concretely:

1. **Paired significance (cheapest, do this):** the macro benchmark already runs SCIP vs.
   the model on identical instances. Compute a **Wilcoxon signed-rank test** on the paired
   per-instance node counts and report the p-value + median/mean reduction with std.
2. **Multiple seeds:** if the 2-day budget allows, train ≥3 seeds and report mean ± std of
   the node reduction; otherwise run the single trained model over a **large** held-out set
   (100+ instances) so the per-instance distribution carries the variation, and note the
   single-training-run as a stated limitation.
3. **Log infrastructure + versions** (`pip freeze`, `nvidia-smi`) at run time for §4.8.
4. **Record hyperparameter ranges** tried for §4.2 (even a short table).
