# Runbook — Training & Evaluating the BnB World Model

Practical commands for running the pipeline. The code has been validated
end-to-end on synthetic data (`smoke_test.py`, 8/8 stages).

---

## 1. Environment

```bash
pip install torch torch_geometric numpy scipy pyyaml tqdm
# Optional: highspy (LP warmstarting; solver falls back to scipy without it)
# For the SCIP benchmark only: ecole, pyscipopt
```

**Windows/conda OpenMP note.** If you see `OMP: Error #15`, set:
```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"     # PowerShell, per session
```
The clean fix (recommended on the training machine, avoids any silent
numerical issues) is to install torch + numpy from one channel so a single
OpenMP runtime is linked:
```bash
conda install pytorch numpy scipy -c pytorch -c conda-forge
```

---

## 2. Validate the code (no dataset needed)

```bash
python smoke_test.py
```
Runs a tiny model through all 5 phases + both metrics + the solver on synthetic
data. Expect `8/8 stages passed`. The numbers are meaningless (random data) —
this only proves the code paths execute.

---

## 3. Main system — real data

### 3a. Schema smoke FIRST (catches data-layout mismatches)

```bash
python train.py --config configs/default.yaml \
                --data_root /path/to/data_with_cuts \
                --max_files 8 --phases 1,2,3,4
```
If this runs, the real `.npz` layout matches `build_pyg_data`. If it throws a
shape error in `build_pyg_data`, the collector stored `edge_indices` as
`(variable, constraint)` instead of `(constraint, variable)` — swap the two
rows in `bnb_wm/data/datasets.py::build_pyg_data` (the `ei[0]`/`ei[1]` use).

### 3b. Full training

```bash
python train.py --config configs/default.yaml \
                --data_root /path/to/data_with_cuts \
                --with_cuts --phases 1,2,3,4,5
```
Checkpoints are written to `checkpoints/` (`phaseN_best.pt`, `model_final.pt`).
Tune epochs / batch size / lr in `configs/default.yaml`.

### 3c. Benchmark vs SCIP (needs ecole + pyscipopt)

```python
from bnb_wm.evaluate.benchmark import run_macro_benchmark
# load model, then:
run_macro_benchmark(model, device, problem="set_cover", n_instances=100)
```

---

## 4. Phase reference

| Phase | What trains | Freezes |
|-------|-------------|---------|
| 1 policy    | everything (imitation)         | — |
| 2 value     | value head                     | rest |
| 3 dynamics  | dynamics + dyn_bound + dyn_reward | rest |
| 4 joint     | everything (policy+value+integrality+cost-to-go+value-consistency) | — |
| 5 cuts      | cutting-plane head             | rest |

Run a subset with `--phases 1,2` etc. Phase 3 pre-encodes trajectories with the
frozen encoder, so run it after 1–2.

---

## 5. Solver / rollout knobs (configs/default.yaml → solver:)

| Knob | Default | Meaning |
|------|---------|---------|
| `lookahead_depth` | 3 | latent rollout steps per candidate |
| `branch_factor` | 2 | rollout tree width (1 = single path) |
| `ctg_weight` | 1.0 | cost-to-go penalty in rollout score (0 = value only) |
| `use_reward_return` | true | MuZero return `Σγ^t r_t + γ^k V(leaf)` (false = value-sum) |
| `node_selection` | cost_to_go | learned best-first (`bound` = LP best-bound) |
| `size_weight` | 0.0 | subtree-size penalty (off: needs DFS-ordered data) |
| `use_global_context` | false | global scalars into z (off: needs frontier data) |

**Ablation matrix for the paper:** toggle `ctg_weight` (0 vs 1),
`branch_factor` (1 vs 2), `use_reward_return` (false vs true),
`node_selection` (bound vs cost_to_go). Report node count vs SCIP pseudocost.

---

## 6. Known watch-items on real data

- **Edge orientation** — see 3a. The one thing synthetic data can't verify.
- **fp16 + infinities** — `build_pyg_data` clips `+inf` to `1e6`, which is out
  of fp16 range. If real features contain `inf` and you get `nan`/`inf` losses
  under AMP, clip to `6e4` instead (or set `training.amp: false`).
- **Phase 3 memory** — per-variable sequences scale with `batch_size` ×
  `max_vars_recon` × `T`. If Phase 3 OOMs, lower the batch size or
  `max_vars_recon` (in the SequenceDataset construction).
