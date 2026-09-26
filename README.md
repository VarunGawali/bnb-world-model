# CutWorld: A Latent World Model for Branch-and-Cut

A learned planning system for Mixed Integer Programs (MIPs) that jointly selects
branching variables and Gomory cutting planes inside a Branch-and-Cut solver.

CutWorld encodes each B&C node as a bipartite graph (variables × constraints),
maintains a latent state with a causal Dynamics Transformer, and uses a
3-step latent rollout to score branching decisions at runtime — without calling
the full solver forward.

On fixed Set Cover benchmarks, neural configurations exhaust their search tree
on **100% of medium instances** (500×1000, 300 s) and **100% of hard instances**
(1000×2000, 600 s) while every classical rule times out. A 3-step latent rollout
reduces the medium-tier shifted-geometric-mean node count by **44%** relative to
imitation (SGM 3,197 vs. 5,733); combining rollout and root cuts at hard scale
reaches SGM 2,603 vs. 4,633 for imitation.

## Architecture

```
B&C Node (bipartite graph: variables × constraints)
        │
        ▼
  GATv2 Encoder
  (bidirectional, 3 layers, hidden=128)
  CrossAttentionPool → latent z ∈ R^d
        │
   ┌────┴──────────────────────┐
   ▼                           ▼
PolicyHead                 CausalDynamicsTransformer
(branching logits)         (4-layer, predicts next z)
                               │
                           ValueHead / IntegralityHead
```

**Cut selection** (at root node): enumerate violated Gomory fractional cuts,
score by attention over the latent, commit the top-k, re-solve and re-encode.

**Branch selection** (at each node): run a 3-step latent rollout under the
dynamics model (both branch directions, depth K=3), combine rollout value with
policy logits via z-score standardisation, pick argmax.

## Repository Structure

```
bnb-world-model/
├── bnb_wm/
│   ├── model/        # GATv2 encoder, heads, DynamicsTransformer, BnBWorldModel
│   ├── data/         # TransitionDataset, SequenceDataset, collate, generate
│   ├── training/     # losses, trainer (4 phases), checkpoint utils
│   ├── evaluate/     # SGM metrics, rank CDF, macro benchmark
│   └── solver/       # standalone Branch-and-Cut solver (neural_bnb.py)
├── scripts/          # generate_instances.py, collect_with_cuts_v2.py, evaluate.py
├── tools/            # ablation runners, correctness checkers
├── configs/          # default.yaml (all hyperparameters)
├── notebooks/        # exploration and result plots
├── paper/            # LaTeX source for the paper
└── tests/            # pytest smoke tests
```

## Setup

```bash
# 1. Clone
git clone https://github.com/varungawali/bnb-world-model.git
cd bnb-world-model

# 2. Install core dependencies
pip install -e .

# 3. Install solver dependencies
conda install -c conda-forge pyscipopt
pip install highspy torch-geometric
```

## Quickstart

### 1. Generate instances
```bash
python scripts/generate_instances.py \
  --out-root data/instances \
  --n-train 2000 --n-val 200 --n-test 200 --jitter
```

### 2. Collect trajectories (branching + valid Gomory cut labels)
```bash
python scripts/collect_with_cuts_v2.py \
  --instances-root data/instances \
  --data-dir data/trajectories \
  --split train --n-instances-per-class 2000
```

### 3. Train
```bash
# Phases 1→4: policy imitation → value → dynamics → joint fine-tuning
python train.py --config configs/default.yaml --data_root data/trajectories

# Fast smoke pass to validate the pipeline
python train.py --data_root data/trajectories --max_files 8 --max_epochs 1
```

### 4. Evaluate
```bash
python scripts/evaluate.py --checkpoint checkpoints/phase4_best.pt
```

## Training Phases

| Phase | What trains | Frozen | Key metric |
|-------|-------------|--------|------------|
| 1 | Policy head | — | Top-1 accuracy vs. strong branching |
| 2 | Value head | Encoder + Policy | Spearman ρ on dual bound |
| 3 | Dynamics Transformer | Encoder | Latent MSE + cosine + overshoot (K=1→3) |
| 4 | All (joint) | — | Total loss, Top-1 accuracy |

Each phase is warm-started from the previous checkpoint and early-stopped on its
own validation metric. Full hyperparameters are in `configs/default.yaml`.

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Lookahead depth K | 3 |
| Discount γ | 0.95 |
| Branching factor b | 2 |
| Latent dimension d | 128 |
| SGM shift (nodes / time) | 10 / 1 |

## Results (Set Cover benchmarks, single seed)

### Termination rate (% of instances solved within budget)

| Method | Medium (500×1000, 300 s) | Hard (1000×2000, 600 s) |
|--------|--------------------------|--------------------------|
| Most-fractional | 0% | 0% |
| Pseudocost | 0% | 0% |
| Imitation (Neural-Base) | 100% | 100% |
| **Neural-Best (rollout + cuts)** | **100%** | **100%** |

### Node count — SGM (shift=10), solved instances only

| Method | Medium | Hard |
|--------|--------|------|
| Imitation (Neural-Base) | 5,733 | 4,633 |
| Neural-Rollout (D3) | 3,197 | — |
| **Neural-Best (D3 + cuts)** | — | **2,603** |

Neural-Best is up to **41% faster** than most-fractional on hard instances.

> **Note:** This is a controlled feasibility study on a fixed benchmark. The
> evaluation uses a single random seed and small instance counts; results
> establish proof-of-concept, not statistical superiority over industrial solvers.

## Checkpoints

Checkpoints are not stored in this repository due to size.

| File | Phase | Val metric |
|------|-------|------------|
| `phase1_best.pt` | Policy imitation | Top-1 accuracy |
| `phase2_best.pt` | Value | Spearman ρ |
| `phase3_best.pt` | Dynamics | Latent MSE |
| `phase4_best.pt` | Joint fine-tuning | Total loss |

## Citation

```bibtex
@mastersthesis{gawali2026cutworld,
  title  = {CutWorld: A Latent World Model for Branch-and-Cut},
  author = {Anonymous},
  school = {Anonymous},
  year   = {2026}
}
```
