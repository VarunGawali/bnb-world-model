"""
train_highs.py — Orchestrate all training phases on HiGHS trajectory data.

Usage:
    python scripts/train_highs.py \
        --data_dir data/highs_trajectories \
        --ckpt_dir checkpoints/highs_retrain \
        [--warm_start checkpoints/phase3_best.pt] \
        [--phase 1 2 3 4] \
        [--cut_transitions_dir data/cut_transitions] \
        [--phase3_train_encoder]   # joint encoder+dynamics (recommended)

The script reads configs/default.yaml and allows per-field overrides via
--key value (dot-separated path, e.g. --training.lr_phase1 5e-4).

Warm-start: loads the GNN encoder + policy head weights, then re-initialises
dynamics + value_head + integrality_logit so Phase 3 trains a fresh dynamics
model while keeping strong policy priors.
"""

import argparse
import sys
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split, Sampler
from torch.utils.data.distributed import DistributedSampler
import yaml

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bnb_wm.model.world_model import BnBWorldModel
from bnb_wm.data.datasets import (
    TransitionDataset,
    SequenceDataset,
    RawSequenceDataset,
    CutTransitionDataset,
    make_raw_collate,
)
from bnb_wm.training.trainer import Trainer
from bnb_wm.training.checkpoint import load_weights_only
import bnb_wm.training.checkpoint as _ckpt_module
from torch.utils.data import Dataset


# ── DDP helpers ──────────────────────────────────────────────────────────────

def _ddp_setup():
    """Initialize process group for DDP. Called only when torchrun launches us."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


def _is_main() -> bool:
    """True on rank 0 (or when not running DDP)."""
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def _ddp_print(*args, **kwargs):
    if _is_main():
        print(*args, **kwargs)


# ── sharded sampler ───────────────────────────────────────────────────────────

class ShardedSampler(Sampler):
    """
    Shuffles trajectory files, then yields all item indices from each file
    consecutively. This keeps TransitionDataset's single-file LRU cache at
    ~100% hit rate instead of ~0% under random shuffling across thousands of
    files. One file is decompressed once per epoch instead of once per item.
    """
    def __init__(self, dataset, seed: int = 0):
        self.dataset = dataset
        self.seed    = seed
        self._epoch  = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        # Group item indices by file index
        from collections import defaultdict
        file_to_items: dict[int, list[int]] = defaultdict(list)
        for item_idx, (fi, _) in enumerate(self.dataset.index):
            file_to_items[fi].append(item_idx)
        file_order = list(file_to_items.keys())
        rng.shuffle(file_order)
        indices = []
        for fi in file_order:
            items = file_to_items[fi]
            rng.shuffle(items)
            indices.extend(items)
        return iter(indices)


# ── helpers ───────────────────────────────────────────────────────────────────

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _apply_overrides(cfg: dict, overrides: list[str]):
    """Apply --key value pairs (key may be dot-separated) to cfg in-place."""
    i = 0
    while i < len(overrides):
        key = overrides[i].lstrip("-")
        val = overrides[i + 1]
        i += 2
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        # Try to parse as number/bool
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                if val.lower() in ("true", "false"):
                    val = val.lower() == "true"
        d[parts[-1]] = val


def _build_model(cfg: dict, device: torch.device) -> BnBWorldModel:
    mc = cfg["model"]
    return BnBWorldModel(
        hidden_dim=mc["hidden_dim"],
        n_gnn_layers=mc["n_gnn_layers"],
        n_gnn_heads=mc["n_gnn_heads"],
        n_dyn_layers=mc["n_dyn_layers"],
        n_dyn_heads=mc.get("n_dyn_heads", 4),
        max_seq=mc.get("max_seq", 512),
        dyn_residual=mc.get("dyn_residual", True),
        dyn_heteroscedastic=mc.get("dyn_heteroscedastic", False),
    ).to(device)


def _warm_start(model: BnBWorldModel, ckpt_path: Path, device: torch.device):
    """Load encoder + policy; reset dynamics, value_head, integrality weights."""
    print(f"[warm_start] loading from {ckpt_path}")
    load_weights_only(model, ckpt_path, device=device, strict=False)

    # Re-initialise components that need fresh training on HiGHS data.
    def _reset(module):
        for p in module.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    _reset(model.dynamics)
    _reset(model.value)
    _reset(model.integrality)
    print("[warm_start] dynamics + value + integrality reset to random")


def _split_files(files: list[Path], train_frac: float, val_frac: float, seed: int):
    rng = random.Random(seed)
    files = list(files)
    rng.shuffle(files)
    n = len(files)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    return files[:n_train], files[n_train:n_train + n_val], files[n_train + n_val:]


def _transition_loaders(data_dirs: list[Path], cfg: dict, seed: int):
    """Build TransitionDataset train/val loaders from .npz files in one or more dirs."""
    files = []
    for d in data_dirs:
        files.extend(sorted(d.glob("**/*.npz")))
    files = [f for f in files if not f.name.endswith("_cut.npz")]
    if not files:
        raise FileNotFoundError(f"No .npz trajectory files found in {data_dirs}")
    print(f"Found {len(files)} trajectory files for TransitionDataset "
          f"(from {len(data_dirs)} dir(s))")

    dc   = cfg["data"]
    tc   = cfg["training"]
    tr_f, va_f, _ = _split_files(
        files, dc["train_split"], dc["val_split"], seed
    )

    tr_ds = TransitionDataset(tr_f)
    va_ds = TransitionDataset(va_f)
    bs    = tc["batch_size"]

    def _collate(batch):
        from torch_geometric.data import Batch
        graphs, metas = zip(*batch)
        return Batch.from_data_list(graphs), list(metas)

    ddp_active = dist.is_available() and dist.is_initialized()
    if ddp_active:
        tr_sampler = DistributedSampler(tr_ds, shuffle=True, seed=seed)
        va_sampler = DistributedSampler(va_ds, shuffle=False)
    else:
        tr_sampler = ShardedSampler(tr_ds, seed=seed)
        va_sampler = None

    tr_loader = DataLoader(tr_ds, batch_size=bs, sampler=tr_sampler,
                           collate_fn=_collate, num_workers=8, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=bs, sampler=va_sampler,
                           shuffle=(va_sampler is None and False),
                           collate_fn=_collate, num_workers=8, pin_memory=True)
    return tr_loader, va_loader


def _sequence_loaders(data_dirs: list[Path], cfg: dict, seed: int,
                      raw: bool = False):
    """Build Sequence(Raw)Dataset train/val loaders."""
    files = []
    for d in data_dirs:
        files.extend(sorted(d.glob("**/*.npz")))
    files = [f for f in files if not f.name.endswith("_cut.npz")]
    if not files:
        raise FileNotFoundError(f"No .npz trajectory files found in {data_dirs}")

    dc = cfg["data"]
    tc = cfg["training"]
    tr_f, va_f, _ = _split_files(files, dc["train_split"], dc["val_split"], seed)

    DS = RawSequenceDataset if raw else SequenceDataset
    tr_ds = DS(tr_f)
    va_ds = DS(va_f)
    if raw:
        # Joint encoder+dynamics: GNN runs with gradients on large graphs.
        # Medium/hard (500x1000+) OOM at bs=4; use bs=1 by default.
        bs = max(1, tc.get("phase3_seq_batch_size", 1))
    else:
        bs = max(4, tc["batch_size"] // 4)  # sequences are longer; shrink batch

    if raw:
        collate_fn = make_raw_collate()
    else:
        def collate_fn(batch):
            from torch.utils.data.dataloader import default_collate
            if isinstance(batch[0], dict):
                keys = batch[0].keys()
                out = {}
                for k in keys:
                    vals = [b[k] for b in batch]
                    try:
                        out[k] = torch.stack(vals) if isinstance(vals[0], torch.Tensor) else vals
                    except Exception:
                        out[k] = vals
                return out
            return default_collate(batch)

    ddp_active = dist.is_available() and dist.is_initialized()
    if ddp_active:
        tr_sampler = DistributedSampler(tr_ds, shuffle=True)
        va_sampler = DistributedSampler(va_ds, shuffle=False)
        tr_loader = DataLoader(tr_ds, batch_size=bs, sampler=tr_sampler,
                               collate_fn=collate_fn, num_workers=4)
        va_loader = DataLoader(va_ds, batch_size=bs, sampler=va_sampler,
                               collate_fn=collate_fn, num_workers=4)
    else:
        tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True,
                               collate_fn=collate_fn, num_workers=8)
        va_loader = DataLoader(va_ds, batch_size=bs, shuffle=False,
                               collate_fn=collate_fn, num_workers=8)
    return tr_loader, va_loader


class CachedLatentDataset(Dataset):
    """Reads pre-encoded latent sequences from chunk_NNNNN.pt files in cache_dir."""

    def __init__(self, cache_dir: Path):
        self.seqs: list[dict] = []
        for chunk_file in sorted(cache_dir.glob("chunk_*.pt")):
            chunk = torch.load(chunk_file, map_location="cpu", weights_only=False)
            self.seqs.extend(chunk)
        if not self.seqs:
            raise FileNotFoundError(f"No cached latent sequences found in {cache_dir}")

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]


def _cached_latent_collate(batch: list[dict]) -> dict:
    """Collate pre-encoded latent dicts — tensors are already fixed-length."""
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            try:
                out[k] = torch.stack(vals)
            except RuntimeError:
                # Variable-length sequences — pad to longest
                max_len = max(v.shape[0] for v in vals)
                padded = torch.zeros(len(vals), max_len, *vals[0].shape[1:])
                for i, v in enumerate(vals):
                    padded[i, :v.shape[0]] = v
                out[k] = padded
        else:
            out[k] = vals
    return out


def _cached_sequence_loaders(cache_dir: Path, cfg: dict, seed: int):
    """Build train/val DataLoaders from pre-encoded latent cache."""
    ds = CachedLatentDataset(cache_dir)
    n = len(ds)
    n_val = max(1, int(n * cfg["data"]["val_split"]))
    n_train = n - n_val
    rng = torch.Generator().manual_seed(seed)
    tr_ds, va_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=rng)
    bs = max(4, cfg["training"].get("phase3_seq_batch_size", 4))

    ddp_active = dist.is_available() and dist.is_initialized()
    if ddp_active:
        tr_sampler = DistributedSampler(tr_ds, shuffle=True)
        va_sampler = DistributedSampler(va_ds, shuffle=False)
        tr_loader = DataLoader(tr_ds, batch_size=bs, sampler=tr_sampler,
                               collate_fn=_cached_latent_collate, num_workers=4)
        va_loader = DataLoader(va_ds, batch_size=bs, sampler=va_sampler,
                               collate_fn=_cached_latent_collate, num_workers=4)
    else:
        tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True,
                               collate_fn=_cached_latent_collate, num_workers=4)
        va_loader = DataLoader(va_ds, batch_size=bs, shuffle=False,
                               collate_fn=_cached_latent_collate, num_workers=4)
    print(f"[CachedLatent] {n_train} train / {n_val} val sequences from {cache_dir}")
    return tr_loader, va_loader


def _cut_loader(cut_dir: Path, cfg: dict):
    if cut_dir is None or not cut_dir.exists():
        return None
    files = sorted(cut_dir.glob("**/*_cut.npz"))
    if not files:
        return None
    print(f"Found {len(files)} cut-transition files")
    ds = CutTransitionDataset(files)
    bs = max(4, cfg["training"]["batch_size"] // 4)
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=1,
                      collate_fn=CutTransitionDataset.collate)


# ── main phases ───────────────────────────────────────────────────────────────

def run(args, cfg, device):
    ckpt_dir = Path(args.ckpt_dir)
    if _is_main():
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = _build_model(cfg, device)
    tc    = cfg["training"]

    if args.warm_start:
        _warm_start(model, Path(args.warm_start), device)

    # Wrap in DDP after warm-start so weight loading happens on the raw model.
    ddp_active = dist.is_available() and dist.is_initialized()
    if ddp_active:
        local_rank = dist.get_rank()
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Trainer always sees the raw model for checkpoint saving; unwrap if needed.
    raw_model = model.module if ddp_active else model
    trainer = Trainer(raw_model, device, ckpt_dir, amp=tc["amp"])
    trainer.ddp_active = ddp_active          # trainer uses this to skip non-rank-0 saves
    seed    = 42

    phases   = set(args.phase) if args.phase else {1, 2, 3, 4}
    data_dirs = [Path(d) for d in args.data_dir]

    # ── Phase 1: Policy ───────────────────────────────────────────────────────
    if 1 in phases:
        print("\n" + "=" * 60)
        print("PHASE 1 — Policy imitation (HiGHS strong branching labels)")
        print("=" * 60)
        tr_l, va_l = _transition_loaders(data_dirs, cfg, seed)
        trainer.train_policy(
            tr_l, va_l,
            epochs=tc["epochs_phase1"],
            lr=tc["lr_phase1"],
            patience=tc.get("patience", 6),
        )
        print("[Phase 1] Done. Best checkpoint: phase1_best.pt")

    # ── Phase 2: Value ────────────────────────────────────────────────────────
    if 2 in phases:
        print("\n" + "=" * 60)
        print("PHASE 2 — Value head (encoder frozen)")
        print("=" * 60)
        # Load best phase1 if it exists (from this run or a previous one).
        p1_ckpt = ckpt_dir / "phase1_best.pt"
        if p1_ckpt.exists():
            load_weights_only(model, p1_ckpt, device=device, strict=False)
            print(f"  Loaded {p1_ckpt}")
        tr_l, va_l = _transition_loaders(data_dirs, cfg, seed)
        trainer.train_value(
            tr_l, va_l,
            epochs=tc["epochs_phase2"],
            lr=tc["lr_phase2"],
            patience=tc.get("patience", 6),
        )
        print("[Phase 2] Done. Best checkpoint: phase2_best.pt")

    # ── Phase 3: Dynamics (joint encoder+dynamics when --phase3_train_encoder) ─
    if 3 in phases:
        print("\n" + "=" * 60)
        mode = "joint encoder+dynamics" if args.phase3_train_encoder else "dynamics (encoder frozen)"
        print(f"PHASE 3 — Dynamics Transformer ({mode})")
        print("=" * 60)
        p2_ckpt = ckpt_dir / "phase2_best.pt"
        if p2_ckpt.exists():
            load_weights_only(model, p2_ckpt, device=device, strict=False)
            print(f"  Loaded {p2_ckpt}")

        use_disk_cache = bool(args.phase3_latent_cache)
        if use_disk_cache:
            cache_dir = Path(args.phase3_latent_cache)
            print(f"  Using pre-encoded disk cache: {cache_dir}")
            tr_l, va_l = _cached_sequence_loaders(cache_dir, cfg, seed)
            # Encoder is frozen; dynamics-only training; cache refreshed externally.
            also_train_encoder = False
            encode_cache_refresh = 0
        else:
            raw = args.phase3_train_encoder
            tr_l, va_l = _sequence_loaders(data_dirs, cfg, seed, raw=raw)
            also_train_encoder = raw
            encode_cache_refresh = tc.get("encode_cache_refresh_every", 5)

        # Unfreeze encoder if joint training is requested (no disk cache).
        if also_train_encoder:
            for p in raw_model.encoder.parameters():
                p.requires_grad_(True)

        trainer.train_dynamics(
            tr_l, va_l,
            epochs=tc["epochs_phase3"],
            lr=tc["lr_phase3"],
            patience=tc.get("patience_phase3", tc.get("patience", 12)),
            overshoot_depth=tc.get("overshoot_depth", 3),
        )
        print("[Phase 3] Done. Best checkpoint: phase3_best.pt")

    # ── Phase 4: Joint fine-tune ───────────────────────────────────────────────
    if 4 in phases:
        print("\n" + "=" * 60)
        print("PHASE 4 — Joint fine-tune (all parameters)")
        print("=" * 60)
        p3_ckpt = ckpt_dir / "phase3_best.pt"
        if p3_ckpt.exists():
            load_weights_only(model, p3_ckpt, device=device, strict=False)
            print(f"  Loaded {p3_ckpt}")
        tr_l, va_l = _transition_loaders(data_dirs, cfg, seed)
        trainer.train_joint(
            tr_l, va_l,
            epochs=tc["epochs_phase4"],
            lr=tc["lr_phase4"],
            patience=tc.get("patience", 6),
        )
        print("[Phase 4] Done. Best checkpoint: phase4_best.pt")

    print("\nAll selected phases complete.")
    print(f"Checkpoints saved to: {ckpt_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True, nargs="+",
                        help="One or more directories of HiGHS trajectory .npz files "
                             "(e.g. --data_dir data/highs_trajectories/easy "
                             "data/highs_trajectories/medium data/highs_trajectories/hard)")
    parser.add_argument("--ckpt_dir", default="checkpoints/highs_retrain",
                        help="Where to save checkpoints")
    parser.add_argument("--warm_start", default=None,
                        help="Path to existing checkpoint to warm-start from "
                             "(encoder + policy loaded; dynamics reset)")
    parser.add_argument("--phase", type=int, nargs="+", default=None,
                        help="Phases to run (default: 1 2 3 4)")
    parser.add_argument("--cut_transitions_dir", default=None,
                        help="Directory of *_cut.npz files for cut-dynamics training")
    parser.add_argument("--phase3_train_encoder", action="store_true",
                        help="Joint encoder+dynamics training in Phase 3 "
                             "(NextLat Theorem 3.2 requirement). Uses RawSequenceDataset.")
    parser.add_argument("--phase3_latent_cache", default=None,
                        help="Path to pre-encoded latent cache dir (from prebuild_latent_cache.py). "
                             "When set, Phase 3 skips live GNN encoding — minutes/epoch instead of hours. "
                             "Re-run prebuild_latent_cache.py every ~5 epochs to refresh the encoder.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"),
                        help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (auto-detected if omitted)")

    args, overrides = parser.parse_known_args()

    cfg = _load_config(Path(args.config))
    if overrides:
        _apply_overrides(cfg, overrides)

    # DDP: torchrun sets LOCAL_RANK. If present, initialise the process group.
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        rank, world_size = _ddp_setup()
        device = torch.device(f"cuda:{local_rank}")
        _set_seed(args.seed + rank)          # different shuffle per rank
        if _is_main():
            print(f"DDP: rank {rank}/{world_size} on {device}")
        # Non-rank-0 processes must not write checkpoints.
        if not _is_main():
            _orig_save = _ckpt_module.save_checkpoint
            _ckpt_module.save_checkpoint = lambda *a, **kw: None
    else:
        if args.device:
            device = torch.device(args.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _set_seed(args.seed)
        print(f"Device: {device}")

    run(args, cfg, device)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
