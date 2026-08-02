"""
repro.py — Reproducibility & provenance (P2.3).

Two helpers used by the training entry point:

    seed_everything(seed)   — seed Python / NumPy / torch / CUDA and (optionally)
                              request deterministic algorithms, plus return a
                              DataLoader worker_init_fn so worker RNGs are seeded.
    write_provenance(dir, extra) — dump a JSON record of the git commit, library
                              versions, platform, seed and config, so any run's
                              artifacts can be traced back to exact code + inputs.

Neither is required for correctness, but without them a "result" is not
reproducible and cannot be defended in a paper.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def seed_everything(seed: int = 0, deterministic: bool = False):
    """
    Seed all RNGs that affect a training run.

    Args:
        seed          : base seed for Python / NumPy / torch / CUDA.
        deterministic : if True, also set cuDNN deterministic + torch
                        deterministic algorithms (slower, fully reproducible).
    Returns:
        worker_init_fn : pass to DataLoader(worker_init_fn=...) so each worker
                         gets a distinct, deterministic seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # opt-in: some ops have no deterministic kernel and will raise.
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    except Exception:
        pass

    def worker_init_fn(worker_id: int):
        s = seed + worker_id
        random.seed(s)
        try:
            import numpy as np
            np.random.seed(s % (2 ** 32))
        except Exception:
            pass

    return worker_init_fn


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parent, stderr=subprocess.DEVNULL)
        return bool(out.decode().strip())
    except Exception:
        return None


def _versions() -> dict:
    vers = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "numpy", "torch_geometric", "ecole", "pyscipopt",
                "highspy"):
        try:
            m = __import__(mod)
            vers[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            vers[mod] = None
    try:
        import torch
        vers["cuda"] = torch.version.cuda
        vers["cuda_available"] = torch.cuda.is_available()
    except Exception:
        pass
    return vers


def write_provenance(out_dir, extra: dict | None = None) -> Path:
    """
    Write `<out_dir>/provenance.json` capturing commit, versions, and `extra`
    (seed, config path, data root, dataset hash, ...). Returns the path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "argv": sys.argv,
        "versions": _versions(),
    }
    if extra:
        record.update(extra)
    path = out_dir / "provenance.json"
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    return path
