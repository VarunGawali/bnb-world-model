"""
checkpoint.py — Save and load model checkpoints.

Handles the "_orig_mod." prefix that torch.compile() adds to state dict keys,
so compiled and non-compiled checkpoints are interchangeable.
"""

import torch
from pathlib import Path


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """
    Save model + optimizer state to disk.

    Args:
        model     : nn.Module
        optimizer : torch.optim.Optimizer
        epoch     : int
        metrics   : dict of metric name -> value
        path      : str or Path
    """
    state = model.state_dict()
    # Strip torch.compile() prefix if present
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    torch.save(
        {
            "epoch":     epoch,
            "model":     state,
            "optimizer": optimizer.state_dict(),
            "metrics":   metrics,
        },
        path,
    )


def load_checkpoint(model, optimizer, path, device=None):
    """
    Load a checkpoint into model and optimizer.

    Args:
        model     : nn.Module (must match architecture)
        optimizer : torch.optim.Optimizer
        path      : str or Path
        device    : torch.device (defaults to CPU)

    Returns:
        epoch   : int   epoch the checkpoint was saved at
        metrics : dict  metrics saved with the checkpoint
    """
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)

    state = ckpt["model"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    return ckpt.get("epoch", 0), ckpt.get("metrics", {})


def load_weights_only(model, path, device=None):
    """
    Load only model weights (no optimizer state).
    Useful for evaluation and fine-tuning.

    Args:
        model  : nn.Module
        path   : str or Path
        device : torch.device

    Returns:
        model (in-place modified)
    """
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return model
