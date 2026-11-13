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


def _from_split_var_dynamics(model, state):
    """Convert checkpoints whose per-variable transition stores its first layer
    split by input block (W_h with bias, W_z, W_a, then out) to this model's
    single-MLP layout (net.0 on [h, z, a], net.3). Exact: net.0([h, z, a]) =
    W_h h + W_z z + W_a a + b.

    Such checkpoints also carry dynamics.h_cons_proj, which this model does
    not have; it is dropped and its largest absolute weight is reported,
    because a non-zero value means the training code used it.
    """
    p = "dynamics.var_dynamics."
    model_keys = set(model.state_dict())
    report = {}
    if f"{p}W_h.weight" in state and f"{p}net.0.weight" in model_keys:
        state = dict(state)
        blocks = [state.pop(f"{p}{n}.weight") for n in ("W_h", "W_z", "W_a")]
        bias = sum(state.pop(f"{p}{n}.bias") for n in ("W_h", "W_z", "W_a")
                   if f"{p}{n}.bias" in state)
        state[f"{p}net.0.weight"] = torch.cat(blocks, dim=1)
        state[f"{p}net.0.bias"] = bias
        state[f"{p}net.3.weight"] = state.pop(f"{p}out.weight")
        state[f"{p}net.3.bias"] = state.pop(f"{p}out.bias")
        report["var_dynamics"] = "converted W_h/W_z/W_a/out to net.0/net.3 (exact)"
    k = "dynamics.h_cons_proj.weight"
    if k in state and k not in model_keys:
        state = dict(state)
        w = state.pop(k)
        report["h_cons_proj_dropped_max_abs"] = float(w.abs().max())
        if report["h_cons_proj_dropped_max_abs"] > 0:
            import warnings
            warnings.warn(
                "checkpoint has a non-zero dynamics.h_cons_proj, which this model "
                "does not implement; dynamics outputs differ from the training code")
    return state, report


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
    from bnb_wm.model.dynamics import _VarDynamics

    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    # Migrate legacy _VarDynamics checkpoint (net.0/net.3) to decomposed form,
    # only when the model itself no longer uses the net.* layout.
    model_has_net = any("var_dynamics.net." in k for k in model.state_dict())
    if any("var_dynamics.net." in k for k in state) and not model_has_net:
        _, state = _VarDynamics.from_legacy_state(
            state, hidden_dim=model.dynamics.hidden_dim
        )

    state, report = _from_split_var_dynamics(model, state)
    model._load_report = report
    if report:
        print(f"[load_weights_only] {report}")

    # h_cons_proj is zero-initialised by design (no-op at load time).
    # Allow it to be absent from old checkpoints.
    _ZERO_INIT_OK = {"dynamics.h_cons_proj.weight"}

    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_bad = [k for k in missing if k not in _ZERO_INIT_OK]
    if missing_bad:
        raise RuntimeError(f"[load_weights_only] missing keys after migration: {missing_bad}")
    if unexpected:
        raise RuntimeError(f"[load_weights_only] unexpected keys: {unexpected}")
    if missing:
        print(f"[load_weights_only] zero-init kept for: {missing}")

    # Sanitize encoder std buffers: checkpoints saved before the clamp was
    # added may contain literal zeros, which cause divide-by-zero in forward.
    enc = model.encoder
    n_zero_var = (enc.var_std == 0).sum().item()
    n_zero_con = (enc.con_std == 0).sum().item()
    if n_zero_var or n_zero_con:
        enc.var_std.clamp_(min=1e-6)
        enc.con_std.clamp_(min=1e-6)
        print(f"[load_weights_only] clamped {n_zero_var} zero var_std "
              f"and {n_zero_con} zero con_std entries to 1e-6")

    return model
