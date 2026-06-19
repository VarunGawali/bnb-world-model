"""
losses.py — Loss functions for each training phase.

Phase 1 — policy_loss_masked   : cross-entropy over candidate action set
Phase 2 — value_loss           : Huber loss on normalised dual bound
Phase 3 — dynamics_loss        : MSE + cosine on latent transitions
Phase 4 — integrality_loss     : weighted BCE on leaf prediction
Phase 5 — cutting_plane_loss   : weighted BCE on cut selection
"""

import torch
import torch.nn.functional as F


def policy_loss_masked(scores, action_set, local_label):
    """
    Masked cross-entropy for one graph.

    Args:
        scores       : [n_vars]  raw policy logits
        action_set   : [k]       indices of valid branching candidates
        local_label  : int       index into action_set of expert choice

    Returns:
        loss : scalar tensor
        acc  : float  1.0 if top-1 matches expert
        rand : float  1/k  (random baseline)
    """
    masked = torch.full_like(scores, -1e9)
    masked[action_set] = scores[action_set]

    target = action_set[local_label]
    loss = F.cross_entropy(masked.unsqueeze(0), target.unsqueeze(0))

    acc  = float(masked.argmax() == target)
    rand = 1.0 / len(action_set)
    return loss, acc, rand


def value_loss(v_pred, target):
    """
    Huber loss for dual-bound regression.

    Args:
        v_pred : [batch] predicted normalised dual bounds
        target : [batch] true normalised dual bounds
    """
    return F.huber_loss(v_pred.squeeze(), target.squeeze(), delta=1.0)


def dynamics_loss(z_pred, z_target):
    """
    MSE + cosine loss for latent transition prediction.

    Args:
        z_pred   : [B, T, H] or [batch, H]  predicted next latent state
        z_target : same shape                true next latent state
    """
    mse = F.mse_loss(z_pred, z_target)
    cos = 1.0 - F.cosine_similarity(
        z_pred.reshape(-1, z_pred.size(-1)),
        z_target.reshape(-1, z_target.size(-1)),
        dim=-1,
    ).mean()
    return mse + 0.1 * cos


def integrality_loss(logit, target, pos_weight):
    """
    Weighted BCE for leaf-node prediction.

    Args:
        logit      : [batch]  raw logits from IntegralityHead
        target     : [batch]  0/1 labels (1 = next node is leaf)
        pos_weight : scalar tensor — n_neg / n_pos
    """
    return F.binary_cross_entropy_with_logits(
        logit.squeeze(), target.float(), pos_weight=pos_weight
    )


def cutting_plane_loss(scores, labels, pos_weight=None):
    """
    Weighted BCE for cut selection imitation.

    Labels are 1 if a cut was selected by the expert solver (SCIP) and
    meaningfully improved the LP bound, 0 otherwise. pos_weight corrects
    for imbalance (most candidate cuts are not useful).

    Args:
        scores     : [n_cuts]  raw logits from CuttingPlaneHead
        labels     : [n_cuts]  binary labels (1 = good cut)
        pos_weight : scalar tensor or None — n_neg / n_pos

    Returns:
        loss : scalar tensor
    """
    return F.binary_cross_entropy_with_logits(
        scores.squeeze(), labels.float(), pos_weight=pos_weight
    )
