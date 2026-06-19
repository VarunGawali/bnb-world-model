"""
losses.py — Loss functions for each training phase.

Phase 1 — policy_loss_masked  : cross-entropy over the candidate action set
Phase 2 — value_loss          : Huber loss on normalised dual bound
Phase 3 — dynamics_loss       : MSE + cosine similarity on latent transitions
Phase 4 — integrality_loss    : weighted BCE on leaf prediction
"""

import torch
import torch.nn.functional as F


def policy_loss_masked(scores, action_set, local_label):
    """
    Compute masked cross-entropy for one graph.

    Masks out non-candidate variables with -1e9 before softmax so the
    model only competes among the valid branching candidates.

    Args:
        scores       : [n_vars]  raw policy logits for all variables
        action_set   : [k]       indices of valid branching candidates
        local_label  : int       index into action_set of expert choice

    Returns:
        loss : scalar tensor
        acc  : float  1.0 if top-1 prediction matches expert, else 0.0
        rand : float  1/k  (random baseline accuracy for this sample)
    """
    masked = torch.full_like(scores, -1e9)
    masked[action_set] = scores[action_set]

    target = action_set[local_label]
    loss = F.cross_entropy(masked.unsqueeze(0), target.unsqueeze(0))

    pred = masked.argmax()
    acc  = float(pred == target)
    rand = 1.0 / len(action_set)

    return loss, acc, rand


def value_loss(v_pred, target):
    """
    Huber loss for dual-bound regression.
    More robust to outliers than MSE at distant nodes.

    Args:
        v_pred : [batch] predicted normalised dual bounds
        target : [batch] true normalised dual bounds

    Returns:
        loss : scalar tensor
    """
    return F.huber_loss(v_pred.squeeze(), target.squeeze(), delta=1.0)


def dynamics_loss(z_pred, z_target):
    """
    Combined MSE + cosine loss for latent transition prediction.
    MSE penalises magnitude errors; cosine penalises directional errors.

    Args:
        z_pred   : [batch, H] predicted next latent state
        z_target : [batch, H] true next latent state (encoder output)

    Returns:
        loss : scalar tensor
    """
    mse = F.mse_loss(z_pred, z_target)
    cos = 1.0 - F.cosine_similarity(z_pred, z_target, dim=-1).mean()
    return mse + 0.1 * cos


def integrality_loss(logit, target, pos_weight):
    """
    Weighted binary cross-entropy for leaf-node prediction.
    pos_weight corrects for class imbalance (leaf nodes are rare).

    Args:
        logit      : [batch] raw logits from IntegralityHead
        target     : [batch] 0/1 labels (1 = next node is leaf)
        pos_weight : scalar tensor — n_neg / n_pos

    Returns:
        loss : scalar tensor
    """
    return F.binary_cross_entropy_with_logits(
        logit.squeeze(),
        target.float(),
        pos_weight=pos_weight,
    )
