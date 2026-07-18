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
    # Use the dtype's most-negative representable value so the mask works under
    # AMP (fp16, where -1e9 overflows). Compute the loss in fp32 for stability.
    neg_inf = torch.finfo(scores.dtype).min
    masked = torch.full_like(scores, neg_inf)
    masked[action_set] = scores[action_set]

    target = action_set[local_label]
    loss = F.cross_entropy(masked.unsqueeze(0).float(), target.unsqueeze(0))

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


def var_reconstruction_loss(h_pred, h_target, var_mask=None):
    """
    MSE + cosine loss for per-variable latent transition prediction.

    Trains the dynamics model's per-variable head so that the predicted
    future embeddings h_vars_{t+1} stay on the real-encoder manifold — the
    ingredient that makes a latent rollout (policy re-run on predicted state)
    trustworthy rather than drifting out of distribution.

    Args:
        h_pred   : [B, T, V, H]  predicted next per-variable embeddings
        h_target : [B, T, V, H]  true next per-variable embeddings
        var_mask : [B, T, V] bool — valid (non-padding) variable positions,
                   or None to use all positions
    """
    if var_mask is not None:
        m = var_mask.unsqueeze(-1)                       # [B, T, V, 1]
        h_pred   = h_pred * m
        h_target = h_target * m
        denom = m.sum().clamp_min(1.0)
        mse = ((h_pred - h_target) ** 2).sum() / (denom * h_pred.size(-1))
    else:
        mse = F.mse_loss(h_pred, h_target)

    cos = 1.0 - F.cosine_similarity(
        h_pred.reshape(-1, h_pred.size(-1)),
        h_target.reshape(-1, h_target.size(-1)),
        dim=-1,
    ).mean()
    return mse + 0.1 * cos


def subtree_size_loss(pred_log_size, target_size):
    """
    Huber loss on log1p subtree size.

    The SubtreeSizeHead predicts log1p(node count); targets come directly from
    the collected B&B traces (the true number of nodes in each node's subtree),
    so this is a fully supervised regression — no proxy. Log space keeps the
    loss well-conditioned across subtrees spanning several orders of magnitude.

    Args:
        pred_log_size : [batch]  predicted log1p(subtree size), already >= 0
        target_size   : [batch]  true subtree node counts (raw, >= 1)
    """
    target_log = torch.log1p(target_size.clamp_min(0.0))
    return F.huber_loss(pred_log_size.squeeze(), target_log.squeeze(), delta=1.0)


def cost_to_go_loss(pred_log_ctg, target_steps):
    """
    Huber loss on log1p cost-to-go (remaining B&B node count).

    The target is a Monte-Carlo return read directly from the trajectory:
    steps_to_go(t) = n_steps - t. It requires no DFS ordering, so it is
    trainable on the collected non-DFS traces — unlike subtree size. Log space
    keeps the loss well-conditioned across nodes with very different amounts of
    remaining work.

    Args:
        pred_log_ctg : [batch]  predicted log1p(remaining nodes), already >= 0
        target_steps : [batch]  true remaining node counts (n_steps - t, >= 0)
    """
    target_log = torch.log1p(target_steps.clamp_min(0.0))
    return F.huber_loss(pred_log_ctg.squeeze(), target_log.squeeze(), delta=1.0)


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
