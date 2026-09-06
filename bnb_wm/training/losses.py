"""
losses.py — Loss functions for each training phase.

Phase 1 — policy_loss_masked      : cross-entropy over candidate action set
Phase 1 — policy_loss_soft        : soft/ranking KL blend using full SB scores
Phase 2 — value_loss              : Huber loss on normalised dual bound
Phase 3 — dynamics_loss           : masked MSE + masked cosine on latent transitions
Phase 3 — candidate_ranking_loss  : listwise ranking over imagined next states
Phase 4 — integrality_loss        : weighted BCE on leaf prediction
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


def policy_loss_soft(scores, action_set, sb_scores, local_label,
                     alpha=0.5, temp=1.0):
    """Soft/ranking imitation for one graph: match SB's *preference order*.

    Strong branching produces a score for every candidate, and those scores have
    many near-ties, so training the policy only on the argmax (hard CE) throws
    away most of the signal and caps top-1 accuracy artificially. Here we build a
    soft target distribution from the full SB scores and minimise KL to the
    policy's candidate distribution, blended with the hard CE for a stable anchor:

        loss = alpha * CE(policy, argmax) + (1 - alpha) * KL(target || policy)

    The SB scores are standardised per node before the softmax so the target is
    scale-invariant (SB magnitudes vary wildly across nodes); `temp` sharpens
    (<1) or softens (>1) the target.

    Args:
        scores      : [n_vars] raw policy logits
        action_set  : [k]      candidate indices
        sb_scores   : [k]      SB scores aligned with action_set
        local_label : int      index into action_set of the SB argmax
        alpha       : weight on the hard-CE term (0=pure soft, 1=pure hard)
        temp        : softmax temperature for the soft target
    Returns:
        loss, acc (top-1 vs SB argmax), rand (1/k)
    """
    neg_inf = torch.finfo(scores.dtype).min
    masked = torch.full_like(scores, neg_inf)
    masked[action_set] = scores[action_set]
    target_idx = action_set[local_label]

    hard = F.cross_entropy(masked.unsqueeze(0).float(), target_idx.unsqueeze(0))

    # Soft target over candidates: standardise SB scores, then temp-softmax.
    sb = sb_scores.float()
    if sb.numel() > 1 and torch.isfinite(sb).all():
        sb = (sb - sb.mean()) / (sb.std() + 1e-6)
        tgt = F.softmax(sb / max(temp, 1e-3), dim=0)              # [k]
        logp = F.log_softmax(scores[action_set].float(), dim=0)  # [k]
        soft = F.kl_div(logp, tgt, reduction="sum")
        loss = alpha * hard + (1.0 - alpha) * soft
    else:
        loss = hard  # degenerate (single candidate / bad scores): hard only

    acc  = float(masked.argmax() == target_idx)
    rand = 1.0 / len(action_set)
    return loss, acc, rand


def value_loss(v_pred, target):
    """
    Huber loss for dual-bound regression.

    delta=0.1 keeps robustness meaningful when targets are normalised to [0,1]
    — with delta=1.0 all residuals fall in the quadratic region and Huber
    degenerates to MSE.

    Args:
        v_pred : [batch] predicted normalised dual bounds
        target : [batch] true normalised dual bounds
    """
    return F.huber_loss(v_pred.squeeze(-1), target.squeeze(-1), delta=0.1)


def dynamics_loss(z_pred, z_target, step_mask=None):
    """
    Masked MSE + masked cosine loss for latent transition prediction.

    Both terms are masked so padding timesteps (zero vectors) do not dilute
    gradient signal. Without masking, padded zero rows contribute 0 to MSE
    numerator while inflating the denominator, and contribute cos=0 → term=1.0
    to the cosine mean — the maximum possible cosine penalty on every pad step.

    Args:
        z_pred    : [B, T, H] or [batch, H]  predicted next latent state
        z_target  : same shape                true next latent state
        step_mask : [B, T] bool or None — True where the timestep is real
                    (non-padding). If None, all positions are treated as real.

    Returns:
        scalar loss
    """
    H = z_pred.size(-1)
    flat_pred   = z_pred.reshape(-1, H)
    flat_target = z_target.reshape(-1, H)

    if step_mask is not None:
        m = step_mask.reshape(-1).float()          # [B*T]
        n = m.sum().clamp_min(1.0)
        # Masked MSE: sum over valid positions, normalise by valid count * H
        mse = ((flat_pred - flat_target) ** 2 * m.unsqueeze(-1)).sum() / (n * H)
        # Masked cosine: average cos-distance over valid positions only
        cos_per = 1.0 - F.cosine_similarity(flat_pred, flat_target, dim=-1)  # [B*T]
        cos = (cos_per * m).sum() / n
    else:
        mse = F.mse_loss(flat_pred, flat_target)
        cos = (1.0 - F.cosine_similarity(flat_pred, flat_target, dim=-1)).mean()

    return mse + 0.1 * cos


def candidate_ranking_loss(
    z_imagined: torch.Tensor,
    sb_scores: torch.Tensor,
    temp: float = 1.0,
) -> torch.Tensor:
    """Listwise ranking loss on imagined next latent states.

    This is the missing objective for counterfactual planning. The dynamics
    model is trained on expert-only trajectories (one action per state), so the
    cheapest MSE solution is to ignore the action and predict from position
    alone — exactly what action-sensitivity measurements confirm.

    This loss directly supervises the *ordering* that rollout_top_k_batched
    produces: given k candidates with SB scores, the dynamics rolls each one
    step, a downstream head decodes a scalar quality estimate from each
    resulting latent, and we train that ordering to match the SB ordering via
    a listwise softmax cross-entropy.

    The caller is responsible for running the dynamics step and decoding a
    per-candidate scalar (e.g. negative predicted subtree size, or value head
    output). This function only computes the ranking loss given those scalars.

    Args:
        z_imagined : [k] float — decoded scalar for each candidate's next
                     latent state (higher = better candidate predicted)
        sb_scores  : [k] float — SB quality scores, same order as z_imagined
                     (higher = SB thinks this candidate is better)
        temp       : temperature for the soft SB target (default 1.0)

    Returns:
        scalar loss — listwise KL(SB_dist || imagined_dist)

    Usage in trainer:
        # For one node with k candidates:
        z_next_preds = [dynamics.step(z, a_cand[i], ...) for i in range(k)]
        scalars = value_head(torch.stack(z_next_preds))   # [k]
        loss = candidate_ranking_loss(scalars, sb_scores_for_node)
    """
    if z_imagined.size(0) < 2:
        return z_imagined.new_zeros(())

    sb = sb_scores.float()
    if not torch.isfinite(sb).all():
        return z_imagined.new_zeros(())

    # Standardise SB scores per node (magnitudes vary wildly across nodes)
    sb = (sb - sb.mean()) / (sb.std() + 1e-6)
    tgt = F.softmax(sb / max(temp, 1e-3), dim=0)                    # [k]
    logp = F.log_softmax(z_imagined.float(), dim=0)                 # [k]
    # KL(SB_dist || imagined_dist): train imagined ordering → SB ordering
    return F.kl_div(logp, tgt, reduction="sum")


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

    # P2.6: cosine term must average over valid positions only. Padding rows are
    # zero vectors (cosine 0 -> term 1.0), so including them dilutes the signal.
    cos_per = 1.0 - F.cosine_similarity(
        h_pred.reshape(-1, h_pred.size(-1)),
        h_target.reshape(-1, h_target.size(-1)),
        dim=-1,
    )                                                    # [B*T*V]
    if var_mask is not None:
        mflat = var_mask.reshape(-1)
        cos = cos_per[mflat].mean() if bool(mflat.any()) else cos_per.new_zeros(())
    else:
        cos = cos_per.mean()
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
    return F.huber_loss(pred_log_size.squeeze(-1), target_log.squeeze(-1), delta=1.0)


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
    return F.huber_loss(pred_log_ctg.squeeze(-1), target_log.squeeze(-1), delta=1.0)


def integrality_loss(logit, target, pos_weight):
    """
    Weighted BCE for leaf-node prediction.

    Args:
        logit      : [batch]  raw logits from IntegralityHead
        target     : [batch]  0/1 labels (1 = next node is leaf)
        pos_weight : scalar tensor — n_neg / n_pos
    """
    return F.binary_cross_entropy_with_logits(
        logit.reshape(-1), target.float().reshape(-1), pos_weight=pos_weight
    )


def cutting_plane_loss(scores, labels, pos_weight=None):
    """
    Weighted BCE for cut selection imitation.

    Retained for backward compatibility with any existing scripts that import
    it. Phase 5 is now a no-op (ZeroShotCutScorer has no trainable parameters),
    so this function is no longer called by the trainer.

    Args:
        scores     : [n_cuts]  raw logits
        labels     : [n_cuts]  binary labels (1 = good cut)
        pos_weight : scalar tensor or None — n_neg / n_pos
    """
    return F.binary_cross_entropy_with_logits(
        scores.reshape(-1), labels.float().reshape(-1), pos_weight=pos_weight
    )
