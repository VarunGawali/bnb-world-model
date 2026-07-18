"""
metrics.py — Evaluation metrics for the BnB World Model.

topk_accuracy   : Top-1/3/5 accuracy of policy head vs. strong branching
rank_cdf        : CDF of expert variable rank under the learned policy
compute_spearman: Spearman correlation for value head evaluation
"""

import numpy as np
import torch
from scipy.stats import spearmanr


def topk_accuracy(model, val_loader, device, ks=(1, 3, 5)):
    """
    Compute Top-k accuracy of the policy head on a validation loader.

    For each B&B node, checks if the strong-branching expert choice
    falls within the model's top-k ranked candidates.

    Args:
        model      : BnBWorldModel (eval mode)
        val_loader : DataLoader yielding (pyg_batch, metas)
        device     : torch.device
        ks         : tuple of k values to evaluate

    Returns:
        dict mapping k -> accuracy (float in [0, 1])
    """
    model.eval()
    counts = {k: 0 for k in ks}
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            pyg_batch, metas = batch
            pyg_batch = pyg_batch.to(device)

            scores, _ = model(pyg_batch)
            offset = 0

            for meta in metas:
                n_v    = meta["n_vars"]
                logits = scores[offset : offset + n_v]
                aset   = meta["action_set"].to(device)
                lbl    = meta["local_label"]
                target = aset[lbl]

                cand_scores = logits[aset]
                ranked = aset[torch.argsort(cand_scores, descending=True)]

                for k in ks:
                    if target in ranked[:k]:
                        counts[k] += 1

                total += 1
                offset += n_v

    return {k: counts[k] / total for k in ks}


def rank_cdf(model, val_loader, device):
    """
    Compute the CDF of the expert variable's rank under the policy.

    Rank 1 means the model's top choice matches the expert.
    Rank k means the expert was the k-th highest scored candidate.

    Args:
        model      : BnBWorldModel (eval mode)
        val_loader : DataLoader
        device     : torch.device

    Returns:
        x   : np.ndarray  rank positions [1, 2, ..., max_rank]
        cdf : np.ndarray  cumulative fraction of samples at each rank
    """
    model.eval()
    ranks = []

    with torch.no_grad():
        for batch in val_loader:
            pyg_batch, metas = batch
            pyg_batch = pyg_batch.to(device)

            scores, _ = model(pyg_batch)
            offset = 0

            for meta in metas:
                n_v    = meta["n_vars"]
                logits = scores[offset : offset + n_v]
                aset   = meta["action_set"].to(device)
                lbl    = meta["local_label"]
                target = aset[lbl]

                cand_scores = logits[aset]
                order  = torch.argsort(cand_scores, descending=True)
                ranked = aset[order]

                pos = (ranked == target).nonzero(as_tuple=True)[0]
                ranks.append(int(pos[0]) + 1 if len(pos) > 0 else len(aset))

                offset += n_v

    ranks = np.array(ranks)
    x = np.arange(1, ranks.max() + 1)
    cdf = np.array([(ranks <= k).mean() for k in x])

    print(f"Top-1 : {(ranks <= 1).mean():.3f}")
    print(f"Top-3 : {(ranks <= 3).mean():.3f}")
    print(f"Top-5 : {(ranks <= 5).mean():.3f}")
    print(f"Median rank: {np.median(ranks):.1f}")

    return x, cdf


def compute_spearman(model, val_loader, device):
    """
    Compute Spearman correlation between predicted and true dual bounds.

    Args:
        model      : BnBWorldModel (eval mode)
        val_loader : DataLoader
        device     : torch.device

    Returns:
        rho : float  Spearman correlation coefficient
    """
    model.eval()
    preds_all = []
    tgts_all  = []

    with torch.no_grad():
        for batch in val_loader:
            pyg_batch, metas = batch
            pyg_batch = pyg_batch.to(device)

            h_vars, z = model.encode(pyg_batch)
            var_mask = pyg_batch.node_type == 0
            bvec     = pyg_batch.batch[var_mask]
            x_var    = pyg_batch.x[var_mask]
            frac_mask = (x_var[:, 14] > 0.05) if x_var.size(1) > 14 else None
            preds = model.value_pred(z, h_vars, bvec, frac_mask)

            preds_all.extend(preds.cpu().numpy().tolist())
            tgts_all.extend([m["norm_db"] for m in metas])

    rho, _ = spearmanr(preds_all, tgts_all)
    print(f"Spearman ρ: {rho:.4f}")
    return float(rho)
