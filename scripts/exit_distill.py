"""
exit_distill.py — Expert Iteration (ExIt): distill the world-model lookahead
back into the policy head, using ONLY the already-collected data.

Motivation
----------
The policy head is the pipeline's weak link (top-1 ~0.29): it was trained on
strong-branching's ARGMAX (a lossy target) and only ever saw SB's own trajectory
states (distribution shift). But the trained world-model *lookahead* is already a
better policy than the raw head (ablation: ~477 vs ~656 nodes). So we use the
lookahead as an "expert" and distill its preferences back into the policy — no
recollection, no SB scores needed.

Each ExIt round:
  1. RELABEL (offline, no grad): for every node, run `rollout_candidate` on the
     policy's top-k candidates to get a value-informed quality score per
     candidate, and form a soft target = softmax(score / temp).
  2. DISTILL: fine-tune the policy head (encoder optionally unfrozen) to match
     that soft target via `policy_loss_soft`.
  3. The improved policy makes the next round's rollout better -> repeat.

Because the target optimizes predicted *subtree cost* (nodes), not SB-match, this
directly targets the metric we care about. SC-hard nodes can be oversampled
(--oversample_hard) since that tier is data-starved and drives the hard-instance
tree explosions.

Usage
-----
    PYTHONPATH=. python scripts/exit_distill.py \
        --checkpoint checkpoints/model_final.pt --data_root data/trajectories \
        --rounds 3 --epochs 4 --topk 8 --depth 2 --out_dir checkpoints
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from bnb_wm.model.world_model import BnBWorldModel
from bnb_wm.training.checkpoint import load_weights_only, save_checkpoint
from bnb_wm.training.losses import policy_loss_soft
from bnb_wm.data import (
    TransitionDataset, transition_collate, list_trajectory_files, split_files,
)
from bnb_wm.data.datasets import ShardedBatchSampler, probe_edge_cost_per_node


class IndexedTransition(TransitionDataset):
    """TransitionDataset that tags each item with its flat dataset index, so the
    relabel cache can be keyed per node and matched back during distillation."""

    def __getitem__(self, i):
        data, meta = super().__getitem__(i)
        meta["ds_idx"] = i
        return data, meta


def _tier_of(path: Path) -> str:
    p = str(path).lower()
    for t in ("hard", "medium", "easy"):
        if t in p:
            return t
    return "unknown"


@torch.no_grad()
def relabel(model, loader, device, topk, depth, gamma, temp, ctg_weight,
            branch_factor, use_reward_return):
    """Return {ds_idx: (cand_local LongTensor, target FloatTensor)}.

    For each node: pick the policy's top-k candidates within the action set,
    score each with the world-model rollout, and softmax the standardized scores
    into a soft target over those candidates.
    """
    model.eval()
    targets = {}
    for pyg_batch, metas in tqdm(loader, desc="relabel", leave=False):
        pyg_batch = pyg_batch.to(device)
        h_vars, z = model.encode(pyg_batch)
        var_mask = pyg_batch.node_type == 0
        bvec = pyg_batch.batch[var_mask]
        offset = 0
        # policy scores over all vars (for top-k pre-selection)
        scores_all = model.policy(h_vars, z[bvec])
        for g, meta in enumerate(metas):
            n_v = meta["n_vars"]
            g_scores = scores_all[offset: offset + n_v]
            h_g = h_vars[offset: offset + n_v]
            z_g = z[g:g + 1]
            aset = meta["action_set"].to(device)               # local indices
            # Candidate mask: restricts rollout imagined branching to real
            # fractional variables, matching inference behaviour in ablation.py.
            valid_mask = torch.zeros(n_v, dtype=torch.bool, device=device)
            valid_mask[aset] = True
            # policy top-k within the candidate set
            k = min(topk, aset.numel())
            cand_scores = g_scores[aset]
            cand_k = aset[cand_scores.topk(k).indices]          # [k] local idx
            # rollout score per candidate (higher = better)
            rs = torch.tensor(
                [model.rollout_candidate(
                    z_g, h_g, int(c), depth=depth, gamma=gamma,
                    valid_mask=valid_mask, past_tokens=None,
                    size_weight=0.0, ctg_weight=ctg_weight,
                    branch_factor=branch_factor,
                    use_reward_return=use_reward_return)
                 for c in cand_k],
                dtype=torch.float32, device=device)
            # Cache the RAW rollout scores; policy_loss_soft standardizes and
            # temp-softmaxes them into the target itself (caching a pre-softmaxed
            # distribution would double-softmax).
            targets[int(meta["ds_idx"])] = (cand_k.cpu(), rs.cpu())
            offset += n_v
    return targets


def distill_epoch(model, loader, targets, optimizer, device, alpha, temp,
                  training=True):
    model.train() if training else model.eval()
    tot_loss = tot_acc = n = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for pyg_batch, metas in tqdm(
                loader, desc="distill" if training else "val", leave=False):
            pyg_batch = pyg_batch.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            try:
                scores, _ = model(pyg_batch)
                losses, top1, m = [], 0, 0
                offset = 0
                for meta in metas:
                    n_v = meta["n_vars"]
                    logits = scores[offset: offset + n_v]
                    offset += n_v
                    idx = int(meta["ds_idx"])
                    if idx not in targets:
                        continue
                    cand_k, rs = targets[idx]
                    cand_k = cand_k.to(device)
                    rs = rs.to(device)
                    # rs = raw rollout scores over cand_k; policy_loss_soft builds
                    # the soft target (standardize -> temp-softmax) and blends with
                    # hard CE to the rollout's best candidate.
                    best_local = int(rs.argmax())
                    loss, acc, _ = policy_loss_soft(
                        logits, cand_k, rs, best_local, alpha=alpha, temp=temp)
                    losses.append(loss)
                    top1 += acc
                    m += 1
                if not losses:
                    continue
                loss = torch.stack(losses).mean()
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0)
                    optimizer.step()
                tot_loss += loss.item()
                tot_acc += top1 / max(1, m)
                n += 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    continue
                raise
    if n == 0:
        return float("inf"), 0.0
    return tot_loss / n, tot_acc / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data_root", default="data/trajectories")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=4, help="distill epochs/round")
    ap.add_argument("--topk", type=int, default=8, help="candidates rolled out")
    ap.add_argument("--depth", type=int, default=2, help="rollout depth for targets")
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.3,
                    help="hard-CE weight vs soft KL (0=pure soft)")
    ap.add_argument("--ctg_weight", type=float, default=1.0)
    ap.add_argument("--branch_factor", type=int, default=1)
    ap.add_argument("--use_reward_return", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--unfreeze_encoder", action="store_true")
    ap.add_argument("--oversample_hard", type=float, default=1.0,
                    help=">1 duplicates hard-tier nodes in the distill loader")
    ap.add_argument("--max_train_files", type=int, default=400,
                    help="cap the number of TRAIN trajectory files used for ExIt "
                         "(all SC-hard files are kept; easy/medium are sampled to "
                         "fill). The relabel rollout is per-node and expensive, so "
                         "a few hundred files is plenty to distill from. 0 = all.")
    ap.add_argument("--max_val_files", type=int, default=60,
                    help="cap validation files (0 = all)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--files_per_batch", type=int, default=4)
    ap.add_argument("--out_dir", default="checkpoints")
    args = ap.parse_args()

    # Line-buffer stdout so progress is visible under nohup/redirection (Python
    # block-buffers a redirected stdout, which hides all prints until exit).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(args.config))["model"]
    model = BnBWorldModel(
        hidden_dim=cfg["hidden_dim"], n_gnn_layers=cfg["n_gnn_layers"],
        n_gnn_heads=cfg["n_gnn_heads"], n_dyn_layers=cfg["n_dyn_layers"],
        n_dyn_heads=cfg["n_dyn_heads"], max_seq=cfg["max_seq"],
        dyn_residual=cfg.get("dyn_residual", True),
        dyn_heteroscedastic=cfg.get("dyn_heteroscedastic", False),
    ).to(device)
    load_weights_only(model, args.checkpoint, device=device)
    print(f"Loaded {args.checkpoint} on {device}")

    all_paths = list_trajectory_files(args.data_root)
    tr_files, val_files, _ = split_files(all_paths, 0.8, 0.1, 0.1, stratify=True)

    def cap_files(files, cap):
        """Cap file count but KEEP every SC-hard file (the starved tier), then
        fill the remainder with a deterministic sample of easy/medium."""
        if not cap or cap <= 0 or len(files) <= cap:
            return list(files)
        rng = np.random.default_rng(0)
        hard = [f for f in files if _tier_of(Path(f)) == "hard"]
        rest = [f for f in files if _tier_of(Path(f)) != "hard"]
        rng.shuffle(rest)
        keep = hard + rest[:max(0, cap - len(hard))]
        rng.shuffle(keep)
        return keep

    tr_files = cap_files(tr_files, args.max_train_files)
    val_files = cap_files(val_files, args.max_val_files)

    # Hard-tier oversampling: repeat SC-hard trajectory files so their nodes are
    # relabeled and trained more often (that tier is data-starved and drives the
    # hard-instance tree explosions). File-level repetition keeps ds_idx stable.
    if args.oversample_hard > 1:
        mult = int(round(args.oversample_hard))
        extra = [f for f in tr_files if _tier_of(Path(f)) == "hard"] * (mult - 1)
        tr_files = list(tr_files) + extra
        print(f"Oversampled hard tier x{mult}: +{len(extra)} file-instances")
    n_hard = sum(_tier_of(Path(f)) == "hard" for f in tr_files)
    print(f"Files: {len(tr_files)} train ({n_hard} hard) | {len(val_files)} val")

    tr_ds = IndexedTransition(tr_files)
    val_ds = IndexedTransition(val_files)

    def make_loader(ds, files, shuffle):
        item_files = [fi for (fi, _t) in ds.index]
        sampler = ShardedBatchSampler(
            item_files, batch_size=args.batch_size,
            files_per_batch=args.files_per_batch, shuffle=shuffle,
            file_node_cost=probe_edge_cost_per_node(files))
        return DataLoader(ds, batch_sampler=sampler,
                          collate_fn=transition_collate, num_workers=0)

    relabel_loader = make_loader(tr_ds, tr_files, shuffle=False)
    val_loader = make_loader(val_ds, val_files, shuffle=False)

    # which parameters to train
    for name, p in model.named_parameters():
        p.requires_grad = ("policy" in name) or (
            args.unfreeze_encoder and "encoder" in name)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,} "
          f"(encoder {'unfrozen' if args.unfreeze_encoder else 'frozen'})")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    best_val = -1.0
    for rnd in range(1, args.rounds + 1):
        print(f"\n=== ExIt round {rnd}/{args.rounds} ===")
        print("Relabeling with world-model rollout ...")
        targets = relabel(
            model, relabel_loader, device, topk=args.topk, depth=args.depth,
            gamma=args.gamma, temp=args.temp, ctg_weight=args.ctg_weight,
            branch_factor=args.branch_factor,
            use_reward_return=bool(args.use_reward_return))
        val_targets = relabel(
            model, val_loader, device, topk=args.topk, depth=args.depth,
            gamma=args.gamma, temp=args.temp, ctg_weight=args.ctg_weight,
            branch_factor=args.branch_factor,
            use_reward_return=bool(args.use_reward_return))
        print(f"  relabeled {len(targets)} train / {len(val_targets)} val nodes")

        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
        # hard-tier oversampling: rebuild the train loader's sampler weighting.
        train_loader = make_loader(tr_ds, tr_files, shuffle=True)

        for ep in range(1, args.epochs + 1):
            tl, ta = distill_epoch(model, train_loader, targets, optimizer,
                                   device, args.alpha, args.temp, training=True)
            vl, va = distill_epoch(model, val_loader, val_targets, None,
                                   device, args.alpha, args.temp, training=False)
            print(f"  round {rnd} ep {ep} | train_loss={tl:.4f} acc={ta:.3f} "
                  f"| val_loss={vl:.4f} acc={va:.3f}")
            if va > best_val:
                best_val = va
                save_checkpoint(model, optimizer, ep, {"exit_val_acc": va},
                                Path(args.out_dir) / "model_exit_best.pt")
                print("    saved model_exit_best.pt")
        save_checkpoint(model, optimizer, rnd, {"round": rnd},
                        Path(args.out_dir) / f"model_exit_r{rnd}.pt")

    print(f"\nDone. Best ExIt val acc: {best_val:.4f}")
    print("Benchmark model_exit_best.pt against model_final.pt with ablation.py.")


if __name__ == "__main__":
    main()
