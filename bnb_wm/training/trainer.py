"""
trainer.py — Training and validation loops for all four phases.

Phase 1 : Policy head — imitation learning from strong branching.
Phase 2 : Value head  — dual bound regression (encoder + policy frozen).
Phase 3 : Dynamics    — latent transition prediction (encoder frozen).
Phase 4 : Joint       — end-to-end fine-tuning of all components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from collections import defaultdict
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from .losses import policy_loss_masked, value_loss as _value_loss, integrality_loss
from .checkpoint import save_checkpoint


# ---------------------------------------------------------------------------
# Batch runner helpers
# ---------------------------------------------------------------------------

def _run_policy_batch(model, batch, device):
    """Forward pass + policy loss for one transition batch."""
    pyg_batch, metas = batch
    pyg_batch = pyg_batch.to(device)

    scores, z = model(pyg_batch)

    losses = []
    top1 = 0
    offset = 0

    for meta in metas:
        n_v    = meta["n_vars"]
        logits = scores[offset : offset + n_v]
        aset   = meta["action_set"].to(device)
        lbl    = meta["local_label"]

        loss, acc, _ = policy_loss_masked(logits, aset, lbl)
        losses.append(loss)
        top1 += acc
        offset += n_v

    return torch.stack(losses).mean(), top1 / len(metas)


def _run_value_batch(model, batch, device):
    """Forward pass + value loss for one transition batch."""
    pyg_batch, metas = batch
    pyg_batch = pyg_batch.to(device)

    _, z = model(pyg_batch)

    targets = torch.tensor(
        [m["norm_db"] for m in metas], dtype=torch.float32, device=device
    )
    preds = model.value_pred(z).squeeze(-1)

    return _value_loss(preds, targets), preds.detach().cpu(), targets.cpu()


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------

class Trainer:
    """
    Unified trainer for all four training phases.

    Args:
        model      : BnBWorldModel
        device     : torch.device
        ckpt_dir   : Path — where to save checkpoints
        amp        : bool — enable automatic mixed precision (recommended on GPU)
    """

    def __init__(self, model, device, ckpt_dir, amp=True):
        self.model    = model
        self.device   = device
        self.ckpt_dir = ckpt_dir
        self.amp      = amp and (device.type == "cuda")
        self.scaler   = GradScaler("cuda", enabled=self.amp)
        self.history  = defaultdict(list)

    # ------------------------------------------------------------------
    # Phase 1 — Policy
    # ------------------------------------------------------------------
    def train_policy(self, train_loader, val_loader, epochs, lr=1e-3):
        """
        Imitation learning: train policy head to mimic strong branching.
        All model parameters are trainable.
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )
        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._epoch_policy(
                train_loader, optimizer, training=True
            )
            val_loss, val_acc = self._epoch_policy(val_loader, None, training=False)

            scheduler.step()

            self.history["p1_train_loss"].append(train_loss)
            self.history["p1_train_acc"].append(train_acc)
            self.history["p1_val_loss"].append(val_loss)
            self.history["p1_val_acc"].append(val_acc)

            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"[Phase1] Epoch {epoch:02d} | "
                f"TrainAcc={train_acc:.3f} | ValAcc={val_acc:.3f} | "
                f"ValLoss={val_loss:.3f} | LR={lr_now:.1e}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(
                    self.model, optimizer, epoch,
                    {"val_acc": val_acc},
                    self.ckpt_dir / "phase1_best.pt",
                )
                print("  ✅ Saved best Phase 1 model")

        save_checkpoint(
            self.model, optimizer, epochs, {}, self.ckpt_dir / "phase1_final.pt"
        )
        print(f"\nBest Val Acc (Phase 1): {best_val_acc:.4f}")

    def _epoch_policy(self, loader, optimizer, training):
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = total_acc = n = 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            pbar = tqdm(loader, desc="Train" if training else "Val", leave=False)
            for batch in pbar:
                if training:
                    optimizer.zero_grad(set_to_none=True)

                with autocast("cuda", enabled=self.amp):
                    loss, acc = _run_policy_batch(self.model, batch, self.device)

                if training:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()

                total_loss += loss.item()
                total_acc  += acc
                n += 1
                pbar.set_postfix(loss=f"{total_loss/n:.3f}", acc=f"{total_acc/n:.3f}")

        return total_loss / n, total_acc / n

    # ------------------------------------------------------------------
    # Phase 2 — Value
    # ------------------------------------------------------------------
    def train_value(self, train_loader, val_loader, epochs, lr=5e-4):
        """
        Train value head with encoder + policy frozen.
        """
        for name, param in self.model.named_parameters():
            param.requires_grad = "encoder" not in name and "policy" not in name

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        print(f"Trainable params (Phase 2): {sum(p.numel() for p in trainable):,}")

        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )
        best_spearman = -1.0

        for epoch in range(1, epochs + 1):
            train_loss = self._epoch_value_train(train_loader, optimizer)
            spearman_r = self._epoch_value_val(val_loader)

            scheduler.step()

            self.history["p2_train_loss"].append(train_loss)
            self.history["p2_val_spearman"].append(spearman_r)

            print(
                f"[Phase2] Epoch {epoch:02d} | "
                f"TrainLoss={train_loss:.4f} | ValSpearman={spearman_r:.3f} | "
                f"LR={optimizer.param_groups[0]['lr']:.1e}"
            )

            if spearman_r > best_spearman:
                best_spearman = spearman_r
                save_checkpoint(
                    self.model, optimizer, epoch,
                    {"val_spearman": spearman_r},
                    self.ckpt_dir / "phase2_best.pt",
                )
                print("  ✅ Saved best Phase 2 model")

        save_checkpoint(
            self.model, optimizer, epochs, {}, self.ckpt_dir / "phase2_final.pt"
        )
        print(f"\nBest Spearman (Phase 2): {best_spearman:.4f}")

        # Unfreeze all params for subsequent phases
        for p in self.model.parameters():
            p.requires_grad = True

    def _epoch_value_train(self, loader, optimizer):
        self.model.train()
        total_loss = n = 0

        for batch in tqdm(loader, desc="Value Train", leave=False):
            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=self.amp):
                loss, _, _ = _run_value_batch(self.model, batch, self.device)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.0
            )
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n += 1

        return total_loss / n

    def _epoch_value_val(self, loader):
        self.model.eval()
        preds_all, tgts_all = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Value Val", leave=False):
                _, preds, tgts = _run_value_batch(self.model, batch, self.device)
                preds_all.extend(preds.numpy().tolist())
                tgts_all.extend(tgts.numpy().tolist())

        r, _ = spearmanr(preds_all, tgts_all)
        return float(r)

    # ------------------------------------------------------------------
    # Phase 4 — Joint fine-tuning
    # ------------------------------------------------------------------
    def train_joint(self, train_loader, val_loader, epochs, lr=1e-4,
                    pos_weight=None):
        """
        End-to-end fine-tuning: policy + value + integrality losses combined.
        """
        for p in self.model.parameters():
            p.requires_grad = True

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )

        pw = pos_weight.to(self.device) if pos_weight is not None else None
        bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = total_acc = n = 0

            for batch in tqdm(train_loader, desc=f"Joint Epoch {epoch}", leave=False):
                pyg_batch, metas = batch
                pyg_batch = pyg_batch.to(self.device)

                optimizer.zero_grad(set_to_none=True)

                with autocast("cuda", enabled=self.amp):
                    scores, z = self.model(pyg_batch)

                    p_losses, top1 = [], 0
                    offset = 0
                    for meta in metas:
                        n_v = meta["n_vars"]
                        logits = scores[offset : offset + n_v]
                        aset = meta["action_set"].to(self.device)
                        ploss, acc, _ = policy_loss_masked(logits, aset, meta["local_label"])
                        p_losses.append(ploss)
                        top1 += acc
                        offset += n_v

                    p_loss = torch.stack(p_losses).mean()

                    targets_v = torch.tensor(
                        [m["norm_db"] for m in metas], dtype=torch.float32, device=self.device
                    )
                    v_loss = _value_loss(self.model.value_pred(z), targets_v)

                    targets_i = torch.tensor(
                        [m["is_leaf"] for m in metas], dtype=torch.float32, device=self.device
                    )
                    i_logit = self.model.integrality_logit(z)
                    i_loss = bce(i_logit.squeeze(), targets_i)

                    loss = p_loss + 0.5 * v_loss + 0.1 * i_loss

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()

                total_loss += loss.item()
                total_acc  += top1 / len(metas)
                n += 1

            scheduler.step()
            train_loss = total_loss / n
            train_acc  = total_acc  / n

            _, val_acc = self._epoch_policy(val_loader, None, training=False)

            self.history["p4_train_loss"].append(train_loss)
            self.history["p4_val_acc"].append(val_acc)

            print(
                f"[Phase4] Epoch {epoch:02d} | "
                f"TotalLoss={train_loss:.4f} | ValAcc={val_acc:.3f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(
                    self.model, optimizer, epoch,
                    {"val_acc": val_acc},
                    self.ckpt_dir / "phase4_best.pt",
                )
                print("  ✅ Saved best Phase 4 model")

        save_checkpoint(
            self.model, optimizer, epochs, {}, self.ckpt_dir / "phase4_final.pt"
        )
