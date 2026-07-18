"""
trainer.py — Training and validation loops for all five phases.

Phase 1 : Policy head     — imitation learning from strong branching.
Phase 2 : Value head      — dual bound regression (encoder + policy frozen).
Phase 3 : Dynamics        — latent transition prediction (encoder frozen).
Phase 4 : Joint           — end-to-end fine-tuning of all components.
Phase 5 : Cut selection   — cut imitation from SCIP (encoder frozen).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from collections import defaultdict
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from .losses import (
    policy_loss_masked,
    value_loss as _value_loss,
    integrality_loss,
    dynamics_loss as _dynamics_loss,
    var_reconstruction_loss as _var_recon_loss,
    subtree_size_loss as _subtree_size_loss,
    cost_to_go_loss as _cost_to_go_loss,
    cutting_plane_loss,
)
from .checkpoint import save_checkpoint


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _var_mask_and_batch(pyg_batch):
    """Return (var_mask, batch_vec_for_vars) from a PyG batch."""
    var_mask  = pyg_batch.node_type == 0
    batch_vec = pyg_batch.batch[var_mask]
    return var_mask, batch_vec


def _frac_mask_from_features(x_var: torch.Tensor) -> torch.Tensor | None:
    """
    Fractional variable mask from Ecole NodeBipartite variable features.

    Ecole feature layout (19-dim):
        index 13 = sol_val   (LP solution value)
        index 14 = sol_frac  (|sol_val - round(sol_val)|, pre-computed by Ecole)

    Using sol_frac directly (index 14) is preferred; fall back to computing
    from sol_val (index 13) if the tensor is narrower than 15 columns.
    """
    if x_var.size(1) > 14:
        return x_var[:, 14] > 0.05          # sol_frac pre-computed by Ecole
    if x_var.size(1) > 13:
        lp_vals = x_var[:, 13]              # sol_val
        return (lp_vals - lp_vals.round()).abs() > 0.05
    return None


def _run_policy_batch(model, batch, device):
    """Forward pass + policy loss for one transition batch."""
    pyg_batch, metas = batch
    pyg_batch = pyg_batch.to(device)

    scores, z = model(pyg_batch)

    losses, top1 = [], 0
    offset = 0
    for meta in metas:
        n_v    = meta["n_vars"]
        logits = scores[offset : offset + n_v]
        aset   = meta["action_set"].to(device)
        lbl    = meta["local_label"]
        loss, acc, _ = policy_loss_masked(logits, aset, lbl)
        losses.append(loss)
        top1  += acc
        offset += n_v

    return torch.stack(losses).mean(), top1 / len(metas)


def _run_value_batch(model, batch, device):
    """Forward pass + value loss for one transition batch."""
    pyg_batch, metas = batch
    pyg_batch = pyg_batch.to(device)

    h_vars, z      = model.encode(pyg_batch)
    var_mask, bvec = _var_mask_and_batch(pyg_batch)
    frac_mask      = _frac_mask_from_features(pyg_batch.x[var_mask])

    targets = torch.tensor(
        [m["norm_db"] for m in metas], dtype=torch.float32, device=device
    )
    preds = model.value_pred(z, h_vars, bvec, frac_mask)

    return _value_loss(preds, targets), preds.detach().cpu(), targets.cpu()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Unified trainer for all five training phases.

    Args:
        model    : BnBWorldModel
        device   : torch.device
        ckpt_dir : Path — where to save checkpoints
        amp      : bool — enable AMP (recommended on GPU)
    """

    def __init__(self, model, device, ckpt_dir, amp=True):
        self.model    = model
        self.device   = device
        self.ckpt_dir = ckpt_dir
        self.amp      = amp and (device.type == "cuda")
        self.scaler   = GradScaler("cuda", enabled=self.amp)
        self.history  = defaultdict(list)
        # Latent-overshooting horizon for Phase 3 (0 = one-step teacher forcing
        # only). Set by train_dynamics from config.
        self.overshoot_depth = 0

    # ------------------------------------------------------------------
    # Phase 1 — Policy
    # ------------------------------------------------------------------
    def train_policy(self, train_loader, val_loader, epochs, lr=1e-3):
        """Imitation learning: all params trainable."""
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
            val_loss, val_acc = self._epoch_policy(
                val_loader, None, training=False
            )
            scheduler.step()

            self.history["p1_train_loss"].append(train_loss)
            self.history["p1_train_acc"].append(train_acc)
            self.history["p1_val_acc"].append(val_acc)

            print(
                f"[Phase1] Epoch {epoch:02d} | "
                f"TrainAcc={train_acc:.3f} | ValAcc={val_acc:.3f} | "
                f"ValLoss={val_loss:.3f} | LR={optimizer.param_groups[0]['lr']:.1e}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(
                    self.model, optimizer, epoch,
                    {"val_acc": val_acc},
                    self.ckpt_dir / "phase1_best.pt",
                )
                print("  Saved best Phase 1 model")

        save_checkpoint(
            self.model, optimizer, epochs, {}, self.ckpt_dir / "phase1_final.pt"
        )
        print(f"\nBest Val Acc (Phase 1): {best_val_acc:.4f}")

    def _epoch_policy(self, loader, optimizer, training):
        self.model.train() if training else self.model.eval()
        total_loss = total_acc = n = 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for batch in tqdm(loader, desc="Train" if training else "Val", leave=False):
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

        return total_loss / n, total_acc / n

    # ------------------------------------------------------------------
    # Phase 2 — Value
    # ------------------------------------------------------------------
    def train_value(self, train_loader, val_loader, epochs, lr=5e-4):
        """Train value head with encoder + policy frozen."""
        for name, p in self.model.named_parameters():
            p.requires_grad = "value" in name

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
                print("  Saved best Phase 2 model")

        save_checkpoint(
            self.model, optimizer, epochs, {}, self.ckpt_dir / "phase2_final.pt"
        )
        print(f"\nBest Spearman (Phase 2): {best_spearman:.4f}")
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
    # Phase 3 — Dynamics
    # ------------------------------------------------------------------
    def _dynamics_batch_loss(self, batch):
        """
        Compute the Phase-3 dynamics loss for one batch.

        Accepts either a tuple (legacy) or a dict (extensible). All terms
        beyond the base latent-transition loss activate only when their inputs
        are present, so the same code path serves whatever the loader supplies.

        Tuple forms (backward compatible):
            (z_seq, a_seq, z_next_seq)
            (z_seq, a_seq, z_next_seq, hv_seq, hv_next_seq, var_mask)

        Dict form (preferred) keys:
            z_seq, a_seq, z_next_seq                 (required)
            hv_seq, hv_next_seq, var_mask            (optional: per-var recon)
            bound_next_seq                           (optional: Gap-2 grounding)

        Loss terms:
            latent transition   (always)
            per-variable recon  (if hv_* present)   — enables the rollout
            grounded dual bound (if bound present)  — Gap 2, anchors the latent
        """
        # Normalise to a dict.
        if isinstance(batch, dict):
            d = batch
        elif len(batch) == 3:
            d = dict(zip(("z_seq", "a_seq", "z_next_seq"), batch))
        else:
            d = dict(zip(
                ("z_seq", "a_seq", "z_next_seq", "hv_seq", "hv_next_seq",
                 "var_mask"),
                batch,
            ))

        z_seq      = d["z_seq"].to(self.device)
        a_seq      = d["a_seq"].to(self.device)
        z_next_seq = d["z_next_seq"].to(self.device)

        has_vars = d.get("hv_seq") is not None
        if has_vars:
            hv_seq = d["hv_seq"].to(self.device)
            z_pred, hv_pred = self.model.dynamics.forward_with_vars(
                z_seq, a_seq, hv_seq
            )
        else:
            z_pred = self.model.dynamics_forward(z_seq, a_seq)

        # Time-padding mask (present when batching variable-length trajectories).
        tmask = d.get("time_mask")
        if tmask is not None:
            tmask = tmask.to(self.device)

        if tmask is None:
            loss = _dynamics_loss(z_pred, z_next_seq)
        else:
            # Masked MSE over valid time positions only.
            m = tmask.unsqueeze(-1).float()                 # [B, T, 1]
            denom = m.sum().clamp_min(1.0) * z_pred.size(-1)
            loss = ((z_pred - z_next_seq) ** 2 * m).sum() / denom

        if has_vars:
            # var_mask already spans only valid time positions.
            loss = loss + _var_recon_loss(
                hv_pred,
                d["hv_next_seq"].to(self.device),
                d["var_mask"].to(self.device),
            )

        # Latent overshooting (Technique 1): unroll the dynamics autoregressively
        # from z_0, feeding its own predictions back in, and supervise each
        # predicted step against the real future latent. Trains the model in the
        # same compounding regime it faces during the inference rollout.
        if self.overshoot_depth and self.overshoot_depth > 0:
            k = min(self.overshoot_depth, a_seq.size(1))
            if k > 0:
                preds = self.model.dynamics.rollout(
                    z_seq[:, 0], a_seq[:, :k]
                )                                        # [B, k, H]
                tgt = z_next_seq[:, :k]                   # real z_1 .. z_k
                if tmask is not None:
                    om = tmask[:, :k].unsqueeze(-1).float()
                    denom = om.sum().clamp_min(1.0) * preds.size(-1)
                    loss = loss + ((preds - tgt) ** 2 * om).sum() / denom
                else:
                    loss = loss + F.mse_loss(preds, tgt)

        # Gap 2: ground the predicted latent against the next real dual bound.
        if d.get("bound_next_seq") is not None:
            bound_pred = self.model.dynamics_bound_pred(z_pred)   # [B, T]
            bound_tgt  = d["bound_next_seq"].to(self.device)
            if tmask is None:
                loss = loss + 0.5 * F.huber_loss(bound_pred, bound_tgt, delta=1.0)
            else:
                per = F.huber_loss(bound_pred, bound_tgt, delta=1.0,
                                   reduction="none")
                loss = loss + 0.5 * (per * tmask.float()).sum() / \
                    tmask.float().sum().clamp_min(1.0)

        # Fix 3: train the reward head to predict the per-step dual-bound
        # improvement, so the MuZero-style rollout return is grounded.
        if d.get("reward_seq") is not None:
            r_pred = self.model.dynamics_reward_pred(z_pred)      # [B, T]
            r_tgt  = d["reward_seq"].to(self.device)
            if tmask is None:
                loss = loss + 0.5 * F.huber_loss(r_pred, r_tgt, delta=1.0)
            else:
                per = F.huber_loss(r_pred, r_tgt, delta=1.0, reduction="none")
                loss = loss + 0.5 * (per * tmask.float()).sum() / \
                    tmask.float().sum().clamp_min(1.0)

        return loss

    def train_dynamics(self, train_loader, val_loader, epochs, lr=5e-4,
                       overshoot_depth=0):
        """
        Train DynamicsTransformer on pre-computed trajectory sequences.

        The encoder is frozen. Each batch yields:
            z_seq      : [B, T, H]  encoder outputs along trajectory
            a_seq      : [B, T, H]  action embeddings (branching var h_vars)
            z_next_seq : [B, T, H]  true next encoder outputs (targets)

        The SequenceDataset is responsible for pre-computing z and a
        values using the frozen encoder from Phase 1/2.

        If the loader also yields per-variable embedding sequences, the
        dynamics model's per-variable head is supervised jointly. This is
        what keeps predicted future h_vars on the real-encoder manifold, so
        the policy can be re-run on predicted states during the latent
        rollout without distribution drift. Loader batch forms supported:

            (z_seq, a_seq, z_next_seq)
            (z_seq, a_seq, z_next_seq, hv_seq, hv_next_seq, var_mask)
        """
        self.overshoot_depth = overshoot_depth
        for name, p in self.model.named_parameters():
            p.requires_grad = "dynamics" in name

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        print(f"Trainable params (Phase 3): {sum(p.numel() for p in trainable):,}"
              f" | overshoot_depth={overshoot_depth}")

        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = n = 0

            for batch in tqdm(
                train_loader, desc=f"Dyn Train Epoch {epoch}", leave=False
            ):
                optimizer.zero_grad(set_to_none=True)
                with autocast("cuda", enabled=self.amp):
                    loss = self._dynamics_batch_loss(batch)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()

                total_loss += loss.item()
                n += 1

            # Validation
            self.model.eval()
            val_loss_sum = val_n = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Dyn Val", leave=False):
                    val_loss_sum += self._dynamics_batch_loss(batch).item()
                    val_n += 1

            train_loss = total_loss / n
            val_loss   = val_loss_sum / val_n if val_n > 0 else float("inf")
            scheduler.step()

            self.history["p3_train_loss"].append(train_loss)
            self.history["p3_val_loss"].append(val_loss)

            print(
                f"[Phase3] Epoch {epoch:02d} | "
                f"TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    self.model, optimizer, epoch,
                    {"val_loss": val_loss},
                    self.ckpt_dir / "phase3_best.pt",
                )
                print("  Saved best Phase 3 model")

        save_checkpoint(
            self.model, optimizer, epochs, {}, self.ckpt_dir / "phase3_final.pt"
        )
        for p in self.model.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    # Phase 4 — Joint fine-tuning
    # ------------------------------------------------------------------
    def train_joint(self, train_loader, val_loader, epochs, lr=1e-4,
                    pos_weight=None):
        """End-to-end fine-tuning: policy + value + integrality losses."""
        for p in self.model.parameters():
            p.requires_grad = True

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        pw = pos_weight.to(self.device) if pos_weight is not None else None
        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = total_acc = n = 0

            for batch in tqdm(train_loader, desc=f"Joint Epoch {epoch}", leave=False):
                pyg_batch, metas = batch
                pyg_batch = pyg_batch.to(self.device)

                optimizer.zero_grad(set_to_none=True)

                with autocast("cuda", enabled=self.amp):
                    h_vars, z      = self.model.encode(pyg_batch)
                    var_mask, bvec = _var_mask_and_batch(pyg_batch)
                    frac_mask      = _frac_mask_from_features(pyg_batch.x[var_mask])

                    # Broadcast z to each variable for Pointer Network
                    z_per_var = z[pyg_batch.batch[var_mask]]
                    scores    = self.model.policy(h_vars, z_per_var)

                    # Policy loss (and collect expert-chosen var embeddings)
                    p_losses, top1 = [], 0
                    offset = 0
                    chosen_idx = []
                    for meta in metas:
                        n_v    = meta["n_vars"]
                        logits = scores[offset : offset + n_v]
                        aset   = meta["action_set"].to(self.device)
                        ploss, acc, _ = policy_loss_masked(
                            logits, aset, meta["local_label"]
                        )
                        p_losses.append(ploss)
                        top1  += acc
                        chosen_idx.append(offset + int(aset[meta["local_label"]]))
                        offset += n_v
                    p_loss = torch.stack(p_losses).mean()

                    # Value loss (on real encoder latents)
                    targets_v = torch.tensor(
                        [m["norm_db"] for m in metas],
                        dtype=torch.float32, device=self.device,
                    )
                    v_pred_real = self.model.value_pred(z, h_vars, bvec, frac_mask)
                    v_loss = _value_loss(v_pred_real, targets_v)

                    # Value consistency on dynamics-PREDICTED latents.
                    # Roll one dynamics step forward from the expert action and
                    # require the value head to read the predicted latent the
                    # same way it reads the real one. This trains the value head
                    # on its own distribution, removing the OOD gap it would
                    # otherwise face during the latent rollout at inference.
                    a_chosen = h_vars[torch.tensor(chosen_idx, device=self.device)]
                    z_pred1, _ = self.model.dynamics_step(z, a_chosen)
                    bvec_g   = torch.zeros(z.size(0), dtype=torch.long, device=self.device)
                    v_on_pred = self.model.value_pred(z_pred1, z_pred1, bvec_g, None)
                    v_consist = F.mse_loss(v_on_pred, v_pred_real.detach())

                    # Integrality loss
                    targets_i = torch.tensor(
                        [m["is_leaf"] for m in metas],
                        dtype=torch.float32, device=self.device,
                    )
                    depth = torch.tensor(
                        [m.get("depth", 0) for m in metas],
                        dtype=torch.float32, device=self.device,
                    )
                    n_frac = torch.tensor(
                        [m.get("n_frac", 0) for m in metas],
                        dtype=torch.float32, device=self.device,
                    )
                    i_logit = self.model.integrality_logit(z, depth, n_frac)
                    i_loss  = integrality_loss(i_logit, targets_i, pw)

                    # Subtree-size loss (supervised on true node counts from the
                    # collected traces). Trained here so it shares the encoder
                    # with the value head. Skipped if the data lacks the target.
                    if all("subtree_size" in m for m in metas):
                        targets_s = torch.tensor(
                            [m["subtree_size"] for m in metas],
                            dtype=torch.float32, device=self.device,
                        )
                        s_pred = self.model.subtree_size_pred(
                            z, h_vars, bvec, frac_mask
                        )
                        s_loss = _subtree_size_loss(s_pred, targets_s)
                    else:
                        s_loss = torch.zeros((), device=self.device)

                    # Cost-to-go loss (Gap 3): Monte-Carlo return n_steps - t.
                    # Needs no DFS ordering, so it trains on the non-DFS traces.
                    if all("steps_to_go" in m for m in metas):
                        targets_c = torch.tensor(
                            [m["steps_to_go"] for m in metas],
                            dtype=torch.float32, device=self.device,
                        )
                        c_pred = self.model.cost_to_go_pred(
                            z, h_vars, bvec, frac_mask
                        )
                        c_loss = _cost_to_go_loss(c_pred, targets_c)
                    else:
                        c_loss = torch.zeros((), device=self.device)

                    loss = (
                        p_loss + 0.5 * v_loss + 0.1 * i_loss
                        + 0.1 * v_consist + 0.3 * s_loss + 0.5 * c_loss
                    )

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
                print("  Saved best Phase 4 model")

        save_checkpoint(
            self.model, optimizer, epochs, {}, self.ckpt_dir / "phase4_final.pt"
        )

    # ------------------------------------------------------------------
    # Phase 5 — Cut selection
    # ------------------------------------------------------------------
    def train_cuts(self, train_loader, val_loader, epochs, lr=5e-4,
                   pos_weight=None):
        """
        Train CuttingPlaneHead to imitate SCIP's cut selection.

        Encoder is frozen; only CuttingPlaneHead parameters are updated.

        Loader yields (pyg_batch, metas) where each meta contains:
            cut_features : Tensor [n_cuts, 6]   per-cut feature vectors
            cut_labels   : Tensor [n_cuts]       1 = cut selected by SCIP
                                                  and improved LP bound

        After training, all parameters are unfrozen for Phase 4 joint
        fine-tuning if it has not already been run.
        """
        for name, p in self.model.named_parameters():
            p.requires_grad = "cutting_planes" in name

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        print(f"Trainable params (Phase 5): {sum(p.numel() for p in trainable):,}")

        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )
        pw = pos_weight.to(self.device) if pos_weight is not None else None
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = n = 0

            for batch in tqdm(
                train_loader, desc=f"Cuts Train Epoch {epoch}", leave=False
            ):
                pyg_batch, metas = batch
                pyg_batch = pyg_batch.to(self.device)

                optimizer.zero_grad(set_to_none=True)

                with autocast("cuda", enabled=self.amp):
                    _, z = self.model.encode(pyg_batch)

                    cut_losses = []
                    for b_idx, meta in enumerate(metas):
                        cut_feats  = meta["cut_features"].to(self.device)   # [n_cuts, 6]
                        cut_labels = meta["cut_labels"].to(self.device)     # [n_cuts]
                        if cut_feats.size(0) == 0:
                            continue
                        scores = self.model.cut_scores(cut_feats, z[b_idx])
                        cut_losses.append(
                            cutting_plane_loss(scores, cut_labels, pw)
                        )

                    if not cut_losses:
                        continue
                    loss = torch.stack(cut_losses).mean()

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()

                total_loss += loss.item()
                n += 1

            # Validation
            self.model.eval()
            val_loss_sum = val_n = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Cuts Val", leave=False):
                    pyg_batch, metas = batch
                    pyg_batch = pyg_batch.to(self.device)
                    _, z = self.model.encode(pyg_batch)
                    for b_idx, meta in enumerate(metas):
                        cut_feats  = meta["cut_features"].to(self.device)
                        cut_labels = meta["cut_labels"].to(self.device)
                        if cut_feats.size(0) == 0:
                            continue
                        scores = self.model.cut_scores(cut_feats, z[b_idx])
                        val_loss_sum += cutting_plane_loss(
                            scores, cut_labels, pw
                        ).item()
                        val_n += 1

            train_loss = total_loss / max(n, 1)
            val_loss   = val_loss_sum / max(val_n, 1)
            scheduler.step()

            self.history["p5_train_loss"].append(train_loss)
            self.history["p5_val_loss"].append(val_loss)

            print(
                f"[Phase5] Epoch {epoch:02d} | "
                f"TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    self.model, optimizer, epoch,
                    {"val_loss": val_loss},
                    self.ckpt_dir / "phase5_best.pt",
                )
                print("  Saved best Phase 5 model")

        save_checkpoint(
            self.model, optimizer, epochs, {}, self.ckpt_dir / "phase5_final.pt"
        )
        for p in self.model.parameters():
            p.requires_grad = True
