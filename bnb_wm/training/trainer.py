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
    policy_loss_soft,
    value_loss as _value_loss,
    integrality_loss,
    dynamics_loss as _dynamics_loss,
    candidate_ranking_loss as _cand_rank_loss,
    var_reconstruction_loss as _var_recon_loss,
    subtree_size_loss as _subtree_size_loss,
    cost_to_go_loss as _cost_to_go_loss,
)
import json
from .checkpoint import save_checkpoint, load_weights_only


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_oom(exc: Exception) -> bool:
    """True if an exception is a CUDA out-of-memory error.

    Graph sizes vary a lot across instances (SC-hard nodes have far more edges),
    so an unlucky batch can transiently exceed GPU memory. Rather than let one
    such batch kill a multi-hour run, the training loops skip it (freeing its
    memory) and continue — a handful of skipped batches per epoch is harmless.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def _recover_oom(optimizer=None):
    """Free the failed batch's memory so the loop can continue."""
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def _run_policy_batch(model, batch, device, soft_alpha=0.5, soft_temp=1.0):
    """Forward pass + policy loss for one transition batch.

    Uses soft/ranking imitation (KL to the full SB score distribution, blended
    with hard CE) when the batch carries `sb_scores`; otherwise falls back to
    plain masked cross-entropy. `soft_alpha`/`soft_temp` tune the blend/target.
    """
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
        if "sb_scores" in meta:
            loss, acc, _ = policy_loss_soft(
                logits, aset, meta["sb_scores"].to(device), lbl,
                alpha=soft_alpha, temp=soft_temp)
        else:
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
    def train_policy(self, train_loader, val_loader, epochs, lr=1e-3,
                     patience=None):
        """Imitation learning: all params trainable."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )
        best_val_acc = 0.0
        no_improve = 0

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
                no_improve = 0
                save_checkpoint(
                    self.model, optimizer, epoch,
                    {"val_acc": val_acc},
                    self.ckpt_dir / "phase1_best.pt",
                )
                print("  Saved best Phase 1 model")
            else:
                no_improve += 1
                if patience and no_improve >= patience:
                    print(f"  Early stop at epoch {epoch} "
                          f"(no val improvement for {patience} epochs)")
                    break

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
                try:
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
                except RuntimeError as e:
                    if _is_oom(e):
                        _recover_oom(optimizer if training else None)
                        continue
                    raise

        # Guard empty loaders (e.g. a tiny stratified val split) — inf so an
        # empty val never masquerades as the best checkpoint.
        if n == 0:
            return float("inf"), 0.0
        return total_loss / n, total_acc / n

    # ------------------------------------------------------------------
    # Phase 2 — Value
    # ------------------------------------------------------------------
    def train_value(self, train_loader, val_loader, epochs, lr=5e-4,
                    patience=None):
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
        no_improve = 0

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
                no_improve = 0
                save_checkpoint(
                    self.model, optimizer, epoch,
                    {"val_spearman": spearman_r},
                    self.ckpt_dir / "phase2_best.pt",
                )
                print("  Saved best Phase 2 model")
            else:
                no_improve += 1
                if patience and no_improve >= patience:
                    print(f"  Early stop at epoch {epoch} "
                          f"(no val improvement for {patience} epochs)")
                    break

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
            try:
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
            except RuntimeError as e:
                if _is_oom(e):
                    _recover_oom(optimizer)
                    continue
                raise
        return total_loss / n if n else float("inf")

    def _epoch_value_val(self, loader):
        self.model.eval()
        preds_all, tgts_all = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Value Val", leave=False):
                try:
                    _, preds, tgts = _run_value_batch(self.model, batch, self.device)
                except RuntimeError as e:
                    if _is_oom(e):
                        _recover_oom()
                        continue
                    raise
                preds_all.extend(preds.numpy().tolist())
                tgts_all.extend(tgts.numpy().tolist())
        r, _ = spearmanr(preds_all, tgts_all)
        return float(r)

    # ------------------------------------------------------------------
    # Phase 3 — Dynamics
    # ------------------------------------------------------------------
    def _transition_loss(self, z_pred, z_next, logvar, tmask):
        """Base one-step transition loss: masked MSE+cosine, or Gaussian NLL.

        When tmask is given, both the MSE and cosine terms are restricted to
        real (non-padding) timesteps. Previously only MSE was masked here while
        cosine was dropped entirely in the masked path — padding zeros contributed
        cos=0 → penalty=1.0 (maximum) per pad step, diluting valid gradients.
        """
        if logvar is not None:
            logvar = logvar.clamp(-8.0, 8.0)     # numerical stability
            per = 0.5 * (torch.exp(-logvar) * (z_pred - z_next) ** 2 + logvar)
            if tmask is None:
                return per.mean()
            m = tmask.unsqueeze(-1).float()
            return (per * m).sum() / (m.sum().clamp_min(1.0) * z_pred.size(-1))
        # _dynamics_loss now accepts step_mask and masks both MSE and cosine.
        return _dynamics_loss(z_pred, z_next, step_mask=tmask)

    def _overshoot_and_ground(self, z_seq, a_seq, d_seq, z_next_seq, tmask,
                              bound_tgt):
        """Multi-anchor latent overshooting with dual-bound grounding.

        Rolls the dynamics forward `_overshoot_k` steps from several anchor
        positions and supervises the free-running predictions (and their
        predicted bound) against the real future. Returns a scalar loss (0 if
        overshooting is disabled).

        Vectorised over anchors: instead of A separate rollout() calls (each
        over B items), we do one call over B*A items so the Transformer sees a
        single large batch — MUCH better GPU utilisation when A >= 3.
        """
        k_max = int(getattr(self, "_overshoot_k", 0) or 0)
        if k_max <= 0:
            return z_seq.new_zeros(())
        B, T, H = z_seq.shape

        # Build anchor list (Fix E: randomised each call, always includes 0).
        n_random = int(getattr(self, "_overshoot_n_random_anchors", 2))
        fixed_set = {0, T // 3, (2 * T) // 3}
        if n_random > 0 and T > 2:
            import random
            random_anchors = random.sample(range(1, T - 1), min(n_random, T - 2))
            fixed_set.update(random_anchors)
        explicit = getattr(self, "_overshoot_anchors", None)
        raw = explicit if explicit is not None else sorted(fixed_set)
        anchors = [a for a in raw if a < T - 1]
        if not anchors:
            return z_seq.new_zeros(())
        A = len(anchors)

        # Scheduled sampling (Issue 6): p_free_run ramps 0→0.5 over Phase 3.
        p_free = float(getattr(self, "_p_free_run", 0.0))

        # --- build start states [B, A, H] ---
        # Default: real encoder states at each anchor.
        anc_t = torch.tensor(anchors, dtype=torch.long, device=self.device)
        z_starts = z_seq[:, anc_t]                               # [B, A, H]

        # Scheduled sampling: for anchor a0 > 0 stochastically replace with
        # model's own one-step prediction from the previous state.
        if p_free > 0.0:
            import random
            for ai, a0 in enumerate(anchors):
                if a0 > 0 and random.random() < p_free:
                    with torch.no_grad():
                        d_prev = d_seq[:, a0 - 1] if d_seq is not None else None
                        z_free, _, _ = self.model.dynamics.step(
                            z_seq[:, a0 - 1], a_seq[:, a0 - 1],
                            past_tokens=None, d_t=d_prev)
                    z_starts[:, ai] = z_free.detach()

        # --- build action/direction buffers [B, A, k_max, …] (zero-padded) ---
        a_buf = torch.zeros(B, A, k_max, H, device=self.device)
        d_buf = (torch.zeros(B, A, k_max, device=self.device)
                 if d_seq is not None else None)
        kk_list = []
        for ai, a0 in enumerate(anchors):
            kk = min(k_max, T - a0)
            kk_list.append(kk)
            a_buf[:, ai, :kk] = a_seq[:, a0:a0 + kk]
            if d_seq is not None:
                d_buf[:, ai, :kk] = d_seq[:, a0:a0 + kk]

        # Reshape [B, A, k_max, H] → [B*A, k_max, H] for one batched rollout.
        z_starts_flat = z_starts.permute(1, 0, 2).reshape(B * A, H)
        a_buf_flat    = a_buf.permute(1, 0, 2, 3).reshape(B * A, k_max, H)
        d_buf_flat    = (d_buf.permute(1, 0, 2).reshape(B * A, k_max)
                         if d_buf is not None else None)

        preds_flat = self.model.dynamics.rollout(
            z_starts_flat, a_buf_flat, d_seq=d_buf_flat)         # [B*A, k_max, H]
        # Restore to [B, A, k_max, H]
        preds = preds_flat.reshape(A, B, k_max, H).permute(1, 0, 2, 3)

        # --- accumulate loss per anchor (slice to each anchor's kk) ---
        total = z_seq.new_zeros(())
        for ai, (a0, kk) in enumerate(zip(anchors, kk_list)):
            p_ai = preds[:, ai, :kk]                             # [B, kk, H]
            tgt  = z_next_seq[:, a0:a0 + kk]                    # [B, kk, H]
            if tmask is not None:
                om = tmask[:, a0:a0 + kk].unsqueeze(-1).float()
                total = total + ((p_ai - tgt) ** 2 * om).sum() / \
                    (om.sum().clamp_min(1.0) * H)
            else:
                total = total + F.mse_loss(p_ai, tgt)
            if bound_tgt is not None:
                bp = self.model.dynamics_bound_pred(p_ai)        # [B, kk]
                bt = bound_tgt[:, a0:a0 + kk]
                if tmask is not None:
                    om2 = tmask[:, a0:a0 + kk].float()
                    per = F.huber_loss(bp, bt, delta=1.0, reduction="none")
                    total = total + 0.5 * (per * om2).sum() / \
                        om2.sum().clamp_min(1.0)
                else:
                    total = total + 0.5 * F.huber_loss(bp, bt, delta=1.0)

        return total / A

    @torch.no_grad()
    def dynamics_rollout_diagnostic(self, loader, max_depth=8, save_path=None):
        """
        k-step rollout drift diagnostic (multi-step-lookahead health check).

        Free-runs the dynamics from each sequence's start — feeding predictions
        back in — and measures, at every rollout depth k, how far the predicted
        latent has drifted from the real one and how wrong the decoded dual bound
        has become. A model with good multi-step lookahead keeps both roughly
        flat as k grows; a drifting model's curves blow up. This is the single
        number that tells us whether the residual/overshoot/grounding upgrades
        actually flattened the drift — and it doubles as a paper figure.

        Returns (and optionally saves as JSON) a dict:
            depth       : [1..K]
            latent_mse  : mean squared error of z_hat_k vs real z_k, per depth
            bound_mae   : mean |bound_pred(z_hat_k) - real bound|, per depth
            count       : number of (masked) samples contributing at each depth
        """
        self.model.eval()
        K = int(max_depth)
        lat_se = [0.0] * K   # summed latent squared error per depth
        bnd_ae = [0.0] * K   # summed bound absolute error per depth
        cnt    = [0.0] * K   # sample counts per depth

        for batch in loader:
            d = batch if isinstance(batch, dict) else dict(zip(
                ("z_seq", "a_seq", "z_next_seq"), batch))
            z_seq      = d["z_seq"].to(self.device)
            a_seq      = d["a_seq"].to(self.device)
            z_next_seq = d["z_next_seq"].to(self.device)
            d_seq = d.get("dir_seq")
            d_seq = d_seq.to(self.device) if d_seq is not None else None
            tmask = d.get("time_mask")
            tmask = tmask.to(self.device) if tmask is not None else None
            bnd_t = d.get("bound_next_seq")
            bnd_t = bnd_t.to(self.device) if bnd_t is not None else None

            T = z_seq.size(1)
            k = min(K, T)
            if k <= 0:
                continue
            preds = self.model.dynamics.rollout(
                z_seq[:, 0], a_seq[:, :k],
                d_seq=d_seq[:, :k] if d_seq is not None else None)   # [B, k, H]

            for j in range(k):
                m = (tmask[:, j].float() if tmask is not None
                     else torch.ones(z_seq.size(0), device=self.device))
                lat = ((preds[:, j] - z_next_seq[:, j]) ** 2).mean(-1)   # [B]
                lat_se[j] += float((lat * m).sum().item())
                cnt[j]    += float(m.sum().item())
                if bnd_t is not None:
                    bp = self.model.dynamics_bound_pred(preds[:, j])     # [B]
                    bnd_ae[j] += float(((bp - bnd_t[:, j]).abs() * m).sum().item())

        depth = list(range(1, K + 1))
        latent_mse = [lat_se[j] / cnt[j] if cnt[j] > 0 else float("nan")
                      for j in range(K)]
        bound_mae = [bnd_ae[j] / cnt[j] if cnt[j] > 0 else float("nan")
                     for j in range(K)]
        out = {"depth": depth, "latent_mse": latent_mse,
               "bound_mae": bound_mae, "count": cnt}

        print("\n[Phase3] k-step rollout drift (free-running):")
        print("  depth |  latent_mse |  bound_mae |   n")
        for j in range(K):
            print(f"  {depth[j]:5d} | {latent_mse[j]:11.5f} | "
                  f"{bound_mae[j]:10.5f} | {int(cnt[j])}")
        if save_path is not None:
            with open(save_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"  saved -> {save_path}")
        return out

    def _online_encode_raw_batch(self, raw):
        """
        Convert a RawSequenceDataset batch (batch_graphs + branch_vars) into
        the standard dict form that _dynamics_batch_loss expects.

        Called only when also_train_encoder=True.  Runs WITH gradients so that
        encoder parameters receive gradients through z_seq / z_next_seq.

        Fully vectorised: no Python loops over sequences or timesteps —
        uses cumulative offsets + fancy indexing to scatter z_all / h_vars_all
        into [B, Tmax, H] tensors in O(1) kernel calls.
        """
        gb      = raw["batch_graphs"].to(self.device)
        sizes   = raw["batch_sizes"].to(self.device)   # [B] steps per path
        bvars   = raw["branch_vars"].to(self.device)   # [sum_P] local var idx
        bound   = raw["bound_seq"].to(self.device)
        dseq    = raw["dir_seq"].to(self.device)
        tmask   = raw["time_mask"].to(self.device)
        iw      = raw.get("instance_weight")

        # Encode ALL steps in one batched call.
        h_vars_all, z_all = self.model.encode(gb)     # [sum_P_vars,H],[sum_P,H]

        B     = sizes.size(0)
        H     = z_all.size(-1)
        Tmax  = int(tmask.size(1))
        total_steps = int(sizes.sum().item())

        # ---- step-index tensor [B, Tmax] --------------------------------
        # For path i, step s, the global step index is: step_offsets[i] + s.
        sizes_cpu    = sizes.cpu().to(torch.long)
        step_offsets = torch.cat([
            torch.zeros(1, dtype=torch.long),
            sizes_cpu.cumsum(0)[:-1],
        ]).to(self.device)                             # [B]

        t_range      = torch.arange(Tmax, device=self.device)          # [Tmax]
        step_idx     = step_offsets.unsqueeze(1) + t_range.unsqueeze(0)  # [B,Tmax]
        valid_mask   = t_range.unsqueeze(0) < sizes.unsqueeze(1)          # [B,Tmax]
        step_idx_cl  = step_idx.clamp(0, total_steps - 1)

        # ---- z_seq and z_next_seq via fancy indexing --------------------
        z_seq      = z_all[step_idx_cl]               # [B, Tmax, H]
        z_seq[~valid_mask] = 0.0

        next_idx_cl = (step_idx + 1).clamp(0, total_steps - 1)
        z_next_seq  = z_all[next_idx_cl]              # [B, Tmax, H]
        # Zero out both the padding positions AND the last real step (no next).
        # "last valid step for path i" is s = sizes[i]-1, which maps to the
        # same z_all entry as s (because we clamped), so mask it explicitly.
        valid_next  = valid_mask & (t_range.unsqueeze(0) < (sizes - 1).unsqueeze(1))
        z_next_seq[~valid_next] = 0.0

        # ---- a_seq: h_vars of the chosen branch variable ----------------
        # variable nodes have node_type==0; gb.batch says which step each belongs to.
        var_mask_all   = gb.node_type == 0
        var_step_batch = gb.batch[var_mask_all]        # [sum_P_vars]

        # Number of variable nodes per step, and cumulative start indices.
        n_vars_per_step = torch.bincount(
            var_step_batch, minlength=total_steps)     # [total_steps]
        var_starts = torch.cat([
            torch.zeros(1, dtype=torch.long, device=self.device),
            n_vars_per_step.cumsum(0)[:-1],
        ])                                             # [total_steps]

        # Global index of the chosen variable for every step.
        # bvars is [total_steps] with LOCAL indices within each step's vars.
        # Clamp so an out-of-range bvars entry doesn't crash (uses first var).
        bvars_clamped   = bvars.clamp(0, n_vars_per_step.clamp_min(1) - 1)
        chosen_var_glob = var_starts + bvars_clamped  # [total_steps]
        a_all           = h_vars_all[chosen_var_glob] # [total_steps, H]

        # Scatter a_all into [B, Tmax, H] using the same step_idx_cl.
        a_seq      = a_all[step_idx_cl]               # [B, Tmax, H]
        a_seq[~valid_mask] = 0.0

        out = {
            "z_seq":          z_seq,
            "a_seq":          a_seq,
            "z_next_seq":     z_next_seq,
            "bound_next_seq": bound,
            "dir_seq":        dseq,
            "time_mask":      tmask,
        }
        if iw is not None:
            out["instance_weight"] = iw.to(self.device)
        return out

    def _dynamics_batch_loss(self, batch, return_components=False):
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
        # Branch direction per transition (+1/-1/0). Absent for legacy batches.
        d_seq = d.get("dir_seq")
        if d_seq is not None:
            d_seq = d_seq.to(self.device)

        # Time-padding mask (present when batching variable-length trajectories).
        tmask = d.get("time_mask")
        if tmask is not None:
            tmask = tmask.to(self.device)

        # Base transition prediction. A heteroscedastic model also predicts a
        # per-dim log-variance and is trained with Gaussian NLL, modelling the
        # transition's aleatoric uncertainty instead of forcing a mean fit.
        hetero = getattr(self.model.dynamics, "heteroscedastic", False)
        logvar = None
        if hetero:
            z_pred, logvar = self.model.dynamics.forward(
                z_seq, a_seq, d_seq, return_logvar=True)
        else:
            z_pred = self.model.dynamics_forward(z_seq, a_seq, d_seq)

        has_vars = d.get("hv_seq") is not None
        if has_vars:
            hv_seq = d["hv_seq"].to(self.device)
            # Same as forward_with_vars but reusing the z_pred computed above.
            hv_pred = self.model.dynamics.var_dynamics(hv_seq, z_pred, a_seq)

        comps: dict[str, float] = {}

        L_trans = self._transition_loss(z_pred, z_next_seq, logvar, tmask)
        comps["transition"] = L_trans.item()
        loss = L_trans

        if has_vars:
            L_var = _var_recon_loss(
                hv_pred,
                d["hv_next_seq"].to(self.device),
                d["var_mask"].to(self.device),
            )
            comps["var_recon"] = L_var.item()
            loss = loss + L_var

        # Multi-anchor latent overshooting (+ rollout grounding). Unroll the
        # dynamics autoregressively — feeding its own predictions back in — from
        # SEVERAL start anchors (not just z_0), and supervise each predicted step
        # against the real future latent. This trains the free-running-from-
        # anywhere regime inference actually uses. The rolled-out latents are
        # also grounded against the real dual bound so compounding predictions
        # stay decodable (anti-drift). The horizon `_overshoot_k` follows an
        # epoch curriculum set in train_dynamics.
        bound_tgt = d.get("bound_next_seq")
        if bound_tgt is not None:
            bound_tgt = bound_tgt.to(self.device)
        L_over = self._overshoot_and_ground(
            z_seq, a_seq, d_seq, z_next_seq, tmask, bound_tgt)
        comps["overshoot"] = L_over.item()
        loss = loss + L_over

        # Gap 2: ground the predicted latent against the next real dual bound.
        if d.get("bound_next_seq") is not None:
            bound_pred = self.model.dynamics_bound_pred(z_pred)   # [B, T]
            bound_tgt  = d["bound_next_seq"].to(self.device)
            if tmask is None:
                L_bnd = 0.5 * F.huber_loss(bound_pred, bound_tgt, delta=1.0)
            else:
                per = F.huber_loss(bound_pred, bound_tgt, delta=1.0,
                                   reduction="none")
                L_bnd = 0.5 * (per * tmask.float()).sum() / \
                    tmask.float().sum().clamp_min(1.0)
            comps["bound"] = L_bnd.item()
            loss = loss + L_bnd

        # Fix 3: train the reward head to predict the per-step dual-bound
        # improvement, so the MuZero-style rollout return is grounded.
        if d.get("reward_seq") is not None:
            r_pred = self.model.dynamics_reward_pred(z_pred)      # [B, T]
            r_tgt  = d["reward_seq"].to(self.device)
            if tmask is None:
                L_rew = 0.5 * F.huber_loss(r_pred, r_tgt, delta=1.0)
            else:
                per = F.huber_loss(r_pred, r_tgt, delta=1.0, reduction="none")
                L_rew = 0.5 * (per * tmask.float()).sum() / \
                    tmask.float().sum().clamp_min(1.0)
            comps["reward"] = L_rew.item()
            loss = loss + L_rew

        # Candidate ranking loss — the missing counterfactual objective.
        #
        # The dynamics is trained on expert-only trajectories: one action per
        # state. The minimum-MSE solution is to ignore the action and predict
        # from position alone. candidate_ranking_loss directly supervises the
        # *ordering* that rollout_top_k_batched uses: roll each of K candidates
        # one step, decode value, train the induced ordering to match SB scores.
        #
        # Activated when the batch carries "cand_actions_seq" [B, T, K, H] and
        # "cand_sb_scores_seq" [B, T, K] — pre-computed by the DAgger collector
        # from sb_scores, which it already writes. Only DAgger files have these;
        # the loss simply skips on base trajectories.
        if d.get("cand_actions_seq") is not None and \
                d.get("cand_sb_scores_seq") is not None:
            cand_a  = d["cand_actions_seq"].to(self.device)    # [B, T, K, H]
            cand_sb = d["cand_sb_scores_seq"].to(self.device)  # [B, T, K]
            B, T, K, H = cand_a.shape

            # Vectorised candidate ranking: replace the B×T×K Python loop with
            # one batched dynamics step over all valid (b,t) positions at once.
            # Valid positions: finite SB scores, at least 2 candidates, not masked.
            finite_mask = torch.isfinite(cand_sb).all(dim=-1)  # [B, T]
            if tmask is not None:
                finite_mask = finite_mask & tmask.bool()
            valid_bt = finite_mask.nonzero(as_tuple=False)  # [N_valid, 2]

            if valid_bt.size(0) > 0 and K >= 2:
                bs_idx, ts_idx = valid_bt[:, 0], valid_bt[:, 1]
                N = valid_bt.size(0)

                # Gather z and actions for all valid (b,t) positions.
                z_bt  = z_seq[bs_idx, ts_idx]          # [N, H]
                a_bt  = cand_a[bs_idx, ts_idx]         # [N, K, H]
                sb_bt = cand_sb[bs_idx, ts_idx]         # [N, K]
                d_bt  = d_seq[bs_idx, ts_idx] if d_seq is not None else None  # [N]

                # Expand to [N*K, H] for a single batched step call.
                z_exp = z_bt.unsqueeze(1).expand(-1, K, -1).reshape(N * K, H)
                a_exp = a_bt.reshape(N * K, H)
                d_exp = d_bt.unsqueeze(1).expand(-1, K).reshape(-1) \
                    if d_bt is not None else None

                z_nexts_flat, _, _ = self.model.dynamics.step(
                    z_exp, a_exp, d_t=d_exp,
                )  # [N*K, H]

                z_nexts = z_nexts_flat.reshape(N, K, H)  # [N, K, H]

                # Decode value per candidate (use z as h_vars placeholder).
                z_flat = z_nexts.reshape(N * K, H)
                bvec   = torch.arange(N, device=self.device).repeat_interleave(K)
                scalars_flat = self.model.value(
                    z_flat, z_flat, bvec,
                )  # [N*K] → higher = better bound

                scalars = scalars_flat.reshape(N, K)  # [N, K]

                rank_losses = [
                    _cand_rank_loss(scalars[i], sb_bt[i]) for i in range(N)
                ]
                cand_rank_w = getattr(self, "cand_rank_weight", 0.5)
                L_rank = cand_rank_w * torch.stack(rank_losses).mean()
                comps["ranking"] = L_rank.item()
                loss = loss + L_rank

        # Value consistency in Phase 3: predicted latents should decode through
        # the (frozen) value head to match the target latent's value. Prevents
        # dynamics from drifting into a region the value head cannot decode.
        # The value head is used in no_grad (teacher) — it stays frozen here.
        v_consist_w = getattr(self, "v_consist_weight", 0.1)
        if v_consist_w > 0.0:
            with torch.no_grad():
                B_s, T_s, H_s = z_next_seq.shape
                z_tgt_flat = z_next_seq.reshape(B_s * T_s, H_s)
                bvec_flat  = torch.zeros(B_s * T_s, dtype=torch.long,
                                         device=self.device)
                v_tgt = self.model.value(
                    z_tgt_flat, z_tgt_flat, bvec_flat,
                ).reshape(B_s, T_s).detach()

            z_pred_flat = z_pred.reshape(B_s * T_s, H_s)
            v_pred = self.model.value(
                z_pred_flat, z_pred_flat, bvec_flat,
            ).reshape(B_s, T_s)

            if tmask is not None:
                per = F.huber_loss(v_pred, v_tgt, reduction="none")
                L_vc = v_consist_w * (per * tmask.float()).sum() / \
                    tmask.float().sum().clamp_min(1.0)
            else:
                L_vc = v_consist_w * F.huber_loss(v_pred, v_tgt)
            comps["value_consist"] = L_vc.item()
            loss = loss + L_vc

        # Fix C: self-supervised consistency on free-running predictions.
        # The free-running rollout from anchor 0 produces z_hat_1..T.  These
        # are the states the model actually visits at inference.  Supervising
        # them against the real z_{1..T} (MSE, same as the overshoot loss but
        # here from a FIXED anchor-0 rollout) teaches the model to recover from
        # its own compounding errors without any extra data.  Weight 0.3: strong
        # enough to matter but subordinate to teacher-forced MSE.
        consist_w = getattr(self, "free_run_consist_weight", 0.3)
        if consist_w > 0.0:
            B_c, T_c, H_c = z_seq.shape
            if T_c >= 2:
                with torch.no_grad():
                    free_preds = self.model.dynamics.rollout(
                        z_seq[:, 0], a_seq, d_seq=d_seq,
                    )  # [B, T, H]
                if tmask is not None:
                    m = tmask.unsqueeze(-1).float()
                    per = ((free_preds - z_next_seq) ** 2 * m).sum()
                    L_cons = consist_w * per / (m.sum().clamp_min(1.0) * H_c)
                else:
                    L_cons = consist_w * F.mse_loss(free_preds, z_next_seq)
                comps["free_run"] = L_cons.item()
                loss = loss + L_cons

        # Fix F: cut transition MSE loss (raw graph → encode on-the-fly →
        # Dynamics(z_before, cut_action_embed(cut_feats), d=0) ≈ z_after).
        # When also_train_encoder=True, encode WITHOUT no_grad so the cut loss
        # trains the encoder too (the primary reason we store raw graphs).
        if d.get("graph_before") is not None:
            gb  = d["graph_before"].to(self.device)
            ga  = d["graph_after"].to(self.device)
            cut_phi = d["cut_feats"].to(self.device)        # [N, cut_feat_dim]
            also_enc = getattr(self, "_also_train_encoder", False)
            _enc_ctx = torch.enable_grad() if also_enc else torch.no_grad()
            with _enc_ctx:
                _, z_cut_zb = self.model.encode(gb)        # [N, H]
                _, z_cut_za = self.model.encode(ga)        # [N, H]
            if not also_enc:
                z_cut_zb = z_cut_zb.detach()
                z_cut_za = z_cut_za.detach()
            a_cut  = self.model.cut_action_embed(cut_phi)  # [N, H]
            d_zeros = torch.zeros(gb.num_graphs, device=self.device)
            z_cut_pred, _, _ = self.model.dynamics.step(
                z_cut_zb, a_cut, d_t=d_zeros)              # [N, H]
            cut_w = getattr(self, "cut_transition_weight", 0.1)
            L_cut = cut_w * F.mse_loss(z_cut_pred, z_cut_za)
            comps["cut"] = L_cut.item()
            loss = loss + L_cut

        # Fix D: counterfactual contrastive loss.
        if d.get("a_cf_seq") is not None:
            a_cf = d["a_cf_seq"].to(self.device)          # [B, T, H]
            z_pred_cf = self.model.dynamics_forward(z_seq, a_cf, d_seq)
            err_expert = ((z_pred    - z_next_seq) ** 2).mean(-1)
            err_cf     = ((z_pred_cf - z_next_seq) ** 2).mean(-1)
            margin = 0.1
            hinge = F.relu(margin + err_expert - err_cf)
            cf_w  = getattr(self, "cf_contrastive_weight", 0.3)
            if tmask is not None:
                L_cf = cf_w * (hinge * tmask.float()).sum() / \
                    tmask.float().sum().clamp_min(1.0)
            else:
                L_cf = cf_w * hinge.mean()
            comps["cf_contrast"] = L_cf.item()
            loss = loss + L_cf

        if return_components:
            return loss, comps
        return loss

    def train_dynamics(self, train_loader, val_loader, epochs, lr=5e-4,
                       overshoot_depth=0, patience=None,
                       cand_rank_weight: float = 0.5,
                       also_train: tuple[str, ...] = (),
                       also_train_encoder: bool = False,
                       encoder_lr_scale: float = 0.1,
                       cut_loader=None,
                       cut_val_loader=None,
                       v_consist_weight: float = 0.1,
                       cut_weight: float = 0.1,
                       cf_weight: float = 0.3,
                       free_run_weight: float = 0.3):
        """
        Train DynamicsTransformer (+ optionally the encoder) on trajectory seqs.

        When also_train_encoder=True the loader MUST yield raw PyG graphs
        (RawSequenceDataset / make_raw_collate) rather than pre-encoded
        latents.  The encoder is then run at every training step so that
        z_seq / z_next_seq always reflect the CURRENT encoder weights.  The
        encoder is trained at encoder_lr_scale * lr to avoid destroying the
        Phase-1/2 representations.

        When also_train_encoder=False (default) the loader provides pre-
        encoded latents (SequenceDataset / make_sequence_collate) and the
        encoder is frozen — identical to the original Phase-3 behaviour.

        Loss components logged each epoch (both modes):
            transition, var_recon, overshoot, bound, reward, ranking,
            value_consist, cut   (each weighted as configured)
        """
        self.overshoot_depth        = overshoot_depth
        self.cand_rank_weight       = cand_rank_weight
        self._also_train_encoder    = also_train_encoder
        self.v_consist_weight       = v_consist_weight
        self.cut_transition_weight  = cut_weight
        self.cf_contrastive_weight  = cf_weight
        self.free_run_consist_weight = free_run_weight

        # Determine trainable set.
        _always = {"dynamics", "dyn_bound", "dyn_reward", "cut_action_embed"}
        if also_train_encoder:
            _always.add("encoder")
        _extra = set(also_train)
        _all_prefixes = _always | _extra
        for name, p in self.model.named_parameters():
            p.requires_grad = any(tok in name for tok in _all_prefixes)

        # Separate param groups: encoder gets a lower LR to avoid destroying
        # Phase-1/2 representations.
        if also_train_encoder:
            enc_params  = [p for n, p in self.model.named_parameters()
                           if p.requires_grad and "encoder" in n]
            dyn_params  = [p for n, p in self.model.named_parameters()
                           if p.requires_grad and "encoder" not in n]
            param_groups = [
                {"params": dyn_params, "lr": lr},
                {"params": enc_params, "lr": lr * encoder_lr_scale},
            ]
            n_enc = sum(p.numel() for p in enc_params)
            n_dyn = sum(p.numel() for p in dyn_params)
            print(f"Trainable (Phase 3 + encoder): "
                  f"dynamics={n_dyn:,}  encoder={n_enc:,} "
                  f"(lr×{encoder_lr_scale}) | overshoot_depth={overshoot_depth}")
        else:
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            param_groups = trainable
            extra_str = f" + {sorted(_extra)}" if _extra else ""
            print(f"Trainable params (Phase 3{extra_str}): "
                  f"{sum(p.numel() for p in trainable):,}"
                  f" | overshoot_depth={overshoot_depth}")

        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )
        best_val_loss = float("inf")
        no_improve = 0

        # Overshoot curriculum: ramp the rollout horizon 1 -> overshoot_depth
        # over the first ~60% of epochs, so long-horizon free-running is trained
        # only once the one-step model is reasonable (stable, standard practice).
        ramp_epochs = max(1, int(0.6 * epochs))

        for epoch in range(1, epochs + 1):
            if self.overshoot_depth and self.overshoot_depth > 0:
                self._overshoot_k = max(1, min(
                    self.overshoot_depth,
                    round(self.overshoot_depth * epoch / ramp_epochs)))
            else:
                self._overshoot_k = 0

            # Scheduled sampling curriculum: ramp p_free_run from 0 to 0.5 over
            # the second half of training (after the 1-step model is reasonable).
            # 0.5 means "half the overshoot anchors use model-predicted starts"
            # rather than real encoder states — the same regime as inference.
            free_start_epoch = max(1, int(0.5 * epochs))
            if epoch > free_start_epoch and epochs > free_start_epoch:
                self._p_free_run = min(
                    0.5,
                    0.5 * (epoch - free_start_epoch) / (epochs - free_start_epoch),
                )
            else:
                self._p_free_run = 0.0

            self.model.train()
            total_loss = n = oom_count = oom_samples = 0
            comp_sums = defaultdict(float)  # per-component loss sums

            # Interleave cut-transition batches with sequence batches when
            # cut_loader is provided.  We cycle through cut batches so every
            # sequence batch has a paired cut batch regardless of relative size.
            cut_iter = iter(cut_loader) if cut_loader is not None else None

            for batch in tqdm(
                train_loader,
                desc=f"Dyn Train Epoch {epoch} (k={self._overshoot_k},"
                     f" pfree={self._p_free_run:.2f})",
                leave=False,
            ):
                optimizer.zero_grad(set_to_none=True)
                try:
                    # Online encoding: when also_train_encoder=True the batch
                    # carries raw PyG graphs (RawSequenceDataset); encode them
                    # here with the CURRENT encoder so z_seq/z_next_seq are never
                    # stale.  The standard SequenceDataset path (pre-encoded
                    # latents) is unchanged.
                    if also_train_encoder and "batch_graphs" in batch:
                        batch = self._online_encode_raw_batch(batch)

                    # Inject cut-transition fields into the batch dict so
                    # _dynamics_batch_loss computes the cut MSE loss.
                    if cut_iter is not None:
                        try:
                            cut_batch = next(cut_iter)
                        except StopIteration:
                            cut_iter = iter(cut_loader)
                            cut_batch = next(cut_iter)
                        if isinstance(batch, dict):
                            batch["graph_before"] = cut_batch["graph_before"]
                            batch["graph_after"]  = cut_batch["graph_after"]
                            batch["cut_feats"]    = cut_batch["cut_feats"]

                    with autocast("cuda", enabled=self.amp):
                        loss, comps = self._dynamics_batch_loss(
                            batch, return_components=True)

                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    all_trainable = [p for p in self.model.parameters()
                                     if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()

                    total_loss += loss.item()
                    for k_c, v_c in comps.items():
                        comp_sums[k_c] += v_c
                    n += 1
                except RuntimeError as e:
                    if _is_oom(e):
                        _recover_oom(optimizer)
                        oom_count += 1
                        # Estimate skipped samples from batch structure.
                        try:
                            oom_samples += int(
                                batch.get("time_mask", batch.get("z_seq")).shape[0])
                        except Exception:
                            oom_samples += 1
                        continue
                    raise

            if oom_count:
                import warnings
                warnings.warn(
                    f"[Phase3] Epoch {epoch}: {oom_count} OOM batches / "
                    f"~{oom_samples} samples skipped "
                    f"({100*oom_count/(n+oom_count):.1f}% of batches). "
                    "Consider reducing batch size or sequence length.",
                    RuntimeWarning, stacklevel=2,
                )

            # Validation — instance-weighted, tracked per k-horizon and per
            # loss component so large trees don't dominate and per-horizon
            # curves stay comparable across epochs.
            self.model.eval()
            val_loss_sum  = val_w_sum = 0
            val_comp_sums = defaultdict(float)
            # Per-horizon latent MSE: always evaluate at k=1 and the final
            # horizon so we can track improvement independently of curriculum.
            horizons = sorted({1, max(1, self._overshoot_k)})
            val_horizon_se  = {h: 0.0 for h in horizons}
            val_horizon_cnt = {h: 0   for h in horizons}

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Dyn Val", leave=False):
                    if also_train_encoder and "batch_graphs" in batch:
                        batch = self._online_encode_raw_batch(batch)
                    w = batch.get("instance_weight") if isinstance(batch, dict) \
                        else None
                    batch_w = float(w.sum()) if w is not None else 1.0
                    vl, vcomps = self._dynamics_batch_loss(
                        batch, return_components=True)
                    val_loss_sum += vl.item() * batch_w
                    val_w_sum    += batch_w
                    for k_c, v_c in vcomps.items():
                        val_comp_sums[k_c] += v_c * batch_w

                    # Per-horizon latent MSE (start from t=0 and t=T//3).
                    z_seq = batch.get("z_seq")
                    z_nxt = batch.get("z_next_seq")
                    a_seq = batch.get("a_seq")
                    d_seq = batch.get("dir_seq")
                    tmask = batch.get("time_mask")
                    if z_seq is not None and a_seq is not None:
                        z_seq = z_seq.to(self.device)
                        a_seq = a_seq.to(self.device)
                        z_nxt = z_nxt.to(self.device) if z_nxt is not None else None
                        d_seq = d_seq.to(self.device) if d_seq is not None else None
                        tmask_dev = tmask.to(self.device) if tmask is not None else None
                        for h in horizons:
                            preds = self.model.dynamics.rollout(
                                z_seq[:, 0], a_seq[:, :h],
                                d_seq=d_seq[:, :h] if d_seq is not None else None)
                            if z_nxt is not None and preds.size(1) > 0:
                                j = min(h - 1, z_nxt.size(1) - 1)
                                m = (tmask_dev[:, j].float()
                                     if tmask_dev is not None
                                     else torch.ones(z_seq.size(0),
                                                     device=self.device))
                                se = ((preds[:, -1] - z_nxt[:, j]) ** 2
                                      ).mean(-1)
                                val_horizon_se[h]  += float((se * m).sum().item())
                                val_horizon_cnt[h] += float(m.sum().item())

            # --- separate cut-transition validation (does not affect early stop) ---
            val_cut_loss = float("nan")
            val_cut_dlb  = float("nan")
            if cut_val_loader is not None:
                cut_val_sum = cut_val_n = 0
                dlb_sum = 0.0
                with torch.no_grad():
                    for cbatch in cut_val_loader:
                        gb  = cbatch["graph_before"].to(self.device)
                        ga  = cbatch["graph_after"].to(self.device)
                        phi = cbatch["cut_feats"].to(self.device)
                        try:
                            _, zb = self.model.encode(gb)
                            _, za = self.model.encode(ga)
                            a_cut = self.model.cut_action_embed(phi)
                            d_z   = torch.zeros(gb.num_graphs, device=self.device)
                            z_pred_cut, _, _ = self.model.dynamics.step(zb, a_cut, d_t=d_z)
                            cut_val_sum += F.mse_loss(z_pred_cut, za).item()
                            cut_val_n   += 1
                            if "delta_lb" in cbatch:
                                dlb_sum += float(cbatch["delta_lb"].mean())
                        except RuntimeError as e:
                            if _is_oom(e):
                                _recover_oom()
                                continue
                            raise
                if cut_val_n > 0:
                    val_cut_loss = cut_val_sum / cut_val_n
                    val_cut_dlb  = dlb_sum / cut_val_n

            train_loss = total_loss / n if n else float("inf")
            val_loss   = val_loss_sum / val_w_sum if val_w_sum > 0 else float("inf")
            scheduler.step()

            self.history["p3_train_loss"].append(train_loss)
            self.history["p3_val_loss"].append(val_loss)
            self.history["p3_val_cut_loss"].append(val_cut_loss)

            # Per-component log line.
            comp_str = "  ".join(
                f"{k}={v/n:.4f}" for k, v in sorted(comp_sums.items()) if n > 0)
            horizon_str = "  ".join(
                f"mse@{h}={val_horizon_se[h]/val_horizon_cnt[h]:.4f}"
                for h in horizons if val_horizon_cnt[h] > 0)
            cut_str = (f"  val_cut={val_cut_loss:.4f}  mean_ΔLB={val_cut_dlb:.4f}"
                       if not (val_cut_loss != val_cut_loss) else "")  # not nan

            print(
                f"[Phase3] Epoch {epoch:02d} | "
                f"TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f}"
                + (f"\n  train: {comp_str}" if comp_str else "")
                + (f"\n  val:   {horizon_str}" if horizon_str else "")
                + (f"\n  cut:   {cut_str}" if cut_str else "")
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                save_checkpoint(
                    self.model, optimizer, epoch,
                    {"val_loss": val_loss},
                    self.ckpt_dir / "phase3_best.pt",
                )
                print("  Saved best Phase 3 model")
            else:
                no_improve += 1
                if patience and no_improve >= patience:
                    print(f"  Early stop at epoch {epoch} "
                          f"(no val improvement for {patience} epochs)")
                    break

        save_checkpoint(
            self.model, optimizer, epochs, {}, self.ckpt_dir / "phase3_final.pt"
        )

        # k-step rollout drift diagnostic on the BEST weights — tells us whether
        # the residual/overshoot/grounding upgrades flattened the multi-step
        # drift curve, and is saved for the paper figure.
        best = self.ckpt_dir / "phase3_best.pt"
        if best.exists():
            load_weights_only(self.model, best, device=self.device)
        diag_depth = max(8, (self.overshoot_depth or 0) + 2)
        self._overshoot_k = 0     # diagnostic must not affect training state
        self.history["p3_rollout_diag"] = self.dynamics_rollout_diagnostic(
            val_loader, max_depth=diag_depth,
            save_path=self.ckpt_dir / "phase3_rollout_diag.json")

        for p in self.model.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    # Phase 4 — Joint fine-tuning
    # ------------------------------------------------------------------
    def train_joint(self, train_loader, val_loader, epochs, lr=1e-4,
                    pos_weight=None, patience=None):
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
        no_improve = 0

        # Adaptive loss scale: calibrated once from first batch so each term
        # starts at ~1.0 in relative magnitude. Stored per-term so the
        # scales can be inspected in logs. Raw weights are still applied on
        # top (0.5 v_loss, etc.) — calibration just removes the order-of-
        # magnitude mismatch between terms at initialisation.
        _p4_scales: dict[str, float] | None = None  # filled after first batch

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

                    var_batch_idx = pyg_batch.batch[var_mask]
                    scores        = self.model.policy_scores(h_vars, z, var_batch_idx)

                    # Policy loss (and collect expert-chosen var embeddings)
                    p_losses, top1 = [], 0
                    offset = 0
                    chosen_idx = []
                    for meta in metas:
                        n_v    = meta["n_vars"]
                        logits = scores[offset : offset + n_v]
                        aset   = meta["action_set"].to(self.device)
                        if "sb_scores" in meta:
                            ploss, acc, _ = policy_loss_soft(
                                logits, aset,
                                meta["sb_scores"].to(self.device),
                                meta["local_label"],
                            )
                        else:
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
                    z_pred1, _, _kv = self.model.dynamics_step(z, a_chosen)
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

                    # Adaptive calibration: estimate scales once from the
                    # first batch (no-grad), then hold fixed for the run.
                    if _p4_scales is None:
                        with torch.no_grad():
                            _p4_scales = {
                                "p":  max(p_loss.item(),  1e-6),
                                "v":  max(v_loss.item(),  1e-6),
                                "i":  max(i_loss.item(),  1e-6),
                                "vc": max(v_consist.item(), 1e-6),
                                "s":  max(s_loss.item(),  1e-6),
                                "c":  max(c_loss.item(),  1e-6),
                            }

                    sp, sv, si, svc, ss, sc = (
                        _p4_scales["p"], _p4_scales["v"], _p4_scales["i"],
                        _p4_scales["vc"], _p4_scales["s"], _p4_scales["c"],
                    )
                    loss = (
                        p_loss / sp
                        + 0.5 * v_loss / sv
                        + 0.1 * i_loss / si
                        + 0.1 * v_consist / svc
                        + 0.3 * s_loss / ss
                        + 0.5 * c_loss / sc
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
            train_loss = total_loss / n if n else float("inf")
            _, val_acc = self._epoch_policy(val_loader, None, training=False)

            self.history["p4_train_loss"].append(train_loss)
            self.history["p4_val_acc"].append(val_acc)

            print(
                f"[Phase4] Epoch {epoch:02d} | "
                f"TotalLoss={train_loss:.4f} | ValAcc={val_acc:.3f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
                save_checkpoint(
                    self.model, optimizer, epoch,
                    {"val_acc": val_acc},
                    self.ckpt_dir / "phase4_best.pt",
                )
                print("  Saved best Phase 4 model")
            else:
                no_improve += 1
                if patience and no_improve >= patience:
                    print(f"  Early stop at epoch {epoch} "
                          f"(no val improvement for {patience} epochs)")
                    break

        save_checkpoint(
            self.model, optimizer, epochs, {}, self.ckpt_dir / "phase4_final.pt"
        )

    # ------------------------------------------------------------------
    # Phase 5 — Cut selection
    # ------------------------------------------------------------------
    @staticmethod
    def _cut_diversity_loss(
        cut_feats: "torch.Tensor",
        scores: "torch.Tensor",
        top_k: int = 8,
        margin: float = 0.5,
        weight: float = 0.1,
    ) -> "torch.Tensor":
        """Pairwise cosine repulsion among the top-k scored cuts.

        Penalises near-parallel cut directions so the selected pool covers
        diverse constraint directions. Operates on cut_embeds [C, H] — the
        GNN-native H-dim embeddings (Σ coeff_j * h_vars[j]) rather than the
        6-dim feature vectors, so the diversity signal lives in the same space
        as the cosine similarity score itself.

        L_div = weight * mean(ReLU(|cos(c_i, c_j)| - margin)) over pairs i<j
        """
        import torch
        import torch.nn.functional as F
        if cut_feats.size(0) < 2:
            return torch.tensor(0.0, device=cut_feats.device, requires_grad=True)
        k = min(top_k, cut_feats.size(0))
        _, top_idx = scores.detach().topk(k)
        selected = cut_feats[top_idx]                            # [k, H]
        normed = F.normalize(selected, dim=-1)                   # [k, H]
        sim = normed @ normed.T                                  # [k, k]
        # upper triangle, exclude diagonal
        mask = torch.triu(torch.ones(k, k, dtype=torch.bool, device=sim.device), diagonal=1)
        repulsion = torch.relu(sim.abs()[mask] - margin)
        return weight * repulsion.mean()

    def train_cuts(self, train_loader=None, val_loader=None, epochs=0, lr=5e-4,
                   pos_weight=None, patience=None, div_weight: float = 0.1):
        """Phase 5 is a no-op: ZeroShotCutScorer has no trainable parameters.

        Cut scoring is done entirely via cosine similarity between GNN-native
        cut embeddings (Σ coeff_j * h_vars[j]) and the graph embedding z,
        plus a violation bonus. No parameters → no training required.

        The diversity repulsion penalty (_cut_diversity_loss) runs at solve
        time inside rollout_cut_branch_beam to prune near-parallel cuts from
        the beam — it is not a training objective.
        """
        print("[Phase 5] ZeroShotCutScorer requires no training — skipping.")
        for p in self.model.parameters():
            p.requires_grad = True
