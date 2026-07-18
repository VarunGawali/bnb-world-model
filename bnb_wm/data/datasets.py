"""
datasets.py — Dataset and collate functions for the five training phases.

Consumes the collected trajectory `.npz` files. Confirmed schema (per file,
one B&B trajectory; ragged per-step arrays stored as object arrays):

    n_steps                int
    var_features           [T] of [n_vars, 19]   Ecole column features
    con_features           [T] of [n_cons, 5]    Ecole row features
    edge_indices           [T] of [2, E]         (constraint_idx, variable_idx)
    edge_values            [T] of [E]            constraint coefficients
    action_sets            [T] of [k]            candidate branching variables
    branching_vars         [T] int               chosen variable (global idx)
    local_branching_label  [T] int               index into action_set
    dual_bounds            [T] float
    norm_dual_bounds       [T] float             normalised (value-head target)
    next_is_leaf           [T] float             1 if next node is a leaf
    depths                 [T] int
    cut_features           [T] of [n_cuts, 6]
    cut_labels             [T] of [n_cuts]
    cut_scores             [T] of [n_cuts]        (optional extra)
    cut_lhs                [T] of [n_cuts, n_vars]
    cut_rhs                [T] of [n_cuts]
    n_cuts                 [T] int

Two dataset views:
    TransitionDataset — one item per B&B node. Feeds Phases 1 (policy),
                        2 (value), 4 (joint), 5 (cuts). Yields (PyG Data, meta).
    SequenceDataset   — one item per trajectory, pre-encoded with the frozen
                        encoder into latent sequences. Feeds Phase 3 (dynamics).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

from .labels import steps_to_go, subtree_sizes_from_depths


# ---------------------------------------------------------------------------
# File discovery / splitting
# ---------------------------------------------------------------------------

def list_trajectory_files(data_root, pattern="traj_*.npz"):
    """Return sorted trajectory file paths under `data_root` (searched recursively)."""
    root = Path(data_root)
    files = sorted(root.rglob(pattern))
    return files


def compute_label_stats(files, with_cuts=False):
    """
    Scan trajectory files for class imbalance, to set pos_weight in the
    integrality (leaf) and cut-selection BCE losses.

    Returns a dict with:
        leaf_pos_weight : float or None   (n_non_leaf / n_leaf)
        cut_pos_weight  : float or None   (n_neg_cut / n_pos_cut)
    None means no positives were found (pos_weight left unset).
    """
    leaf_pos = leaf_tot = 0
    cut_pos = cut_tot = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        nil = np.asarray(d["next_is_leaf"], dtype=np.float32)
        leaf_pos += int((nil > 0.5).sum())
        leaf_tot += int(nil.size)
        if with_cuts:
            for t in range(int(d["n_steps"])):
                if int(d["n_cuts"][t]) > 0:
                    cl = np.asarray(d["cut_labels"][t], dtype=np.float32)
                    cut_pos += int((cl > 0.5).sum())
                    cut_tot += int(cl.size)

    def _pw(pos, tot):
        neg = tot - pos
        return float(neg) / float(pos) if pos > 0 else None

    return {
        "leaf_pos_weight": _pw(leaf_pos, leaf_tot),
        "cut_pos_weight":  _pw(cut_pos, cut_tot) if with_cuts else None,
    }


def split_files(files, train=0.8, val=0.1, test=0.1, seed=0):
    """Deterministically split a file list into (train, val, test)."""
    files = list(files)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(files))
    n_train = int(round(train * len(files)))
    n_val   = int(round(val   * len(files)))
    tr = [files[i] for i in idx[:n_train]]
    va = [files[i] for i in idx[n_train:n_train + n_val]]
    te = [files[i] for i in idx[n_train + n_val:]]
    return tr, va, te


# ---------------------------------------------------------------------------
# PyG graph construction (mirrors evaluate/benchmark._format_obs exactly)
# ---------------------------------------------------------------------------

def build_pyg_data(vf, cf, ei, ev):
    """
    Build a bipartite PyG Data object from one node's raw arrays.

    Layout matches the encoder / benchmark:
        x          : [n_vars + n_cons, 19]  (constraint features padded to 19)
        node_type  : 0 = variable, 1 = constraint
        edge_index : [2, E]  variable and constraint nodes, constraints offset
        edge_attr  : [E, 3]  [coeff, coeff / |RHS|, sign(coeff)]
    """
    vf = np.nan_to_num(np.asarray(vf, dtype=np.float32), nan=0.0,
                       posinf=1e6, neginf=-1e6)
    cf = np.nan_to_num(np.asarray(cf, dtype=np.float32), nan=0.0,
                       posinf=1e6, neginf=-1e6)
    ei = np.asarray(ei, dtype=np.int64)                      # [2, E]
    ev = np.asarray(ev, dtype=np.float32).reshape(-1)
    ev = np.nan_to_num(ev, nan=0.0, posinf=1e6, neginf=-1e6)

    n_vars = vf.shape[0]
    n_cons = cf.shape[0]

    # Edge features: normalise coefficient by constraint RHS (con feature idx 1).
    con_src = ei[0]
    rhs = cf[con_src, 1] if cf.shape[1] > 1 else np.ones(len(con_src), np.float32)
    norm_ev = ev / (np.abs(rhs) + 1e-8)
    sign_ev = np.sign(ev)
    edge_attr = np.stack([ev, norm_ev, sign_ev], axis=1).astype(np.float32)

    vf_t = torch.from_numpy(vf)
    cf_t = torch.from_numpy(cf)
    ei_t = torch.from_numpy(ei)
    ea_t = torch.from_numpy(edge_attr)

    cf_pad = F.pad(cf_t, (0, 14))                            # 5 -> 19 dims
    x = torch.cat([vf_t, cf_pad], dim=0)

    node_type = torch.cat([
        torch.zeros(n_vars, dtype=torch.long),
        torch.ones(n_cons,  dtype=torch.long),
    ])
    # Constraints placed after variables; ei[1] = variable idx, ei[0] = con idx.
    edge_index = torch.stack([ei_t[0] + n_vars, ei_t[1]], dim=0)

    return Data(x=x, edge_index=edge_index, node_type=node_type, edge_attr=ea_t)


# ---------------------------------------------------------------------------
# TransitionDataset — one B&B node per item (Phases 1, 2, 4, 5)
# ---------------------------------------------------------------------------

class TransitionDataset(Dataset):
    """
    Flattens every trajectory into individual B&B nodes.

    Each item is (Data, meta) where meta carries the per-node training targets:
        n_vars, action_set, local_label            (policy)
        norm_db                                     (value)
        is_leaf, depth, n_frac                      (integrality / joint)
        steps_to_go                                 (cost-to-go, Gap 3)
        subtree_size                                (only if the trajectory is
                                                     DFS-ordered; else absent)
        cut_features, cut_labels                    (cuts, if present)
    """

    def __init__(self, files, with_cuts=False):
        self.files = list(files)
        self.with_cuts = with_cuts
        # Build a flat index of (file_idx, step) without holding files open.
        self.index = []
        for fi, f in enumerate(self.files):
            with np.load(f, allow_pickle=True) as d:
                n = int(d["n_steps"])
            self.index.extend((fi, t) for t in range(n))
        self._cache_fi = None
        self._cache_d = None

    def __len__(self):
        return len(self.index)

    def _load(self, fi):
        if fi != self._cache_fi:
            self._cache_d = np.load(self.files[fi], allow_pickle=True)
            self._cache_fi = fi
        return self._cache_d

    def __getitem__(self, i):
        fi, t = self.index[i]
        d = self._load(fi)

        vf = d["var_features"][t]
        data = build_pyg_data(vf, d["con_features"][t],
                              d["edge_indices"][t], d["edge_values"][t])

        n_vars = int(vf.shape[0])
        n_steps = int(d["n_steps"])

        # n_frac from sol_frac (Ecole var feature idx 14), falling back to none.
        if vf.shape[1] > 14:
            n_frac = int((np.asarray(vf[:, 14], dtype=np.float32) > 0.05).sum())
        else:
            n_frac = 0

        meta = {
            "n_vars":      n_vars,
            "action_set":  torch.as_tensor(
                np.asarray(d["action_sets"][t], dtype=np.int64),
                dtype=torch.long),
            "local_label": int(d["local_branching_label"][t]),
            "norm_db":     float(d["norm_dual_bounds"][t]),
            "is_leaf":     float(d["next_is_leaf"][t]),
            "depth":       int(d["depths"][t]),
            "n_frac":      n_frac,
            "steps_to_go": float(n_steps - t),          # Gap 3 target (no DFS)
        }

        # Subtree size only when the trajectory is a valid DFS pre-order.
        sizes = subtree_sizes_from_depths(d["depths"])
        if sizes is not None:
            meta["subtree_size"] = float(sizes[t])

        if self.with_cuts:
            # Always present (possibly empty) so the Phase-5 loop, which reads
            # cut_features unconditionally and skips size-0 entries, never
            # KeyErrors on a node that generated no cuts.
            if int(d["n_cuts"][t]) > 0:
                cf_t = np.asarray(d["cut_features"][t], dtype=np.float32)
                cl_t = np.asarray(d["cut_labels"][t],   dtype=np.float32)
            else:
                cf_t = np.zeros((0, 6), dtype=np.float32)
                cl_t = np.zeros((0,),   dtype=np.float32)
            meta["cut_features"] = torch.as_tensor(cf_t, dtype=torch.float32)
            meta["cut_labels"]   = torch.as_tensor(cl_t, dtype=torch.float32)

        return data, meta


def transition_collate(batch):
    """Collate (Data, meta) items into (PyG Batch, [meta, ...])."""
    datas = [b[0] for b in batch]
    metas = [b[1] for b in batch]
    return Batch.from_data_list(datas), metas


# ---------------------------------------------------------------------------
# SequenceDataset — one trajectory per item, pre-encoded (Phase 3)
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):
    """
    Pre-encodes each trajectory into latent sequences for the dynamics model.

    Uses the frozen encoder (from Phase 1/2) to turn every node into (z, h_vars),
    then assembles one-step-shifted sequences. Returns a dict per trajectory:

        z_seq          [T-1, H]        latents  z_0 .. z_{T-2}
        a_seq          [T-1, H]        action embeddings (chosen var's h_vars)
        z_next_seq     [T-1, H]        targets  z_1 .. z_{T-1}
        bound_next_seq [T-1]           next norm dual bound (Gap 2 grounding)

      and, when include_vars is set (subsampled to keep memory bounded):
        hv_seq         [T-1, K, H]     per-variable embeddings at t
        hv_next_seq    [T-1, K, H]     per-variable embeddings at t+1
        var_mask       [T-1, K] bool   valid (non-padding) positions

    Encoding one trajectory is a single batched encoder call.
    """

    def __init__(self, files, model, device, include_vars=True,
                 max_vars_recon=64, seed=0):
        self.files = list(files)
        self.model = model
        self.device = device
        self.include_vars = include_vars
        self.max_vars_recon = max_vars_recon
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.files)

    @torch.no_grad()
    def __getitem__(self, i):
        d = np.load(self.files[i], allow_pickle=True)
        T = int(d["n_steps"])
        # Need at least 2 nodes to form one transition.
        if T < 2:
            # Return a trivially-masked single step so collate stays uniform.
            H = self.model.hidden_dim
            return {
                "z_seq": torch.zeros(1, H), "a_seq": torch.zeros(1, H),
                "z_next_seq": torch.zeros(1, H),
                "bound_next_seq": torch.zeros(1),
                "reward_seq": torch.zeros(1),
                "valid_len": 0,
            }

        datas = [
            build_pyg_data(d["var_features"][t], d["con_features"][t],
                           d["edge_indices"][t], d["edge_values"][t])
            for t in range(T)
        ]
        batch = Batch.from_data_list(datas).to(self.device)
        h_vars, z = self.model.encode(batch)          # h_vars [sumV,H], z [T,H]
        z = z.cpu()

        # Split h_vars per step (n_vars constant within a trajectory).
        var_mask_all = batch.node_type == 0
        var_batch = batch.batch[var_mask_all].cpu()
        h_all = h_vars.cpu()

        branch = np.asarray(d["branching_vars"]).astype(np.int64)
        # Action embedding at step t = chosen variable's embedding.
        a_list = []
        per_step_h = []
        for t in range(T):
            ht = h_all[var_batch == t]                # [n_vars_t, H]
            per_step_h.append(ht)
            bv = int(branch[t])
            bv = bv if 0 <= bv < ht.size(0) else 0
            a_list.append(ht[bv])
        a_all = torch.stack(a_list, dim=0)            # [T, H]

        ndb = np.asarray(d["norm_dual_bounds"], dtype=np.float32)
        out = {
            "z_seq":          z[:-1],
            "a_seq":          a_all[:-1],
            "z_next_seq":     z[1:],
            "bound_next_seq": torch.as_tensor(ndb[1:], dtype=torch.float32),
            # Per-step reward = dual-bound improvement (Fix 3 target).
            "reward_seq":     torch.as_tensor(ndb[1:] - ndb[:-1],
                                              dtype=torch.float32),
            "valid_len":      T - 1,
        }

        if self.include_vars:
            n_vars = per_step_h[0].size(0)
            K = min(self.max_vars_recon, n_vars)
            # Fixed variable subset for the whole trajectory (same set each step,
            # since the var_dynamics head is shared across variables).
            sub = self.rng.choice(n_vars, size=K, replace=False)
            sub = torch.as_tensor(np.sort(sub), dtype=torch.long)
            hv = torch.stack([h[sub] for h in per_step_h], dim=0)   # [T, K, H]
            out["hv_seq"]      = hv[:-1]
            out["hv_next_seq"] = hv[1:]
            out["var_mask"]    = torch.ones(T - 1, K, dtype=torch.bool)

        return out


def make_sequence_collate(include_vars=True):
    """
    Build a collate that pads trajectories to the batch's max length and
    stacks them into [B, T, ...], with a time_mask marking valid positions.
    """
    def _collate(batch):
        B = len(batch)
        H = batch[0]["z_seq"].size(-1)
        lengths = [int(b.get("valid_len", b["z_seq"].size(0))) for b in batch]
        Tmax = max(max(lengths), 1)

        z_seq   = torch.zeros(B, Tmax, H)
        a_seq   = torch.zeros(B, Tmax, H)
        z_next  = torch.zeros(B, Tmax, H)
        bound   = torch.zeros(B, Tmax)
        reward  = torch.zeros(B, Tmax)
        tmask   = torch.zeros(B, Tmax, dtype=torch.bool)

        has_vars = include_vars and ("hv_seq" in batch[0])
        if has_vars:
            K = max(b["hv_seq"].size(1) for b in batch if "hv_seq" in b)
            hv_seq  = torch.zeros(B, Tmax, K, H)
            hv_next = torch.zeros(B, Tmax, K, H)
            vmask   = torch.zeros(B, Tmax, K, dtype=torch.bool)

        for i, b in enumerate(batch):
            L = int(b.get("valid_len", b["z_seq"].size(0)))
            if L <= 0:
                continue
            z_seq[i, :L]  = b["z_seq"][:L]
            a_seq[i, :L]  = b["a_seq"][:L]
            z_next[i, :L] = b["z_next_seq"][:L]
            bound[i, :L]  = b["bound_next_seq"][:L]
            if "reward_seq" in b:
                reward[i, :L] = b["reward_seq"][:L]
            tmask[i, :L]  = True
            if has_vars and "hv_seq" in b:
                k = b["hv_seq"].size(1)
                hv_seq[i, :L, :k]  = b["hv_seq"][:L]
                hv_next[i, :L, :k] = b["hv_next_seq"][:L]
                vmask[i, :L, :k]   = b["var_mask"][:L]

        out = {
            "z_seq": z_seq, "a_seq": a_seq, "z_next_seq": z_next,
            "bound_next_seq": bound, "reward_seq": reward, "time_mask": tmask,
        }
        if has_vars:
            out["hv_seq"] = hv_seq
            out["hv_next_seq"] = hv_next
            out["var_mask"] = vmask
        return out

    return _collate
