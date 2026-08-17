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
    node_ids               [T] int               SCIP node id (P0.3 path recon)
    parent_ids             [T] int               parent's SCIP node id
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

from .labels import (
    steps_to_go, subtree_sizes_from_depths, subtree_sizes_from_tree,
)


# ---------------------------------------------------------------------------
# File discovery / splitting
# ---------------------------------------------------------------------------

# Feature clip range, shared by build_pyg_data and compute_feature_stats so
# training and deployment clip node features identically before standardisation.
FEATURE_CLIP = 1e4


def compute_feature_stats(files, max_files=200):
    """
    Per-feature mean/std of the variable (19-dim) and constraint (5-dim) node
    features over a sample of trajectory files, for input standardisation
    ("prenorm"). Features are clipped to FEATURE_CLIP first, matching
    build_pyg_data. Returns (var_mean, var_std, con_mean, con_std) float32.

    Cheap: one pass over up to `max_files` files. std is floored to 1e-6 so
    constant features (e.g. always-zero flags) don't divide by ~0.
    """
    vsum = vsq = csum = csq = None
    vcnt = ccnt = 0
    for f in list(files)[:max_files]:
        d = np.load(f, allow_pickle=True)
        for t in range(int(d["n_steps"])):
            vf = np.clip(np.nan_to_num(np.asarray(d["var_features"][t], np.float64)),
                         -FEATURE_CLIP, FEATURE_CLIP)
            cf = np.clip(np.nan_to_num(np.asarray(d["con_features"][t], np.float64)),
                         -FEATURE_CLIP, FEATURE_CLIP)
            vsum = vf.sum(0) if vsum is None else vsum + vf.sum(0)
            vsq  = (vf**2).sum(0) if vsq is None else vsq + (vf**2).sum(0)
            vcnt += vf.shape[0]
            csum = cf.sum(0) if csum is None else csum + cf.sum(0)
            csq  = (cf**2).sum(0) if csq is None else csq + (cf**2).sum(0)
            ccnt += cf.shape[0]
    if vcnt == 0 or ccnt == 0:
        return None
    var_mean = vsum / vcnt
    var_std  = np.sqrt(np.maximum(vsq / vcnt - var_mean**2, 0.0))
    con_mean = csum / ccnt
    con_std  = np.sqrt(np.maximum(csq / ccnt - con_mean**2, 0.0))
    return (var_mean.astype(np.float32), np.maximum(var_std, 1e-6).astype(np.float32),
            con_mean.astype(np.float32), np.maximum(con_std, 1e-6).astype(np.float32))


def gap_to_primal_norm(d):
    """
    P0.11: fixed cross-instance value target — fraction of the root->optimum gap
    that a node's dual bound has closed:

        norm[t] = (dual_bounds[t] - root_bound) / (primal_bound - root_bound)

    This is 0 at the root and 1 at the primal (optimum), on the SAME scale for
    every instance — unlike the old per-trajectory min-max `norm_dual_bounds`,
    whose 0/1 endpoints meant a different absolute bound in every file, so the
    value head was regressed against an instance-dependent target.

    Falls back to the stored `norm_dual_bounds` when the anchors are absent
    (legacy files) or degenerate (primal == root).

    Args:
        d : an opened npz (np.load) for one trajectory
    Returns:
        [T] float32 in ~[0, 1]
    """
    # Only trust the primal anchor when the collector actually proved optimality
    # (P0.11 `optimal_valid`); otherwise `primal_bound` is a fallback, not the
    # true optimum, and would give a meaningless scale.
    optimal_valid = bool(np.asarray(d["optimal_valid"])) if "optimal_valid" in d else True
    if (optimal_valid and "root_bound" in d and "primal_bound" in d
            and "dual_bounds" in d):
        db = np.asarray(d["dual_bounds"], dtype=np.float64)
        root = float(np.asarray(d["root_bound"]))
        primal = float(np.asarray(d["primal_bound"]))
        denom = primal - root
        if abs(denom) > 1e-9:
            return np.clip((db - root) / denom, 0.0, 1.0).astype(np.float32)
    return np.asarray(d["norm_dual_bounds"], dtype=np.float32)


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


def split_files(files, train=0.8, val=0.1, test=0.1, seed=0, stratify=True):
    """
    Deterministically split a file list into (train, val, test).

    P1.2: by default STRATIFY by difficulty tier (the file's parent directory,
    e.g. SC-easy / SC-medium / SC-hard) so every tier is proportionally present
    in each split — a plain random split can starve the thin SC-hard tier from
    val/test. Set stratify=False for a single-group random split.
    """
    files = list(files)
    if stratify:
        groups = {}
        for f in files:
            groups.setdefault(Path(f).parent.name, []).append(f)
        # If everything is in one group, stratification is a no-op; fall through.
        if len(groups) > 1:
            tr, va, te = [], [], []
            for _, gfiles in sorted(groups.items()):
                g_tr, g_va, g_te = split_files(
                    gfiles, train, val, test, seed, stratify=False)
                tr += g_tr; va += g_va; te += g_te
            return tr, va, te
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
    # Clip to a sane range (was ±1e6): the encoder standardises features with
    # fixed per-feature stats, and a single 1e6 outlier would otherwise dominate
    # the scale. FEATURE_CLIP is shared with compute_feature_stats so train and
    # deploy clip identically.
    vf = np.clip(np.nan_to_num(np.asarray(vf, dtype=np.float32), nan=0.0,
                               posinf=FEATURE_CLIP, neginf=-FEATURE_CLIP),
                 -FEATURE_CLIP, FEATURE_CLIP)
    cf = np.clip(np.nan_to_num(np.asarray(cf, dtype=np.float32), nan=0.0,
                               posinf=FEATURE_CLIP, neginf=-FEATURE_CLIP),
                 -FEATURE_CLIP, FEATURE_CLIP)
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
    # P0.2: emit edges in BOTH directions. The encoder splits edges into a
    # constraint->variable pass and a variable->constraint pass by matching
    # node_type[src]/node_type[dst]; if only the con->var direction is present
    # the variable->constraint pass sees zero edges and constraints never
    # aggregate from their variables. Duplicate edge_attr for the reverse edges.
    con_to_var = torch.stack([ei_t[0] + n_vars, ei_t[1]], dim=0)
    var_to_con = torch.stack([ei_t[1], ei_t[0] + n_vars], dim=0)
    edge_index = torch.cat([con_to_var, var_to_con], dim=1)
    edge_attr  = torch.cat([ea_t, ea_t], dim=0)

    return Data(x=x, edge_index=edge_index, node_type=node_type, edge_attr=edge_attr)


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
        steps_to_go                                 (cost-to-go: tree subtree
                                                     size, or n_steps-t legacy)
        subtree_size                                (tree subtree size, when
                                                     node/parent ids present)
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
        self._cache_sizes = None        # per-file tree subtree sizes (or None)

    def __len__(self):
        return len(self.index)

    def _load(self, fi):
        if fi != self._cache_fi:
            self._cache_d = np.load(self.files[fi], allow_pickle=True)
            self._cache_fi = fi
            self._cache_sizes = self._compute_subtree_sizes(self._cache_d)
        return self._cache_d

    @staticmethod
    def _compute_subtree_sizes(d):
        """Subtree sizes for the recorded nodes, best source first. Returns [T].

        1. FULL tree (`full_*`, collector v3): sizes over EVERY processed node —
           includes fathomed leaves — mapped back to the recorded nodes. Most
           accurate cost-to-go / subtree-size target.
        2. Recorded tree (`node_ids`/`parent_ids`): recorded-node subtree count
           (undercounts fathomed leaves), exact for any order.
        3. DFS-depth heuristic: legacy fallback for id-less files (or None).
        """
        if "full_node_ids" in d and "full_parent_ids" in d and "node_ids" in d:
            full_sizes = subtree_sizes_from_tree(
                d["full_node_ids"], d["full_parent_ids"])
            f2i = {int(nid): i for i, nid in enumerate(np.asarray(d["full_node_ids"]))}
            rec = np.asarray(d["node_ids"], dtype=np.int64)
            return np.array(
                [int(full_sizes[f2i[int(nid)]]) if int(nid) in f2i else 1
                 for nid in rec], dtype=np.int64)
        if "node_ids" in d and "parent_ids" in d:
            return subtree_sizes_from_tree(d["node_ids"], d["parent_ids"])
        return subtree_sizes_from_depths(d["depths"])

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
            "norm_db":     float(gap_to_primal_norm(d)[t]),   # P0.11
            # True terminal leaf label from the full tree (collector v3) when
            # present; else the "no recorded child" proxy.
            "is_leaf":     float((d["true_next_is_leaf"] if "true_next_is_leaf" in d
                                  else d["next_is_leaf"])[t]),
            "depth":       int(d["depths"][t]),
            "n_frac":      n_frac,
        }

        # Cost-to-go and subtree-size targets from the TRUE tree when available
        # (recorded-subtree node count), falling back to the visitation-order
        # `n_steps - t` proxy only for id-less legacy files. Using the tree
        # target trains the cost-to-go head (which drives node selection and the
        # rollout score) on real remaining-work, not visitation position.
        sizes = self._cache_sizes
        if sizes is not None:
            meta["steps_to_go"]  = float(sizes[t])   # nodes to close this subtree
            meta["subtree_size"] = float(sizes[t])
        else:
            meta["steps_to_go"]  = float(n_steps - t)   # legacy proxy (Gap 3)

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

    P0.3 fix (Option A — root->node paths). The dynamics model is a *causal*
    Transformer, so a training sequence is only meaningful if consecutive
    positions are a genuine parent->child lineage. SCIP visits nodes in
    best-/depth-first order, so the raw recorded order (z_0, z_1, ...) is NOT a
    path — z_{t+1} is usually not the child of z_t. Training on it would feed
    each transition a causal context of unrelated nodes.

    Instead we reconstruct, from the collector's `node_ids`/`parent_ids`, every
    true **root->leaf path** through the recorded tree and emit one sequence per
    path. Along a path, position t's action a_t (the branching variable chosen
    at that node) really does lead to the child latent at position t+1, so the
    causal Transformer sees exactly the ancestry that produced each transition —
    which is what makes the multi-step latent-rollout claim well-posed.

    One dataset *item* is therefore one root->leaf path (not one file); a single
    trajectory file yields as many items as it has recorded leaves. Paths are
    capped to the `max_path_len` nearest ancestors so deep trees stay within the
    Transformer's positional-embedding capacity and sequence lengths stay bounded.

    Each item is a dict (P = path length, transitions = P-1):
        z_seq          [P-1, H]        latents at the path's nodes (root..parent)
        a_seq          [P-1, H]        action embeddings (chosen var's h_vars)
        z_next_seq     [P-1, H]        child latents (the true next state)
        bound_next_seq [P-1]           child norm dual bound (Gap 2 grounding)
        reward_seq     [P-1]           child - parent dual-bound improvement
        dir_seq        [P-1]           branch direction +1/-1/0 (P0.12)
        valid_len      int             P-1

      and, when include_vars is set (subsampled to keep memory bounded):
        hv_seq         [P-1, K, H]     per-variable embeddings at parent nodes
        hv_next_seq    [P-1, K, H]     per-variable embeddings at child nodes
        var_mask       [P-1, K] bool   valid (non-padding) positions

    Files without `node_ids`/`parent_ids` are REFUSED by default (a `ValueError`),
    because visitation order would train the dynamics on wrong transitions;
    `allow_visitation_fallback=True` opts into the legacy single-sequence
    behaviour for old data / the unit-test fixture, knowingly accepting it.

    Encoding one trajectory is a single batched encoder call; the per-file
    encoded bundle is computed once and every path from that file slices it.
    """

    def __init__(self, files, model, device, include_vars=True,
                 max_vars_recon=64, max_path_len=64, seed=0, cache_dir=None,
                 allow_visitation_fallback=False):
        self.files = list(files)
        self.model = model
        self.device = device
        self.include_vars = include_vars
        self.max_vars_recon = max_vars_recon
        self.max_path_len = max_path_len
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # P0.3: files without node_ids/parent_ids cannot form true parent->child
        # paths. Silently falling back to SCIP visitation order trains the
        # dynamics model on WRONG transitions, so by default we refuse. Set
        # allow_visitation_fallback=True only for legacy data / the unit-test
        # fixture, knowingly accepting incorrect (visitation-order) dynamics.
        self.allow_visitation_fallback = allow_visitation_fallback

        # Build a flat index of (file_idx, path) so each __getitem__ returns one
        # root->leaf path. Path reconstruction reads only node_ids/parent_ids —
        # no GNN forward — so this pass is cheap.
        self.index = []            # list of (file_idx, path) ; path = list[int]
        for fi, f in enumerate(self.files):
            with np.load(f, allow_pickle=True) as d:
                T = int(d["n_steps"])
                if "node_ids" in d and "parent_ids" in d:
                    paths = _root_to_leaf_paths(
                        np.asarray(d["node_ids"]), np.asarray(d["parent_ids"]),
                        self.max_path_len)
                elif self.allow_visitation_fallback:
                    # Legacy: treat the whole visitation order as one sequence.
                    paths = [list(range(T))] if T >= 2 else []
                else:
                    raise ValueError(
                        f"{f} has no node_ids/parent_ids, so true parent->child "
                        "paths cannot be built and visitation order would train "
                        "the dynamics on WRONG transitions (P0.3). Recollect with "
                        "the current collector, or pass "
                        "allow_visitation_fallback=True to knowingly accept it.")
            for p in paths:
                self.index.append((fi, p))

        # In-memory bundle cache for the most recently encoded file, so the
        # several paths of one file reuse a single encoder forward when the
        # loader draws them consecutively.
        self._bundle_fi = None
        self._bundle = None

        # Encode-once disk cache of the per-file bundle. In Phase 3 the encoder
        # is frozen, so the bundle is identical every epoch and re-encoding
        # through the GNN each epoch is the dominant cost. Keyed by a fingerprint
        # of the encoder weights + settings, so it invalidates automatically if
        # any of those change.
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            fp = self._encoder_fingerprint()
            self.cache_dir = self.cache_dir / (
                f"enc{fp}_v{int(include_vars)}_k{max_vars_recon}_s{seed}")
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _encoder_fingerprint(self):
        """Short stable hash of encoder weights (frozen during Phase 3)."""
        import hashlib
        h = hashlib.md5()
        for k, v in sorted(self.model.encoder.state_dict().items()):
            h.update(k.encode())
            h.update(v.detach().cpu().float().numpy().tobytes())
        return h.hexdigest()[:12]

    def _cache_path(self, fi):
        # Key by file path (not list index) so train/val loaders sharing one
        # cache dir never collide on their 0-based indices.
        import hashlib
        key = hashlib.md5(str(self.files[fi]).encode()).hexdigest()[:16]
        return self.cache_dir / f"{key}.pt"

    def __len__(self):
        return len(self.index)

    @torch.no_grad()
    def __getitem__(self, i):
        fi, path = self.index[i]
        bundle = self._get_bundle(fi)
        return self._slice_path(bundle, path)

    def _get_bundle(self, fi):
        """Return the per-file encoded bundle, using memory then disk cache."""
        if fi == self._bundle_fi:
            return self._bundle
        bundle = None
        if self.cache_dir is not None:
            cp = self._cache_path(fi)
            if cp.exists():
                bundle = torch.load(cp, map_location="cpu")
        if bundle is None:
            bundle = self._encode_file(fi)
            if self.cache_dir is not None:
                torch.save(bundle, self._cache_path(fi))
        self._bundle_fi = fi
        self._bundle = bundle
        return bundle

    @torch.no_grad()
    def _encode_file(self, fi):
        """
        Encode every recorded node of one trajectory once. Returns a bundle of
        per-node tensors (indexed by recorded step) that any path can slice:
            z   [T, H]        graph latents
            a   [T, H]        action embeddings (chosen branching var)
            ndb [T]           normalised dual bounds
            hv  [T, K, H]     per-variable embeddings (only if include_vars)
        """
        d = np.load(self.files[fi], allow_pickle=True)
        T = int(d["n_steps"])
        H = self.model.hidden_dim

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
        a_list = []
        per_step_h = []
        for t in range(T):
            ht = h_all[var_batch == t]                # [n_vars_t, H]
            per_step_h.append(ht)
            bv = int(branch[t])
            bv = bv if 0 <= bv < ht.size(0) else 0
            a_list.append(ht[bv])
        a_all = torch.stack(a_list, dim=0)            # [T, H]

        ndb = torch.as_tensor(gap_to_primal_norm(d))          # P0.11

        # Branch direction of each node relative to its parent (+1 up / -1 down /
        # 0 root). Legacy files without it fall back to 0 (a constant feature).
        if "branch_dirs" in d:
            bdir = torch.as_tensor(
                np.asarray(d["branch_dirs"], dtype=np.float32))
        else:
            bdir = torch.zeros(T, dtype=torch.float32)

        bundle = {"z": z, "a": a_all, "ndb": ndb, "dir": bdir, "H": H}

        if self.include_vars and T > 0:
            n_vars = per_step_h[0].size(0)
            K = min(self.max_vars_recon, n_vars)
            # Fixed variable subset for the whole trajectory (same set each step,
            # since the var_dynamics head is shared across variables). Seed with
            # the file index so the subset is deterministic across epochs/cache.
            sub = np.random.default_rng(self.seed + fi).choice(
                n_vars, size=K, replace=False)
            sub = torch.as_tensor(np.sort(sub), dtype=torch.long)
            bundle["hv"] = torch.stack([h[sub] for h in per_step_h], dim=0)  # [T,K,H]

        return bundle

    def _slice_path(self, bundle, path):
        """Slice a root->leaf path out of a per-file bundle into a training item."""
        H = bundle["H"]
        # Need >= 2 nodes (one transition). Emit a trivially-masked item otherwise
        # so collate stays uniform.
        if len(path) < 2:
            return {
                "z_seq": torch.zeros(1, H), "a_seq": torch.zeros(1, H),
                "z_next_seq": torch.zeros(1, H),
                "bound_next_seq": torch.zeros(1),
                "reward_seq": torch.zeros(1),
                "dir_seq": torch.zeros(1),
                "valid_len": 0,
            }
        src = torch.as_tensor(path[:-1], dtype=torch.long)   # root..parent
        dst = torch.as_tensor(path[1:],  dtype=torch.long)   # child..leaf
        z, a, ndb, bdir = bundle["z"], bundle["a"], bundle["ndb"], bundle["dir"]
        out = {
            "z_seq":          z[src],
            "a_seq":          a[src],
            "z_next_seq":     z[dst],
            "bound_next_seq": ndb[dst],
            # Per-step reward = child-vs-parent dual-bound improvement (Fix 3).
            "reward_seq":     ndb[dst] - ndb[src],
            # Direction of the branch that leads parent(src) -> child(dst): it is
            # the child's own incoming branch direction (P0.12).
            "dir_seq":        bdir[dst],
            "valid_len":      int(src.numel()),
        }
        if self.include_vars and "hv" in bundle:
            hv = bundle["hv"]
            out["hv_seq"]      = hv[src]
            out["hv_next_seq"] = hv[dst]
            out["var_mask"]    = torch.ones(src.numel(), hv.size(1),
                                            dtype=torch.bool)
        return out


def _root_to_leaf_paths(node_ids, parent_ids, max_path_len):
    """
    Reconstruct every root->leaf path through the recorded B&B tree.

    node_ids[t] / parent_ids[t] are SCIP's node id and its parent's id for the
    t-th recorded node. Only a subset of tree nodes is recorded (SB branched
    there, within max_steps), so an edge exists between recorded nodes u,v iff
    v.parent_id == u.node_id. A leaf is a recorded node that is nobody's
    recorded parent; walking parent pointers up from each leaf (stopping when the
    parent is not itself recorded) yields that leaf's path. Fragments whose root
    has an unrecorded parent simply start at that fragment root.

    Returns a list of paths, each a list of recorded-step indices ordered
    root..leaf, truncated to the last `max_path_len` nodes and dropped if it has
    fewer than 2 nodes (no transition to learn).
    """
    node_ids = np.asarray(node_ids, dtype=np.int64)
    parent_ids = np.asarray(parent_ids, dtype=np.int64)
    T = len(node_ids)
    id2idx = {int(nid): t for t, nid in enumerate(node_ids)}  # last wins on dup

    parent_of = [None] * T          # parent's step index, or None if unrecorded
    is_parent = [False] * T
    for t in range(T):
        pid = int(parent_ids[t])
        if pid in id2idx and id2idx[pid] != t:
            pidx = id2idx[pid]
            parent_of[t] = pidx
            is_parent[pidx] = True

    paths = []
    for leaf in range(T):
        if is_parent[leaf]:
            continue                # not a leaf
        path = []
        cur = leaf
        seen = set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            path.append(cur)
            cur = parent_of[cur]
        path.reverse()              # root..leaf
        if len(path) >= 2:
            if len(path) > max_path_len:
                path = path[-max_path_len:]
            paths.append(path)
    return paths


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
        direction = torch.zeros(B, Tmax)
        tmask   = torch.zeros(B, Tmax, dtype=torch.bool)

        # P2.6: detect var-recon from ANY item, not just batch[0] — a degenerate
        # (valid_len=0) first item lacks hv_seq and would otherwise silently drop
        # var reconstruction for the whole batch.
        has_vars = include_vars and any("hv_seq" in b for b in batch)
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
            if "dir_seq" in b:
                direction[i, :L] = b["dir_seq"][:L]
            tmask[i, :L]  = True
            if has_vars and "hv_seq" in b:
                k = b["hv_seq"].size(1)
                hv_seq[i, :L, :k]  = b["hv_seq"][:L]
                hv_next[i, :L, :k] = b["hv_next_seq"][:L]
                vmask[i, :L, :k]   = b["var_mask"][:L]

        out = {
            "z_seq": z_seq, "a_seq": a_seq, "z_next_seq": z_next,
            "bound_next_seq": bound, "reward_seq": reward,
            "dir_seq": direction, "time_mask": tmask,
        }
        if has_vars:
            out["hv_seq"] = hv_seq
            out["hv_next_seq"] = hv_next
            out["var_mask"] = vmask
        return out

    return _collate
