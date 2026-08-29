"""
verify_vectorization.py — prove the vectorized pool/head ops equal the old loops.

Run on Tyrone (needs torch_geometric):  python scripts/verify_vectorization.py

Rebuilds the ORIGINAL per-graph Python loops as reference implementations and
checks the new vectorized CrossAttentionPool and _frac_mean produce identical
outputs (to fp tolerance) on random multi-graph batches. Then runs a full
BnBWorldModel forward to confirm the wiring is intact.
"""
import torch
import torch.nn.functional as F

from bnb_wm.model.encoder import CrossAttentionPool
from bnb_wm.model.heads import _frac_mean
from bnb_wm.model.world_model import BnBWorldModel

torch.manual_seed(0)
H = 128


def ref_pool(pool, h_vars, batch_vec):
    """Original per-b loop for CrossAttentionPool."""
    batch_size = int(batch_vec.max().item()) + 1
    keys, values = pool.W_k(h_vars), pool.W_v(h_vars)
    out = []
    for b in range(batch_size):
        mask = batch_vec == b
        k, v = keys[mask], values[mask]
        attn = F.softmax((pool.query @ k.T) * pool.scale, dim=-1)
        out.append((attn @ v).squeeze(0))
    return torch.stack(out, 0)


def ref_frac_mean(z, h_vars, batch_vec, frac_mask):
    """Original per-b loop for the head fractional-mean."""
    batch_size = z.size(0)
    if frac_mask is not None and frac_mask.any():
        fm = torch.zeros_like(z)
        for b in range(batch_size):
            sel = (batch_vec == b) & frac_mask
            fm[b] = h_vars[sel].mean(0) if sel.any() else z[b]
        return fm
    return z


def build_batch(sizes):
    total = sum(sizes)
    h = torch.randn(total, H)
    bvec = torch.cat([torch.full((n,), i) for i, n in enumerate(sizes)])
    return h, bvec


# ---- CrossAttentionPool ----
pool = CrossAttentionPool(H).eval()
for sizes in ([7, 3, 11, 1], [5], [4, 4, 4]):
    h, bvec = build_batch(sizes)
    got, exp = pool(h, bvec), ref_pool(pool, h, bvec)
    err = (got - exp).abs().max().item()
    assert err < 1e-5, f"pool mismatch {err} for {sizes}"
    print(f"pool  {str(sizes):16} max_err={err:.2e}  OK")

# ---- _frac_mean (incl. graph with zero fractional vars) ----
for sizes in ([7, 3, 11], [5, 5, 5]):
    total = sum(sizes)
    h, bvec = build_batch(sizes)
    z = torch.randn(len(sizes), H)
    fm = torch.rand(total) > 0.5
    # force graph 0 to have NO fractional vars -> must fall back to z[0]
    fm[bvec == 0] = False
    got, exp = _frac_mean(z, h, bvec, fm), ref_frac_mean(z, h, bvec, fm)
    err = (got - exp).abs().max().item()
    assert err < 1e-5, f"frac_mean mismatch {err} for {sizes}"
    print(f"frac  {str(sizes):16} max_err={err:.2e}  OK")

# also the all-None / empty-mask fallback path
z = torch.randn(3, H)
h, bvec = build_batch([4, 4, 4])
assert torch.equal(_frac_mean(z, h, bvec, None), z)
assert torch.equal(_frac_mean(z, h, bvec, torch.zeros(12, dtype=torch.bool)), z)
print("frac  none/empty-mask fallback   OK")

# ---- full model forward smoke ----
from torch_geometric.data import Data, Batch
def rand_graph(nv=10, nc=6):
    x = torch.zeros(nv + nc, 19)
    x[:nv] = torch.randn(nv, 19)
    x[nv:, :5] = torch.randn(nc, 5)
    node_type = torch.cat([torch.zeros(nv), torch.ones(nc)]).long()
    ei = torch.stack([torch.randint(nv, nv + nc, (20,)), torch.randint(0, nv, (20,))])
    ei = torch.cat([ei, ei.flip(0)], dim=1)
    ea = torch.randn(ei.size(1), 3)
    return Data(x=x, edge_index=ei, node_type=node_type, edge_attr=ea)

batch = Batch.from_data_list([rand_graph(), rand_graph(8, 5), rand_graph(12, 7)])
model = BnBWorldModel().eval()
with torch.no_grad():
    scores, z = model(batch)
assert z.shape == (3, H), z.shape
print(f"model forward  scores={tuple(scores.shape)} z={tuple(z.shape)}  OK")

# ---- ShardedBatchSampler: covers every item exactly once, len is exact ----
from bnb_wm.data.datasets import ShardedBatchSampler
# 5 files with uneven node counts -> item_files like [0,0,0, 1,1, 2,...]
counts = [17, 3, 40, 1, 25]
item_files = [fi for fi, n in enumerate(counts) for _ in range(n)]
for fpb in (1, 4, 8):
    smp = ShardedBatchSampler(item_files, batch_size=32,
                              files_per_batch=fpb, shuffle=True)
    batches = list(iter(smp))
    seen = [gi for b in batches for gi in b]
    assert sorted(seen) == list(range(len(item_files))), \
        f"coverage broken fpb={fpb}: {len(seen)} vs {len(item_files)}"
    assert len(batches) == len(smp), \
        f"len mismatch fpb={fpb}: {len(batches)} vs {len(smp)}"
    files_touched = max(len({item_files[gi] for gi in b}) for b in batches)
    assert files_touched <= fpb, f"too many files/batch fpb={fpb}: {files_touched}"
    print(f"sampler fpb={fpb}  {len(batches)} batches, "
          f"all {len(item_files)} items once, <= {fpb} files/batch  OK")

# ---- size-aware batching: hard files -> smaller chunks, uniform edge budget --
counts = [40] * 6
item_files2 = [fi for fi, n in enumerate(counts) for _ in range(n)]
cost = {0: 1.0, 1: 1.0, 2: 3.0, 3: 3.0, 4: 9.0, 5: 9.0}   # median 3.0
smp = ShardedBatchSampler(item_files2, batch_size=32, files_per_batch=4,
                          shuffle=True, file_node_cost=cost)
sa_batches = list(iter(smp))
seen = [gi for b in sa_batches for gi in b]
assert sorted(seen) == list(range(len(item_files2))), "size-aware coverage"
assert len(sa_batches) == len(smp), "size-aware len"
assert smp.per_file_of[0] > smp.per_file_of[2] > smp.per_file_of[4], \
    f"cost->chunk-size not monotone: {smp.per_file_of}"
print(f"size-aware  per_file_of={smp.per_file_of} "
      f"(easy>med>hard), coverage+len OK")

# ---- soft/ranking policy loss ------------------------------------------------
from bnb_wm.training.losses import policy_loss_soft, policy_loss_masked
sc = torch.randn(20, requires_grad=True)
aset = torch.tensor([2, 5, 7, 11, 14])
sb = torch.tensor([0.1, 0.9, 0.3, 0.2, 0.05])   # candidate 5 (local idx 1) best
ls, acc, _ = policy_loss_soft(sc, aset, sb, 1)
ls.backward()
assert torch.isfinite(sc.grad).all(), "soft-loss grad not finite"
# aligned policy -> lower loss than a random one
al = torch.full((20,), -5.0); al[aset] = torch.tensor([0., 3., 0., 0., 0.])
al.requires_grad_(True)
la, aa, _ = policy_loss_soft(al, aset, sb, 1)
assert la.item() < ls.item() and aa == 1.0, "aligned policy not rewarded"
# single-candidate degenerate -> falls back to hard, finite
sd = torch.randn(20, requires_grad=True)
ld, _, _ = policy_loss_soft(sd, torch.tensor([3]), torch.tensor([0.5]), 0)
assert torch.isfinite(ld), "degenerate soft-loss not finite"
# alpha extremes differ
lp, _, _ = policy_loss_soft(sc, aset, sb, 1, alpha=0.0)
lh, _, _ = policy_loss_soft(sc, aset, sb, 1, alpha=1.0)
assert abs(lp.item() - lh.item()) > 1e-3, "alpha has no effect"
print(f"soft-loss  grad OK, aligned<random ({la.item():.3f}<{ls.item():.3f}), "
      f"pure_soft={lp.item():.3f} pure_hard={lh.item():.3f}  OK")

print("\nALL CHECKS PASSED — vectorized ops match the old loops exactly.")
