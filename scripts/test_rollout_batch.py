"""Sanity-check the batched rollout for multiple (depth, branch_factor) configs."""
import time
import torch

from bnb_wm.model.world_model import BnBWorldModel
from bnb_wm.training.checkpoint import load_weights_only

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BnBWorldModel(hidden_dim=128, n_gnn_layers=3).to(device).eval()
load_weights_only(model, "checkpoints/model_rl_best.pt", device)

torch.manual_seed(42)

V, H = 32, 128
z = torch.randn(1, H, device=device)
h = torch.randn(V, H, device=device)

mask = torch.zeros(V, dtype=torch.bool, device=device)
mask[:12] = True

print("Device:", device)
print("Testing batched rollout over multiple configs...")

configs = [
    (2, 1, True),
    (2, 2, True),
    (3, 2, True),
    (2, 1, False),
    (2, 2, False),
]

for depth, branch_factor, reward_return in configs:
    with torch.no_grad():
        score = model.rollout_candidate(
            z, h, cand_idx=0,
            depth=depth,
            gamma=0.95,
            valid_mask=mask,
            size_weight=1.0,
            ctg_weight=1.0,
            branch_factor=branch_factor,
            use_reward_return=reward_return,
            expand_both_children=True,
        )
    assert torch.isfinite(torch.tensor(score)), f"non-finite score for {depth}/{branch_factor}"
    print(
        f"depth={depth}, b={branch_factor}, rr={reward_return}: score={score:.6f}  OK"
    )

print("\nAll configs: PASS")

# ---------------------------------------------------------
# Speed check
# ---------------------------------------------------------
depth, branch_factor = 3, 2

def run():
    with torch.no_grad():
        return model.rollout_candidate(
            z, h, cand_idx=0,
            depth=depth, gamma=0.95, valid_mask=mask,
            size_weight=1.0, ctg_weight=1.0,
            branch_factor=branch_factor,
            use_reward_return=True, expand_both_children=True,
        )

# Warm-up
run(); run()

if device.type == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    run()
if device.type == "cuda":
    torch.cuda.synchronize()
avg_t = (time.perf_counter() - t0) / 5

print(f"\nBatched rollout avg ({depth}-deep, b={branch_factor}): {avg_t:.4f}s")
print("\nROLLOUT TEST PASSED")
