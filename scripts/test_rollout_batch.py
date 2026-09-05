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
print("Testing recursive vs batched rollout...")

configs = [
    (2, 1, True),
    (2, 2, True),
    (3, 2, True),
]

for depth, branch_factor, reward_return in configs:
    with torch.no_grad():

        # Reference implementation
        old = model.rollout_candidate(
            z, h, cand_idx=0,
            depth=depth,
            gamma=0.95,
            valid_mask=mask,
            size_weight=1.0,
            ctg_weight=1.0,
            branch_factor=branch_factor,
            use_reward_return=reward_return,
            expand_both_children=True,
            batched=False,
        )

        # Batched implementation
        new = model.rollout_candidate(
            z, h, cand_idx=0,
            depth=depth,
            gamma=0.95,
            valid_mask=mask,
            size_weight=1.0,
            ctg_weight=1.0,
            branch_factor=branch_factor,
            use_reward_return=reward_return,
            expand_both_children=True,
            batched=True,
        )

    diff = abs(old - new)

    print(
        f"depth={depth}, b={branch_factor}: "
        f"old={old:.6f}  new={new:.6f}  diff={diff:.2e}"
    )

    assert diff < 1e-4, f"MISMATCH: diff={diff}"

print("\nEquivalence: PASS")

# ---------------------------------------------------------
# Tiny speed comparison
# ---------------------------------------------------------
depth, branch_factor = 3, 2

def run(batched):
    with torch.no_grad():
        return model.rollout_candidate(
            z, h, cand_idx=0,
            depth=depth,
            gamma=0.95,
            valid_mask=mask,
            size_weight=1.0,
            ctg_weight=1.0,
            branch_factor=branch_factor,
            use_reward_return=True,
            expand_both_children=True,
            batched=batched,
        )

# Warm-up
run(False)
run(True)

if device.type == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()

for _ in range(3):
    run(False)

if device.type == "cuda":
    torch.cuda.synchronize()
old_t = (time.perf_counter() - t0) / 3

if device.type == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()

for _ in range(3):
    run(True)

if device.type == "cuda":
    torch.cuda.synchronize()
new_t = (time.perf_counter() - t0) / 3

print(f"Recursive avg: {old_t:.4f}s")
print(f"Batched   avg: {new_t:.4f}s")
print(f"Speedup:      {old_t / new_t:.2f}x")

print("\nROLLOUT BATCH TEST PASSED")
