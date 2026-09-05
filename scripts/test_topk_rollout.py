"""Verify rollout_top_k_batched matches K sequential rollout_candidate calls."""
import torch
from bnb_wm.model.world_model import BnBWorldModel
from bnb_wm.training.checkpoint import load_weights_only

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BnBWorldModel(hidden_dim=128, n_gnn_layers=3).to(device).eval()
load_weights_only(model, "checkpoints/model_rl_best.pt", device)

torch.manual_seed(0)
V, H = 32, 128
z = torch.randn(1, H, device=device)
h = torch.randn(V, H, device=device)
mask = torch.zeros(V, dtype=torch.bool, device=device)
mask[:16] = True

cands = torch.tensor([0, 2, 5, 8], dtype=torch.long, device=device)

configs = [
    dict(depth=2, branch_factor=1, use_reward_return=True,  ctg_weight=1.0),
    dict(depth=2, branch_factor=2, use_reward_return=True,  ctg_weight=1.0),
    dict(depth=3, branch_factor=2, use_reward_return=False, ctg_weight=0.0),
]

print(f"Device: {device}")
for cfg in configs:
    with torch.no_grad():
        # Sequential reference
        seq = torch.tensor([
            model.rollout_candidate(
                z, h, int(c),
                gamma=0.95, valid_mask=mask, size_weight=0.0,
                **cfg,
            )
            for c in cands
        ], device=device)

        # Batched
        bat = model.rollout_top_k_batched(
            z, h, cands, gamma=0.95, valid_mask=mask, size_weight=0.0,
            **cfg,
        )

    diff = (seq - bat).abs().max().item()
    tag = cfg
    print(f"depth={cfg['depth']} b={cfg['branch_factor']} rr={cfg['use_reward_return']}: "
          f"max_diff={diff:.2e}  {'PASS' if diff < 1e-3 else 'FAIL'}")
    assert diff < 1e-3, f"MISMATCH {diff}"

print("\nAll configs: PASS")
