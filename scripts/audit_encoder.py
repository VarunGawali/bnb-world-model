"""
Encoder compatibility audit: checks which encoder keys are present,
missing, or unexpected in the checkpoint vs the current architecture.

Run: PYTHONPATH=. python scripts/audit_encoder.py

Exit code 0 = encoder fully compatible (safe to encode dataset).
Exit code 1 = mismatches found (resolve before encoding).
"""
import sys, torch, yaml

CKPT   = "checkpoints/model_rl_best.pt"
CONFIG = "configs/default.yaml"

cfg = yaml.safe_load(open(CONFIG))
from bnb_wm.model.world_model import BnBWorldModel
model = BnBWorldModel(**cfg["model"])

sd = torch.load(CKPT, map_location="cpu", weights_only=False)
ckpt_state = sd["model"] if "model" in sd else sd

# Separate encoder keys from everything else.
ckpt_enc  = {k[len("encoder."):]: v for k, v in ckpt_state.items()
             if k.startswith("encoder.")}
model_enc = dict(model.encoder.state_dict())

missing    = sorted(set(model_enc) - set(ckpt_enc))
unexpected = sorted(set(ckpt_enc)  - set(model_enc))
shape_mismatch = [
    k for k in set(model_enc) & set(ckpt_enc)
    if model_enc[k].shape != ckpt_enc[k].shape
]

print("=" * 60)
print(f"Checkpoint : {CKPT}")
print(f"Encoder keys in checkpoint : {len(ckpt_enc)}")
print(f"Encoder keys in model      : {len(model_enc)}")
print("=" * 60)

if missing:
    print(f"\n[MISSING — will be randomly initialised] ({len(missing)} keys):")
    for k in missing:
        print(f"  encoder.{k}  {tuple(model_enc[k].shape)}")

if unexpected:
    print(f"\n[UNEXPECTED — in checkpoint but not in model] ({len(unexpected)} keys):")
    for k in unexpected:
        print(f"  encoder.{k}  {tuple(ckpt_enc[k].shape)}")

if shape_mismatch:
    print(f"\n[SHAPE MISMATCH] ({len(shape_mismatch)} keys):")
    for k in shape_mismatch:
        print(f"  encoder.{k}  ckpt={tuple(ckpt_enc[k].shape)}  "
              f"model={tuple(model_enc[k].shape)}")

n_issues = len(missing) + len(unexpected) + len(shape_mismatch)
if n_issues == 0:
    print("\n✓ Encoder fully compatible — safe to run encode_all.py")
    sys.exit(0)
else:
    total_missing_params = sum(model_enc[k].numel() for k in missing)
    total_model_params   = sum(v.numel() for v in model_enc.values())
    pct = 100 * total_missing_params / max(total_model_params, 1)
    print(f"\n✗ {n_issues} issue(s) found.")
    print(f"  Missing params: {total_missing_params:,} / {total_model_params:,} "
          f"({pct:.1f}% of encoder randomly initialised)")
    print("\nDo NOT run encode_all.py until the encoder is compatible.")
    sys.exit(1)
