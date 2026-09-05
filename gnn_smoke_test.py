import torch
import torch.nn.functional as F
from bnb_wm.model.world_model import BnBWorldModel


def check(name, condition):
    if not condition:
        raise RuntimeError(f"FAIL: {name}")
    print(f"PASS: {name}")


def finite(name, x):
    check(name, torch.isfinite(x).all().item())


torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Small model so this is a cheap structural test.
model = BnBWorldModel(
    hidden_dim=32,
    n_gnn_layers=2,
    n_gnn_heads=2,
    n_dyn_layers=2,
    n_dyn_heads=2,
    max_seq=16,
    dyn_residual=True,
    dyn_heteroscedastic=True,
).to(device).eval()

B, T, V, H = 2, 4, 7, 32

z_seq = torch.randn(B, T, H, device=device)
a_seq = torch.randn(B, T, H, device=device)
h_vars_seq = torch.randn(B, T, V, H, device=device)

# Direction-conditioned transitions.
d_seq = torch.tensor(
    [
        [1.0, -1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0, 1.0],
    ],
    device=device,
)

print("\n=== 1. Parallel dynamics ===")

z_pred, logvar = model.dynamics(
    z_seq,
    a_seq,
    d_seq,
    return_logvar=True,
)

check("z_pred shape", z_pred.shape == (B, T, H))
check("logvar shape", logvar.shape == (B, T, H))
finite("z_pred finite", z_pred)
finite("logvar finite", logvar)

print("\n=== 2. Parallel dynamics + variable states ===")

z_pred2, h_pred = model.dynamics.forward_with_vars(
    z_seq,
    a_seq,
    h_vars_seq,
    d_seq,
)

check("forward_with_vars z shape", z_pred2.shape == (B, T, H))
check("forward_with_vars h shape", h_pred.shape == (B, T, V, H))
finite("h_pred finite", h_pred)

print("\n=== 3. Single-step dynamics ===")

z_t = z_seq[:, 0]
a_t = a_seq[:, 0]
d_t = d_seq[:, 0]

z_next, tokens = model.dynamics.step(
    z_t,
    a_t,
    None,
    d_t,
)

check("step z shape", z_next.shape == (B, H))
check("step token shape", tokens.shape == (B, 1, H))
finite("step z finite", z_next)
finite("step tokens finite", tokens)

print("\n=== 4. Multi-step token buffer ===")

past = None
z_cur = z_seq[:, 0]

for t in range(T):
    z_cur, past = model.dynamics.step(
        z_cur,
        a_seq[:, t],
        past,
        d_seq[:, t],
    )

    check(
        f"token buffer step {t}",
        past.shape == (B, t + 1, H),
    )
    finite(f"rollout latent step {t}", z_cur)

print("\n=== 5. Full latent step with variable embeddings ===")

# step_full is written for single-graph variable states.
z_single = z_seq[:1, 0]
a_single = a_seq[:1, 0]
h_single = h_vars_seq[:1, 0]

z_full, h_full, tok_full = model.dynamics.step_full(
    z_single,
    a_single,
    h_single,
    None,
    1.0,
)

check("step_full z shape", z_full.shape == (1, H))
check("step_full h shape", h_full.shape == (V, H))
check("step_full token shape", tok_full.shape == (1, 1, H))

finite("step_full z finite", z_full)
finite("step_full h finite", h_full)

print("\n=== 6. World-model helper heads ===")

bound = model.dynamics_bound_pred(z_pred)
reward = model.dynamics_reward_pred(z_pred)

check("bound shape", bound.shape == (B, T))
check("reward shape", reward.shape == (B, T))

finite("bound finite", bound)
finite("reward finite", reward)

print("\n=== 7. Global context ===")

global_ctx = torch.randn(B, 6, device=device)

z_context = model.add_global_context(
    z_seq[:, 0],
    global_ctx,
)

check("global context shape", z_context.shape == (B, H))
finite("global context finite", z_context)

# The projection is zero-initialised, so it must initially be an exact no-op.
check(
    "zero-init global context is no-op",
    torch.allclose(z_context, z_seq[:, 0]),
)

print("\n=== 8. Autoregressive rollout ===")

rollout_actions = torch.randn(B, 3, H, device=device)
rollout_dirs = torch.tensor(
    [
        [1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
    ],
    device=device,
)

rollout_pred = model.dynamics.rollout(
    z_seq[:, 0],
    rollout_actions,
    d_seq=rollout_dirs,
)

check("rollout shape", rollout_pred.shape == (B, 3, H))
finite("rollout finite", rollout_pred)

print("\n=== 9. Backpropagation / gradient sanity ===")

model.train()

z_pred_train, logvar_train = model.dynamics(
    z_seq,
    a_seq,
    d_seq,
    return_logvar=True,
)

loss = (
    F.mse_loss(z_pred_train, z_seq)
    + 0.01 * logvar_train.mean()
)

check("loss finite", torch.isfinite(loss).item())

loss.backward()

grad_count = 0

for name, param in model.dynamics.named_parameters():
    if param.requires_grad and param.grad is not None:
        finite(f"gradient finite: {name}", param.grad)
        grad_count += 1

check("dynamics gradients exist", grad_count > 0)

print("\n" + "=" * 60)
print("ALL UPDATED DYNAMICS / WORLD-MODEL TESTS PASSED")
print("=" * 60)
