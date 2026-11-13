import torch

from bnb_wm.model.world_model import BnBWorldModel
from bnb_wm.training.checkpoint import load_weights_only


def _split_layout(state, hcons=None):
    p = "dynamics.var_dynamics."
    s = dict(state)
    W = s.pop(f"{p}net.0.weight")
    H = W.shape[1] // 3
    s[f"{p}W_h.weight"], s[f"{p}W_z.weight"], s[f"{p}W_a.weight"] = (
        W[:, :H].clone(), W[:, H:2 * H].clone(), W[:, 2 * H:].clone())
    s[f"{p}W_h.bias"] = s.pop(f"{p}net.0.bias")
    s[f"{p}out.weight"] = s.pop(f"{p}net.3.weight")
    s[f"{p}out.bias"] = s.pop(f"{p}net.3.bias")
    s["dynamics.h_cons_proj.weight"] = hcons if hcons is not None else torch.zeros(H, H)
    return s


def test_split_var_dynamics_checkpoint_loads_exactly(tmp_path):
    torch.manual_seed(0)
    src = BnBWorldModel().eval()
    path = tmp_path / "split.pt"
    torch.save({"model": _split_layout(src.state_dict())}, path)

    dst = BnBWorldModel().eval()
    load_weights_only(dst, path)
    assert dst._load_report["h_cons_proj_dropped_max_abs"] == 0.0

    H = src.dynamics.hidden_dim
    h, z, a = torch.randn(7, H), torch.randn(H), torch.randn(H)
    with torch.no_grad():
        assert torch.equal(src.dynamics.var_dynamics(h, z, a), dst.dynamics.var_dynamics(h, z, a))
    for k, v in src.state_dict().items():
        assert torch.equal(v, dst.state_dict()[k]), k


def test_nonzero_h_cons_proj_is_reported(tmp_path):
    src = BnBWorldModel().eval()
    path = tmp_path / "split.pt"
    H = src.dynamics.hidden_dim
    torch.save({"model": _split_layout(src.state_dict(), torch.full((H, H), 0.5))}, path)
    dst = BnBWorldModel().eval()
    import pytest
    with pytest.warns(UserWarning, match="h_cons_proj"):
        load_weights_only(dst, path)
    assert dst._load_report["h_cons_proj_dropped_max_abs"] == 0.5
