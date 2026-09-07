import numpy as np
from pathlib import Path
from scripts.gen_cut_transitions import _reconstruct_lp, _solve_lp
import os; os.makedirs("/tmp/cut_test", exist_ok=True)

f = next(Path("data/trajectories").rglob("traj_sc_*.npz"))
d = np.load(f, allow_pickle=True)
print("file:", f, " T=", int(d["n_steps"]))

vf = d["var_features"][0]
cf = d["con_features"][0]
ei = d["edge_indices"][0]
ev = d["edge_values"][0]

A, b, c, x_lp = _reconstruct_lp(vf, cf, ei, ev)
print("A:", A.shape, " b[:3]:", b[:3], " c[:5]:", c[:5])
print("x_lp all_zero:", np.all(x_lp == 0))

print("Solving LP...")
obj, xs = _solve_lp(A, b, c)
print("obj:", obj, " xs is None:", xs is None)
if xs is not None:
    print("xs[:10]:", xs[:10])
    print("xs all_zero:", np.all(xs == 0))
    x_lp = xs

cl = np.asarray(d["cut_lhs"][0], dtype=np.float64)
cr = np.asarray(d["cut_rhs"][0], dtype=np.float64)
print("cl shape:", cl.shape, " cr:", cr)
if cl.ndim == 2: cl = cl[0]
rhs = float(cr[0]) if cr.ndim > 0 else float(cr)
n = vf.shape[0]
if cl.shape[0] > n: cl = cl[:n]
elif cl.shape[0] < n: cl = np.pad(cl, (0, n - cl.shape[0]))
viol = float(np.dot(cl, x_lp) - rhs)
print("violation:", viol, " (need > 1e-6)")
