"""
collector_instances.py — Feed training-distribution instances to an Ecole env.

The training data was collected by a custom generator (Bernoulli(density) A,
U(1,10) costs, RHS = 1) whose instances differ from Ecole's SetCoverGenerator
in column density (~25 vs ~37 rows/col at 500×1000) and cost spread.

This module provides a generator that builds instances from the same
distribution the collector used and wraps them as Ecole SCIP models, so
ablation.py can run an unconfounded measurement of the feature fix on
matched instances.

Usage in ablation.py:
    --instance_source collector   # default: ecole (SetCoverGenerator)

API:
    gen = CollectorInstanceGenerator(n_rows, n_cols, density, seed)
    ecole_model = next(gen)   # drop-in for ecole.instance.SetCoverGenerator
"""

from __future__ import annotations

import numpy as np

try:
    import ecole
    from pyscipopt import Model as SCIPModel
except ImportError:
    ecole = None
    SCIPModel = None


class CollectorInstanceGenerator:
    """
    Infinite generator of set-cover instances matching the collector's
    distribution (Bernoulli A, U(1,10) costs, RHS=1), wrapped as Ecole
    SCIP models.

    Parameters match the collector's _gen_instance():
        n_rows:  number of constraints (sets to cover)
        n_cols:  number of variables (set elements)
        density: Bernoulli probability for each A entry
        seed:    base seed; instances are seeded consecutively
    """

    def __init__(
        self,
        n_rows: int = 500,
        n_cols: int = 1000,
        density: float = 0.05,
        seed: int = 0,
    ):
        if ecole is None or SCIPModel is None:
            raise ImportError("ecole and pyscipopt are required")
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.density = density
        self._rng = np.random.default_rng(seed)
        self._count = 0

    def seed(self, s: int) -> None:
        self._rng = np.random.default_rng(s)

    def __iter__(self):
        return self

    def __next__(self):
        return self._build()

    def _build(self):
        rng = self._rng
        nr, nc, d = self.n_rows, self.n_cols, self.density

        # A ∈ {0,1}^{nr×nc}, Bernoulli(d); ensure each row has ≥1 nonzero.
        A = (rng.random((nr, nc)) < d).astype(np.float64)
        empty_rows = np.where(A.sum(axis=1) == 0)[0]
        for r in empty_rows:
            A[r, rng.integers(nc)] = 1.0

        # c ~ U(1, 10), minimise sum c_j x_j
        c = rng.uniform(1.0, 10.0, nc)

        # Build pyscipopt model: min c^T x  s.t. Ax >= 1, x ∈ {0,1}
        m = SCIPModel()
        m.hideOutput(True)
        xs = [m.addVar(f"x{j}", vtype="B", obj=float(c[j])) for j in range(nc)]
        for i in range(nr):
            nonzeros = np.where(A[i] > 0)[0]
            m.addCons(
                sum(xs[j] for j in nonzeros) >= 1.0,
                name=f"c{i}",
            )
        m.setMinimize()

        self._count += 1
        return ecole.scip.Model.from_pyscipopt(m)


def make_generator(
    source: str,
    n_rows: int,
    n_cols: int,
    density: float = 0.05,
    seed: int = 0,
):
    """
    Factory used by ablation.py.

    source: "ecole"     -> ecole.instance.SetCoverGenerator (default, community benchmark)
            "collector" -> CollectorInstanceGenerator (training distribution)
    """
    if source == "collector":
        return CollectorInstanceGenerator(n_rows, n_cols, density, seed)
    # default: Ecole's own generator
    gen = ecole.instance.SetCoverGenerator(
        n_rows=n_rows, n_cols=n_cols, density=density)
    gen.seed(seed)
    return gen
