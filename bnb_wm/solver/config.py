"""
config.py — SolverConfig dataclass for NeuralBnBSolver.

One flag per component. Two key properties:

  exact=True (default): forces Gates 2 and 6 off, so every prune is
  bound-justified and the solver is certifiably correct. is_exact rides
  into every SolveResult so results always carry their regime.

  ladder() / heuristic_tier(): canonical ablation rows, one component added
  at a time, so the ablation is a config sweep not eight forked solvers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Optional


@dataclass
class SolverConfig:
    # ------------------------------------------------------------------ #
    # Correctness / exactness                                              #
    # ------------------------------------------------------------------ #
    exact: bool = True
    gap_tolerance: float = 1e-4
    time_limit: float = 300.0
    node_limit: int = 500_000

    # ------------------------------------------------------------------ #
    # Branching mode                                                       #
    # ------------------------------------------------------------------ #
    # "most_fractional" | "random" | "policy" | "rollout"
    branch_mode: str = "rollout"

    # ------------------------------------------------------------------ #
    # Lookahead (rollout)                                                  #
    # ------------------------------------------------------------------ #
    lookahead_k: int = 5
    lookahead_depth: int = 3
    lookahead_gamma: float = 0.95
    size_weight: float = 0.0
    ctg_weight: float = 1.0
    branch_factor: int = 2
    use_reward_return: bool = True

    # Skip lookahead if the integrality head says leaf_prob >= this.
    leaf_prob_skip: float = 0.8
    # Skip lookahead if policy top-1 probability >= this (None = off).
    skip_confident: Optional[float] = None

    # ------------------------------------------------------------------ #
    # ORS cascade                                                         #
    # ------------------------------------------------------------------ #
    ors_cascade: bool = False
    ors_shallow_depth: int = 1
    ors_margin: float = 0.05
    # Node-level significance prune (UNSAFE — fires before LP).
    node_significance_prune: bool = False
    ors_sig_thresh: float = 0.1
    ors_p_explore: float = 0.05
    significance_fn: Optional[Callable] = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    # Katz blend                                                          #
    # ------------------------------------------------------------------ #
    katz_weight: float = 0.0
    katz_alpha: float = 0.1
    katz_iters: int = 3

    # ------------------------------------------------------------------ #
    # Cuts                                                                 #
    # ------------------------------------------------------------------ #
    # "none" | "heuristic" | "latent"
    cut_mode: str = "none"
    force_root_cuts: bool = False
    cut_budget_cap: int = 200
    cut_pool_max: int = 200
    max_cuts_per_node: int = 10
    cut_rounds: int = 3
    cut_beam: int = 4

    # Gate 7 sub-thresholds
    cut_integrality_thresh: float = 0.7   # near-leaf gate (leaf_prob)
    cut_min_nfrac: int = 3                # min fractional variables
    cut_depth_max: int = 20               # don't cut deep nodes
    cut_min_gain: float = 1e-4            # skip if last cut barely moved obj
    cut_subtree_thresh: float = 0.0       # predicted subtree >= this (0 = off)

    # Cut embedding normalisation: "none" | "mean" | "l2match"
    cut_embed_norm: str = "mean"

    # ------------------------------------------------------------------ #
    # Node selection                                                       #
    # ------------------------------------------------------------------ #
    # "bound" | "cost_to_go" | "subtree"
    node_selection: str = "bound"

    # ------------------------------------------------------------------ #
    # Primal heuristic                                                     #
    # ------------------------------------------------------------------ #
    primal_heuristic: bool = True
    primal_heuristic_every: int = 50

    # ------------------------------------------------------------------ #
    # Neural prune (UNSAFE)                                               #
    # ------------------------------------------------------------------ #
    neural_prune: bool = False
    neural_prune_margin: float = 0.02
    neural_prune_s_thresh: float = 2.0
    neural_prune_v_thresh: float = 0.5

    # ------------------------------------------------------------------ #
    # Derived: is_exact reflects which unsafe gates are actually on       #
    # ------------------------------------------------------------------ #
    @property
    def is_exact(self) -> bool:
        if self.exact:
            return True
        return not (self.node_significance_prune or self.neural_prune)

    def __post_init__(self):
        if self.exact:
            self.node_significance_prune = False
            self.neural_prune = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("significance_fn", None)
        d["is_exact"] = self.is_exact
        return d

    # ------------------------------------------------------------------ #
    # Ablation ladder                                                      #
    # ------------------------------------------------------------------ #

    @classmethod
    def mf(cls, **kw) -> "SolverConfig":
        """most_fractional baseline."""
        return cls(branch_mode="most_fractional", cut_mode="none",
                   ors_cascade=False, katz_weight=0.0,
                   node_selection="bound", **kw)

    @classmethod
    def policy(cls, **kw) -> "SolverConfig":
        """GNN policy only (no rollout)."""
        return cls(branch_mode="policy", cut_mode="none",
                   ors_cascade=False, katz_weight=0.0,
                   node_selection="bound", **kw)

    @classmethod
    def rollout(cls, **kw) -> "SolverConfig":
        """Policy + rollout lookahead."""
        return cls(branch_mode="rollout", cut_mode="none",
                   ors_cascade=False, katz_weight=0.0,
                   node_selection="bound", **kw)

    @classmethod
    def rollout_ors(cls, **kw) -> "SolverConfig":
        """rollout + ORS cascade."""
        return cls(branch_mode="rollout", cut_mode="none",
                   ors_cascade=True, katz_weight=0.0,
                   node_selection="bound", **kw)

    @classmethod
    def rollout_ors_katz(cls, **kw) -> "SolverConfig":
        """rollout + ORS + Katz blend."""
        return cls(branch_mode="rollout", cut_mode="none",
                   ors_cascade=True, katz_weight=0.3,
                   node_selection="bound", **kw)

    @classmethod
    def cuts_heur(cls, **kw) -> "SolverConfig":
        """rollout + ORS + Katz + heuristic cuts."""
        return cls(branch_mode="rollout", cut_mode="heuristic",
                   ors_cascade=True, katz_weight=0.3,
                   node_selection="bound", **kw)

    @classmethod
    def cuts_latent(cls, **kw) -> "SolverConfig":
        """rollout + ORS + Katz + latent cut beam."""
        return cls(branch_mode="rollout", cut_mode="latent",
                   ors_cascade=True, katz_weight=0.3,
                   node_selection="bound", **kw)

    @classmethod
    def nodesel_ctg(cls, **kw) -> "SolverConfig":
        """Full stack + cost-to-go node selection."""
        return cls(branch_mode="rollout", cut_mode="latent",
                   ors_cascade=True, katz_weight=0.3,
                   node_selection="cost_to_go", **kw)

    @classmethod
    def heuristic_tier(cls, tier: str, **kw) -> "SolverConfig":
        """Convenience: 'fast' = mf, 'standard' = rollout, 'full' = nodesel_ctg."""
        return {
            "fast": cls.mf,
            "standard": cls.rollout,
            "full": cls.nodesel_ctg,
        }[tier](**kw)
