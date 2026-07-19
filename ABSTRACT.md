# AAAI 2027 Submission — Title & Abstract

> Formatting notes (from the AAAI anonymous-submission template):
> - Author block must read **"Anonymous Submission"** — no names/affiliations.
> - Clear the PDF metadata before submitting (identity leak prevention).
> - Anonymize any self-referential citations and the Code/Datasets links below.
> - No AAAI copyright footer on page 1 for the anonymous version.
> - Abstract is a single paragraph; no citations inside the abstract.

## Title

**Learning to Branch by Imagining: A World Model for Branch-and-Bound**

## Abstract

Branch-and-Bound (B&B) is the core of exact Mixed-Integer Programming solvers,
and its efficiency hinges on variable selection. Strong branching yields
excellent decisions but is prohibitively expensive, solving child linear
programs (LPs) for every candidate; learned branchers imitate it cheaply, yet
score each candidate from the current node alone, with no model of the subtree
a decision induces. We introduce a *world model* for B&B: a learned latent
dynamics model that simulates the consequences of a branching decision in
latent space, approximating strong branching's subtree evaluation without
solving any LPs. A bipartite graph encoder maps each node to a latent state,
and a causal Transformer predicts future latent states together with
per-variable embeddings, enabling multi-step, tree-structured rollouts in which
the policy is re-applied to imagined states. Candidates are scored by a
MuZero-style return that combines a learned per-step reward with a cost-to-go
value directly estimating remaining tree size. The same world model
additionally guides node and cut selection while preserving optimality. We
evaluate on Set Cover across difficulty tiers, measuring node-count reduction
against a strong solver baseline and ablating each planning component.

## Header fields (AAAI abstract page — anonymized for submission)

    Code — https://anonymous.4open.science/r/bnb-world-model   (anonymize)
    Datasets — included as supplementary / anonymized link
    Extended version — (optional appendix)

## Sentence to add for the camera-ready / full paper (once results exist)

Replace the final sentence's tail with the measured result, e.g.:
"…On Set Cover across difficulty tiers, our latent planner reduces explored
nodes by **X%** (mean over N seeds) relative to SCIP pseudocost branching
(Wilcoxon signed-rank p < 0.05), and ablations confirm that each planning
component contributes to the reduction."
