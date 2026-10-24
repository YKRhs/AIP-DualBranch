# Explainability analyses

## Purpose

Interpret each model level with a method appropriate to that level rather than
forcing one attribution method onto the entire pipeline.

## Completed analyses

### PLM branch

- attention over relative peptide position on the independent test set;
- strict-OOF attention over the training set;
- single-residue replacement sensitivity;
- local sequence-window sensitivity.

These analyses identify residues and local sequence regions to which the PLM
representation is sensitive. Attention is not itself a causal importance
score, so replacement and window perturbations provide complementary evidence.

### DDE branch

- strict five-fold OOF permutation feature importance over all 400 dipeptides;
- top-dipeptide summaries and the 20 x 20 DDE importance matrix.

Each validation column is permuted while its fold model remains fixed. Ranking
therefore uses training OOF data, not independent-test labels.

### Fusion layer

- seven-input logistic-regression contribution analysis;
- SHAP analysis in logit space;
- branch-level aggregation of PLM and DDE contributions.

### Cross-branch link

PLM analyses localize sensitive residues and positions, whereas DDE importance
identifies composition/order patterns such as specific dipeptides. Agreement is
assessed by checking whether residues highlighted by PLM perturbation or
attention participate in high-ranked DDE pairs. The two branches are related
but not expected to produce identical rankings because one models contextual
sequence representation and the other models global dipeptide statistics.

## Reproduction status

The completed source scripts currently depend on the large archived checkpoint,
TabICLv2 offload, and intermediate feature layout. They are not copied into this
release until those paths are converted to public configuration arguments and
the generated-cache requirements are documented. This directory records the
analysis contract without claiming a portable command that does not yet exist.
