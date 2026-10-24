# Ablation experiments

## Purpose

The ablation study separates two questions:

1. Structural ablation: PLM branch, DDE branch, two raw branch probabilities,
   and the complete seven-input fusion.
2. Sampling analysis: original QC-SMOTE geometry versus the finalized
   KMeans-constrained Minkowski-distance variant for each branch.

## Evaluation design

- Use the same fixed five folds and threshold `0.5` for every configuration.
- Fit scaling and sampling only on an outer-training fold.
- Report mean fold metrics and pooled OOF metrics separately.
- Use independent-test labels only after a configuration is fixed.
- Treat the two single branches as branch models; do not pass a single
  probability through a second-layer logistic regression merely to create an
  ablation row.

## Publication status

The completed historical ablation script contains `C=0.12`, while the deployed
final model and its manifest use `C=0.11`. The script is withheld from this
release until it is rerun or corrected and its exported metrics are checked.
This prevents an apparently runnable script from silently evaluating a model
different from the one reported as final.

Sampling comparisons belong in the sampling-method section of the paper. The
structural comparison belongs in the model-ablation section.
