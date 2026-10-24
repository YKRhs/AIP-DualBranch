# Seven-input probability fusion

## Purpose

Fit the finalized logistic-regression meta model from strict OOF branch
probabilities. This stage is lightweight and does not require rerunning the PLM,
GRANDE, or TabICLv2.

## Inputs

- `evaluation/base_probabilities/training_oof.csv`;
- `evaluation/base_probabilities/independent_test.csv`.

The seven inputs are `p_plm`, `p_dde`, absolute disagreement, mean probability,
probability product, PLM confidence distance, and DDE confidence distance.

## Command

```bash
python experiments/fusion/fit_fixed_fusion.py
```

## Outputs

The command writes the fitted model, five-fold training metrics, independent
test metrics, and test probabilities to `results/fusion_reproduction/`.

The hyperparameters are fixed to the deployed model (`C=0.11`, positive class
weight `1.56`, L2 penalty, `intercept_scaling=2.0`). This command verifies the
locked configuration; it does not search hyperparameters on test labels.
