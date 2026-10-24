# Branch models

## Purpose

Train the two first-layer classifiers and generate aligned probabilities:

- PLM representation -> improved QC-SMOTE -> GRANDE;
- DDE representation -> improved QC-SMOTE -> TabICLv2.

Training probabilities must be strict out-of-fold predictions. Each sample is
predicted by a model that was not fitted on that sample. Independent-test
probabilities are averaged over the five fold-specific fits.

## Final GRANDE settings

```text
depth = 3
n_estimators = 128
selected_variables = 50
dropout = 0.30
data_subset_fraction = 1.0
learning_rate_weights = 0.0005
learning_rate_index = 0.005
learning_rate_values = 0.03
learning_rate_leaf = 0.03
epochs = 80
early_stopping_epochs = 15
random_state = 42
```

Within each outer-training fold, 15% of raw training rows are reserved for
GRANDE early stopping before scaling and QC-SMOTE. This subset is not sampled.

## Final TabICLv2 settings

```text
n_estimators = 8
softmax_temperature = 0.9
average_logits = true
checkpoint = tabicl-classifier-v2-20260212.ckpt
random_state = 42
```

## Archived outputs

The aligned final outputs are tracked as:

```text
evaluation/base_probabilities/training_oof.csv
evaluation/base_probabilities/independent_test.csv
```

They are the inputs to `experiments/fusion/fit_fixed_fusion.py`.

## Current limitation

The exact five-fold PLM fine-tuning entry point was not recovered. Publishing a
nominal branch-training launcher would therefore imply a raw-data-to-model
reproduction that is not currently supported. The verified feature extraction,
OOF probabilities, model settings, and finalized inference remain available.
