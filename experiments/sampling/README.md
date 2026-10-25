# Improved QC-SMOTE

## Purpose

`qc_smote_exact.py` contains the reusable QC-SMOTE implementation used in the
study. `resample_features.py` is a standalone utility that applies the locked
settings to one already-defined training partition.

The complete strict-OOF experiment is launched from
`experiments/branch_models/train_strict_branches.py`. That script imports the
same sampler and fits scaling plus QC-SMOTE separately inside every outer
training fold before fitting GRANDE or TabICLv2. Do not run
`resample_features.py` once on the complete training set and then use that
matrix for cross-validation.

## Leakage rule

Scaling and QC-SMOTE must be fitted inside each training fold. Never resample
the complete dataset before cross-validation, and never resample a validation
or independent-test set.

## Command

```bash
python experiments/sampling/resample_features.py \
  --features results/features/train_plm.npy \
  --labels results/features/train_labels.npy \
  --branch plm \
  --output results/sampling/plm_full_fit
```

Use `--branch dde` for DDE. The command saves the fitted scaler and resampled
matrix. For a manual strict-CV audit, call the same transformation separately
for each outer training fold. For the normal end-to-end reproduction, run:

```bash
python experiments/branch_models/train_strict_branches.py \
  --branch both --device cuda
```
