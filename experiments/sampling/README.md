# Improved QC-SMOTE

## Purpose

`qc_smote_exact.py` contains the sampling implementation used in the study.
`resample_features.py` applies the locked settings for either branch to one
training matrix.

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
matrix. For strict CV, call the same transformation separately for each outer
training fold.
