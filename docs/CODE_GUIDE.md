# Code guide

This page explains what each code group does and which files a user should run.
Files under `src/aip_predictor/` are imported modules; they are not separate
programs that must be run manually.

## Research workflow

### `experiments/feature_extraction/`

Purpose: convert the fixed FASTA split into either 400-dimensional DDE vectors
or five-checkpoint averaged 1280-dimensional PLM vectors.

```bash
python experiments/feature_extraction/extract_features.py --feature dde
python experiments/feature_extraction/extract_features.py --feature plm --device cuda
```

The PLM command requires the paths configured in `configs/inference.local.json`.

### `experiments/sampling/`

Purpose: provide the exact QC-SMOTE implementation and branch-specific final
settings. Sampling must be fitted only on the training portion of a fold.

```bash
python experiments/sampling/resample_features.py \
  --features path/to/fold_train_features.npy \
  --labels path/to/fold_train_labels.npy \
  --branch plm \
  --output results/resampled_plm_fold
```

### `experiments/branch_models/`

Purpose: document the finalized PLM-GRANDE and DDE-TabICLv2 settings, strict OOF
rules, inputs, and outputs. The original research scripts are not exposed as a
single public command yet because they still depend on the unrecovered PLM
fine-tuning entry point and large external models.

### `experiments/fusion/`

Purpose: reconstruct the final seven inputs, fit the fixed logistic-regression
fusion model on strict training OOF probabilities, and evaluate the independent
test once.

```bash
python experiments/fusion/fit_fixed_fusion.py
```

### `experiments/ablation/`

Purpose: record the structural and sampling ablation design and the rules that
prevent leakage. This directory currently documents the completed experiments;
its older monolithic script is not published until the `C=0.12` historical
setting is reconciled with the final deployed `C=0.11` setting.

### `experiments/baselines/`

Purpose: describe fixed-split comparison runs and distinguish exact public
implementations from repository-specific adaptations. Results are stored in
`evaluation/baseline_metrics.csv`.

### `experiments/explainability/`

Purpose: document PLM attention/sensitivity, DDE strict-OOF permutation
importance, and fusion-level SHAP analyses. Expensive generated caches are not
stored in Git.

## Predictor code

### User-facing entry points

- `scripts/predict.py`: predict a FASTA file from the command line.
- `scripts/check_installation.py`: check whether all external model files are
  configured and available.
- `app/web_app.py`: launch the optional local web interface.
- `start_web.bat` and `start_web.sh`: convenience launchers for the web app.

### Imported modules

- `src/aip_predictor/fasta.py`: FASTA parsing and sequence validation.
- `src/aip_predictor/dde.py`: exact 400-dimensional DDE extraction.
- `src/aip_predictor/plm.py`: five-checkpoint PLM representation extraction.
- `src/aip_predictor/qc_smote_exact.py`: QC-SMOTE implementation.
- `src/aip_predictor/fusion.py`: construction of the seven fusion inputs.
- `src/aip_predictor/pipeline.py`: end-to-end finalized-model inference.
- `src/aip_predictor/feature_cache.py`: versioned local feature and prediction
  caches.
- `src/aip_predictor/config.py`: portable path configuration.
- `src/aip_predictor/artifacts.py`: fitted-artifact loading.
- `src/aip_predictor/grande_compat.py`: compatibility support for GRANDE
  deserialization.

Only an entry point should normally be run. The modules are separated so they
can be tested and reused without duplicating model logic.
