# Branch models

## Purpose

Train the two first-layer classifiers and generate aligned probabilities:

- PLM representation -> improved QC-SMOTE -> GRANDE;
- DDE representation -> improved QC-SMOTE -> TabICLv2.

Sampling and classifier fitting are separate responsibilities in the code.
The implementation is in `experiments/sampling/qc_smote_exact.py`, while this
script performs the strict fold loop and calls that sampler before fitting the
branch classifier. `experiments/sampling/resample_features.py` is only a
standalone one-partition utility and is not a prerequisite command for this
workflow.

Training probabilities are strict out-of-fold predictions. Each sample is
predicted by a branch classifier that was not fitted on that sample. The
independent test set is predicted by a final branch classifier trained from
the fixed training split; the PLM classifier uses the five-checkpoint averaged
representations. GRANDE reserves 15% of the training rows for internal early
stopping, while TabICLv2 is fitted on the full training split.

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

## Command

After generating strict fold-specific PLM features and the DDE matrices, run:

```bash
python experiments/branch_models/train_strict_branches.py \
  --branch both \
  --device cuda
```

For a fresh end-to-end rerun, create strict PLM features from the new
fine-tuning run explicitly:

```bash
python experiments/plm_finetuning/finetune_plm_tnfa_5fold.py \
  --base-model models/plm_base --seed 42 --device cuda
python experiments/feature_extraction/extract_strict_plm_fold_features.py \
  --checkpoint-dir checkpoints/training_run --seed 42 --device cuda
python experiments/feature_extraction/extract_features.py --feature dde
python experiments/branch_models/train_strict_branches.py --branch both --device cuda
```

The two branches can also be run separately with `--branch plm` and
`--branch dde`. When both outputs are available, the script creates
`training_oof.csv` and `independent_test.csv` for the fusion stage.

Use `--save-models` to save the full-training scalers and fitted branch models
under the selected output directory. It is omitted by default because the
TabICLv2 artifact and its cached training context can be large.

## Reproducibility note

The recovered five-fold PLM fine-tuning program is now provided under
`experiments/plm_finetuning/`. The public scripts preserve the archived model
architecture and fixed settings but use portable arguments and English text.
Long GPU reruns may not be bitwise identical across library and CUDA versions;
the tracked archived OOF probabilities remain the reference records.
