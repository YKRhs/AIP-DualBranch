# Reproduction scope

## Status matrix

| Stage | Status | Public entry point |
|---|---|---|
| Fixed data split | Available | `data/splits/` |
| DDE extraction | Runnable | `experiments/feature_extraction/extract_features.py` |
| PLM feature extraction from five weights | Runnable with external files | same script with `--feature plm` |
| Exact QC-SMOTE implementation | Runnable | `experiments/sampling/resample_features.py` |
| PLM five-fold fine-tuning from the base PLM | Not yet packaged | original exact entry point was not recovered |
| GRANDE and TabICLv2 branch reconstruction | Documented, external dependencies required | `experiments/branch_models/README.md` |
| Fixed seven-input fusion | Runnable | `experiments/fusion/fit_fixed_fusion.py` |
| Final inference | Runnable with external files | `scripts/predict.py` or `app/web_app.py` |
| Ablation protocol | Documented | `experiments/ablation/README.md` |
| Baseline protocol | Documented | `experiments/baselines/README.md` |
| Explainability protocol | Documented | `experiments/explainability/README.md` |

## External files

The following files are not stored in ordinary Git history:

- the PLM base model;
- five fine-tuned PLM checkpoints (approximately 13 GB total);
- the TabICLv2 pretrained checkpoint and disk-offload cache;
- fitted deployment artifacts when their size exceeds normal Git limits;
- generated PLM feature arrays and explanation caches.

Use `configs/inference.example.json` as the path template. A public release
should provide checksums and a stable archive or model-repository URL for every
external file.

## Missing PLM training entry point

Five checkpoint files named
`fold{1..5}_best_last4_cls_l2_finetune.pt` are available and their architecture
is verified by inference. The experimental workspace also contains a quick
single-split prototype. That prototype uses a 20% validation split and a
different classifier head, so it cannot truthfully reproduce these five
checkpoints. It is intentionally excluded from the public workflow.

Until the exact original five-fold training entry point is recovered or
independently reconstructed and validated against the checkpoints, this
repository must not be described as a complete raw-data-to-weights
reproduction.

## Evaluation policy

The fixed training set contains 3,775 sequences and the independent test set
contains 419 sequences. Model selection, early stopping, sampling, fusion
fitting, and threshold selection must use training data only. Independent-test
labels are used once for final reporting.

The branch probability files in `evaluation/base_probabilities/` contain strict
training OOF predictions and independent-test predictions. They permit the
fusion stage to be reproduced without rerunning the expensive PLM and TabICLv2
branches.
