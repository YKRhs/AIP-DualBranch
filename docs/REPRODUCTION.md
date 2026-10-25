# Reproduction scope

## Status matrix

| Stage | Status | Public entry point |
|---|---|---|
| Fixed data split | Available | `data/splits/` |
| DDE extraction | Runnable | `experiments/feature_extraction/extract_features.py` |
| PLM feature extraction from five weights | Runnable with external files | same script with `--feature plm` |
| Exact QC-SMOTE implementation | Runnable | `experiments/sampling/resample_features.py` |
| PLM five-fold fine-tuning from the base PLM | Runnable with external base model | `experiments/plm_finetuning/finetune_plm_tnfa_5fold.py` |
| GRANDE and TabICLv2 branch reconstruction | Runnable with external dependencies | `experiments/branch_models/train_strict_branches.py` |
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

## Recovered PLM training entry point

The five checkpoint files are external and are not present in this source
checkout. The recovered five-fold training program was ported to
`experiments/plm_finetuning/finetune_plm_tnfa_5fold.py` with portable paths and
English text. Its architecture, stratified splitter, penultimate-layer CLS
head, last-four-layer unfreezing, optimizer settings, validation-MCC selection,
and checkpoint names match the recovered script.

The recovered source and archived checkpoints match in architecture and
checkpoint naming, but available timestamps do not prove that this exact source
revision generated those weights. The expensive training run has not been
repeated inside this release directory. Results can vary slightly with
PyTorch, Transformers, CUDA, and hardware versions; archived checkpoints and
OOF probabilities remain the numerical reference artifacts.

## Evaluation policy

The fixed training set contains 3,775 sequences and the independent test set
contains 419 sequences. Model selection, early stopping, sampling, fusion
fitting, and threshold selection must use training data only. Independent-test
labels are used once for final reporting.

The branch probability files in `evaluation/base_probabilities/` contain strict
training OOF predictions and independent-test predictions. They permit the
fusion stage to be reproduced without rerunning the expensive PLM and TabICLv2
branches.
