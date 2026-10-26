# AIP-DualBranch

Research code and evaluation records for a dual-branch anti-inflammatory
peptide (AIP) classifier. The method combines complementary protein language
model (PLM) and dipeptide deviation from expected mean (DDE) representations.
The local predictor in `app/` is an optional demonstration, not the main scope
of this repository.

## Method overview

```text
Peptide + TNF-alpha
  -> five-fold fine-tuned PLM
  -> average 1280D penultimate-layer CLS representation
  -> branch-specific improved QC-SMOTE
  -> GRANDE
  -> p_plm

Peptide
  -> 400D NeXtMD-DDE representation
  -> branch-specific improved QC-SMOTE
  -> TabICLv2
  -> p_dde

p_plm + p_dde
  -> seven probability-derived inputs
  -> fixed logistic regression
  -> final AIP probability (threshold = 0.5)
```

Model selection uses training-side cross-validation and strict out-of-fold
(OOF) predictions. Independent-test labels are reserved for final evaluation.

## Fixed data split

| Split | Samples | Positive | Negative | Maximum length |
|---|---:|---:|---:|---:|
| Training | 3,775 | 1,498 | 2,277 | 25 |
| Independent test | 419 | 180 | 239 | 25 |

The two splits have no sequence overlap. The exact FASTA files are provided in
`data/splits/` because this split is required for comparable evaluation.

## Final independent-test result

| AUC | ACC | MCC | Sn | Sp | Precision | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.8559 | 0.8091 | 0.6095 | 0.7667 | 0.8410 | 0.7841 | 0.7753 |

The row-level predictions and branch probabilities are stored in
`evaluation/independent_test_predictions.csv`.

## Repository layout

```text
AIP-DualBranch/
|-- data/                  Fixed train/test sequence split
|-- experiments/           Research workflow, code, and method notes
|-- evaluation/            Reported metrics and row-level predictions
|-- src/aip_predictor/      Reusable inference implementation
|-- scripts/               Command-line prediction and installation checks
|-- app/                   Optional local Gradio predictor
|-- tests/                 Lightweight regression and unit tests
|-- artifacts/             Fitted small/downstream artifacts (external)
|-- checkpoints/           Five fine-tuned PLM weights (external)
|-- models/                Base PLM files (external)
`-- docs/                  Architecture, code guide, and reproducibility scope
```

Start with [docs/CODE_GUIDE.md](docs/CODE_GUIDE.md) for the purpose and command
of every code group. The staged research workflow is documented in
[experiments/README.md](experiments/README.md).

## Reproduce lightweight stages

Create the environment:

```bash
conda env create -f environment.yml
conda activate aip-dual-branch
```

Extract DDE features from the fixed split:

```bash
python experiments/feature_extraction/extract_features.py --feature dde
```

Reproduce the finalized seven-input fusion from archived strict OOF branch
probabilities:

```bash
python experiments/fusion/fit_fixed_fusion.py
```

The PLM feature route additionally requires the external base model and five
fold checkpoints. Full branch retraining requires GRANDE and TabICLv2 and is
substantially more expensive. See [docs/REPRODUCTION.md](docs/REPRODUCTION.md)
for the exact supported boundary.

## Full research workflow

The recovered five-fold fine-tuning and strict branch-training entry points are
also included. They require the external PLM-interact base model, substantial
GPU time, and enough disk space for five checkpoints.

```bash
# 1. Fine-tune five PLM folds.
python experiments/plm_finetuning/finetune_plm_tnfa_5fold.py \
  --base-model models/plm_base --device cuda

# 2. Extract fold-specific PLM features and the DDE matrices.
python experiments/feature_extraction/extract_strict_plm_fold_features.py \
  --checkpoint-dir checkpoints/training_run --seed 42 --device cuda
python experiments/feature_extraction/extract_features.py --feature dde

# 3. Train both strict-OOF branch classifiers.
python experiments/branch_models/train_strict_branches.py --branch both --device cuda

# 4. Fit the fixed seven-input fusion from the regenerated probabilities.
python experiments/fusion/fit_fixed_fusion.py \
  --oof results/branch_models/training_oof.csv \
  --test results/branch_models/independent_test.csv
```

## Optional local predictor

After configuring the external model files, run:

```bash
python scripts/check_installation.py
python app/web_app.py
```

The interface accepts pasted sequences or a FASTA upload and runs only on the
local computer. Detailed instructions are in [docs/INFERENCE.md](docs/INFERENCE.md).

## Large model files

Large binary files are distributed through the Hugging Face model repository
instead of Git history:

<https://huggingface.co/moyeyeye/AIP-DualBranch>

Download the five fine-tuned checkpoints and fitted downstream artifacts from
the source repository root with:

```bash
hf download moyeyeye/AIP-DualBranch \
  --include "checkpoints/*" "artifacts/*" \
  --local-dir .
```

The five checkpoints are approximately 13 GB before Hugging Face Xet
deduplication. The public PLM-interact base model is not redistributed here;
download it from
[`danliu1226/PLM-interact-650M-humanV12`](https://huggingface.co/danliu1226/PLM-interact-650M-humanV12):

```bash
hf download danliu1226/PLM-interact-650M-humanV12 \
  --local-dir models/plm_base
```

## Reproducibility statement

This repository supports:

- exact DDE extraction;
- five-fold PLM fine-tuning with the recovered archived architecture and settings;
- feature extraction from the five archived PLM checkpoints;
- strict-OOF GRANDE and TabICLv2 branch training;
- finalized-model inference;
- fixed fusion reconstruction from strict OOF branch probabilities;
- verification of the reported 419-sample independent-test predictions.

The portable fine-tuning script was reconstructed from the recovered original
five-fold program. Its architecture, splitter, optimization settings, stopping
rule, and output names match the archived run. Because a full rerun is costly
and framework/CUDA versions can affect numerical details, the tracked OOF
probabilities remain the reference record rather than a claim of bitwise
checkpoint reproduction.
