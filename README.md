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

## Optional local predictor

After configuring the external model files, run:

```bash
python scripts/check_installation.py
python app/web_app.py
```

The interface accepts pasted sequences or a FASTA upload and runs only on the
local computer. Detailed instructions are in [docs/INFERENCE.md](docs/INFERENCE.md).

## Large model files

The base PLM and five fine-tuned checkpoints are intentionally excluded from
Git history. The five checkpoints are approximately 13 GB in total. Their
release location and checksums must be added before a public archival release.

## Reproducibility statement

This repository currently supports:

- exact DDE extraction;
- feature extraction from the five archived PLM checkpoints;
- finalized-model inference;
- fixed fusion reconstruction from strict OOF branch probabilities;
- verification of the reported 419-sample independent-test predictions.

The original program that produced the five fold-specific PLM checkpoints has
not yet been recovered from the experimental workspace. A separate one-split
prototype exists, but its split and classifier head do not match the archived
five-fold checkpoints and is therefore not published as the training entry
point. This limitation is stated explicitly to avoid overstating the current
reproduction scope.
