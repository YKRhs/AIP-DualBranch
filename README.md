# AIP Dual-Branch Predictor

This repository contains the local inference package for a finalized anti-inflammatory peptide (AIP) classifier. The model combines two complementary branches:

- a five-fold fine-tuned protein language model (PLM) representation followed by scaling and GRANDE;
- a 400-dimensional dipeptide deviation from expected mean (DDE) representation followed by scaling and TabICLv2;
- a fixed logistic-regression fusion layer built from seven probability-derived inputs.

The default decision threshold is 0.5. All inference runs locally. The web interface binds to `127.0.0.1` and does not send sequences to an external service.

## Repository layout

```text
AIP-DualBranch/
|-- app/                 Local Gradio web interface
|-- artifacts/           Fitted downstream classifiers and scalers
|-- cache/               Generated PLM feature cache (not committed)
|-- checkpoints/         Five fine-tuned PLM checkpoints (not committed)
|-- configs/             Portable and private inference configurations
|-- docs/                Inference, architecture, and reproduction notes
|-- examples/            Example FASTA input and expected output
|-- models/              PLM base model (not committed)
|-- results/             Generated prediction files
|-- scripts/             User-facing command-line utilities
|-- src/aip_predictor/   Internal inference modules
`-- tests/               Lightweight unit tests
```

Files under `src/aip_predictor` are library modules. They are imported by the two user-facing entry points and are not intended to be run individually.

## Quick start on the configured Windows computer

1. Double-click `start_web.bat`.
2. Wait until the terminal reports `Running on local URL`.
3. Open `http://127.0.0.1:7860` if the browser does not open automatically.
4. Paste one or more peptide sequences, or upload a FASTA file, and select **Run prediction**.

The private file `configs/inference.local.json` already points to the PLM base model and five fine-tuned checkpoints in the current workspace. This file is excluded from Git because its paths are machine-specific.

## Command-line prediction

```powershell
python scripts/predict.py --input examples/example_sequences.fasta
```

The result is written to `results/predictions.csv` unless `--output` is supplied. See [docs/INFERENCE.md](docs/INFERENCE.md) for complete setup and usage instructions.

## Installation on another computer

Create a Python 3.10 environment and install the pinned packages:

```bash
python -m pip install -r requirements.txt
```

Then place the external model files in `models/` and `checkpoints/`, or create `configs/inference.local.json` from `configs/inference.example.json` and point it to their actual locations. Run:

```bash
python scripts/check_installation.py
```

## Model files and GitHub

The PLM base model is several gigabytes, each fine-tuned checkpoint is approximately 2.6 GB, and the five checkpoints together are approximately 13 GB. They should not be committed to ordinary Git history. Use a documented external archive, a model repository, Git LFS subject to its quota, or GitHub Releases subject to current size limits. The lightweight source repository should retain only the path templates and download instructions.

## Reproducibility scope

This package currently supports reproducible inference from the finalized fitted artifacts. It does not yet provide a complete one-command reconstruction from raw training data, including five-fold PLM fine-tuning, improved QC-SMOTE, branch training, and out-of-fold fusion fitting. The exact boundary is documented in [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

No files have been uploaded to GitHub from this working directory.
