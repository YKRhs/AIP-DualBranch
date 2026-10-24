# Feature extraction

## Purpose

Generate the two representations used by the final model:

- 400-dimensional NeXtMD-DDE vectors;
- 1280-dimensional penultimate-layer CLS vectors averaged over five fine-tuned
  PLM checkpoints.

## Inputs

- labeled FASTA files in `data/splits/`;
- for PLM only, the base model and five checkpoints configured in
  `configs/inference.local.json`.

## Commands

```bash
python experiments/feature_extraction/extract_features.py --feature dde
python experiments/feature_extraction/extract_features.py --feature plm --device cuda
python experiments/feature_extraction/extract_features.py --feature both --device cuda
```

## Outputs

The default directory is `results/features/`. Each split produces a feature
matrix, label vector, and CSV index. DDE extraction takes seconds. PLM extraction
requires five forward passes per sequence and can take hours depending on the
GPU.

## Limitation

This script extracts representations from existing checkpoints. It does not
train the five PLM checkpoints.
