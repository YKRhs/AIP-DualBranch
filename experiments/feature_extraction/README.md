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

For strict branch-model OOF training, generate fold-specific PLM features:

```bash
python experiments/feature_extraction/extract_strict_plm_fold_features.py \
  --checkpoint-dir checkpoints/training_run \
  --seed 42 --folds 5 --unfreeze-last 4 --device cuda
```

If the five weights are already configured in `configs/inference.local.json`,
omit `--checkpoint-dir`. When the fine-tuning run's `run_config.json` is
present, the extractor checks the fold count, seed, number of unfrozen layers,
and (when recorded) the training FASTA checksum before creating features.

For fold `f`, the validation features are extracted only with checkpoint `f`,
which was fine-tuned without that validation fold. The script also saves the
five-checkpoint averaged train and test representations used for the final
full-training fit.

## Outputs

The default directory is `results/features/`. Each split produces a feature
matrix, label vector, and CSV index. DDE extraction takes seconds. PLM extraction
requires five forward passes per sequence and can take hours depending on the
GPU.

## Limitation

The strict script extracts representations from existing checkpoints. It does
not train the five PLM checkpoints; use
`experiments/plm_finetuning/finetune_plm_tnfa_5fold.py` for that stage.
