# Five-fold PLM fine-tuning

This stage fine-tunes the PLM-interact human STRING V12 base model for the
binary anti-inflammatory peptide task. Each peptide is paired with the fixed
TNF-alpha sequence. The penultimate-layer CLS representation is passed through
a `1280 -> 256 -> 1` classification head.

The original experiment used stratified five-fold cross-validation. Every fold
starts from the same base model, unfreezes the last four transformer layers,
and selects its checkpoint by validation MCC.

```bash
python experiments/plm_finetuning/finetune_plm_tnfa_5fold.py \
  --base-model models/plm_base \
  --train data/splits/AIP_train.fasta \
  --output checkpoints/training_run \
  --device cuda
```

The five output checkpoints are named
`fold{1..5}_best_last4_cls_l2_finetune.pt`. The archived checkpoints are about
2.6 GB each and should be distributed outside ordinary Git history.

This script is a portable English-language reconstruction of the recovered
experimental program `finetune_plm_tnfa_5fold_cv_cls_l2.py`. Its architecture,
fold splitter, optimization settings, early-stopping rule, and checkpoint names
match the archived run. A full rerun is computationally expensive and exact
bitwise equality is not guaranteed across CUDA, PyTorch, and Transformers
versions.

The recovered source file and archived checkpoints match in architecture and
checkpoint naming, but available file timestamps do not establish that this
exact source revision created the archived weights. The script is therefore
provided as the recovered training procedure, not as a claim of proven
checkpoint-by-checkpoint provenance.
