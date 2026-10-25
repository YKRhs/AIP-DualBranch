# Fine-tuned PLM checkpoints

The public configuration expects these five fine-tuned checkpoint files here:

```text
fold1_best_last4_cls_l2_finetune.pt
fold2_best_last4_cls_l2_finetune.pt
fold3_best_last4_cls_l2_finetune.pt
fold4_best_last4_cls_l2_finetune.pt
fold5_best_last4_cls_l2_finetune.pt
```

This directory contains documentation only in the source checkout; model weights
are external files and are not committed. Each checkpoint is approximately
2.6 GB and should be distributed outside ordinary Git history. Alternatively,
create `configs/inference.local.json` and point `plm_fold_weights` to weights
stored elsewhere.

These files are five independently fine-tuned weight sets, one for each outer
fold. The training script writes them under
`checkpoints/training_run/` by default; configure those output paths for
inference or place/copy them under this directory. They can be regenerated with
`experiments/plm_finetuning/finetune_plm_tnfa_5fold.py` after the external base
model has been installed.
