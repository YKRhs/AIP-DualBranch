# Fine-tuned PLM checkpoints

Place the five fold checkpoints here when using the public configuration:

```text
fold1_best.pt
fold2_best.pt
fold3_best.pt
fold4_best.pt
fold5_best.pt
```

Alternatively, create `configs/inference.local.json` and point `plm_fold_weights` to checkpoint files stored elsewhere. Each checkpoint is approximately 2.6 GB and should be distributed outside ordinary Git history.

