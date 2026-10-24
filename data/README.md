# Data

`splits/AIP_train.fasta` and `splits/AIP_test.fasta` are the fixed sequence
splits used throughout the study.

FASTA headers use this form:

```text
>sample_id|class_label|split_name
```

Class label `1` denotes an anti-inflammatory peptide and `0` denotes a negative
sample. The training and test sets contain no identical sequences.

| File | Samples | Positive | Negative |
|---|---:|---:|---:|
| `AIP_train.fasta` | 3,775 | 1,498 | 2,277 |
| `AIP_test.fasta` | 419 | 180 | 239 |

Do not combine the independent test set with training data. Do not use its
labels for hyperparameter selection, early stopping, sampling, or threshold
selection.
