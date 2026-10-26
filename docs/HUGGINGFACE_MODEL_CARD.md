---
library_name: joblib
base_model: danliu1226/PLM-interact-650M-humanV12
tags:
  - bioinformatics
  - protein-language-model
  - peptide-classification
  - pytorch
  - scikit-learn
---

# AIP-DualBranch model files

This repository contains the released model files for
[AIP-DualBranch](https://github.com/YKRhs/AIP-DualBranch), a research model for
anti-inflammatory peptide classification. The source code, reproducibility
workflow, local predictor, and full usage documentation are maintained in the
GitHub repository.

## Repository contents

### Fine-tuned PLM checkpoints

The `checkpoints/` directory contains five outer-fold model weights:

```text
fold1_best_last4_cls_l2_finetune.pt
fold2_best_last4_cls_l2_finetune.pt
fold3_best_last4_cls_l2_finetune.pt
fold4_best_last4_cls_l2_finetune.pt
fold5_best_last4_cls_l2_finetune.pt
```

Each checkpoint is approximately 2.6 GB before Hugging Face Xet
deduplication. Inference extracts one 1,280-dimensional representation per
checkpoint and averages the five representations.

### Fitted downstream artifacts

The `artifacts/` directory contains:

- `plm_scaler.joblib`: PLM-branch feature scaler;
- `plm_grande.pkl`: fitted GRANDE PLM-branch classifier;
- `dde_scaler.joblib`: DDE-branch feature scaler;
- `dde_tabicl.pkl`: fitted TabICLv2 DDE-branch classifier;
- `meta_lr.joblib`: fitted logistic-regression fusion model;
- `manifest.json`: model metadata, independent-test metrics, and SHA-256
  checksums.

## Download

Install the Hugging Face CLI and download these released files into a clone of
the source repository:

```bash
python -m pip install -U huggingface_hub
hf download moyeyeye/AIP-DualBranch \
  --include "checkpoints/*" "artifacts/*" \
  --local-dir .
```

The public base PLM is maintained by its original authors and is not duplicated
in this repository:

```bash
hf download danliu1226/PLM-interact-650M-humanV12 \
  --local-dir models/plm_base
```

## Use with the source code

```bash
git clone https://github.com/YKRhs/AIP-DualBranch.git
cd AIP-DualBranch
python -m pip install -r requirements.txt

hf download moyeyeye/AIP-DualBranch \
  --include "checkpoints/*" "artifacts/*" \
  --local-dir .
hf download danliu1226/PLM-interact-650M-humanV12 \
  --local-dir models/plm_base

python scripts/check_installation.py
python app/web_app.py
```

Detailed instructions are available in the
[inference guide](https://github.com/YKRhs/AIP-DualBranch/blob/main/docs/INFERENCE.md)
and
[reproduction guide](https://github.com/YKRhs/AIP-DualBranch/blob/main/docs/REPRODUCTION.md).

## Reported independent-test result

On the fixed 419-sequence independent test set, the finalized dual-branch
deployment model recorded AUC 0.8559, accuracy 0.8091, MCC 0.6095, sensitivity
0.7667, specificity 0.8410, precision 0.7841, and F1 0.7753.

## Important notes

- This is a research release and not a clinical decision system.
- The model was developed for peptide sequences using the 20 standard amino
  acids. The observed training and independent-test sequences were at most 25
  residues long.
- Pickle and Joblib files can execute code during deserialization. Load these
  artifacts only from a trusted source and use the dependency versions
  documented by the source repository.
