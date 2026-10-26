# Fitted downstream artifacts

Inference requires the following files:

- `plm_scaler.joblib`: scaler for the 1280-dimensional PLM representation;
- `plm_grande.pkl`: fitted GRANDE classifier for the PLM branch;
- `dde_scaler.joblib`: scaler for the 400-dimensional DDE representation;
- `dde_tabicl.pkl`: fitted TabICLv2 classifier for the DDE branch;
- `meta_lr.joblib`: fitted seven-input logistic-regression fusion model;
- `manifest.json`: model metadata, metrics, and artifact checksums.

Large binary artifacts are excluded from normal Git tracking. Preserve their SHA-256 checksums when distributing them separately.

The released files are hosted in
[`moyeyeye/AIP-DualBranch`](https://huggingface.co/moyeyeye/AIP-DualBranch/tree/main/artifacts).
From the source repository root, download them with:

```bash
hf download moyeyeye/AIP-DualBranch \
  --include "artifacts/*" \
  --local-dir .
```
