# Inference guide

## 1. Prepare the environment

Python 3.10 is recommended. Install the required packages from the repository root:

```bash
python -m pip install -r requirements.txt
```

On the original Windows workstation, `configs/local_runtime.bat` selects the tested Conda interpreter. That file is private and is not included in Git.

## 2. Configure model paths

`configs/inference.example.json` is the portable public template. It expects:

- the PLM base model under `models/plm_base`;
- five fold checkpoints under `checkpoints`;
- fitted branch and fusion artifacts under `artifacts`.

For files stored elsewhere, copy the template to `configs/inference.local.json` and edit its paths. Relative paths are resolved from the configuration file location. Absolute paths and environment variables are also supported.

The configuration selection order is:

1. a path supplied with `--config`;
2. the `AIP_CONFIG` environment variable;
3. `configs/inference.local.json` when present;
4. `configs/inference.example.json`.

Validate all required files before prediction:

```bash
python scripts/check_installation.py
```

## 3. Run the local web interface

Windows users can double-click `start_web.bat`. Linux and macOS users can run:

```bash
bash start_web.sh
```

The default address is `http://127.0.0.1:7860`. Only the local computer can access this address. To suppress automatic browser opening:

```bash
python app/web_app.py --no-browser
```

## 4. Prepare input

The interface accepts a FASTA file, FASTA-formatted text, or one raw sequence per line. Only the 20 standard amino-acid letters are accepted. FASTA identifiers must be unique.

The training and independent-test sequences were at most 25 residues long. Longer sequences can be processed, but the output marks them as outside the observed training length range and their predictions require additional caution.

## 5. Read the output

Each output row contains:

- `PLM_Probability`: probability from the PLM-GRANDE branch;
- `DDE_Probability`: probability from the DDE-TabICLv2 branch;
- `Final_Probability`: probability after fixed logistic-regression fusion;
- `Prediction_Label`: `AIP` when the final probability is at least 0.5, otherwise `Non-AIP`;
- `Length_Note`: a warning for sequences longer than the training range.

The web interface saves a timestamp-independent unique CSV under `results`. The command-line interface writes `results/predictions.csv` by default.

## 6. Command-line options

```bash
python scripts/predict.py \
  --input examples/example_sequences.fasta \
  --output results/example_run.csv \
  --device cuda
```

Use `--device cpu` only when sufficient RAM and substantially longer runtime are acceptable.

## Runtime expectations

The first prediction is intentionally heavier than a conventional small classifier. Each sequence is paired with the TNF-alpha sequence, passed through the PLM five times using the five fine-tuned checkpoints, and then evaluated by TabICLv2. The web process keeps the fitted DDE classifier in memory after its first use. Averaged PLM features are cached locally under `cache/plm_features`. Final branch and fusion probabilities are cached under `cache/predictions`, so an identical sequence can return quickly after its first complete prediction. Both caches use model-version signatures. Changing a checkpoint, fitted artifact, threshold, or relevant PLM configuration automatically selects a different cache namespace. A new sequence still requires the full model on its first run, and submitting new sequences together is more efficient than submitting them one at a time.
