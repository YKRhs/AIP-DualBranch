# Optional local predictor

The Gradio interface is a supplementary demonstration of the finalized model.
It is not the main scientific contribution of the repository.

Run from the repository root:

```bash
python app/web_app.py
```

The interface accepts pasted peptide sequences, FASTA-formatted text, or a
FASTA file upload. It reports PLM, DDE, and fused probabilities. See
`docs/INFERENCE.md` for model-file setup and runtime details.
