# PLM base model

Place the complete pretrained PLM directory at `models/plm_base` when using the public configuration. The directory must be loadable by `transformers.AutoTokenizer.from_pretrained` and `transformers.AutoModel.from_pretrained` with `local_files_only=True`.

The archived experiment used the PLM-interact human STRING V12 checkpoint. Its
configuration is an ESM model with 33 transformer layers, hidden size 1280, 20
attention heads, and approximately 650 million parameters. This base model
contains the protein-pair representation learned before the AIP experiment.
The five files in `checkpoints/` are subsequent AIP-task fine-tuned weights,
not copies of the base model.

To keep the repository small, distribute this model through an external model archive or model hosting service rather than ordinary Git history.
