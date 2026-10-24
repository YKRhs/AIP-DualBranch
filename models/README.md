# PLM base model

Place the complete pretrained PLM directory at `models/plm_base` when using the public configuration. The directory must be loadable by `transformers.AutoTokenizer.from_pretrained` and `transformers.AutoModel.from_pretrained` with `local_files_only=True`.

To keep the repository small, distribute this model through an external model archive or model hosting service rather than ordinary Git history.

