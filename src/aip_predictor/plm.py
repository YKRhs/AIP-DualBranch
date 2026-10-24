from __future__ import annotations

import gc
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


class TNFAFineTuneModel(nn.Module):
    def __init__(self, model_path: str | Path):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            str(model_path),
            local_files_only=True,
            output_hidden_states=True,
        )
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return output.hidden_states[-2][:, 0, :]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(input_ids, attention_mask)).squeeze(-1)


class PairDataset(Dataset):
    def __init__(
        self,
        sequences: list[str],
        tokenizer,
        tnfa_sequence: str,
        max_length: int,
    ):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.tnfa_sequence = tnfa_sequence
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.sequences[index],
            self.tnfa_sequence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


def select_device(requested: str | None = None) -> torch.device:
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA GPU is available")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    kwargs = {"map_location": "cpu"}
    if "mmap" in torch.load.__code__.co_varnames:
        kwargs["mmap"] = True
    try:
        return torch.load(path, weights_only=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def _load_compatible_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load checkpoints created by older Transformers ESM implementations.

    Older releases registered an absolute position-embedding table even when
    the configuration selected rotary positions. That unused table is absent
    from newer releases. No other mismatch is accepted.
    """
    incompatible = model.load_state_dict(state, strict=False)
    allowed_unexpected = {"backbone.embeddings.position_embeddings.weight"}
    unexpected = set(incompatible.unexpected_keys)
    missing = set(incompatible.missing_keys)
    if missing or unexpected.difference(allowed_unexpected):
        raise RuntimeError(
            "PLM checkpoint structure mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


@torch.inference_mode()
def extract_ensemble_features(
    sequences: list[str],
    base_model_path: str | Path,
    fold_weights: list[str | Path],
    tnfa_sequence: str,
    max_length: int = 384,
    batch_size: int = 4,
    device: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> np.ndarray:
    if not sequences:
        raise ValueError("At least one sequence is required")
    reporter = progress or (lambda message: None)
    target_device = select_device(device)
    base_model_path = Path(base_model_path)
    weights = [Path(path) for path in fold_weights]

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path), local_files_only=True
    )
    dataset = PairDataset(sequences, tokenizer, tnfa_sequence, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = TNFAFineTuneModel(base_model_path).to(target_device)
    model.eval()

    feature_sum: np.ndarray | None = None
    for fold, weight_path in enumerate(weights, start=1):
        reporter(f"PLM fold {fold}/{len(weights)}: loading weights")
        state = _load_state(weight_path)
        _load_compatible_state(model, state)
        del state
        model.eval()

        fold_features = []
        for batch_index, batch in enumerate(loader, start=1):
            input_ids = batch["input_ids"].to(target_device)
            attention_mask = batch["attention_mask"].to(target_device)
            features = model.encode(input_ids, attention_mask)
            fold_features.append(features.float().cpu().numpy())
            reporter(
                f"PLM fold {fold}/{len(weights)}: batch {batch_index}/{len(loader)}"
            )
        current = np.concatenate(fold_features, axis=0).astype(np.float32)
        feature_sum = current if feature_sum is None else feature_sum + current

    del model
    gc.collect()
    if target_device.type == "cuda":
        torch.cuda.empty_cache()
    return (feature_sum / len(weights)).astype(np.float32)
