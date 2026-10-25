from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aip_predictor.fasta import read_fasta  # noqa: E402
from aip_predictor.plm import TNFAFineTuneModel, select_device  # noqa: E402


TNFA_SEQUENCE = (
    "MSTESMIRDVELAEEALPKKTGGPQGSRRCLFLSLFSFLIVAGATTLFCLLHFGVIGPQREEFPRDLSLISPLAQAVRSSSRTPSDKPVAHVVANPQAEGQLQWLNRRANALLANGVELRDNQLVVPSEGLYLIYSQVLFKGQGCPSTHVLLTHTISRIAVSYQTKVNLLSAIKSPCQRETPEGAEAKPWYEPIYLGGVFQLEKGDRLSAEINRPDYLDFAESGQVYFGIIAL"
)
METRICS = ("AUC", "ACC", "MCC", "Sn", "Sp", "Pre", "F1")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def label_from_identifier(identifier: str) -> int:
    fields = identifier.split("|")
    if len(fields) < 2 or fields[1] not in {"0", "1"}:
        raise ValueError(
            f"Cannot read a binary label from FASTA header: {identifier!r}"
        )
    return int(fields[1])


def calculate_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "AUC": float(roc_auc_score(labels, probabilities)),
        "ACC": float(accuracy_score(labels, predictions)),
        "MCC": float(matthews_corrcoef(labels, predictions)),
        "Sn": float(tp / (tp + fn)),
        "Sp": float(tn / (tn + fp)),
        "Pre": float(precision_score(labels, predictions, zero_division=0)),
        "F1": float(f1_score(labels, predictions, zero_division=0)),
    }


class LabeledPairDataset(Dataset):
    def __init__(
        self,
        sequences: list[str],
        labels: np.ndarray,
        tokenizer,
        max_length: int,
    ) -> None:
        self.sequences = sequences
        self.labels = np.asarray(labels, dtype=np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.sequences[index],
            TNFA_SEQUENCE,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[index], dtype=torch.float32),
        }


def transformer_layers(backbone: nn.Module):
    candidates = (
        lambda model: model.encoder.layer,
        lambda model: model.encoder.layers,
        lambda model: model.esm.encoder.layer,
        lambda model: model.esm.encoder.layers,
    )
    for getter in candidates:
        try:
            layers = getter(backbone)
            if len(layers):
                return layers
        except AttributeError:
            continue
    raise RuntimeError("Cannot locate transformer encoder layers in the base model")


def prepare_trainable_parameters(model: TNFAFineTuneModel, last_layers: int) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    layers = transformer_layers(model.backbone)
    if not 1 <= last_layers <= len(layers):
        raise ValueError(f"--unfreeze-last must be between 1 and {len(layers)}")
    for layer in layers[-last_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


@torch.inference_mode()
def evaluate(
    model: TNFAFineTuneModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    labels, probabilities = [], []
    for batch in loader:
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        labels.append(batch["label"].numpy())
        probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
    y_true = np.concatenate(labels).astype(int)
    y_probability = np.concatenate(probabilities).astype(float)
    return calculate_metrics(y_true, y_probability), y_true, y_probability


def train_fold(
    fold: int,
    train_index: np.ndarray,
    valid_index: np.ndarray,
    sequences: list[str],
    labels: np.ndarray,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray]:
    train_loader = DataLoader(
        LabeledPairDataset(
            [sequences[index] for index in train_index],
            labels[train_index],
            tokenizer,
            args.max_length,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        LabeledPairDataset(
            [sequences[index] for index in valid_index],
            labels[valid_index],
            tokenizer,
            args.max_length,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # Each fold must start from the same unmodified base PLM.
    model = TNFAFineTuneModel(args.base_model).to(device)
    prepare_trainable_parameters(model, args.unfreeze_last)
    backbone_parameters, head_parameters = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = head_parameters if name.startswith("classifier") else backbone_parameters
        target.append(parameter)
    optimizer = torch.optim.AdamW(
        (
            {"params": backbone_parameters, "lr": args.backbone_lr},
            {"params": head_parameters, "lr": args.head_lr},
        ),
        weight_decay=args.weight_decay,
    )
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([args.positive_weight], device=device)
    )
    amp_enabled = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_mcc = -np.inf
    best_epoch = 0
    best_state = None
    best_metrics = None
    best_probability = None
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
                loss = loss_function(logits, batch["label"].to(device))
                scaled_loss = loss / args.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            running_loss += float(loss.detach().cpu())
            if step % args.gradient_accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        if len(train_loader) % args.gradient_accumulation:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        metrics, _, probability = evaluate(model, valid_loader, device)
        print(
            f"[fold {fold}] epoch {epoch}/{args.epochs} "
            f"loss={running_loss / len(train_loader):.5f} "
            f"val_MCC={metrics['MCC']:.4f} val_AUC={metrics['AUC']:.4f}",
            flush=True,
        )
        if metrics["MCC"] > best_mcc:
            best_mcc = metrics["MCC"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics
            best_probability = probability.copy()
            stale_epochs = 0
            checkpoint = (
                args.output
                / f"fold{fold}_best_last{args.unfreeze_last}_cls_l2_finetune.pt"
            )
            torch.save(best_state, checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is None or best_metrics is None or best_probability is None:
        raise RuntimeError(f"Fold {fold} did not produce a valid checkpoint")
    del best_state, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"Fold": fold, "Best_Epoch": best_epoch, **best_metrics}, best_probability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Five-fold TNF-alpha PLM fine-tuning")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=ROOT / "data/splits/AIP_train.fasta")
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/training_run")
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--unfreeze-last", type=int, default=4)
    parser.add_argument("--backbone-lr", type=float, default=3e-6)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.base_model.exists():
        raise FileNotFoundError(args.base_model)
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = select_device(args.device)

    records = read_fasta(args.train)
    sequences = [record.sequence for record in records]
    labels = np.asarray(
        [label_from_identifier(record.identifier) for record in records], dtype=int
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.base_model), local_files_only=True
    )
    splitter = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    oof_probability = np.full(len(labels), np.nan, dtype=np.float32)
    fold_rows = []

    for fold, (train_index, valid_index) in enumerate(
        splitter.split(np.zeros(len(labels)), labels), start=1
    ):
        row, probability = train_fold(
            fold,
            train_index,
            valid_index,
            sequences,
            labels,
            tokenizer,
            args,
            device,
        )
        fold_rows.append(row)
        oof_probability[valid_index] = probability

    if np.isnan(oof_probability).any():
        raise RuntimeError("OOF probability vector is incomplete")
    oof_metrics = calculate_metrics(labels, oof_probability)
    pd.DataFrame(fold_rows).to_csv(args.output / "fold_metrics.csv", index=False)
    pd.DataFrame(
        {
            "Sample_Index": np.arange(len(labels)),
            "Label": labels,
            "PLM_FineTune_Probability": oof_probability,
        }
    ).to_csv(args.output / "training_oof_predictions.csv", index=False)
    configuration = {
        "base_model": str(args.base_model.resolve()),
        "train": str(args.train.resolve()),
        "train_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
        "tnfa_sequence": TNFA_SEQUENCE,
        **{
            key: value
            for key, value in vars(args).items()
            if key not in {"base_model", "train", "output", "device"}
        },
        "device": str(device),
        "classifier": "penultimate CLS: Linear(1280,256)-ReLU-Dropout(0.2)-Linear(256,1)",
        "oof_metrics": oof_metrics,
    }
    (args.output / "run_config.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame([{**{"Evaluation": "Pooled training OOF"}, **oof_metrics}]).to_string(index=False))
    print(f"Saved five-fold checkpoints and records to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
