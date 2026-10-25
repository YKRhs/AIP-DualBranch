from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aip_predictor.config import configured_path, configured_paths, load_config  # noqa: E402
from aip_predictor.fasta import read_fasta  # noqa: E402
from aip_predictor.plm import extract_ensemble_features  # noqa: E402


def label_from_identifier(identifier: str) -> int:
    fields = identifier.split("|")
    if len(fields) < 2 or fields[1] not in {"0", "1"}:
        raise ValueError(
            f"Cannot read a binary label from FASTA header: {identifier!r}"
        )
    return int(fields[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract fold-specific PLM features for strict OOF training"
    )
    parser.add_argument("--train", type=Path, default=ROOT / "data/splits/AIP_train.fasta")
    parser.add_argument("--test", type=Path, default=ROOT / "data/splits/AIP_test.fasta")
    parser.add_argument("--output", type=Path, default=ROOT / "results/features/plm_strict_folds")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing fold{n}_best_last4_cls_l2_finetune.pt; "
            "defaults to the checkpoint paths in the inference config"
        ),
    )
    parser.add_argument("--unfreeze-last", type=int, default=4)
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configuration = load_config(args.config)
    plm_configuration = configuration["plm"]
    if args.checkpoint_dir is None:
        checkpoints = configured_paths(configuration, "plm_fold_weights")
    else:
        checkpoints = [
            args.checkpoint_dir
            / f"fold{fold}_best_last{args.unfreeze_last}_cls_l2_finetune.pt"
            for fold in range(1, args.folds + 1)
        ]
    if len(checkpoints) != args.folds:
        raise ValueError(
            f"Expected {args.folds} PLM checkpoints, found {len(checkpoints)}"
        )
    missing_checkpoints = [path for path in checkpoints if not path.is_file()]
    if missing_checkpoints:
        raise FileNotFoundError(
            "Missing PLM checkpoints:\n" + "\n".join(map(str, missing_checkpoints))
        )

    if args.checkpoint_dir is not None:
        run_config_path = args.checkpoint_dir / "run_config.json"
        if run_config_path.is_file():
            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            for key, expected in (
                ("folds", args.folds),
                ("seed", args.seed),
                ("unfreeze_last", args.unfreeze_last),
            ):
                actual = run_config.get(key)
                if actual != expected:
                    raise ValueError(
                        f"Fine-tuning metadata mismatch for {key}: "
                        f"checkpoint run has {actual!r}, feature extraction expects {expected!r}"
                    )
            recorded_hash = run_config.get("train_sha256")
            if recorded_hash is not None:
                current_hash = hashlib.sha256(args.train.read_bytes()).hexdigest()
                if current_hash != recorded_hash:
                    raise ValueError(
                        "The training FASTA differs from the one used to create "
                        "these checkpoints (SHA-256 mismatch)."
                    )

    train_records = read_fasta(args.train)
    test_records = read_fasta(args.test)
    train_sequences = [record.sequence for record in train_records]
    test_sequences = [record.sequence for record in test_records]
    train_labels = np.asarray(
        [label_from_identifier(record.identifier) for record in train_records],
        dtype=int,
    )
    test_labels = np.asarray(
        [label_from_identifier(record.identifier) for record in test_records],
        dtype=int,
    )
    all_sequences = train_sequences + test_sequences
    splitter = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    splits = list(splitter.split(np.zeros(len(train_labels)), train_labels))
    args.output.mkdir(parents=True, exist_ok=True)

    train_sum = np.zeros(
        (len(train_sequences), int(plm_configuration["feature_dimension"])),
        dtype=np.float32,
    )
    test_sum = np.zeros(
        (len(test_sequences), int(plm_configuration["feature_dimension"])),
        dtype=np.float32,
    )
    split_records = []

    for fold, ((train_index, valid_index), checkpoint) in enumerate(
        zip(splits, checkpoints, strict=True), start=1
    ):
        print(f"[fold {fold}/{args.folds}] extracting features with {checkpoint.name}")
        # One checkpoint processes all sequences in one pass. The resulting
        # rows are then separated into outer-train, validation, and test sets.
        features = extract_ensemble_features(
            sequences=all_sequences,
            base_model_path=configured_path(configuration, "plm_base_model"),
            fold_weights=[checkpoint],
            tnfa_sequence=plm_configuration["tnfa_sequence"],
            max_length=int(plm_configuration["max_length"]),
            batch_size=int(plm_configuration["batch_size"]),
            device=args.device,
            progress=lambda message: print(f"  {message}", flush=True),
        )
        fold_train_features = features[: len(train_sequences)]
        fold_test_features = features[len(train_sequences) :]
        train_sum += fold_train_features
        test_sum += fold_test_features

        fold_directory = args.output / f"fold{fold}"
        fold_directory.mkdir(parents=True, exist_ok=True)
        np.save(fold_directory / "X_train_plm.npy", fold_train_features[train_index])
        np.save(fold_directory / "y_train.npy", train_labels[train_index])
        np.save(fold_directory / "X_val_plm.npy", fold_train_features[valid_index])
        np.save(fold_directory / "y_val.npy", train_labels[valid_index])
        np.save(fold_directory / "val_idx.npy", valid_index)
        np.save(fold_directory / "X_test_plm.npy", fold_test_features)
        np.save(fold_directory / "y_test.npy", test_labels)
        split_records.append(
            {
                "fold": fold,
                "checkpoint": str(checkpoint.resolve()),
                "train_size": int(len(train_index)),
                "validation_size": int(len(valid_index)),
                "test_size": int(len(test_labels)),
            }
        )
        del features, fold_train_features, fold_test_features

    np.save(args.output / "ensemble_train_plm.npy", train_sum / args.folds)
    np.save(args.output / "ensemble_test_plm.npy", test_sum / args.folds)
    np.save(args.output / "train_labels.npy", train_labels)
    np.save(args.output / "test_labels.npy", test_labels)
    manifest = {
        "feature": "penultimate hidden-layer CLS representation",
        "dimension": int(plm_configuration["feature_dimension"]),
        "folds": args.folds,
        "seed": args.seed,
        "strict_oof_rule": (
            "Fold f validation features are extracted only with checkpoint f, "
            "which was not fine-tuned on those validation samples."
        ),
        "splits": split_records,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Saved strict fold features to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
