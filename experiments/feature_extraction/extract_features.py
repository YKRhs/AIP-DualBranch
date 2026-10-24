from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aip_predictor.config import configured_path, configured_paths, load_config
from aip_predictor.dde import extract_dde_batch
from aip_predictor.fasta import read_fasta
from aip_predictor.plm import extract_ensemble_features


def labels_from_headers(identifiers: list[str]) -> np.ndarray:
    labels = []
    for identifier in identifiers:
        fields = identifier.split("|")
        if len(fields) < 2 or fields[1] not in {"0", "1"}:
            raise ValueError(
                f"Cannot read a binary label from FASTA header: {identifier!r}"
            )
        labels.append(int(fields[1]))
    return np.asarray(labels, dtype=np.int64)


def save_split(
    split: str,
    fasta_path: Path,
    output_dir: Path,
    feature: str,
    config: dict,
    device: str | None,
) -> None:
    records = read_fasta(fasta_path)
    sequences = [record.sequence for record in records]
    identifiers = [record.identifier for record in records]
    labels = labels_from_headers(identifiers)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{split}_labels.npy", labels)
    pd.DataFrame(
        {
            "Sample_Index": np.arange(len(records)),
            "ID": identifiers,
            "Sequence": sequences,
            "Label": labels,
        }
    ).to_csv(output_dir / f"{split}_index.csv", index=False)

    if feature in {"dde", "both"}:
        dde = extract_dde_batch(sequences)
        np.save(output_dir / f"{split}_dde.npy", dde)
        print(f"{split}: DDE {dde.shape}", flush=True)

    if feature in {"plm", "both"}:
        plm_config = config["plm"]
        plm = extract_ensemble_features(
            sequences=sequences,
            base_model_path=configured_path(config, "plm_base_model"),
            fold_weights=configured_paths(config, "plm_fold_weights"),
            tnfa_sequence=plm_config["tnfa_sequence"],
            max_length=int(plm_config["max_length"]),
            batch_size=int(plm_config["batch_size"]),
            device=device,
            progress=lambda message: print(message, flush=True),
        )
        np.save(output_dir / f"{split}_plm.npy", plm)
        print(f"{split}: PLM {plm.shape}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract DDE and/or five-checkpoint averaged PLM features"
    )
    parser.add_argument("--feature", choices=["dde", "plm", "both"], default="dde")
    parser.add_argument("--train", type=Path, default=ROOT / "data/splits/AIP_train.fasta")
    parser.add_argument("--test", type=Path, default=ROOT / "data/splits/AIP_test.fasta")
    parser.add_argument("--output", type=Path, default=ROOT / "results/features")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    save_split("train", args.train, args.output, args.feature, config, args.device)
    save_split("test", args.test, args.output, args.feature, config, args.device)
    print(f"Saved feature files to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
