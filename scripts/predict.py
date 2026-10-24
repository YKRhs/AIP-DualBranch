from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aip_predictor.config import configured_path, load_config
from aip_predictor.fasta import read_fasta
from aip_predictor.pipeline import AIPPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict anti-inflammatory peptides")
    parser.add_argument("--input", required=True, help="Input FASTA file")
    parser.add_argument("--output", default=None, help="Output CSV file")
    parser.add_argument("--config", default=None, help="Inference configuration JSON")
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu")
    args = parser.parse_args()

    records = read_fasta(args.input)
    config = load_config(args.config)
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else configured_path(config, "results_dir") / "predictions.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    predictor = AIPPredictor(
        config_path=args.config,
        device=args.device,
        progress=lambda message: print(message, flush=True),
    )
    result = predictor.predict(records)
    result.to_csv(output, index=False, encoding="utf-8-sig", float_format="%.8f")
    print(result.to_string(index=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
