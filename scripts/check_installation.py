from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aip_predictor.artifacts import required_artifacts
from aip_predictor.config import configured_path, configured_paths, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate local predictor files")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    paths = [
        configured_path(config, "plm_base_model"),
        *configured_paths(config, "plm_fold_weights"),
        *required_artifacts(configured_path(config, "artifacts_dir")),
    ]
    missing = [path for path in paths if not path.exists()]
    for path in paths:
        print(f"{'OK' if path.exists() else 'MISSING':7} {path}")
    if missing:
        raise SystemExit(f"Installation check failed: {len(missing)} path(s) missing")

    artifacts_dir = configured_path(config, "artifacts_dir")
    manifest_path = artifacts_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        expected_hashes = json.load(handle).get("artifact_sha256", {})
    mismatches = []
    for name, expected in expected_hashes.items():
        path = artifacts_dir / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        status = "OK" if actual == expected else "MISMATCH"
        print(f"{status:8} SHA-256 {name}")
        if actual != expected:
            mismatches.append(name)
    if mismatches:
        raise SystemExit(
            f"Checksum validation failed for {len(mismatches)} artifact(s)"
        )
    print("Installation check passed.")


if __name__ == "__main__":
    main()
