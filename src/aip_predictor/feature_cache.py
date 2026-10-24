from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def build_plm_signature(
    base_model_path: Path,
    fold_weights: list[Path],
    tnfa_sequence: str,
    max_length: int,
) -> str:
    """Create a lightweight signature that invalidates stale PLM features."""
    files = []
    base_config = base_model_path / "config.json"
    if base_config.exists():
        stat = base_config.stat()
        files.append((str(base_config.resolve()), stat.st_size, stat.st_mtime_ns))
    for path in fold_weights:
        stat = path.stat()
        files.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    payload = {
        "files": files,
        "tnfa_sequence": tnfa_sequence,
        "max_length": max_length,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


class PLMFeatureCache:
    """Store exact averaged PLM features by sequence and model signature."""

    def __init__(self, root: Path, signature: str, feature_dimension: int):
        self.directory = root / "plm_features" / signature
        self.directory.mkdir(parents=True, exist_ok=True)
        self.feature_dimension = feature_dimension

    def _path(self, sequence: str) -> Path:
        key = hashlib.sha256(sequence.encode("ascii")).hexdigest()
        return self.directory / f"{key}.npy"

    def get(self, sequence: str) -> np.ndarray | None:
        path = self._path(sequence)
        if not path.exists():
            return None
        try:
            feature = np.load(path, allow_pickle=False)
        except (OSError, ValueError):
            return None
        if feature.shape != (self.feature_dimension,) or not np.isfinite(feature).all():
            return None
        return feature.astype(np.float32, copy=False)

    def put(self, sequence: str, feature: np.ndarray) -> None:
        value = np.asarray(feature, dtype=np.float32)
        if value.shape != (self.feature_dimension,) or not np.isfinite(value).all():
            raise ValueError("Invalid PLM feature vector cannot be cached")
        path = self._path(sequence)
        temporary = path.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
        temporary.replace(path)


def build_prediction_signature(
    model_files: list[Path],
    plm_signature: str,
    threshold: float,
) -> str:
    files = []
    for path in model_files:
        stat = path.stat()
        files.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    payload = {
        "files": files,
        "plm_signature": plm_signature,
        "threshold": threshold,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


class PredictionCache:
    """Store deterministic branch and fusion probabilities by sequence."""

    FIELDS = ("p_plm", "p_dde", "p_final")

    def __init__(self, root: Path, signature: str):
        self.directory = root / "predictions" / signature
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, sequence: str) -> Path:
        key = hashlib.sha256(sequence.encode("ascii")).hexdigest()
        return self.directory / f"{key}.json"

    def get(self, sequence: str) -> dict[str, float] | None:
        path = self._path(sequence)
        if not path.exists():
            return None
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
            result = {field: float(values[field]) for field in self.FIELDS}
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in result.values()):
            return None
        return result

    def put(self, sequence: str, values: dict[str, float]) -> None:
        payload = {field: float(values[field]) for field in self.FIELDS}
        if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in payload.values()):
            raise ValueError("Invalid probabilities cannot be cached")
        path = self._path(sequence)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(path)
