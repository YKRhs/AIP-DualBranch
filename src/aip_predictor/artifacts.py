from __future__ import annotations

import pickle
from pathlib import Path

import joblib

from .grande_compat import install_grande_compatibility


PLM_SCALER_FILE = "plm_scaler.joblib"
PLM_MODEL_FILE = "plm_grande.pkl"
DDE_SCALER_FILE = "dde_scaler.joblib"
DDE_MODEL_FILE = "dde_tabicl.pkl"
META_MODEL_FILE = "meta_lr.joblib"
MANIFEST_FILE = "manifest.json"


def required_artifacts(directory: str | Path) -> list[Path]:
    root = Path(directory)
    return [
        root / PLM_SCALER_FILE,
        root / PLM_MODEL_FILE,
        root / DDE_SCALER_FILE,
        root / DDE_MODEL_FILE,
        root / META_MODEL_FILE,
        root / MANIFEST_FILE,
    ]


def save_grande(model, path: str | Path) -> None:
    with Path(path).open("wb") as handle:
        pickle.dump(model, handle, protocol=5)


def load_grande(path: str | Path):
    install_grande_compatibility()
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def save_joblib(value, path: str | Path) -> None:
    joblib.dump(value, path, compress=3)


def load_joblib(path: str | Path):
    return joblib.load(path)
