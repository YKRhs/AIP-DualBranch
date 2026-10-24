from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CONFIG = PACKAGE_ROOT / "configs" / "inference.local.json"
PUBLIC_CONFIG = PACKAGE_ROOT / "configs" / "inference.example.json"


def default_config_path() -> Path:
    """Return the private local config when present, otherwise the public template."""
    environment_path = os.environ.get("AIP_CONFIG")
    if environment_path:
        return Path(os.path.expandvars(environment_path)).expanduser().resolve()
    return LOCAL_CONFIG if LOCAL_CONFIG.exists() else PUBLIC_CONFIG


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve() if path else default_config_path()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    return config


def resolve_path(config: dict[str, Any], value: str) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    if path.is_absolute():
        return path
    return (Path(config["_config_dir"]) / path).resolve()


def configured_path(config: dict[str, Any], key: str) -> Path:
    return resolve_path(config, config["paths"][key])


def configured_paths(config: dict[str, Any], key: str) -> list[Path]:
    return [resolve_path(config, item) for item in config["paths"][key]]
