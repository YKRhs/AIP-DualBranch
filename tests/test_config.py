import json
from pathlib import Path

from aip_predictor.config import configured_path, load_config


def test_relative_paths_are_resolved_from_config_directory(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"paths": {"artifacts_dir": "../artifacts"}}),
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert configured_path(config, "artifacts_dir") == (tmp_path / "../artifacts").resolve()

