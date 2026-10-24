from pathlib import Path

import numpy as np

from aip_predictor.feature_cache import (
    PLMFeatureCache,
    PredictionCache,
    build_plm_signature,
)


def test_feature_cache_round_trip(tmp_path: Path) -> None:
    cache = PLMFeatureCache(tmp_path, "model-version", 4)
    feature = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    assert cache.get("ACDE") is None
    cache.put("ACDE", feature)
    assert np.array_equal(cache.get("ACDE"), feature)
    assert cache.get("ACDF") is None


def test_signature_changes_with_checkpoint_metadata(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "fold.pt"
    checkpoint.write_bytes(b"first")
    first = build_plm_signature(base, [checkpoint], "TNFA", 384)
    checkpoint.write_bytes(b"second-version")
    second = build_plm_signature(base, [checkpoint], "TNFA", 384)
    assert first != second


def test_prediction_cache_round_trip(tmp_path: Path) -> None:
    cache = PredictionCache(tmp_path, "model-version")
    values = {"p_plm": 0.7, "p_dde": 0.8, "p_final": 0.9}
    assert cache.get("ACDE") is None
    cache.put("ACDE", values)
    assert cache.get("ACDE") == values
