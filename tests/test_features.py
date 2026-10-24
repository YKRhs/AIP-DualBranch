import numpy as np
import pytest

from aip_predictor.dde import DIPEPTIDES, extract_dde, extract_dde_batch
from aip_predictor.fusion import make_meta_features


def test_dde_shape_and_dtype() -> None:
    feature = extract_dde("ACDEFG")
    assert feature.shape == (400,)
    assert feature.dtype == np.float32
    assert len(DIPEPTIDES) == 400
    assert np.isfinite(feature).all()


def test_single_residue_dde_is_zero() -> None:
    assert np.array_equal(extract_dde("A"), np.zeros(400, dtype=np.float32))


def test_dde_batch_shape() -> None:
    assert extract_dde_batch(["ACD", "WYS"]).shape == (2, 400)


def test_fusion_feature_definition() -> None:
    features = make_meta_features(np.array([0.8]), np.array([0.6]))
    expected = np.array([[0.8, 0.6, 0.2, 0.7, 0.48, 0.3, 0.1]])
    assert features == pytest.approx(expected)

