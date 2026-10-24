from __future__ import annotations

import numpy as np


FEATURE_NAMES = (
    "p_plm",
    "p_dde",
    "absolute_difference",
    "mean_probability",
    "probability_product",
    "plm_confidence_distance",
    "dde_confidence_distance",
)


def make_meta_features(p_plm: np.ndarray, p_dde: np.ndarray) -> np.ndarray:
    p_plm = np.asarray(p_plm, dtype=float)
    p_dde = np.asarray(p_dde, dtype=float)
    if p_plm.shape != p_dde.shape:
        raise ValueError("PLM and DDE probability arrays must have the same shape")
    return np.column_stack(
        [
            p_plm,
            p_dde,
            np.abs(p_plm - p_dde),
            (p_plm + p_dde) / 2.0,
            p_plm * p_dde,
            np.abs(p_plm - 0.5),
            np.abs(p_dde - 0.5),
        ]
    )
